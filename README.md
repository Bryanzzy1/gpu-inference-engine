# GPU Inference Engine for Trading Signals

[![ci](https://github.com/Bryanzzy1/gpu-inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Bryanzzy1/gpu-inference-engine/actions/workflows/ci.yml)

Turns a market tick stream into a price-direction prediction four ways (CPU + three
GPU paths) and measures which is fastest at the **p999 tail**.

Pipeline: `parse ticks -> microstructure features -> tiny MLP (4-16-16-1) -> logit`.
The headline artifact is a **2D frontier** over (batch size x arrival rate) showing
which backend wins p999 in each cell. The model is a fixed workload for comparing
backends, not a trading signal.

## Backends and result (RTX 4060)

One `InferenceEngine` interface, timed on identical input.

| Backend | Mechanism | Result |
| --- | --- | --- |
| CPU reference | cache-hot, pinned busy-poll | **wins at batch 1** |
| GPU naive | copy in, launch, copy out per event | narrows the gap at large batch, no stable win |
| GPU + CUDA Graphs | launch sequence recorded, replayed | no win on one tiny kernel (WDDM) |
| GPU persistent megakernel | resident kernel + lock-free pinned ring | ~10x lower p999 than naive |

The CPU wins at batch 1 (in-cache, no launch or PCIe) and holds the p999 grid on this WDDM
laptop; a larger batch narrows the gap but any single-run crossover sits inside the
run-to-run noise (see "Scaling the dataset" below). On top: a jitter autopsy (per-stage
tail) and a closed-loop SLA controller that holds a 150 us p99 on 20/20 ticks under bursty
load.

## Under real load: batching is a stability requirement

The benchmarks above feed one input and wait for the result, so arrival rate never builds
a queue and batching looks like a latency tradeoff. `queue_load` replays the **measured**
per-batch service times under a Poisson arrival stream (a discrete-event model,
`include/queue_sim.hpp`) to show what the synchronous view hides: a batching server has a
throughput ceiling of `batch / service`, and any offered load above it makes the queue
explode.

| Backend, batch | Capacity | p99 at 50k rows/s | p99 at 1M rows/s |
| --- | --- | --- | --- |
| cuda-naive, 1 | 19k rows/s | 6364 us (overloaded) | overloaded |
| cuda-naive, 256 | 2.9M rows/s | 174 us | 174 us |
| cpu, 8 | 3.6M rows/s | 4.2 us | 4.4 us |

cuda-naive at batch 1 collapses above ~19k rows/s (p99 in **milliseconds**); at batch 256
it holds 174 us past 1M rows/s. The control law this implies: at a given offered
load, pick the **smallest** batch whose capacity clears the load. See
`results/queue_load.png`. Only the arrivals are simulated.

## Scaling the dataset, and what a single run can and cannot claim

The results above come from one day (473k trades). Re-running on a **week** (2026-06-21 to
06-27, **7.1M trades, 614 MB**, retrained model `test_acc 0.8168` vs a 0.5184 baseline)
splits into things that reproduce and one that did not.

**What holds up.**

- **The batch-256 GPU crossover does not reproduce.** On one day cuda-naive won p999 at
  batch 256; on the week the CPU holds 53 of 54 frontier cells and the GPU region shrinks to
  one cell. `compare_frontier.py` puts a number on it: 5 of 54 cells flip winner between the
  two runs. So do not lean on that crossover; on this hardware the CPU wins the p999 grid.
- **The CPU keeps its batch-1 edge under volume.** p50 stays ~0.4 us over 7.1M rows, because
  tick inference streams sequentially and the prefetcher stays effective regardless of size.
  Volume erodes a random-access advantage, not a streaming one.
- **The queue/stability result is dataset-independent.** `queue_load` on the week's service
  times gives the same shape: cuda-naive at batch 1 caps near 20k rows/s, batch 256 holds
  ~240 us past 1M rows/s.

**What did not hold up, and this is the real lesson.** A single run's **p999 is not
trustworthy** on this WDDM laptop. `bench_variance.py` runs the batch-1 benchmark five times
and reports the run-to-run spread (`results/variance_day.csv`, `variance_week.csv`):

| Backend | p50 spread | p999 spread |
| --- | --- | --- |
| cpu | 3.0x | 2.4x (day), 5.5x (week) |
| cuda-naive | 1.6x | **15.4x (day)**, 6.0x (week) |
| persistent | 1.2x | 3.4x (day), 6.3x (week) |

The **median is stable** (cpu p50 ~0.3 us, persistent ~6-7 us across runs); the **tail is
not** (cuda-naive p999 ranged 474 us to 7272 us over five day-runs). And more data did **not**
tighten it: cpu p999 has a median near 1.5 us on both datasets, so the earlier reading that a
week gave "cleaner tails" was one lucky run, not a volume effect. The honest rule this forces:
report p999 as a distribution over repeated runs, not a single number, or move to a GPU that
does not share a display. `compare_frontier` and `bench_variance` exist so that claim is
checkable, not asserted.

## Run it

```bash
cd python && python download_data.py BTCUSDT 2026-06-27 && cd ..              # one day
cd python && python download_data.py BTCUSDT 2026-06-21 2026-06-27 && cd ..   # a week, one combined CSV
cmake -S . -B build && cmake --build build     # CPU targets + self-checks
./build/test_ring && ./build/test_router && ./build/test_controller && ./build/test_latency_stats
```

`bench_hist` (CPU, no GPU) dumps the full latency distribution as a histogram CSV plus
extended tail stats (p9999, IQR, stddev); `python python/plot_hist.py` plots it.

GPU backends build with `nvcc` (load `vcvars64.bat` first on Windows so nvcc finds the
MSVC host compiler), e.g.:

```bash
nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude src/bench/bench_all.cpp src/gpu/*.cu \
  src/io/parser.cpp src/io/features.cpp src/cpu/model.cpp src/cpu/latency.cpp -o build/bench_all
./build/bench_all data/BTCUSDT-aggTrades-2026-06-27.csv data/model
```

Because a single p999 is noisy here, quantify it before trusting it:

```bash
python python/bench_variance.py ./build/bench_all data/...csv data/model out.csv 5   # run-to-run spread
python python/compare_frontier.py results/frontier.csv results/frontier_week.csv     # do winners reproduce?
```

Layout: `src/io` (parse, features), `src/cpu` (model, harness, router, controller),
`src/gpu` (CUDA backends), `src/bench` (drivers), `python` (train, plot).
