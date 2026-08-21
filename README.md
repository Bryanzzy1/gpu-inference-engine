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
| GPU naive | copy in, launch, copy out per event | overtakes CPU at batch 128-256 |
| GPU + CUDA Graphs | launch sequence recorded, replayed | no win on one tiny kernel (WDDM) |
| GPU persistent megakernel | resident kernel + lock-free pinned ring | ~10x lower p999 than naive |

The CPU wins at batch 1 (in-cache, no launch or PCIe); the GPU only overtakes once a
larger batch amortizes the fixed cost. On top: a jitter autopsy (per-stage tail) and a
closed-loop SLA controller that holds a 150 us p99 on 20/20 ticks under bursty load.

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

## Scaling the dataset: a week of ticks

The results above come from one day (473k trades). Re-running on a **week** (2026-06-21 to
06-27, **7.1M trades, 614 MB**, retrained model `test_acc 0.8168` vs a 0.5184 baseline)
changes what the numbers say in two honest ways.

**The tails get much cleaner, and the GPU's marginal win disappears.** Unpaced p999, us:

| Backend, batch | one day | one week |
| --- | --- | --- |
| cpu, 1 | 24.8 | 1.1 |
| cuda-naive, 1 | 3318.8 | 173.2 |
| cpu, 256 | 1349.8 | 319.1 |
| cuda-naive, 256 | 1017.0 | 422.0 |

On one day cuda-naive won p999 at batch 256 (1017 vs the CPU's 1349). On a week the CPU's
tail tightens to 319 and it **wins batch 256 too**; the CPU now holds 53 of 54 cells, and
the GPU's whole top-right region collapses to a single noisy cell. The lesson: that
crossover was small-sample tail noise, not a real regime, and more data is what exposed it.
A short, contention-affected run produces fat tails; 20k iterations over a 7.1M-row stream
produce tight, trustworthy ones. (Absolute latencies still drift run to run on a WDDM laptop
sharing its GPU with the display, so the frontier's value is the shape and its stability,
both of which improve with volume.)

**At batch 1 the CPU does not lose its edge to volume.** p50 stays ~0.4 us even over 7.1M
rows, because tick inference streams sequentially and the prefetcher keeps it cache-friendly
regardless of total size. Volume erodes a random-access advantage, not a streaming one.

**The stability result is robust to the dataset.** Re-running `queue_load` on the week's
service times gives the same shape: cuda-naive at batch 1 caps near 20k rows/s and collapses
above it, batch 256 holds ~240 us past 1M rows/s. That the queue conclusion does not move
under 15x more data is the point of checking. Artifacts: `results/frontier_week*.png`,
`results/queue_load_week.png`.

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

Layout: `src/io` (parse, features), `src/cpu` (model, harness, router, controller),
`src/gpu` (CUDA backends), `src/bench` (drivers), `python` (train, plot).
