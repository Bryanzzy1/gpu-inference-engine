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

## Run it

```bash
cd python && python download_data.py BTCUSDT 2026-06-27 && cd ..
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
