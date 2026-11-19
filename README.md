# GPU Inference Engine for Trading Signals

[![ci](https://github.com/Bryanzzy1/gpu-inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Bryanzzy1/gpu-inference-engine/actions/workflows/ci.yml)

Turns a market tick stream into a price-direction prediction four ways (CPU and three
GPU paths) and measures which is fastest at the p999 tail.

Pipeline: `parse ticks -> microstructure features -> tiny MLP (4-16-16-1) -> logit`. The
main artifact is a 2D frontier over (batch size, arrival rate) showing which backend wins
p999 in each cell. The model is a fixed workload for comparing backends, not a trading
signal.

## Backends (RTX 4060)

One `InferenceEngine` interface, timed on identical input.

| Backend | Mechanism | Result |
| --- | --- | --- |
| CPU reference | cache-hot, pinned busy-poll | wins at batch 1 |
| GPU naive | copy in, launch, copy out per event | narrows the gap at large batch |
| GPU + CUDA Graphs | launch sequence recorded, replayed | no win on one tiny kernel |
| GPU persistent megakernel | resident kernel + lock-free pinned ring | 10x lower p99, 7x lower p999 than naive |

## Latency (batch 1)

Median of a 5-run benchmark on one day of trades (microseconds):

| Backend | p50 | p99 | p999 |
| --- | --- | --- | --- |
| cpu | 0.5 | 1.1 | 2.0 |
| persistent | 7.3 | 26.7 | 61 |
| cuda-naive | 124 | 286 | 440 |
| cuda-graphs | 175 | 363 | 551 |

## Day vs week

The numbers above come from one day (473k trades). On a week (7.1M trades) the CPU wins
53 of 54 frontier cells and the GPU region shrinks to one cell. `compare_frontier.py`
reports 5 of 54 cells flip winner between the day and week runs, so the CPU wins the p999
grid. CPU p50 stays around 0.5 us across both datasets.

A single p999 varies between runs, so `bench_variance.py` runs the benchmark five times and
reports the spread (`results/variance_day.csv`, `results/variance_week.csv`):

| Backend | p50 spread | p999 spread |
| --- | --- | --- |
| cpu | 1.2x | 2.2x day, 1.7x week |
| cuda-naive | 1.7x | 1.1x day, 1.8x week |
| persistent | 1.1x | 2.1x day, 1.6x week |

The medians are steady across runs and across both datasets. The p999 moves more between
runs, so it is reported as a spread over five runs.

## Offered load

The latency numbers above feed one input at a time. `queue_load` replays the measured
per-batch service times under a Poisson arrival stream (`include/queue_sim.hpp`) to show
throughput under load. A batching server has a capacity of `batch / service`, and load
above that capacity backs the queue up.

| Backend, batch | Capacity | p99 at 50k rows/s | p99 at 1M rows/s |
| --- | --- | --- | --- |
| cuda-naive, 1 | 19k rows/s | 6364 us | overloaded |
| cuda-naive, 256 | 2.9M rows/s | 174 us | 174 us |
| cpu, 8 | 3.6M rows/s | 4.2 us | 4.4 us |

At batch 1 cuda-naive caps near 19k rows/s. At batch 256 it holds 174 us past 1M rows/s.
Pick the smallest batch whose capacity clears the load. Only the arrivals are simulated.
See `results/queue_load.png`.

## Data

`python/download_data.py` ingests Binance spot aggTrades over a date range: it fetches and
unzips each UTC day and concatenates them into one combined CSV (the week of 2026-06-21 to
06-27 is 7.1M trades, 614 MB). Microsecond timestamps, 8 columns, no order-book depth. The
C++ parser reads the whole file once and parses fields with `std::from_chars`, with no
per-field allocation.

## Run it

```bash
cd python && python download_data.py BTCUSDT 2026-06-27 && cd ..              # one day
cd python && python download_data.py BTCUSDT 2026-06-21 2026-06-27 && cd ..   # a week, one combined CSV
cmake -S . -B build && cmake --build build     # CPU targets + self-checks
./build/test_ring && ./build/test_router && ./build/test_controller && ./build/test_latency_stats
```

`bench_hist` (CPU, no GPU) writes the full latency distribution as a histogram CSV plus tail
stats (p9999, IQR, stddev); `python python/plot_hist.py` plots it.

GPU backends build with `nvcc` (load `vcvars64.bat` first on Windows so nvcc finds the MSVC
host compiler):

```bash
nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude src/bench/bench_all.cpp src/gpu/*.cu \
  src/io/parser.cpp src/io/features.cpp src/cpu/model.cpp src/cpu/latency.cpp -o build/bench_all
./build/bench_all data/BTCUSDT-aggTrades-2026-06-27.csv data/model
```

Run-to-run spread and frontier reproducibility:

```bash
python python/bench_variance.py ./build/bench_all data/...csv data/model out.csv 5
python python/compare_frontier.py results/frontier.csv results/frontier_week.csv
```

Layout: `src/io` (parse, features), `src/cpu` (model, harness, router, controller),
`src/gpu` (CUDA backends), `src/bench` (drivers), `python` (train, plot).
