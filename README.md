# GPU Inference Engine for Trading Signals

[![ci](https://github.com/Bryanzzy1/gpu-inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Bryanzzy1/gpu-inference-engine/actions/workflows/ci.yml)

Turns a market tick stream into a price-direction prediction, four ways (CPU +
three GPU paths), and measures which is fastest at the **p999 tail**, not the mean.
For trading the tail is what matters: one slow inference in a thousand is a missed quote.

> At batch size 1, when does a resident-model GPU beat an in-cache CPU, and which
> cost (launch, PCIe, occupancy, contention) decides it?

## Headline result (RTX 4060, measured)

- **CPU wins at batch 1.** In-cache, no launch, no PCIe. Nothing beats it there.
- **GPU overtakes at batch 128-256**, once fixed launch+PCIe cost amortizes over the batch.
- **Persistent megakernel cuts p999 ~10x vs naive** by removing the per-event launch.
- **CUDA Graphs do not help** a single tiny kernel under Windows WDDM.
- **Closed-loop controller holds a 150 us p99 SLA on 20/20 ticks** under bursty load.

Full numbers and charts: [BENCHMARKS.md](BENCHMARKS.md). Design rationale:
[docs/DESIGN.md](docs/DESIGN.md).

## The four backends

One `InferenceEngine` interface, timed on identical input.

| Backend | Mechanism | Result |
| --- | --- | --- |
| CPU reference | cache-hot, pinned busy-poll | wins at batch 1 |
| GPU naive | copy in, launch, copy out per event | pays launch + PCIe each event |
| GPU + CUDA Graphs | launch sequence recorded once, replayed | no win on one tiny kernel |
| GPU persistent megakernel | one resident kernel fed via a lock-free pinned ring | ~10x lower p999 than naive |

## Pipeline

`parse ticks -> microstructure features -> tiny MLP (4-16-16-1) -> price-up logit`,
timed at microsecond granularity. The model is a fixed workload for comparing
backends, not a trading signal. On top: a **2D frontier** (batch x arrival rate)
of which backend wins each cell, a **jitter autopsy** (per-stage tail), and a
**closed-loop SLA controller** that picks batch+backend to hold a p99 target.

## Layout

```
include/   headers (flat)
src/io/    parser, features, CSV builders
src/cpu/   model, latency harness, router, controller, match check
src/gpu/   CUDA backends: naive, graph, persistent, weights
src/bench/ benchmark + frontier + autopsy + controller drivers
src/tools/ self-checks (ring, stage-timing, router, controller)
python/    data download, model training, plotting
data/      raw CSVs (gitignored)
```

## Quick start

Get the data, build the CPU targets, run the self-checks:

```bash
cd python && python download_data.py BTCUSDT 2026-06-27 && cd ..
cmake -S . -B build && cmake --build build
./build/test_ring && ./build/test_router && ./build/test_controller
```

Train the model (needs torch): `python python/train_model.py data/features.csv --out data/model`.

The GPU backends compile with `nvcc` (not CMake: on Windows nvcc needs the MSVC host
compiler, so load `vcvars64.bat` first, and `-arch=sm_89` targets the Ada GPU). Every
GPU driver first checks its output matches the CPU forward pass, then times both.
All build + run commands are in [BENCHMARKS.md](BENCHMARKS.md#reproduce).

## Data

One day of Binance spot **aggTrades, BTCUSDT** (trade prints, no order book).
`timestamp` is microseconds; `is_buyer_maker` gives the trade sign, so buy/sell
pressure without a book.

## Notes

- **WDDM watchdog:** the persistent kernel busy-spins, so Windows' 2s TDR watchdog
  resets a long run. Drivers keep the live window short. Production wants Linux or TCC mode.
- **Batch axis is GPU-only:** on the CPU a "batch of N" is just N sequential calls, so
  `frontier_cpu` is a control that verifies the sweep machinery, not a real result.
