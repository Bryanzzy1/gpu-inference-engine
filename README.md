# GPU Inference Engine for Trading Signals

A C++/CUDA engine that turns a market tick stream into a short-horizon
price-direction prediction, timed at microsecond granularity across a CPU path and
three GPU paths.

The metric is **p999 latency**, not mean throughput. For trading the tail is what
matters: one slow inference in a thousand is a missed quote.

> At batch size one, when does a GPU that keeps the model resident beat a CPU that
> stays in cache, and which cost - launch, PCIe, occupancy, contention - decides it?

## The four backends

One `InferenceEngine` interface, timed on identical input.

| Backend | Mechanism | Result |
| --- | --- | --- |
| **CPU reference** | cache-hot, pinned busy-poll | wins at batch 1 |
| **GPU naive** | copy in, launch, copy out per event | pays launch + PCIe each event |
| **GPU + CUDA Graphs** | launch sequence recorded once, replayed | no win here (single tiny kernel, WDDM sync dominates) |
| **GPU persistent megakernel** | one resident kernel fed via a lock-free pinned ring | cuts p999 ~10x vs naive, still loses to CPU at batch 1 |

## Layout

```
include/   headers (flat)
src/io/    parser, features, CSV builders
src/cpu/   model forward pass, latency harness, match check
src/gpu/   CUDA backends: naive, graph, persistent, weights
src/bench/ benchmark drivers
src/tools/ smoke test, ring self-check
python/    data download/inspection, model training
data/      raw CSVs (gitignored)
```

## Data

One day of Binance spot **aggTrades, BTCUSDT** (trade prints, no order book), from
<https://data.binance.vision/>. `timestamp` is microseconds. `is_buyer_maker` gives
the trade sign, which yields buy/sell pressure without a book.

```bash
cd python
python download_data.py BTCUSDT 2026-06-27   # download, sha256-verify, unzip
python inspect_data.py ../data/BTCUSDT-aggTrades-2026-06-27.csv
```

## Build

CPU targets use CMake (C++17):

```bash
cmake -S . -B build && cmake --build build
```

GPU targets use `nvcc` directly (on Windows nvcc needs the MSVC host compiler, not
the MinGW g++ CMake uses). Load `vcvars64.bat` first. Shared across every command:

```bash
FLAGS="-O2 -arch=sm_89 -std=c++17 -Iinclude"                                 # -arch=sm_89 = local Ada GPU
COMMON="src/io/parser.cpp src/io/features.cpp src/cpu/model.cpp src/cpu/latency.cpp"
```

Every driver first checks its GPU output matches the CPU forward pass, then times both
on the same harness. Run each with `<exe> data/BTCUSDT-aggTrades-2026-06-27.csv data/model`.

**Naive** - copy in, launch, copy out per event. The baseline GPU path.
```bash
nvcc $FLAGS src/bench/bench_gpu.cpp src/gpu/gpu_model.cu $COMMON -o build/bench_gpu.exe
```

**CUDA Graphs** - records the copy-launch-copy sequence once, replays it per call. No
win here: one tiny kernel, so there is no launch overhead to amortize.
```bash
nvcc $FLAGS src/bench/bench_graph.cpp src/gpu/gpu_model.cu src/gpu/graph_model.cu src/gpu/gpu_weights.cu $COMMON -o build/bench_graph.exe
```

**Persistent megakernel** - one kernel that never exits, fed via a lock-free pinned
ring, so no per-event launch. Cuts p999 ~10x vs naive.
```bash
nvcc $FLAGS src/bench/bench_persistent.cpp src/gpu/gpu_model.cu src/gpu/persistent_model.cu $COMMON -o build/bench_persistent.exe
```

**All four** - the headline table. Every backend gated on matching the CPU, timed together.
```bash
nvcc $FLAGS src/bench/bench_all.cpp src/gpu/gpu_model.cu src/gpu/graph_model.cu src/gpu/gpu_weights.cu src/gpu/persistent_model.cu $COMMON -o build/bench_all.exe
```

**Windows note:** the persistent kernel busy-spins, so the 2s WDDM TDR watchdog resets
the GPU if it runs too long. The driver keeps its live window under 2s. Production
wants TCC mode or Linux, where there is no watchdog.
