# GPU Inference Engine for Trading Signals

A C++ / CUDA inference engine for market tick data. It reads a stream of ticks,
computes microstructure features, runs a small ML model's forward pass, and emits
short-horizon price-direction predictions. Every stage is timed at microsecond
granularity across a CPU path and several GPU paths.

The metric is **p999 latency and jitter**, not mean throughput. For trading, the
tail is what matters: one p999 spike is a missed quote or a bad fill, so the mean
hides the cases you care about.

The question it answers:

> At batch size one (tick by tick), when does a GPU that keeps the model resident
> and streams ticks in beat a CPU that stays in cache, and which cost - kernel
> launch, PCIe transfer, occupancy, or contention - decides where the line falls?

The output is a **frontier** over (batch size × arrival rate): which backend has
the lowest p999 in each region. On top of that, a closed-loop controller adjusts
batch size at runtime to hold a latency target.

## The four backends

All four implement the same `InferenceEngine` interface, so the harness runs them
on identical input.

| Backend | Description |
| --- | --- |
| **CPU reference** | cache-friendly, SIMD, pinned busy-poll thread. The baseline, hard to beat at batch size one. |
| **GPU request-response (naive)** | copy in, launch, copy out per event. Pays the launch and PCIe cost each time. |
| **GPU + CUDA Graphs** | the launch sequence recorded once and replayed, which cuts launch overhead. |
| **GPU persistent megakernel** | one resident kernel fed tick by tick through a lock-free pinned ring. No per-event launch. |

## Repository layout

```
gpu-inference-trading/
  CMakeLists.txt      # build
  include/            # headers (flat)
  src/
    io/               # parser, features, CSV builders
    cpu/              # CPU model forward pass, latency harness, match check
    gpu/              # CUDA backends: naive, graph, persistent, weights
    bench/            # benchmark drivers per backend
    tools/            # hello smoke test, ring self-check
  python/             # data download/inspection and model training
  data/               # raw CSVs (gitignored, re-fetch with download_data.py)
  README.md
```

## Data

Source: **Binance public data**, <https://data.binance.vision/>
(format & scripts: [`binance/binance-public-data`](https://github.com/binance/binance-public-data)).

We use one day of **spot aggTrades for BTCUSDT**. aggTrades columns (no header
row in the file):

```
agg_trade_id, price, qty, first_trade_id, last_trade_id,
timestamp, is_buyer_maker, is_best_match
```

- `timestamp` is in **microseconds** since epoch.
- `is_buyer_maker` gives the trade sign: when the buyer is the maker, the
  aggressor was a seller (sell pressure), and vice versa. That lets us compute
  buy/sell pressure **without an order book**, which is enough for M1.

### Reproduce the dataset

```bash
cd python
python download_data.py BTCUSDT 2026-06-27   # downloads, sha256-verifies, unzips
python inspect_data.py ../data/BTCUSDT-aggTrades-2026-06-27.csv
```

## Build (C++)

Requires a C++17 compiler and CMake ≥ 3.16.

```bash
cmake -S . -B build
cmake --build build
./build/hello
```

### GPU backend (nvcc)

The GPU backend builds separately from the CMake targets, because `nvcc` on Windows
needs the MSVC host compiler, not the MinGW g++ the CMake build uses. Load the MSVC
environment (`vcvars64.bat`) first, then compile everything in one `nvcc` call:

```bash
nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
  src/bench/bench_gpu.cpp src/gpu/gpu_model.cu src/io/parser.cpp src/io/features.cpp \
  src/cpu/model.cpp src/cpu/latency.cpp -o build/bench_gpu.exe
./build/bench_gpu.exe data/BTCUSDT-aggTrades-2026-06-27.csv data/model
```

`-arch=sm_89` targets the local GPU (Ada) natively so the driver loads SASS directly.
`bench_gpu` first checks the GPU forward pass matches the CPU one, then times both on
the same harness.

### Persistent megakernel (nvcc)

The persistent backend launches one kernel that never exits: it spins on a lock-free
ring in pinned host memory, runs the forward pass on each input the host enqueues, and
writes the logit back, so it pays no per-event launch cost. Build it alongside the
naive path:

```bash
nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
  src/bench/bench_persistent.cpp src/gpu/gpu_model.cu src/gpu/persistent_model.cu \
  src/io/parser.cpp src/io/features.cpp src/cpu/model.cpp src/cpu/latency.cpp -o build/bench_persistent.exe
./build/bench_persistent.exe data/BTCUSDT-aggTrades-2026-06-27.csv data/model
```

`bench_persistent` checks CPU, naive GPU, and persistent GPU all agree, then times all
three. On this dataset the persistent kernel cuts p999 to roughly a tenth of the naive
path by removing launch overhead, though the in-cache CPU still wins at batch size one.

Windows note: the kernel busy-spins, so the 2s WDDM TDR watchdog resets the GPU if it
runs too long. The backend exposes `start()` / `stop()` and the benchmark keeps that
window under 2s and caps the persistent iteration count. A production persistent kernel
wants the GPU in TCC mode or on Linux, where there is no display watchdog.

### CUDA Graphs (nvcc)

The graph backend records the copy-launch-copy sequence once and replays it with a
single `cudaGraphLaunch` per call, so the driver schedules the sequence once instead of
on every event. Build and run:

```bash
nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
  src/bench/bench_graph.cpp src/gpu/gpu_model.cu src/gpu/graph_model.cu src/gpu/gpu_weights.cu \
  src/io/parser.cpp src/io/features.cpp src/cpu/model.cpp src/cpu/latency.cpp -o build/bench_graph.exe
./build/bench_graph.exe data/BTCUSDT-aggTrades-2026-06-27.csv data/model
```

Measured finding: for this workload the graph is correct but not faster than the naive
path. The graph holds a single tiny kernel, so there is almost no per-launch cost to
amortize, and on WDDM the per-call synchronization dominates both paths. CUDA Graphs pay
off on many-kernel pipelines or on Linux, not on a single fused kernel behind a per-call
sync. The launch-overhead win in this project comes from the persistent megakernel, not
graphs.

### All four backends in one run (nvcc)

`bench_all` is the defensible latency table: it loads all four backends, gates each on
agreeing with the CPU forward pass within tolerance, then times them on the same
harness at batch size one. Use this for the headline comparison.

```bash
nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
  src/bench/bench_all.cpp src/gpu/gpu_model.cu src/gpu/graph_model.cu src/gpu/gpu_weights.cu \
  src/gpu/persistent_model.cu src/io/parser.cpp src/io/features.cpp src/cpu/model.cpp \
  src/cpu/latency.cpp -o build/bench_all.exe
./build/bench_all.exe data/BTCUSDT-aggTrades-2026-06-27.csv data/model
```

It prints a per-backend p50/p99/p999 table in microseconds with the transfer note per
backend. No backend's numbers are shown unless it first matches the CPU logit.

## The 2D frontier

The headline artifact: a sweep over `(batch size x arrival rate)` recording each
backend's p999, then a heatmap of which backend wins each cell. `frontier.hpp` holds
the cell type, the sweep axes, and the CSV writer that every driver shares.

`frontier_cpu` sweeps the CPU backend alone. It builds and runs with no GPU, so the
sweep loop, batching, arrival-rate pacing, and CSV format are all verified on the CPU
path before the GPU work. The four-backend GPU sweep reuses the same header and axes on
a machine with nvcc.

```bash
cmake --build build --target frontier_cpu
# frontier_cpu <trades.csv> <model-stem> <out.csv> [iters]
./build/frontier_cpu data/BTCUSDT-aggTrades-2026-06-27.csv data/model data/frontier.csv
python python/plot_frontier.py data/frontier.csv   # winner map + per-backend p999 maps
```

Axes default to batch `1..256` and rate `0` (unpaced) through `250 kHz`. The plot
writes `frontier_winner.png` (the winning-backend heatmap) and one p999 heatmap per
backend.

Caveat: on the CPU a batch of N is just N sequential `forward()` calls, so its batch
axis scales linearly and is not a real batching result. `frontier_cpu` exists to verify
the sweep, pacing, and CSV machinery without a GPU. The batch axis is only meaningful in
the GPU sweep, where a batch is one kernel over N rows.
