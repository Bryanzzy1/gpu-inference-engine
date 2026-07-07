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

## Milestones

| Milestone | What it adds | Status |
| --- | --- | --- |
| **M1** | CPU tick→prediction pipeline + latency harness recording full distributions and jitter (no CUDA) | In progress |
| M2 | CUDA fundamentals in a separate learning repo; ring-buffer and persistent-kernel patterns | |
| M3 | GPU feature kernels; first 2D (batch × load) frontier: CPU vs naive GPU vs CUDA Graphs | |
| M4 | GPU inference path (cuBLAS + 1 fused kernel), then the **persistent megakernel** and the **SLA controller** | |

Each milestone works on its own. M3 produces the first frontier; M4 adds the
persistent megakernel and the controller.

## Repository layout

```
gpu-inference-trading/
  CMakeLists.txt      # build (Day 1: hello-CMake target)
  src/                # C++ source (parser, features, harness — M1)
  python/             # data download/inspection now; model training later
  data/               # raw CSVs (gitignored — re-fetch with download_data.py)
  README.md
```

## Data

Source: **Binance public data** — <https://data.binance.vision/>
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

## Features implemented so far

The CPU pipeline reads trades, computes features over a rolling window, and
writes a feature + label CSV for training.

- **parse_trades** reads an aggTrades CSV into a `vector<Trade>`.
- **build_features** computes four features per trade and a label:
  - mid price (last trade price)
  - rolling volatility (std dev of the last N log returns)
  - trade-sign imbalance (signed volume over total volume, in [-1, 1])
  - trade intensity (trade count in the last time window)
  - label: 1 if price rose H trades ahead, else 0

```bash
./build/build_features data/BTCUSDT-aggTrades-2026-06-27.csv data/features.csv
python python/plot_features.py data/features.csv
```

Notes: features use only past trades; the label uses a future trade. Trade
intensity uses a fixed-size timestamp ring that caps at 1024, which clips about
0.1% of rows during the busiest bursts.
