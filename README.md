# GPU Inference Engine for Trading Signals

[![ci](https://github.com/Bryanzzy1/gpu-inference-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Bryanzzy1/gpu-inference-engine/actions/workflows/ci.yml)

Turns a market tick stream into a price-direction prediction four ways (CPU + three
GPU paths) and measures which is fastest at the **p999 tail**, not the mean, because
in trading one slow inference in a thousand is a missed quote.

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

## Run it

```bash
cd python && python download_data.py BTCUSDT 2026-06-27 && cd ..
cmake -S . -B build && cmake --build build     # CPU targets + self-checks
./build/test_ring && ./build/test_router && ./build/test_controller && ./build/test_latency_stats
```

`bench_hist` (CPU, no GPU) dumps the full latency distribution as a histogram CSV plus
extended tail stats (p9999, IQR, stddev); `python python/plot_hist.py` plots it.

GPU backends build with `nvcc -arch=sm_89` (load `vcvars64.bat` first on Windows).
All build + run commands: [BENCHMARKS.md](BENCHMARKS.md#reproduce). Layout: `src/io`
(parse, features), `src/cpu` (model, harness, router, controller), `src/gpu` (CUDA
backends), `src/bench` (drivers), `python` (train, plot).
