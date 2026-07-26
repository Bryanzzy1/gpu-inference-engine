# Benchmarks

Latency of one tick-to-prediction inference, batch size one, measured on the
`bench_all` runner in a single process so every number below comes from one run.
The goal is a number that is defensible in an interview, not a flattering one.

## Environment

- CPU: Intel Core Ultra 9 185H
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, driver 581.32, max SM clock 3105 MHz, WDDM mode
- CUDA: Toolkit 13.3 (V13.3.73)
- Compiler: `nvcc -O2 -arch=sm_89 -std=c++17`, host compiler MSVC cl.exe (VS 2022 Build Tools) at -O2
- OS: Windows 11

All four backends in this table are compiled through the one nvcc/MSVC -O2 host
path, so the comparison shares a toolchain. The standalone CMake CPU build uses
MinGW g++ 15.2.0 -O3 and is not what this table measures.

## Method

- The timer is `std::chrono::steady_clock` around the whole `forward()` call, so it
  measures end-to-end host-observed latency **including** the H2D and D2H copies and
  the sync. That is the tick-to-prediction latency, not compute only.
- 1000 warmup iterations run first and are discarded, so cold caches and GPU clock
  ramp do not skew the tail.
- Percentiles come from the full sorted sample (nearest-rank), not a summary.
- Batch size one throughout, which is the latency case that matters for a quote.
- Device-side kernel time is measured separately with Nsight, not with steady_clock.
  Using a host clock for end-to-end and GPU timestamps for kernel-only is the point:
  each tool answers the question it is right for.

Every backend is checked against the CPU logit before timing: naive and graph match
to 6.05e-6, persistent to 1.19e-6.

## Results

| Backend | Iters | p50 (us) | p99 (us) | p99.9 (us) | Transfer in timing |
| --- | --- | --- | --- | --- | --- |
| CPU baseline | 200,000 | 0.30 | 0.90 | 13.10 | n/a (host only) |
| CUDA standard (naive) | 200,000 | 64.00 | 422.30 | 2519.70 | included (H2D + D2H) |
| CUDA Graphs | 200,000 | 134.90 | 1024.30 | 2762.50 | included (H2D + D2H) |
| Persistent megakernel | 100,000 | 5.90 | 19.20 | 207.10 | included (zero-copy) |

Persistent is capped at 100,000 iterations because it busy-spins and must stay under
the 2s Windows WDDM TDR watchdog. The other three run the full 200,000.

Plain reading: the CPU wins at batch size one. It stays in cache and never pays a
launch or a PCIe crossing, so 0.30 us p50 beats every GPU path. Among GPU backends
the persistent megakernel is far ahead because it removes the per-event launch, cutting
p99.9 from 2519.70 to 207.10 us (about 12x) and p50 from 64.00 to 5.90 us. CUDA Graphs
are correct but not faster here, for the reason the profile makes explicit.

## Nsight profile of the naive backend

Nsight Systems trace of the standard CUDA path, per inference:

| Stage | Time | Where |
| --- | --- | --- |
| kernel execution | 3.17 us (median, stddev 18 ns) | GPU |
| H2D copy | 0.44 us | GPU |
| D2H copy | 1.68 us | GPU |
| cudaLaunchKernel | 7.62 us (median) | host API |
| cudaMemcpy (blocking) | 21.86 us (median) | host API |

Real GPU work is about 5.3 us per inference. End-to-end p50 is 64 us. So roughly
90% of the latency is host-side CUDA API and WDDM driver and sync cost, not kernel
execution and not PCIe bandwidth (the payload is 16 bytes up and 4 bytes down, so
the copy time is per-call latency, not throughput). This is why paying the launch
once (persistent) wins and why CUDA Graphs, which only trim host-side launch cost,
do not close the gap on their own.

GPU clocks were not locked (laptop GPU, unsupported). Sampling during the run showed
them varying 210 to 2130 MHz, dominated by the GpuIdle reason (the GPU idles between
tiny launches), with no thermal or power throttling (HwSlowdown never asserted, temp
steady near 56 C). Kernel execution time is stable regardless (stddev 18 ns) and the
end-to-end latency is host-bound, so the metric is insensitive to GPU clock.

## Reproduce

```bash
# GPU backends need nvcc + the MSVC host env (vcvars64.bat), -arch=sm_89 for Ada.
nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
  src/bench_all.cpp src/gpu_model.cu src/graph_model.cu src/persistent_model.cu \
  src/gpu_weights.cu src/parser.cpp src/features.cpp src/model.cpp src/latency.cpp \
  -o build/bench_all.exe
./build/bench_all.exe data/BTCUSDT-aggTrades-2026-06-27.csv data/model 1000 200000

# Nsight profile of the naive path:
nsys profile --trace=cuda --stats=true -o prof_naive \
  ./build/bench_gpu.exe data/BTCUSDT-aggTrades-2026-06-27.csv data/model 200 3000
```
