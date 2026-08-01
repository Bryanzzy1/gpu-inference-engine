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

## The 2D frontier (batch x arrival rate)

Batch 1 is one question; the frontier answers the other one: as the batch grows, one
kernel over N rows amortizes the fixed launch + PCIe cost the CPU never pays, so at
some batch the GPU should overtake the in-cache CPU. `frontier_all` sweeps batch in
{1,2,4,8,16,32,64,128,256} against arrival rate in {unpaced, 1k, 10k, 50k, 100k, 250k}
Hz, times all four backends per cell (20,000 iters, 1,000 warmup discarded), and gates
each GPU backend against the CPU batched reference before timing. The winner of a cell
is the lowest p99.9. This is one run, all four backends in one process, so it trades the
batch-1 isolation above for a real 2D sweep; the numbers below are that run.

Unpaced (rate 0) p99.9 latency in microseconds, per backend and batch:

| Batch | CPU | cuda-naive | cuda-graphs | persistent |
| --- | --- | --- | --- | --- |
| 1   | 24.8   | 3318.8 | 3759.3 | 1293.5 |
| 8   | 39.6   | 1381.4 | 1903.1 | 774.3  |
| 32  | 272.2  | 1365.4 | 1065.4 | 3179.9 |
| 64  | 443.0  | 1785.5 | 1286.7 | 5427.2 |
| 128 | 641.7  | 1256.7 | 1714.4 | 7485.5 |
| 256 | 1349.8 | 1017.0 | 1363.5 | 3000.0 |

The winner map: the CPU holds every cell except the top-right corner. cuda-naive takes
batch 256 unpaced (1017 vs the CPU's 1350 us) and batch 128 at 250k Hz, and wins several
of the batch-256 paced cells. In p50 the crossover is the same shape: the CPU's median
scales with batch (1.0 us at batch 1 to 96.7 us at batch 256) because a CPU "batch" is N
sequential forwards, while cuda-naive stays nearly flat (52.1 to 87.3 us) as the launch
cost spreads over the batch. They cross near batch 256.

The honest reading: on this WDDM laptop GPU with display contention, the in-cache CPU
wins the whole low-to-moderate-batch regime and the GPU only overtakes at large batch.
The crossover exists but sits far to the right; a dedicated GPU without the 2s watchdog
and without a display sharing the device would move it left. That is the same limitation
the batch-1 table shows, measured across two axes instead of one.

Persistent is the best GPU backend at batch 1 (p50 5.5 us) but degrades as the batch
grows: its single resident block processes the batch one row at a time, each with a
system-wide fence and a host handshake, so it is a batch-1 streaming tool, not a
throughput-batch tool. It also runs unpaced only: a host-paced resident kernel stays on
the GPU for iters/rate seconds and the 2s TDR watchdog resets it, so paced persistent is
not measurable here. Its window is capped to a row budget (hence fewer samples at large
batch, shown in the CSV `count` column).

Artifacts: `results/frontier.csv` (every cell) and `results/frontier_winner.png` (the
heatmap), with one p99.9 map per backend beside it.

## Reproduce

```bash
# GPU backends need nvcc + the MSVC host env (vcvars64.bat), -arch=sm_89 for Ada.
# Batch-1 table (all four backends, one process):
nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
  src/bench/bench_all.cpp src/gpu/gpu_model.cu src/gpu/graph_model.cu \
  src/gpu/persistent_model.cu src/gpu/gpu_weights.cu src/io/parser.cpp \
  src/io/features.cpp src/cpu/model.cpp src/cpu/latency.cpp -o build/bench_all.exe
./build/bench_all.exe data/BTCUSDT-aggTrades-2026-06-27.csv data/model 1000 200000

# 2D frontier (batch x arrival rate, all four backends):
nvcc -O2 -arch=sm_89 -std=c++17 -Iinclude \
  src/bench/frontier_all.cpp src/bench/frontier.cpp src/gpu/gpu_model.cu \
  src/gpu/graph_model.cu src/gpu/gpu_weights.cu src/gpu/persistent_model.cu \
  src/io/parser.cpp src/io/features.cpp src/cpu/model.cpp src/cpu/latency.cpp \
  -o build/frontier_all.exe
./build/frontier_all.exe data/BTCUSDT-aggTrades-2026-06-27.csv data/model results/frontier.csv 20000
python python/plot_frontier.py results/frontier.csv

# Nsight profile of the naive path:
nsys profile --trace=cuda --stats=true -o prof_naive \
  ./build/bench_gpu.exe data/BTCUSDT-aggTrades-2026-06-27.csv data/model 200 3000
```
