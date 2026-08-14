# Talking points

What to say about this project, and the follow-up questions to be ready for. Grouped
by theme. Every claim here is backed by code or a measured number in this repo.

## The one-liner

A C++/CUDA engine that runs a tiny trading model four ways (CPU + three GPU paths) and
measures which wins at the **p999 tail latency**, not the mean, then routes and adapts
live to hold a latency SLA. The interesting result is not a speedup; it is *where* each
backend wins and *why*.

## 1. Why tail latency, not throughput

- A quote is placed against a predicted move. A latency spike on 1 event in 1000 means a
  stale quote = missed fill or adverse selection. The mean hides that; the tail is the risk.
- So every number is a full distribution: p50/p99/p999/**p9999**/max, plus IQR and stddev.
- **Follow-up "why p9999?":** at 100k events/sec, the p999 spike happens ~100x/second, so
  the 1-in-10000 point is what a tight SLA actually feels. `bench_hist` reports it and
  dumps the whole histogram so you see the shape, not one number.

## 2. The four backends and the crossover

- **CPU** wins at batch 1: in-cache, no kernel launch, no PCIe crossing. ~0.3 us p50.
- **GPU naive** pays launch + PCIe every event; loses badly at batch 1 (~64 us p50).
- **CUDA Graphs** record the launch sequence once and replay it. **No win here** and that
  is a real finding: one tiny fused kernel has almost no host launch cost to amortize, and
  Windows WDDM's per-call sync dominates both paths.
- **Persistent megakernel** keeps one kernel resident, fed by a lock-free ring; no
  per-event launch. Cuts p999 ~10x vs naive. Still loses to the CPU at batch 1.
- **The crossover** is a 2D frontier over (batch size x arrival rate): CPU wins the grid
  until batch 128-256, where the GPU amortizes its fixed cost and takes over.
- **Follow-up "so the GPU lost?":** at batch 1, yes, and that is the honest answer. The
  point was to find the crossover, not to force a speedup. Knowing where the GPU is NOT
  worth it is as valuable as knowing where it is.

## 3. The persistent megakernel (hardest CUDA)

- One kernel launched once, spinning on a lock-free single-producer/single-consumer ring
  in pinned host memory; host is the producer, GPU is the consumer.
- Correctness is the whole game: `__threadfence_system` + acquire/release ordering on the
  head/tail indices so the GPU never reads a half-written slot and the host never
  overwrites an unconsumed one. It is the CUDA analogue of a C++ acquire/release SPSC queue.
- **Follow-up "why is this hard?":** no debugger story for a data race across the PCIe
  boundary; it fails rarely and nondeterministically. Prototyped in a separate repo first,
  then correctness-gated against the CPU reference on every run.
- **Follow-up "WDDM watchdog?":** a busy-spinning resident kernel trips Windows' 2s TDR
  watchdog, so the live window is capped. Production wants Linux or TCC mode.

## 4. The jitter autopsy (the differentiator)

- Most GPU write-ups give one speedup number. This one decomposes each inference into
  H2D / launch / compute / D2H with CUDA events and reports a per-stage tail.
- **Finding:** no single stage owns the tail; all four blow out together at p999. That
  flat, everywhere-heavy tail is the signature of WDDM scheduling jitter, not one slow op.
- **The honest catch:** the `compute` bucket is inflated by the WDDM queue gap (time the
  command waits in the driver before running), so it is not pure kernel time. I state that
  rather than claim compute is 40 us when the kernel is ~3 us. Instrumentation has observer
  cost; the autopsy is for relative stage weight, not absolute latency.

## 5. Routing and the closed-loop controller

- **Static routing:** the frontier is a lookup table; dispatch each request to the backend
  that wins at the current (batch, rate). Open loop.
- **Closed loop:** measure live p99, adjust batch + backend to hold an SLA, with a headroom
  margin as hysteresis so it does not oscillate. Holds a 150 us p99 SLA on 20/20 ticks
  under a bursty schedule.
- **The best story here:** closing the loop on *real* p99 caught two bugs the static map and
  a simulated test both passed:
  1. **Route on the metric you hold.** It routed on p999 while the SLA was p99; those are
     different order statistics and rank backends differently. Fixed by selecting on p99.
  2. **Do not grow past the feasible batch.** Headroom growth climbed to a batch no backend
     can hold, then oscillated. Fixed by consulting the frontier's modeled p99 before growing.
- **The lesson:** a closed loop on real measurements reveals what an open loop and a clean
  simulation hide. That is the argument for measuring live.

## 6. Measurement discipline (what makes it defensible)

- Warm-up discarded; `steady_clock` (monotonic, portable) for end-to-end; CUDA events for
  per-stage; full sample kept, not a running mean; nearest-rank percentiles.
- Every GPU backend is gated against the CPU logit (< 1e-4) before any number is trusted,
  so a latency comparison can never be between two different computations.
- **Follow-up "how do I know the numbers are real?":** every result reproduces from a clone
  via the commands in BENCHMARKS.md; the host-side math (stats, histogram, router,
  controller) is unit-tested in CI.

## 7. Honest limitations (say these before you are asked)

- One laptop GPU under WDDM: the display driver's scheduling jitter dominates and the
  crossover sits far right. A datacenter GPU on Linux would move it left.
- The model's 0.826 accuracy is the **bid-ask bounce**, not alpha; the model is a fixed
  workload for comparing backends, not a signal. Always caveat it.
- CUDA Graphs and the locked-clock autopsy have known gaps (single-kernel workload;
  clock-lock needs admin). Stated, not hidden.

## Quick numbers to memorize

- CPU p50 ~0.3 us (batch 1). Naive GPU p50 ~64 us. Persistent p50 ~5.9 us, p999 ~207 us.
- Persistent cuts naive p999 ~12x (2520 -> 207 us).
- Crossover batch 128-256. SLA controller: 150 us p99, 20/20 ticks.
- Model 4-16-16-1, 0.826 test acc vs 0.529 baseline (bid-ask bounce).
