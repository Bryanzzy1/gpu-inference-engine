# Design: tail latency, the frontier, and the persistent kernel

This document covers the reasoning behind the project. The `README` is the
orientation; this is the detail.

## The central question

> At batch size one (tick by tick), can a persistent-kernel streaming GPU design
> close the p999/jitter gap that naive GPU inference has - and across the
> **(batch size × arrival rate)** plane, where does each backend have the lowest
> p999, and which cost (kernel launch, PCIe transfer, occupancy, contention)
> decides each boundary?

"Is the GPU faster" has a known answer. Where each design wins on the tail, and
why, does not - it depends on numbers you only get by building and measuring.

## Why the tail, not the mean

A strategy quotes against a predicted move. If inference latency spikes on the
999th event in a thousand, the quote is stale and the result is a missed fill or
adverse selection. The mean does not capture that, so every result here is a
distribution - p50 / p99 / p999 / max / IQR - and p999 is the number that
matters.

## The four backends (one interface)

All implement the same `InferenceEngine` so the harness drives them the same way.

| Backend | Mechanism | Expected behavior |
| --- | --- | --- |
| **CPU reference** | cache-friendly + SIMD, pinned busy-poll thread | wins at batch-1 low load; tight tail |
| **GPU request-response** | per event: copy in → launch → copy out | loses at batch-1 (launch + PCIe cost); fine at large batch |
| **GPU + CUDA Graphs** | recorded launch sequence, replayed | lower launch overhead; still pays transfer |
| **GPU persistent megakernel** | resident kernel polling a lock-free pinned ring | no per-event launch; transfer spread across the stream |

## The persistent megakernel

Instead of launching a kernel per event, one kernel stays resident on the GPU and
spins, reading ticks from a **lock-free single-producer/single-consumer ring in
pinned/mapped host memory** and writing predictions back through a second ring.
The host thread is the producer; the GPU is the consumer. Correctness depends on
memory fences and volatile/atomic head/tail indices, so the GPU never reads a
half-written slot and the host never overwrites an unconsumed one.

This removes the per-event launch cost and turns PCIe transfer into a continuous
stream instead of a per-event round trip. It is also the hardest part to get
right - see Risks.

## The frontier

The crossover between backends is a surface over two axes, not a point:

- **Batch size** - events per inference, from 1 upward.
- **Arrival rate / load** - ticks replayed at controlled inter-arrival times,
  including bursty load taken from the real Binance inter-arrival times.

The main artifact is a heatmap of the (batch × load) plane, colored by which
backend has the lowest p999, with each boundary annotated with the cost that
decides it.

## The jitter autopsy

The differentiating deliverable. Most GPU projects report a speedup and stop. This
one decomposes where the GPU tail comes from and tries to kill each source, as
controlled experiments rather than one number:

- **Launch scheduling** - isolated by comparing naive request-response vs CUDA
  Graphs vs the persistent kernel on the same work.
- **Clock boosting** - the GPU idles at a low clock and ramps under load, fattening
  early-sample latency. Locked with `nvidia-smi --lock-gpu-clocks` and the tail
  change is measured, not assumed.
- **PCIe transfer variance** - the copy itself has a tail; measured per stage.
- **Memory contention and ECC scrubbing** - background hardware activity that adds
  unpredictable spikes.

Each inference is split into H2D / launch / compute / D2H with CUDA events (GPU-side
timestamps), so the output is a per-stage tail distribution, not a single latency.
The harness built in M1 is the sensor; the autopsy extends it onto the GPU. The
honest finding "a pinned CPU thread has a tighter tail than any GPU path at batch-1"
is itself a result, and reporting it truthfully is the point.

## The SLA controller

A closed loop that measures the arrival rate at runtime and adjusts batch size to
hold a p99 target under bursty load. This uses the frontier rather than just
plotting it. Static routing (consult the measured frontier as a lookup table and
dispatch each request to the backend that wins at the current batch and load) is the
simple version; the closed loop that adapts live under drift is the capstone.

## Measurement

- Warm-up iterations discarded.
- Clock pinned / frequency locked where possible; `steady_clock` vs `rdtsc`
  choice stated.
- Full sample retained, not a running mean; p999 computed on a large sample.
- Latency split into H2D / launch / compute / D2H via Nsight Systems.
- `make bench` regenerates every number and chart from raw data on a fresh clone.

## Layering

Each layer works on its own, and the harder layers build on the simpler ones:

1. **Floor:** CPU + naive GPU + CUDA Graphs, the 2D frontier, the tail/jitter
   metric, the reproducible harness.
2. add the persistent megakernel.
3. add the SLA controller.

The project stands if it stops at the floor, and each later layer answers a
further question.

## Depth targets

In priority order:

1. **The persistent megakernel done for real**, not as a label. This is the primary
   CUDA teacher: persistent/grid-resident kernels, the CUDA memory model
   (`__threadfence`, volatile loads, acquire/release), mapped pinned memory, a
   lock-free SPSC ring across the host↔device boundary, and occupancy.
2. **The jitter autopsy as a first-class deliverable** (see above). This is the part
   peers skip and the main source of uniqueness.
3. **Stretch, only if the base kernel is solid: warp specialization** inside the
   persistent kernel - some warps ingest from the ring, some compute, some write
   results out, a producer/consumer pipeline inside one kernel. This is how
   production kernels (FlashAttention, Hopper pipelines) are built, and it teaches
   `__syncwarp`, cooperative groups, and shared-memory queues.

What this deliberately does not do: add more backends, chase model accuracy, or
build the controller before the frontier it rides on is measured and honest.

## Risks

- **The persistent kernel is the hardest part.** Lock-free host↔device sync from
  a zero-CUDA start is fiddly. It sits on top of a working floor rather than being
  a dependency, and a minimal persistent-kernel + ring prototype is built in the
  M2 learning repo before integration.
- **GPU jitter is hard to remove.** Launch scheduling, clock boosting, memory
  contention, and ECC scrubbing all add to it. A pinned CPU thread may have a
  tighter tail; if so, that is a result worth reporting, not a problem.
- **The model is not alpha.** The engineering is the point, not prediction quality.
