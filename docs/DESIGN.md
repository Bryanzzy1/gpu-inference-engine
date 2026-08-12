# Design: tail latency, the frontier, the persistent kernel

Why the project is built the way it is. README = orientation, this = rationale,
BENCHMARKS.md = numbers.

## The question

> At batch size 1, can a resident-kernel GPU close the p999 gap that naive GPU
> inference has, and across (batch size x arrival rate) where does each backend win
> p999, decided by which cost (launch, PCIe, occupancy, contention)?

"Is the GPU faster" has a known answer. Where each design wins on the tail, and why,
does not - you only get it by building and measuring.

## Why the tail, not the mean

A strategy quotes against a predicted move. If latency spikes on the 999th event in a
thousand, the quote is stale = missed fill or adverse selection. The mean hides that,
so every result is a distribution (p50/p99/p999/max/IQR) and p999 is the number.

## The four backends (one `InferenceEngine`)

| Backend | Mechanism | Behavior |
| --- | --- | --- |
| CPU reference | cache-friendly, pinned busy-poll | wins at batch 1; tight tail |
| GPU naive | per event: copy in, launch, copy out | loses at batch 1 (launch+PCIe); fine at large batch |
| GPU + CUDA Graphs | recorded launch sequence, replayed | trims host launch cost; no win on one tiny kernel |
| GPU persistent megakernel | resident kernel polling a lock-free pinned ring | no per-event launch; ~10x lower p999 than naive |

## The persistent megakernel

One kernel stays resident and spins, reading ticks from a lock-free single-producer /
single-consumer ring in pinned host memory and writing logits back. Host = producer,
GPU = consumer. Correctness rests on fences + volatile/atomic head/tail indices so the
GPU never reads a half-written slot and the host never overwrites an unconsumed one.
This removes per-event launch cost. It is the hardest part (see Risks).

## The frontier

The crossover is a surface, not a point, over two axes: batch size (events per
inference) and arrival rate (ticks replayed at controlled inter-arrival times). The
artifact is a heatmap of which backend has the lowest p999 per cell. Measured result:
CPU wins the grid until batch 128-256, where naive GPU overtakes.

## The jitter autopsy

The differentiating deliverable. Most GPU projects report a speedup and stop; this one
decomposes where the GPU tail comes from. Each inference is split into H2D / launch /
compute / D2H with CUDA events, giving a per-stage tail, not one number. A clock-lock
run (`nvidia-smi --lock-gpu-clocks`) separates clock-ramp jitter from real variance.
Measured finding: no single stage owns the tail; it is flat WDDM scheduling jitter.

## Routing and the SLA controller

Static routing consults the measured frontier as a lookup table and dispatches to the
cell winner (open loop). The controller closes the loop: measure live p99, adjust batch
and backend to hold a target, with hysteresis so it does not oscillate. It rides on the
frontier; a controller on a wrong map is worse than none. Measured: holds a 150 us p99
SLA on 20/20 ticks under bursty load.

## Measurement discipline

Warm-up discarded; `steady_clock` (monotonic, portable); full sample kept, not a running
mean; per-stage split via CUDA events / Nsight. All commands reproduce from a clone in
BENCHMARKS.md.

## Layering (degrades gracefully)

1. Floor: CPU + naive GPU + CUDA Graphs + the frontier + the tail metric.
2. + persistent megakernel.
3. + routing and the SLA controller.

Each layer stands alone. Deliberately NOT done: more backends, chasing model accuracy,
or a controller before the frontier it rides on is measured.

## Risks (and how they landed)

- **Persistent kernel is the hardest part** - lock-free host<->device sync. Prototyped
  in the cuda-learning repo before integration; now built and correctness-gated.
- **GPU jitter is hard to remove** - the CPU has a tighter tail at batch 1. That is a
  reported result, not a failure.
- **The model is not alpha** - the engineering is the point. Its 0.826 accuracy is the
  bid-ask bounce, always caveated.
