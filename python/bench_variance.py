"""Run a batch-1 benchmark several times and report the run-to-run spread.

Usage:
    python bench_variance.py <bench_exe> <trades.csv> <model-stem> <out.csv> [runs] [warmup] [iters]

A single benchmark run gives one p50/p99/p999 per backend, but those numbers drift between
runs on a WDDM laptop (display contention, clock ramp, thermals). One run cannot tell a real
change from that drift. This runs the benchmark `runs` times, parses each backend's tail from
stdout, and reports the median plus the spread (max/min ratio) across runs. A large spread
means the single-run number is not trustworthy on its own.

It is a wrapper: it shells out to an existing bench exe (bench_all) and parses the table it
already prints, so it adds the missing repeated-measurement layer without touching the C++.
"""

import statistics
import subprocess
import sys
from pathlib import Path

BACKENDS = {"cpu", "cuda-naive", "cuda-graphs", "persistent"}
METRICS = ("p50", "p99", "p999")


def parse_run(stdout: str) -> dict[str, dict[str, float]]:
    """Pull {backend: {p50,p99,p999}} out of one bench_all run's table."""
    out: dict[str, dict[str, float]] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] in BACKENDS:
            try:
                out[parts[0]] = {"p50": float(parts[2]), "p99": float(parts[3]),
                                 "p999": float(parts[4])}
            except ValueError:
                continue
    return out


def main(exe, trades, model, out_path, runs=5, warmup=1000, iters=50000):
    samples: dict[str, dict[str, list[float]]] = {}
    for i in range(int(runs)):
        r = subprocess.run([exe, trades, model, str(warmup), str(iters)],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            sys.exit(f"run {i} failed:\n{r.stderr[-500:]}")
        parsed = parse_run(r.stdout)
        if not parsed:
            sys.exit(f"run {i}: could not parse a backend table from:\n{r.stdout[-500:]}")
        for backend, metrics in parsed.items():
            for m, v in metrics.items():
                samples.setdefault(backend, {}).setdefault(m, []).append(v)
        print(f"run {i + 1}/{runs} done: " +
              "  ".join(f"{b} p50={parsed[b]['p50']:.1f}" for b in sorted(parsed)))

    rows = []
    print(f"\nrun-to-run spread over {runs} runs (us; spread = max/min):")
    print(f"{'backend':<13}{'metric':<6}{'median':>10}{'min':>10}{'max':>10}{'spread':>8}")
    for backend in sorted(samples):
        for m in METRICS:
            vals = samples[backend][m]
            med, lo, hi = statistics.median(vals), min(vals), max(vals)
            spread = hi / lo if lo > 0 else float("inf")
            print(f"{backend:<13}{m:<6}{med:>10.2f}{lo:>10.2f}{hi:>10.2f}{spread:>8.1f}")
            rows.append((backend, m, med, lo, hi, spread))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("backend,metric,median_us,min_us,max_us,spread\n")
        for backend, m, med, lo, hi, spread in rows:
            f.write(f"{backend},{m},{med:.2f},{lo:.2f},{hi:.2f},{spread:.3f}\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:5], *sys.argv[5:8])
