"""Plot the live SLA-controller trace.

Usage:
    python plot_controller.py ../results/controller_trace.csv

Produces controller_trace.png next to the CSV: measured p99 per control tick against
the SLA line (points colored by whether the tick held the SLA), with the batch size the
controller chose on a second axis. It shows the loop climb to the SLA-limited batch and
where routing on p999 lets a p99 tick slip over the target.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main(path):
    df = pd.read_csv(path)
    df["p99_us"] = df["observed_p99_ns"] / 1000.0
    sla_us = df["sla_ns"].iloc[0] / 1000.0
    out = Path(path).with_suffix(".png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["tick"], df["p99_us"], color="#888", zorder=1)
    held = df[df["under_sla"] == 1]
    over = df[df["under_sla"] == 0]
    ax.scatter(held["tick"], held["p99_us"], color="#2a9d4a", label="held SLA", zorder=3)
    ax.scatter(over["tick"], over["p99_us"], color="#d23", label="over SLA", zorder=3)
    ax.axhline(sla_us, color="#d23", ls="--", lw=1, label=f"SLA p99 = {sla_us:.0f} us")
    ax.set_xlabel("control tick")
    ax.set_ylabel("measured p99 (us)")
    ax.set_title("Live SLA controller: real backend p99 vs the target")
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.step(df["tick"], df["batch"], where="mid", color="#3a6", alpha=0.4, lw=1.5)
    ax2.set_ylabel("batch size (step)", color="#3a6")
    ax2.set_yscale("log", base=2)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python plot_controller.py <controller_trace.csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
