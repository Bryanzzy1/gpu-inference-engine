"""Plot the jitter autopsy from an autopsy CSV.

Usage:
    python plot_autopsy.py ../results/autopsy.csv

For each label in the CSV (e.g. "naive", "naive-locked"), draws a grouped bar chart of
p50/p99/p999 per stage (h2d, launch, compute, d2h), so you can see which stage owns the
tail and how much the tail exceeds the median. Writes autopsy_<label>.png next to the CSV.

Run the autopsy twice, clocks locked vs unlocked, with different labels, to compare how
much of each stage's tail is clock-ramp jitter.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGES = ["h2d", "launch", "compute", "d2h", "total"]


def main(path):
    if not Path(path).exists():
        sys.exit(f"error: no such file: {path}\nrun autopsy first to produce it")
    df = pd.read_csv(path)
    df["p50_us"] = df["p50_ns"] / 1000.0
    df["p99_us"] = df["p99_ns"] / 1000.0
    df["p999_us"] = df["p999_ns"] / 1000.0
    out_dir = Path(path).parent

    for label in df["label"].unique():
        sub = df[df["label"] == label].set_index("stage").reindex(STAGES)
        x = np.arange(len(STAGES))
        w = 0.27

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w, sub["p50_us"], w, label="p50")
        ax.bar(x, sub["p99_us"], w, label="p99")
        ax.bar(x + w, sub["p999_us"], w, label="p999")
        ax.set_xticks(x, STAGES)
        ax.set_ylabel("latency (us)")
        ax.set_title(f"Jitter autopsy: {label}")
        ax.legend()
        fig.tight_layout()

        out = out_dir / f"autopsy_{label}.png"
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")

    labels = list(df["label"].unique())
    parts = [s for s in STAGES if s != "total"]  # stack the four stages, not the total

    # Stacked breakdown at p999: shows how each stage adds up to the tail per label,
    # so the stage that owns the tail is obvious at a glance.
    fig, ax = plt.subplots(figsize=(1.8 * len(labels) + 3, 5))
    bottom = np.zeros(len(labels))
    for stage in parts:
        vals = [float(df[(df["label"] == lb) & (df["stage"] == stage)]["p999_us"].iloc[0])
                if not df[(df["label"] == lb) & (df["stage"] == stage)].empty else 0.0
                for lb in labels]
        ax.bar(labels, vals, bottom=bottom, label=stage)
        bottom += np.array(vals)
    ax.set_ylabel("p999 latency (us)")
    ax.set_title("Stage breakdown at p999 (stacked)")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "autopsy_stacked_p999.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")

    # Locked vs unlocked overlay: only meaningful with more than one label. Grouped
    # bars per stage make the clock-ramp jitter (tail that shrinks when locked) visible.
    if len(labels) > 1:
        x = np.arange(len(parts))
        w = 0.8 / len(labels)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, lb in enumerate(labels):
            vals = [float(df[(df["label"] == lb) & (df["stage"] == s)]["p999_us"].iloc[0])
                    if not df[(df["label"] == lb) & (df["stage"] == s)].empty else 0.0
                    for s in parts]
            ax.bar(x + (i - (len(labels) - 1) / 2) * w, vals, w, label=lb)
        ax.set_xticks(x, parts)
        ax.set_ylabel("p999 latency (us)")
        ax.set_title("Per-stage p999 across runs (e.g. clocks locked vs unlocked)")
        ax.legend()
        fig.tight_layout()
        out = out_dir / "autopsy_overlay_p999.png"
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python plot_autopsy.py <autopsy.csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
