"""Plot the 2D frontier from a frontier CSV.

Usage:
    python plot_frontier.py ../data/frontier.csv

Produces two things next to the CSV:
    frontier_winner.png  the (batch x rate) heatmap colored by the backend with the
                         lowest p999 in each cell, the headline artifact.
    frontier_<backend>_p999.png  one p999 heatmap per backend, for reading absolutes.

With a single-backend CSV (the CPU scaffold), the winner map is one color and the
per-backend map still shows how p999 moves with batch and rate.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pivot(df, backend, value):
    sub = df[df["backend"] == backend]
    return sub.pivot(index="batch", columns="rate_hz", values=value).sort_index()


def main(path):
    df = pd.read_csv(path)
    df["p999_us"] = df["p999_ns"] / 1000.0
    backends = sorted(df["backend"].unique())
    batches = sorted(df["batch"].unique())
    rates = sorted(df["rate_hz"].unique())
    out_dir = Path(path).parent

    # Winner map: index of the backend with the lowest p999 per (batch, rate) cell.
    grid = np.full((len(batches), len(rates)), -1, dtype=int)
    for i, b in enumerate(batches):
        for j, r in enumerate(rates):
            cell = df[(df["batch"] == b) & (df["rate_hz"] == r)]
            if cell.empty:
                continue
            grid[i, j] = backends.index(cell.loc[cell["p999_ns"].idxmin(), "backend"])

    fig, ax = plt.subplots(figsize=(1.6 * len(rates) + 2, 0.6 * len(batches) + 2))
    cmap = plt.get_cmap("tab10", max(len(backends), 1))
    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=max(len(backends) - 1, 1))
    ax.set_xticks(range(len(rates)), [f"{r:.0f}" for r in rates])
    ax.set_yticks(range(len(batches)), [str(b) for b in batches])
    ax.set_xlabel("arrival rate (Hz, 0 = unpaced)")
    ax.set_ylabel("batch size")
    ax.set_title("Frontier: backend with lowest p999 per cell")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(k)) for k in range(len(backends))]
    ax.legend(handles, backends, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    winner = out_dir / "frontier_winner.png"
    fig.savefig(winner, dpi=120)
    print(f"wrote {winner}")

    # Per-backend p999 heatmap in microseconds.
    for backend in backends:
        p = pivot(df, backend, "p999_us")
        fig, ax = plt.subplots(figsize=(1.6 * len(rates) + 2, 0.6 * len(batches) + 2))
        im = ax.imshow(p.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(p.columns)), [f"{c:.0f}" for c in p.columns])
        ax.set_yticks(range(len(p.index)), [str(i) for i in p.index])
        ax.set_xlabel("arrival rate (Hz, 0 = unpaced)")
        ax.set_ylabel("batch size")
        ax.set_title(f"{backend}: p999 latency (us)")
        fig.colorbar(im, ax=ax, label="p999 (us)")
        fig.tight_layout()
        out = out_dir / f"frontier_{backend}_p999.png"
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python plot_frontier.py <frontier.csv>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
