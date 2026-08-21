"""Compare two frontier CSVs cell by cell: where did the winner flip, and how much did the
tail move?

Usage:
    python compare_frontier.py <frontier_a.csv> <frontier_b.csv>

Made to check whether a result reproduces across datasets. It reports, per (batch, rate)
cell, the lowest-p999 backend in each file and flags the cells where that winner changed,
plus the ratio of the two files' best p999 so you can see how much cleaner one run is. A
crossover that only shows up in one file, or a p999 that swings by a large factor, is a sign
the single-run number was noise, not a regime.
"""

import sys
from pathlib import Path

import pandas as pd


def winners(df: pd.DataFrame) -> dict[tuple, tuple[str, float]]:
    """{(batch, rate): (winning_backend, best_p999_us)} for one frontier."""
    out = {}
    for (batch, rate), cell in df.groupby(["batch", "rate_hz"]):
        row = cell.loc[cell["p999_ns"].idxmin()]
        out[(int(batch), float(rate))] = (row["backend"], row["p999_ns"] / 1000.0)
    return out


def main(path_a: str, path_b: str) -> None:
    a, b = pd.read_csv(path_a), pd.read_csv(path_b)
    wa, wb = winners(a), winners(b)
    cells = sorted(set(wa) & set(wb))

    flips = [(c, wa[c][0], wb[c][0]) for c in cells if wa[c][0] != wb[c][0]]
    ratios = [max(wa[c][1], wb[c][1]) / min(wa[c][1], wb[c][1])
              for c in cells if min(wa[c][1], wb[c][1]) > 0]

    name_a, name_b = Path(path_a).stem, Path(path_b).stem
    print(f"comparing {name_a} vs {name_b} over {len(cells)} shared cells\n")

    from collections import Counter
    ca = Counter(w for w, _ in wa.values())
    cb = Counter(w for w, _ in wb.values())
    print(f"winner counts {name_a}: {dict(ca)}")
    print(f"winner counts {name_b}: {dict(cb)}\n")

    if flips:
        print(f"{len(flips)} cell(s) where the winner flipped:")
        for (batch, rate), x, y in flips:
            print(f"  batch {batch:<4} rate {int(rate):<8}  {name_a}={x}  ->  {name_b}={y}")
    else:
        print("no cell changed winner")

    if ratios:
        ratios.sort()
        med = ratios[len(ratios) // 2]
        print(f"\nbest-p999 ratio between the two files: median {med:.1f}x, max {max(ratios):.1f}x")
        print("(a large ratio means the tail is not stable across the two runs)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
