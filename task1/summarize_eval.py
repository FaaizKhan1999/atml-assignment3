"""Build the Task 1 benchmark table (7 conditions x 4 benchmarks) from saved results (CPU only).

Writes results/task1/tables/benchmarks.csv and benchmarks.md: primary scores in %,
raw accuracy where a normalized one is primary, and changes from the shared
baseline in percentage points (pp). GSM8K also gets unextractable and
truncation rates, so answer-format changes can be told apart from correctness.

Usage: python -m task1.summarize_eval [--push]
"""
from __future__ import annotations

import argparse
import csv

from common.config import REPO_ROOT
from common.logging_utils import read_json
from common.publish import try_push
from task1.conditions import CONDITIONS

EVAL = REPO_ROOT / "results" / "task1" / "eval"
OUT = REPO_ROOT / "results" / "task1" / "tables"
COLUMNS = [  # (column, benchmark, json path)
    ("gsm8k_em_flexible", "gsm8k", ("metrics", "exact_match,flexible-extract")),
    ("gsm8k_em_strict", "gsm8k", ("metrics", "exact_match,strict-match")),
    ("gsm8k_unextractable", "gsm8k", ("unextractable_rate",)),
    ("gsm8k_truncated", "gsm8k", ("truncation_rate",)),
    ("hellaswag_acc_norm", "hellaswag", ("metrics", "acc_norm,none")),
    ("hellaswag_acc", "hellaswag", ("metrics", "acc,none")),
    ("boolq_acc", "boolq", ("metrics", "acc,none")),
    ("arc_easy_acc_norm", "arc_easy", ("metrics", "acc_norm,none")),
    ("arc_easy_acc", "arc_easy", ("metrics", "acc,none")),
]
DELTA_COLS = ["gsm8k_em_flexible", "hellaswag_acc_norm", "boolq_acc", "arc_easy_acc_norm"]


def _get(d, path):
    for p in path:
        if d is None:
            return None
        d = d.get(p)
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    table = {}
    for cond in CONDITIONS:
        row = {}
        for col, bench, path in COLUMNS:
            f = EVAL / cond / f"{bench}.json"
            v = _get(read_json(f), path) if f.exists() else None
            row[col] = None if v is None else 100.0 * float(v)
        table[cond] = row
    base = table["baseline"]
    for cond, row in table.items():
        for col in DELTA_COLS:
            row[f"{col}_delta_pp"] = (row[col] - base[col]) if row[col] is not None and base[col] is not None else None
        nonmath = [row[f"{c}_delta_pp"] for c in DELTA_COLS[1:]]
        row["mean_nonmath_delta_pp"] = sum(nonmath) / 3 if all(v is not None for v in nonmath) else None

    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(next(iter(table.values())).keys())
    with open(OUT / "benchmarks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", *cols])
        for cond, row in table.items():
            w.writerow([cond, *["" if row[c] is None else f"{row[c]:.2f}" for c in cols]])

    def cell(row, col):
        v, d = row[col], row.get(f"{col}_delta_pp")
        if v is None:
            return "-"
        return f"{v:.1f}" if d is None or row is base else f"{v:.1f} ({d:+.1f})"

    def pct(v):
        return "-" if v is None else f"{v:.1f}"

    lines = ["| condition | GSM8K EM (flex) | HellaSwag acc_norm | BoolQ acc | ARC-E acc_norm | mean non-math Δpp "
             "| GSM8K unextractable % | GSM8K truncated % |", "|" + "---|" * 8]
    for cond, row in table.items():
        mn = row["mean_nonmath_delta_pp"]
        mean_cell = "-" if mn is None or row is base else f"{mn:+.2f}"
        cells = [cond, cell(row, "gsm8k_em_flexible"), cell(row, "hellaswag_acc_norm"), cell(row, "boolq_acc"),
                 cell(row, "arc_easy_acc_norm"), mean_cell, pct(row["gsm8k_unextractable"]),
                 pct(row["gsm8k_truncated"])]
        lines.append("| " + " | ".join(cells) + " |")
    (OUT / "benchmarks.md").write_text("Scores in %, change from baseline in pp in parentheses.\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.push:
        try_push([OUT], "task1/tables: benchmark summary")


if __name__ == "__main__":
    main()
