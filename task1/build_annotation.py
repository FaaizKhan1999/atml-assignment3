"""Combine the 7 conditions' qualitative outputs (CPU only).

Writes to results/task1/qualitative/:
  all_responses.csv        168 rows (24 prompts x 7 conditions) with automatic flags
  by_question.md           every response grouped by question, then format, then condition
  annotation_sheet.csv     the fixed 42-response subset (first and last plain questions,
                           3 formats, 7 conditions) with automatic flags and EMPTY columns
                           for your own judgements

Usage: python -m task1.build_annotation [--push]
"""
from __future__ import annotations

import argparse
import csv
import json

from common.config import REPO_ROOT, add_config_args, load_config
from common.publish import try_push
from task1.conditions import CONDITIONS
from task1.qualitative import load_prompts, question_order

QUAL_DIR = REPO_ROOT / "results" / "task1" / "qualitative"
FORMATS = ["plain", "json", "template"]
AUTO_COLS = ["n_response_tokens", "truncated", "n_words", "repeated_4gram_fraction", "math_markers",
             "json_fenced", "json_valid_as_is", "json_valid_after_unfencing",
             "template_keys", "template_keys_filled", "template_required_filled"]
MANUAL_COLS = ["format_compliance", "coherence_relevance", "repetition", "unnecessary_math_framing",
               "inappropriate_content", "excerpt", "notes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task1/eval.yaml")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)

    prompts = load_prompts(cfg.qualitative.prompts_file)
    order = question_order(prompts)
    rows, missing = {}, []
    for cond in CONDITIONS:
        path = QUAL_DIR / f"{cond}.jsonl"
        if not path.exists():
            missing.append(cond)
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            rows[(r["question"], r["format"], cond)] = r
    if missing:
        print(f"WARNING: no qualitative outputs yet for {missing}; building with what exists")

    conds = [c for c in CONDITIONS if c not in missing]
    with open(QUAL_DIR / "all_responses.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["question", "format", "condition", "response", *AUTO_COLS])
        for q in order:
            for f in FORMATS:
                for c in conds:
                    r = rows[(q, f, c)]
                    w.writerow([q, f, c, r["response"], *[r.get(k, "") for k in AUTO_COLS]])

    with open(QUAL_DIR / "by_question.md", "w", encoding="utf-8") as fh:
        fh.write("# Task 1 qualitative responses\n\nGreedy, thinking disabled, 512-token cap. "
                 "Grouped by question, format, condition.\n")
        for q in order:
            fh.write(f"\n## {q}\n")
            for f in FORMATS:
                p = next(x for x in prompts if x["question"] == q and x["format"] == f)
                fh.write(f"\n### {q} - {f}\n\n")
                if p["system"]:
                    fh.write(f"System: {p['system']}\n\n")
                fh.write("Prompt:\n\n```text\n" + p["user"] + "\n```\n")
                for c in conds:
                    r = rows[(q, f, c)]
                    tags = []
                    if r.get("truncated"):
                        tags.append("TRUNCATED")
                    if f == "json":
                        tags.append("json ok" if r.get("json_valid_as_is") else
                                    "json ok after unfencing" if r.get("json_valid_after_unfencing") else "json INVALID")
                    if f == "template":
                        tags.append(f"template {r.get('template_keys_filled')}/{r.get('template_keys')} keys")
                    fh.write(f"\n#### {c} ({r['n_response_tokens']} tokens{'; ' + ', '.join(tags) if tags else ''})\n\n")
                    fh.write("```text\n" + r["response"].replace("```", "` ` `") + "\n```\n")

    annotate = list(cfg.qualitative.annotate_questions)
    if annotate != [order[0], order[-1]]:
        print(f"note: configured annotation questions {annotate} differ from first/last {[order[0], order[-1]]}")
    with open(QUAL_DIR / "annotation_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["question", "format", "condition", "response", *AUTO_COLS, *MANUAL_COLS])
        n = 0
        for q in annotate:
            for f in FORMATS:
                for c in conds:
                    r = rows[(q, f, c)]
                    w.writerow([q, f, c, r["response"], *[r.get(k, "") for k in AUTO_COLS], *[""] * len(MANUAL_COLS)])
                    n += 1
    print(f"wrote {len(rows)} responses; annotation sheet has {n} rows (expected 42)")
    if args.push:
        try_push([QUAL_DIR], "task1/qualitative: combined responses and annotation sheet")


if __name__ == "__main__":
    main()
