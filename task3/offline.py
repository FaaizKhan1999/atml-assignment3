"""Task 3 offline experiments (Steps 1, 2, 3 and 5) on the supplied rollouts. CPU only.

    python -m task3.offline --push                  # both datasets
    python -m task3.offline --datasets math --push

Writes splits/task3/<ds>_split.json, <ds>_orders.json and results/task3/<ds>/:
  answer_groups.csv.gz     every rollout's extracted answer, answer group and group grade (the fixed scorer)
  summary.json             data checks, scorer versions, accuracy / oracle coverage / distinct groups /
                           token cost per method and budget (dev and held-out), failure decomposition,
                           paired bootstrap CIs for accuracy differences
  signals.json             N = 32 signal analysis (quartiles, length associations, confidence-vision,
                           tokenwise vs trajectory fusion identity)
  adaptive.json            development grid, frozen settings, held-out results, stopping distributions
  selections.csv.gz        every (question, order, budget, method): selected candidate, answer, correct, cost
  per_question.csv         held-out per-question coverage, fraction correct, groups, SC@32 and stopping
  early_stop_errors.csv    held-out incorrect early stops (with whether fixed-32 would have been right)

Sensitivity check: --datasets mathvista --mathvista-scorer lenient reruns MathVista with numeric pre-cleaning
(units, spacing, $, degree/percent signs, thousands separators) before MathVista's float()/int() rules, for
grouping AND grading; results go to results/task3/mathvista_lenient/ and never replace the official ones.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import time

import numpy as np

from common.config import REPO_ROOT, add_config_args, load_config
from common.logging_utils import get_logger, read_json, run_metadata, write_json
from common.publish import try_push
from task3.data import SPLITS, make_orders, make_splits, prepare, save_groups
from task3.engine import (bootstrap_diffs, evaluate, failure_decomposition, per_question_means, select_policy,
                          signal_analysis, simulate)

log = get_logger()
RES = REPO_ROOT / "results" / "task3"

CONTROLLED_PAIRS = [  # (method, baseline): same aggregation/different score, or same score/different aggregation
    ("selfcert_rwc", "max_selfcert"), ("max_selfcert", "max_logp"), ("vision_rwc", "max_vision"),
    ("visconf_rwc", "max_visconf"), ("max_visconf", "max_vision"), ("max_visconf", "max_selfcert"),
    ("visconf_rwc", "selfcert_rwc"), ("max_visconf_token", "max_visconf"), ("visconf_token_rwc", "visconf_rwc"),
    ("sc", "first")]


def _stable(path, obj, what: str) -> None:
    """Write split/order files; if already committed, require them to be identical (determinism check)."""
    if path.exists():
        old = read_json(path)
        if old != obj:
            raise SystemExit(f"{what} differs from the committed {path.name}; refusing to overwrite")
    else:
        write_json(path, obj)


def run_dataset(ds: str, cfg, args) -> None:
    a, ad = cfg.analysis, cfg.adaptive
    lenient = ds == "mathvista" and args.mathvista_scorer == "lenient"
    out = RES / (f"{ds}_lenient" if lenient else ds)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    prepared = prepare(ds, cfg, rebuild=args.rebuild, lenient=lenient)
    qids = sorted(prepared["questions"])
    split = make_splits(ds, qids, cfg)
    _stable(SPLITS / f"{ds}_split.json", split, "split")
    orders = make_orders(qids, int(a.n_orders), int(a.seed))
    _stable(SPLITS / f"{ds}_orders.json", orders, "orders")
    save_groups(prepared, out / "answer_groups.csv.gz")

    budgets = list(a.budgets)
    quality = ad.quality_score[ds]
    vc = (tuple(a.visconf_quantiles), float(a.visconf_eps))
    log.info("%s: evaluating %d questions x %d orders x %d budgets", ds, len(qids), int(a.n_orders), len(budgets))
    t1 = time.perf_counter()
    ev = evaluate(prepared, orders, budgets, quality, list(ad.checkpoints), vc)
    eval_seconds = time.perf_counter() - t1
    methods = ev["methods"]

    # ---------------------------------------------------------------- Steps 1-2: curves, failures, bootstrap
    curves = {}
    for part in ("dev", "eval"):
        pm = per_question_means(ev, split[part])
        curves[part] = {"N": budgets,
                        "oracle_coverage": pm["coverage"].mean(0).tolist(),
                        "distinct_groups": pm["distinct"].mean(0).tolist(),
                        "accuracy": {m: pm["correct"][:, :, i].mean(0).tolist() for i, m in enumerate(methods)},
                        "token_cost": {m: pm["cost"][:, :, i].mean(0).tolist() for i, m in enumerate(methods)}}
    A_eval = per_question_means(ev, split["eval"])["correct"]
    pairs = [(m, "sc") for m in methods if m != "sc"] + CONTROLLED_PAIRS
    boot = bootstrap_diffs(A_eval, methods, budgets, pairs, int(a.bootstrap), int(a.seed))
    failures = {m: failure_decomposition(ev, split["eval"], m) for m in methods}
    d = prepared["questions"]
    stats = {"n_questions": len(qids), "n_rollouts": 32 * len(qids),
             "no_answer_rate": float(np.mean([(d[q]["groups"] == 0).mean() for q in qids])),
             "capped_rate": float(np.mean([d[q]["hit_max"].mean() for q in qids])),
             "rollout_accuracy": float(np.mean([np.mean([d[q]["correct"][int(g)] for g in d[q]["groups"]]) for q in qids])),
             "mean_tokens": float(np.mean([d[q]["tokens"].mean() for q in qids])),
             "zero_length_rollouts": int(sum((d[q]["tokens"] == 0).sum() for q in qids))}
    summary = {"dataset": ds, "checks": prepared["checks"], "scorer": prepared["scorer"], "data_stats": stats,
               "split_sizes": {k: len(v) for k, v in split.items() if isinstance(v, list)},
               "methods": methods, "curves": curves, "failure_decomposition_eval": failures,
               "bootstrap_eval": boot, "evaluation_seconds": eval_seconds, "environment": run_metadata()}
    write_json(out / "summary.json", summary)

    # ---------------------------------------------------------------- Step 3: signals
    write_json(out / "signals.json", signal_analysis(prepared, split["eval"], vc))

    # ---------------------------------------------------------------- Step 5: adaptive
    tol = float(ad.tolerance_pp) / 100.0
    ad_out = {"quality_score": quality, "checkpoints": list(ad.checkpoints), "tolerance_pp": float(ad.tolerance_pp)}
    fixed = {}
    for n in budgets:
        b = budgets.index(n)
        i = methods.index("sc")
        pm = per_question_means(ev, split["eval"])
        fixed[str(n)] = {"accuracy": float(pm["correct"][:, b, i].mean()), "mean_cost": float(pm["cost"][:, b, i].mean())}
    ad_out["fixed_n_self_consistency_eval"] = fixed
    base_eval = simulate(ev, split["eval"], "fixed32", None, None, list(ad.checkpoints))
    early_rows = []
    nq = len(split["eval"])
    bidx = np.random.default_rng(int(a.seed)).integers(0, nq, size=(int(a.bootstrap), nq))
    per_q_stop = {}
    for family in ("agreement", "quality"):
        sel = select_policy(ev, split["dev"], family, list(ad.taus), list(ad.gammas), list(ad.checkpoints), tol)
        s = sel["selected"]
        res = simulate(ev, split["eval"], s["family"], s["tau"], s["gamma"], list(ad.checkpoints))
        da = np.array(res["per_question_accuracy"]) - np.array(base_eval["per_question_accuracy"])
        dc = np.array(res["per_question_cost"]) - np.array(base_eval["per_question_cost"])
        ad_out[family] = {"dev_selection": sel, "frozen": {"tau": s["tau"], "gamma": s["gamma"], "family": s["family"]},
                          "eval": {k: res[k] for k in ("accuracy", "mean_cost", "mean_rollouts", "stop_distribution")},
                          "eval_minus_fixed32": {
                              "accuracy": {"diff": float(da.mean()), "ci95": [float(np.percentile(da[bidx].mean(1), 2.5)),
                                                                              float(np.percentile(da[bidx].mean(1), 97.5))]},
                              "mean_cost": {"diff": float(dc.mean()), "ci95": [float(np.percentile(dc[bidx].mean(1), 2.5)),
                                                                               float(np.percentile(dc[bidx].mean(1), 97.5))]}}}
        for q, o, stop, g, correct, c32 in res["rows"]:
            per_q_stop.setdefault((family, q), []).append(stop)
            if stop < 32 and not correct:
                early_rows.append([family, q, o, stop, prepared["questions"][q]["reps"][g], int(c32)])
    ad_out["fixed32_eval"] = {k: base_eval[k] for k in ("accuracy", "mean_cost", "mean_rollouts")}
    write_json(out / "adaptive.json", ad_out)

    # ---------------------------------------------------------------- per-question tables
    with open(out / "early_stop_errors.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "sample_id", "order", "stopped_at", "answer", "fixed32_correct"])
        w.writerows(early_rows)
    i_sc = methods.index("sc")
    with open(out / "per_question.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "split", "fraction_correct_rollouts", "n_groups_32", "no_answer_rollouts",
                    "sc32_correct", "mean_stop_agreement", "mean_stop_quality"])
        for part in ("dev", "eval"):
            for q in split[part]:
                dq = d[q]
                fr = np.mean([dq["correct"][int(g)] for g in dq["groups"]])
                sc32 = ev["per_question"][q]["correct"][:, budgets.index(32), i_sc].mean()
                w.writerow([q, part, f"{fr:.4f}", len(set(dq["groups"].tolist())), int((dq["groups"] == 0).sum()),
                            f"{sc32:.3f}",
                            f"{np.mean(per_q_stop[('agreement', q)]):.2f}" if ('agreement', q) in per_q_stop else "",
                            f"{np.mean(per_q_stop[('quality', q)]):.2f}" if ('quality', q) in per_q_stop else ""])
    with gzip.open(out / "selections.csv.gz", "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "order", "N", "method", "candidate_rollout", "answer", "correct", "token_cost"])
        for q in qids:
            r = ev["per_question"][q]
            for o in range(r["correct"].shape[0]):
                for b, n in enumerate(budgets):
                    for mi, m in enumerate(methods):
                        w.writerow([q, o, n, m, int(r["cand"][o, b, mi]), d[q]["reps"][int(r["group"][o, b, mi])],
                                    int(r["correct"][o, b, mi]), int(r["cost"][o, b, mi])])

    c = curves["eval"]
    print(f"\n=== {ds}{' (LENIENT sensitivity scorer)' if lenient else ''}: held-out {len(split['eval'])} questions ===")
    print(f"rollout accuracy {stats['rollout_accuracy']:.3f}, NO_ANSWER {stats['no_answer_rate']:.3f}, "
          f"capped {stats['capped_rate']:.3f}; checks {prepared['checks']}")
    print("N:          " + "".join(f"{n:>8}" for n in budgets))
    print("coverage:   " + "".join(f"{v:>8.3f}" for v in c["oracle_coverage"]))
    for m in methods:
        print(f"{m:<12}" + "".join(f"{v:>8.3f}" for v in c["accuracy"][m]))
    for fam in ("agreement", "quality"):
        f = ad_out[fam]
        print(f"adaptive {fam}: tau={f['frozen']['tau']} gamma={f['frozen']['gamma']} -> acc {f['eval']['accuracy']:.3f}, "
              f"cost {f['eval']['mean_cost']:.0f} (fixed-32 SC acc {base_eval['accuracy']:.3f}, cost {base_eval['mean_cost']:.0f}), "
              f"mean rollouts {f['eval']['mean_rollouts']:.1f}")
    log.info("%s done in %.1f min", ds, (time.perf_counter() - t0) / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task3/base.yaml")
    parser.add_argument("--datasets", nargs="+", default=["math", "mathvista"], choices=["math", "mathvista"])
    parser.add_argument("--rebuild", action="store_true", help="recompute answer groups (cached otherwise)")
    parser.add_argument("--mathvista-scorer", choices=["official", "lenient"], default="official",
                        help="lenient = sensitivity check with numeric pre-cleaning; writes results/task3/mathvista_lenient/")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    for ds in args.datasets:
        run_dataset(ds, cfg, args)
    if args.push:
        try_push([RES, SPLITS], "task3: offline experiments")


if __name__ == "__main__":
    main()
