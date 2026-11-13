"""All Task 3 figures and tables for the report (CPU; reads committed results; the optional trajectory
figure also needs the supplied data, i.e. the Kaggle dataset attached).

    python -m task3.make_report_assets --push

Main body (report/figs/task3/):
  fig1_compute_curves     2 x 2, one shared accuracy axis: accuracy vs candidates N (top) and vs generated-token
                          cost (bottom) for MATH and MathVista; oracle coverage dashed; adaptive policies as stars
                          on the cost row (comparison with fixed budgets at similar cost)
  tab1_methods            compact comparison at N = 8 and 32: accuracy, change vs self-consistency [95% CI]
                          (includes the length-only control and the tokenwise-fusion ablation)
  fig2_failures_signals   (a, b) correct / selection failure / sampling failure vs N for self-consistency and the
                          best RWC rule; (c) top-minus-bottom-quartile correctness per score
  tab2_signals            length associations, confidence-vision correlation, tokenwise-vs-trajectory identity
  tab3_live               single vs parallel vs refinement: accuracy [CI vs single], tokens, time, truncation,
                          extraction failures, revision transitions
  tab4_adaptive           dev-selected thresholds and held-out accuracy / cost / rollouts / stopping distribution
                          next to fixed-N self-consistency
  tab5_examples           aligned examples of selection disagreement, revision and premature stopping
Appendix: tabA1_all_curves, tabA2_scorer_sensitivity, tabA3_bootstrap, tabA4_adaptive_grid, tabA5_data_scorer,
          figA1_all_methods, figA2_stopping_vs_correct, figA3_trajectories (if the data is available)

Example selection rules (fixed in advance, stated in captions): the first held-out question in split order that
satisfies each condition (see tab5 caption).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.config import REPO_ROOT  # noqa: E402
from common.logging_utils import read_json  # noqa: E402
from common.publish import try_push  # noqa: E402
from common.report_utils import FONT_PT, TEXT_WIDTH_IN, rcparams, save_fig, save_table  # noqa: E402

RES = REPO_ROOT / "results" / "task3"
SPL = REPO_ROOT / "splits" / "task3"
OUT = REPO_ROOT / "report" / "figs" / "task3"
DS = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}
LABEL = {"first": "FirstAns", "sc": "Self-consistency", "max_logp": "Max mean log-prob", "max_selfcert": "Max self-certainty",
         "selfcert_rwc": "Self-certainty + RWC", "shortest": "Shortest (length control)", "max_vision": "Max vision",
         "vision_rwc": "Vision + RWC", "max_visconf": "Max VisConf", "visconf_rwc": "VisConf + RWC",
         "max_visconf_token": "Max VisConf (tokenwise)", "visconf_token_rwc": "VisConf tokenwise + RWC"}
COLOR = {"first": "#7f7f7f", "sc": "#1f77b4", "max_logp": "#bcbd22", "max_selfcert": "#ff7f0e", "selfcert_rwc": "#d62728",
         "shortest": "#8c564b", "max_vision": "#17becf", "vision_rwc": "#2ca02c", "max_visconf": "#e377c2",
         "visconf_rwc": "#9467bd", "max_visconf_token": "#c5b0d5", "visconf_token_rwc": "#6b4c9a"}
MAIN_METHODS = {"math": ["first", "sc", "max_selfcert", "selfcert_rwc"],
                "mathvista": ["first", "sc", "max_selfcert", "selfcert_rwc", "max_visconf", "visconf_rwc"]}
BEST_RWC = {"math": "selfcert_rwc", "mathvista": "visconf_rwc"}


def load(ds: str, variant: str = "") -> dict:
    d = RES / f"{ds}{variant}"
    return {"summary": read_json(d / "summary.json"), "signals": read_json(d / "signals.json"),
            "adaptive": read_json(d / "adaptive.json"), "dir": d}


def pct(v, nd=1):
    return "–" if v is None else f"{100 * v:.{nd}f}"


# ============================================================================ figure 1: accuracy vs N and vs cost

def fig1_compute_curves(R: dict) -> None:
    """Printed at 5.45 in x 2.35 in (scale 1). One shared accuracy axis for all four panels; the two N panels
    and the two token panels also share their x scales, so MATH and MathVista compare at a glance."""
    fig, axes = plt.subplots(1, 4, figsize=(TEXT_WIDTH_IN - 0.05, 2.35), sharey=True)
    fig.subplots_adjust(wspace=0.12, left=0.075, right=0.995, top=0.88, bottom=0.42)
    axes[1].sharex(axes[0])
    axes[3].sharex(axes[2])
    for col, ds in enumerate(DS):
        c = R[ds]["summary"]["curves"]["eval"]
        N = c["N"]
        a_n, a_t = axes[col], axes[2 + col]
        a_n.plot(N, c["oracle_coverage"], "k--", lw=1.0, label="Oracle coverage")
        a_t.plot(c["token_cost"]["sc"], c["oracle_coverage"], "k--", lw=1.0)
        for m in MAIN_METHODS[ds]:
            a_n.plot(N, c["accuracy"][m], "-o", color=COLOR[m], ms=2.8, label=LABEL[m])
            a_t.plot(c["token_cost"][m], c["accuracy"][m], "-o", color=COLOR[m], ms=2.8)
        ad = R[ds]["adaptive"]
        for fam, mk in (("agreement", "*"), ("quality", "P")):
            e = ad[fam]["eval"]
            a_t.plot(e["mean_cost"], e["accuracy"], mk, color="k", ms=7 if mk == "*" else 5.5, zorder=5,
                     label=f"Adaptive ({fam})")
        name = DS[ds].split(" ")[0]
        a_n.set_title(f"({'ab'[col]}) {name}: vs. N")
        a_t.set_title(f"({'cd'[col]}) {name}: vs. tokens")
        a_n.set_xscale("log", base=2)
        a_n.set_xticks(N)
        a_n.set_xticklabels([str(n) for n in N])
        a_n.set_xlabel("Candidates N")
        a_t.set_xscale("log")
        a_t.set_xlabel("Tokens / question")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Held-out accuracy")
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = axes[3].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=0.9)
    save_fig(fig, OUT, "fig1_compute_curves")


# ============================================================================ table 1: methods at N = 8, 32

def _boot(R, ds, m, n):
    for b in R[ds]["summary"]["bootstrap_eval"]:
        if b["method"] == m and b["baseline"] == "sc" and b["N"] == n:
            return b
    return None


def tab1_methods(R: dict) -> None:
    methods = R["mathvista"]["summary"]["methods"]
    header = ["Method", "MATH\nN = 8", "MATH\nN = 32", "MATH Δ vs SC\n@32 [95% CI]", "MathVista\nN = 8",
              "MathVista\nN = 32", "MathVista Δ vs SC\n@32 [95% CI]"]
    rows = []
    for m in methods:
        row = [LABEL[m]]
        for ds in DS:
            c = R[ds]["summary"]["curves"]["eval"]
            if m not in c["accuracy"]:
                row += ["–", "–", "–"]
                continue
            i8, i32 = c["N"].index(8), c["N"].index(32)
            b = _boot(R, ds, m, 32)
            row += [pct(c["accuracy"][m][i8]), pct(c["accuracy"][m][i32]),
                    "–" if m == "sc" or b is None else f"{100 * b['diff']:+.1f} [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}]"]
        rows.append(row)
    cov = [pct(R[ds]["summary"]["curves"]["eval"]["oracle_coverage"][R[ds]["summary"]["curves"]["eval"]["N"].index(n)])
           for ds in DS for n in (8, 32)]
    rows.append(["Oracle coverage (upper bound)", cov[0], cov[1], "–", cov[2], cov[3], "–"])
    save_table(OUT, "tab1_methods", header, rows,
               "Held-out accuracy (%) of each selection rule (800 questions per dataset; averaged over ten rollout orders, "
               "then questions). Δ: paired question-level bootstrap (1,000 resamples) of the difference from "
               "self-consistency at N = 32. Tokenwise rows are the fusion ablation (confidence × vision per token); the "
               "shortest-rollout rule is a length-only control. Oracle coverage uses reference answers and is not a method.",
               "tab:t3-methods", align="lrrrrrr")


# ============================================================================ figure 2: failures + signals

def fig2_failures_signals(R: dict) -> None:
    """Printed at 5.45 in x 2.45 in (scale 1). (a) and (b) share the 0-1 share axis."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(TEXT_WIDTH_IN - 0.05, 2.45))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.62, 1.15], wspace=0.08, left=0.075, right=0.99, top=0.88,
                          bottom=0.33)   # an empty column keeps (c)'s row labels clear of (b)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], sharey=axa)
    axc = fig.add_subplot(gs[3])
    cols = {"correct": "#4c9f70", "selection_failure": "#e0a030", "sampling_failure": "#c44e52"}
    for ax, ds in ((axa, "math"), (axb, "mathvista")):
        fd = R[ds]["summary"]["failure_decomposition_eval"]
        N = [r["N"] for r in fd["sc"]]
        x = np.arange(len(N))
        for off, m, hatch in ((-0.2, "sc", ""), (0.2, BEST_RWC[ds], "////")):
            bottom = np.zeros(len(N))
            for k in ("correct", "selection_failure", "sampling_failure"):
                v = np.array([r[k] for r in fd[m]])
                ax.bar(x + off, v, 0.38, bottom=bottom, color=cols[k], hatch=hatch, edgecolor="white", lw=0.3)
                bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N])
        ax.set_xlabel("Candidates N")
        ax.set_title(f"({'a' if ds == 'math' else 'b'}) {DS[ds].split(' ')[0]}")
        ax.set_ylim(0, 1)
    axa.set_ylabel("Share of question-orders")
    plt.setp(axb.get_yticklabels(), visible=False)
    scores = [("K", "Self-certainty"), ("L", "Mean log-prob"), ("shortness", "Shortness"), ("V", "Vision"),
              ("Z", "VisConf"), ("Z_token", "VisConf (tokenwise)")]
    y = np.arange(len(scores))[::-1]
    for off, ds in ((0.15, "math"), (-0.15, "mathvista")):
        q = R[ds]["signals"]["top_minus_bottom_quartile_correctness"]
        qm = R[ds]["signals"]["top_minus_bottom_quartile_correctness_mixed_questions"]
        col = "#1f77b4" if ds == "math" else "#d62728"
        for yi, (k, _) in zip(y, scores):
            if q.get(k):
                axc.plot(100 * q[k]["mean"], yi + off, "o", color=col, ms=4)
                axc.plot(100 * qm[k]["mean"], yi + off, "o", mfc="none", color=col, ms=4)
    axc.axvline(0, color="0.5", lw=0.6)
    axc.set_yticks(y)
    axc.set_yticklabels([x_[1] for x_ in scores])
    axc.set_xlabel("Top $-$ bottom quartile correctness (pp)")
    axc.set_title("(c) Signal quality at N = 32")
    handles = [Patch(color=cols["correct"], label="correct"), Patch(color=cols["selection_failure"], label="correct available, missed"),
               Patch(color=cols["sampling_failure"], label="no correct candidate"),
               Patch(facecolor="white", edgecolor="0.3", hatch="////", label="best RWC rule (hatched)"),
               Line2D([], [], color="#1f77b4", marker="o", ls="none", ms=4, label="MATH"),
               Line2D([], [], color="#d62728", marker="o", ls="none", ms=4, label="MathVista"),
               Line2D([], [], color="0.3", marker="o", mfc="none", ls="none", ms=4, label="mixed questions only")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=0.9)
    save_fig(fig, OUT, "fig2_failures_signals")


def tab2_signals(R: dict) -> None:
    names = [("K", "Self-certainty K"), ("L", "Mean log-prob L"), ("shortness", "Shortness (−length)"), ("V", "Vision V"),
             ("C", "Calibrated confidence C"), ("Z", "VisConf Z = V·C"), ("Z_token", "Tokenwise mean(c·v)")]
    header = ["Score", "MATH quartile Δ (all / mixed), pp", "MathVista quartile Δ (all / mixed), pp",
              "MATH Spearman with length", "MathVista Spearman with length"]
    rows = []
    for k, lab in names:
        row = [lab]
        for ds in DS:
            s = R[ds]["signals"]
            a, m = s["top_minus_bottom_quartile_correctness"].get(k), s["top_minus_bottom_quartile_correctness_mixed_questions"].get(k)
            row.append("–" if not a else f"{100 * a['mean']:+.1f} / {100 * m['mean']:+.1f}")
        for ds in DS:
            sp = R[ds]["signals"]["spearman_with_length"].get(k)
            row.append("–" if not sp else f"{sp['median']:+.2f} [{sp['q25']:+.2f}, {sp['q75']:+.2f}]")
        rows.append(row)
    cv = R["mathvista"]["signals"]["confidence_vision_correlation"]
    pr, pq = cv["per_rollout"], cv["per_question_mean"]
    ident = R["mathvista"]["signals"]["tokenwise_identity_max_abs_error"]
    save_table(OUT, "tab2_signals", header, rows,
               "Signal analysis at N = 32 (held-out). Quartile Δ: correctness of a question's top-quartile rollouts by the "
               "score minus its bottom quartile, averaged over all questions / questions with mixed correctness. Spearman: "
               "within-question rank correlation with rollout length, median [IQR]. MathVista confidence–vision "
               f"correlation (Pearson over tokens of c_t and v_t): per rollout median {pr['median']:+.2f} "
               f"[IQR {pr['q25']:+.2f}, {pr['q75']:+.2f}], n = {pr['n']:,}; per-question mean {pq['mean']:+.2f}. "
               f"Tokenwise − trajectory fusion equals Cov_t(c, v) up to {ident:.1e} (numerical check).",
               "tab:t3-signals", align="lrrrr")


# ============================================================================ table 3: live generation

def tab3_live() -> None:
    s = read_json(RES / "live" / "summary.json")
    header = ["Strategy", "Acc. %", "Δ vs single\n[95% CI]", "Input\ntokens", "Output\ntokens", "Total\ntokens",
              "s /\nquestion", "Truncated\ncalls %"]
    rows = []
    for k, lab in (("single", "Single (≤3,072)"), ("parallel", "Parallel ×3 + SC"),
                   ("refinement", "Refinement")):
        d = s["strategies"][k]
        b = s["paired_bootstrap"].get(f"{k}_minus_single")
        rows.append([lab, pct(d["accuracy"]), "–" if b is None else f"{100 * b['diff']:+.0f} [{100 * b['ci95'][0]:+.0f}, {100 * b['ci95'][1]:+.0f}]",
                     f"{d['mean_input_tokens']:.0f}", f"{d['mean_output_tokens']:.0f}", f"{d['mean_total_tokens']:.0f}",
                     f"{d['mean_seconds']:.1f}", pct(d["call_truncation_rate"])])
    t = s["refinement_transitions"]
    save_table(OUT, "tab3_live", header, rows,
               f"Live generation on {s['n_questions']} held-out MATH questions (Qwen2.5-3B-Instruct, FP16, batch 1, one T4; "
               "same 3,072-token output allowance). Means per question; the shared initial response is charged to both "
               "parallel and refinement, and refinement also pays its longer revision prompt. Time excludes model loading. "
               f"Revision transitions vs the shared initial answer (initial accuracy {100 * t['initial_accuracy']:.0f}%): "
               f"wrong→correct {t['wrong_to_correct']}, correct→wrong {t['correct_to_wrong']}, answer changed "
               f"{t['answer_changed']}. Paired bootstrap over questions (1,000 resamples).",
               "tab:t3-live", align="lrrrrrrr")


# ============================================================================ table 4: adaptive

def tab4_adaptive(R: dict) -> None:
    header = ["Dataset", "Policy", "Setting\n(dev)", "Acc. %", "Δ vs SC@32\n[95% CI]", "Tokens\n/ q.", "Roll-\nouts",
              "Stops at\n4 / 8 / 16 / 32 (%)"]
    rows, breaks = [], []
    for ds in DS:
        ad = R[ds]["adaptive"]
        for n in ("4", "8", "16", "32"):
            f = ad["fixed_n_self_consistency_eval"][n]
            rows.append([DS[ds].split(" ")[0] if n == "4" else "", f"Fixed SC, N = {n}", "–", pct(f["accuracy"]), "–",
                         f"{f['mean_cost']:.0f}", n, "–"])
        for fam in ("agreement", "quality"):
            a = ad[fam]
            fr = a["frozen"]
            setting = f"τ = {fr['tau']}" + (f", γ = {fr['gamma']}" if fam == "quality" and fr["tau"] is not None else "")
            if fr["family"] == "fixed32":
                setting = "fallback (always 32)"
            dd = a["eval_minus_fixed32"]["accuracy"]
            sd = a["eval"]["stop_distribution"]
            rows.append(["", f"Adaptive: {fam}", setting, pct(a["eval"]["accuracy"]),
                         f"{100 * dd['diff']:+.1f} [{100 * dd['ci95'][0]:+.1f}, {100 * dd['ci95'][1]:+.1f}]",
                         f"{a['eval']['mean_cost']:.0f}", f"{a['eval']['mean_rollouts']:.1f}",
                         " / ".join(f"{100 * sd[k]:.0f}" for k in ("4", "8", "16", "32"))])
        breaks.append(len(rows) - 1)
    q = {ds: R[ds]["adaptive"]["quality_score"] for ds in DS}
    save_table(OUT, "tab4_adaptive", header, rows,
               "Adaptive sampling (held-out; settings chosen on the 200 development questions as the cheapest within 1 pp of "
               "fixed-32 self-consistency, then frozen). Both controllers return the current self-consistency answer; "
               "checks after 4, 8, 16 candidates, otherwise 32. Quality-aware stopping adds SC = RWC agreement and an RWC "
               f"margin ≥ γ (scores: {q['math']} for MATH, {q['mathvista']} for MathVista, recomputed on revealed candidates). "
               "Tokens are generated tokens of all consumed candidates (offline proxy).",
               "tab:t3-adaptive", align="llllrrrl", midrules_after=breaks)


# ============================================================================ table 5: examples

def _first(rows, key):
    return rows[0] if rows else None


def tab5_examples(R: dict) -> list[dict]:
    out = []
    split = read_json(SPL / "mathvista_split.json")
    sel = {}
    with gzip.open(RES / "mathvista" / "selections.csv.gz", "rt") as fh:
        for r in csv.DictReader(fh):
            if r["N"] == "32" and r["order"] == "0" and r["method"] in ("sc", "visconf_rwc"):
                sel[(r["sample_id"], r["method"])] = r
    for q in split["eval"]:
        a, b = sel.get((q, "sc")), sel.get((q, "visconf_rwc"))
        if a and b and a["answer"] != b["answer"] and a["correct"] != b["correct"]:
            out.append({"kind": "Selection disagreement (MathVista, N = 32, order 0)", "id": q,
                        "detail": f"SC → {a['answer']} ({'correct' if a['correct'] == '1' else 'wrong'}); "
                                  f"VisConf + RWC → {b['answer']} ({'correct' if b['correct'] == '1' else 'wrong'})"})
            break
    live = read_json(RES / "live" / "per_question.json")
    order = read_json(SPL / "math_split.json")["live"]
    for q in order:
        r = live.get(q)
        if r and r["refinement"]["answer_changed"] and r["initial_correct"] != r["refinement"]["correct"]:
            out.append({"kind": "Revision changes correctness (live MATH)", "id": q,
                        "detail": f"initial {r['initial_answer']} ({'correct' if r['initial_correct'] else 'wrong'}) → revised "
                                  f"{r['refinement']['answer']} ({'correct' if r['refinement']['correct'] else 'wrong'}); gold {r['gold']}"})
            break
    errs = list(csv.DictReader(open(RES / "math" / "early_stop_errors.csv")))
    mo = read_json(SPL / "math_split.json")["eval"]
    rank = {q: i for i, q in enumerate(mo)}
    cands = sorted([e for e in errs if e["family"] == "quality" and e["fixed32_correct"] == "1"],
                   key=lambda e: (rank.get(e["sample_id"], 10**9), int(e["order"])))
    if cands:
        e = cands[0]
        out.append({"kind": "Premature stop (MATH, quality-aware)", "id": e["sample_id"],
                    "detail": f"stopped after {e['stopped_at']} with {e['answer']} (wrong); fixed-32 SC would be correct (order {e['order']})"})
    rows = [[o["kind"], o["id"], o["detail"]] for o in out]
    save_table(OUT, "tab5_examples", ["Behaviour", "Question", "What happened"], rows,
               "Aligned examples, each the FIRST held-out question in split order meeting the condition: (1) at N = 32 "
               "(order 0) self-consistency and VisConf + RWC select different answers and exactly one is correct; (2) the "
               "first live question whose revision changes correctness; (3) the first quality-aware early stop that is wrong "
               "while fixed-32 self-consistency is right.", "tab:t3-examples", align="llp{8.2cm}")
    return out


# ============================================================================ appendix

def tabA1_all_curves(R: dict) -> None:
    for ds in DS:
        c = R[ds]["summary"]["curves"]["eval"]
        header = ["Method", *[f"N={n}" for n in c["N"]], "Tokens @32"]
        rows = [[LABEL[m], *[pct(v) for v in c["accuracy"][m]], f"{c['token_cost'][m][-1]:.0f}"] for m in c["accuracy"]]
        rows.append(["Oracle coverage", *[pct(v) for v in c["oracle_coverage"]], "–"])
        rows.append(["Distinct answer groups", *[f"{v:.2f}" for v in c["distinct_groups"]], "–"])
        save_table(OUT, f"tabA1_curves_{ds}", header, rows,
                   f"{DS[ds]}: held-out accuracy (%) of every rule at every budget, oracle coverage and mean distinct answer "
                   "groups (NO_ANSWER counts as a group).", f"tab:t3-curves-{ds}", align="l" + "r" * (len(c["N"]) + 1))


def tabA2_scorer_sensitivity(R: dict) -> None:
    if "mathvista_lenient" not in R:
        return
    o, l = R["mathvista"]["summary"], R["mathvista_lenient"]["summary"]
    header = ["Method", "Official N=8", "Lenient N=8", "Official N=32", "Lenient N=32"]
    co, cl = o["curves"]["eval"], l["curves"]["eval"]
    i8, i32 = co["N"].index(8), co["N"].index(32)
    rows = [[LABEL[m], pct(co["accuracy"][m][i8]), pct(cl["accuracy"][m][i8]), pct(co["accuracy"][m][i32]), pct(cl["accuracy"][m][i32])]
            for m in co["accuracy"]]
    rows.append(["NO_ANSWER rate (rollouts)", pct(o["data_stats"]["no_answer_rate"]), pct(l["data_stats"]["no_answer_rate"]), "", ""])
    rows.append(["Oracle coverage", pct(co["oracle_coverage"][i8]), pct(cl["oracle_coverage"][i8]),
                 pct(co["oracle_coverage"][i32]), pct(cl["oracle_coverage"][i32])])
    save_table(OUT, "tabA2_scorer_sensitivity", header, rows,
               "MathVista scorer sensitivity (held-out accuracy, %). Official: MathVista's answer-type / precision rules. "
               "Lenient (sensitivity check only): units, spacing commands, $, degree / percent signs and thousands "
               "separators removed before those rules, for grouping and grading alike.", "tab:t3-sensitivity", align="lrrrr")


def tabA3_bootstrap(R: dict) -> None:
    header = ["Dataset", "Comparison", *[f"N={n}" for n in (2, 4, 8, 16, 32)]]
    rows = []
    for ds in DS:
        boots = R[ds]["summary"]["bootstrap_eval"]
        pairs = sorted({(b["method"], b["baseline"]) for b in boots if b["baseline"] != "sc"})
        for m, base in pairs:
            row = [DS[ds].split(" ")[0], f"{LABEL[m]} − {LABEL[base]}"]
            for n in (2, 4, 8, 16, 32):
                b = next((x for x in boots if x["method"] == m and x["baseline"] == base and x["N"] == n), None)
                row.append("–" if b is None else f"{100 * b['diff']:+.1f} [{100 * b['ci95'][0]:+.1f}, {100 * b['ci95'][1]:+.1f}]")
            rows.append(row)
    save_table(OUT, "tabA3_bootstrap", header, rows,
               "Controlled comparisons (pp, paired question-level bootstrap, 1,000 resamples): same aggregation with a "
               "different score, or the same score with a different aggregation.", "tab:t3-boot", align="ll" + "r" * 5)


def tabA4_adaptive_grid(R: dict) -> None:
    header = ["Dataset", "Family", "τ", "γ", "Dev accuracy %", "Dev tokens / q.", "Dev rollouts", "Eligible"]
    rows = []
    for ds in DS:
        for fam in ("agreement", "quality"):
            sel = R[ds]["adaptive"][fam]["dev_selection"]
            base = sel["dev_fixed32_accuracy"]
            tol = R[ds]["adaptive"]["tolerance_pp"] / 100
            for g in sel["grid"]:
                chosen = g == sel["selected"]
                rows.append([DS[ds].split(" ")[0], g["family"], "–" if g["tau"] is None else g["tau"],
                             "–" if g["gamma"] is None else g["gamma"], pct(g["accuracy"]), f"{g['mean_cost']:.0f}",
                             f"{g['mean_rollouts']:.1f}", ("selected" if chosen else "yes") if g["accuracy"] >= base - tol - 1e-12 else "no"])
    save_table(OUT, "tabA4_adaptive_grid", header, rows,
               "Development-set grids for adaptive stopping (eligible = within 1 pp of fixed-32 self-consistency).",
               "tab:t3-grid", align="llrrrrrl")


def tabA5_data_scorer(R: dict) -> None:
    rows = []
    for ds in DS:
        s = R[ds]["summary"]
        st, sc = s["data_stats"], s["scorer"]
        rows += [[DS[ds], "Rollout accuracy / NO_ANSWER / capped",
                  f"{pct(st['rollout_accuracy'])}% / {pct(st['no_answer_rate'])}% / {pct(st['capped_rate'])}%"],
                 ["", "Mean retained tokens per rollout", f"{st['mean_tokens']:.0f}"],
                 ["", "Splits (dev / held-out)", f"{s['split_sizes']['dev']} / {s['split_sizes']['eval']}"]]
        if ds == "math":
            t = sc.get("math_verify_timeouts") or {}
            rows.append(["", f"Math-Verify {sc['math_verify']}: comparisons hitting the time limit (→ not equivalent)",
                         f"{t.get('n_timed_out_comparisons', '–')} in {t.get('n_questions', '–')} questions"])
        else:
            m = sc.get("mathvista_metadata") or {}
            rows.append(["", "MathVista metadata (testmini.json) SHA-256", (m.get("sha256") or "–")[:16] + "…"])
    save_table(OUT, "tabA5_data_scorer", ["Dataset", "Item", "Value"], rows,
               "Data and scorer details (all 1,000 questions per dataset).", "tab:t3-data", align="llr")


def figA1_all_methods(R: dict) -> None:
    """Printed at 5.45 in x 3.1 in. Shared y; the legend sits in its own band below the axis labels."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN - 0.05, 3.1), sharey=True)
    fig.subplots_adjust(wspace=0.08, left=0.08, right=0.99, top=0.92, bottom=0.42)
    for ax, ds in zip(axes, DS):
        c = R[ds]["summary"]["curves"]["eval"]
        ax.plot(c["N"], c["oracle_coverage"], "k--", lw=1.0, label="Oracle coverage")
        for m in c["accuracy"]:
            ax.plot(c["N"], c["accuracy"][m], "-o", color=COLOR[m], ms=2.8, label=LABEL[m])
        ax.set_xscale("log", base=2)
        ax.set_xticks(c["N"])
        ax.set_xticklabels([str(n) for n in c["N"]])
        ax.set_title(DS[ds])
        ax.set_xlabel("Candidates N")
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Held-out accuracy")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    save_fig(fig, OUT, "figA1_all_methods")


def figA2_stopping_vs_correct() -> None:
    """Printed at 5.45 in x 2.5 in. Shared axes; each panel keeps its labels inside its own space."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN - 0.05, 2.5), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.12, left=0.09, right=0.99, top=0.9, bottom=0.3)
    for ax, ds in zip(axes, DS):
        rows = [r for r in csv.DictReader(open(RES / ds / "per_question.csv")) if r["split"] == "eval"]
        x = np.array([float(r["fraction_correct_rollouts"]) for r in rows])
        for fam, col, mk, dy in (("agreement", "#1f77b4", "o", -0.3), ("quality", "#d62728", "^", 0.3)):
            yv = np.array([float(r[f"mean_stop_{fam}"]) for r in rows])
            ax.scatter(x, yv + dy, s=7, alpha=0.4, color=col, marker=mk, lw=0, label={"agreement": "agreement-based stopping", "quality": "quality-aware stopping"}[fam])
        ax.set_xlabel("Correct share of the 32 rollouts")
        ax.set_title(DS[ds])
        ax.set_yticks([4, 8, 16, 32])
        ax.set_xlim(-0.03, 1.03)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Mean rollouts consumed")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), markerscale=2)
    save_fig(fig, OUT, "figA2_stopping_vs_correct")


def figA3_trajectories(cfg_path: str) -> str:
    """Per-token calibrated confidence c_t and visual engagement v_t for the first held-out MathVista question (split
    order) with 25-75% correct rollouts: its first two correct and first two incorrect rollouts (rollout-ID order)."""
    try:
        from common.config import load_config
        from task3.data import prepare
        cfg = load_config(cfg_path)
        prep = prepare("mathvista", cfg)
    except Exception as exc:
        return f"skipped ({type(exc).__name__}: {exc})"
    split = read_json(SPL / "mathvista_split.json")
    for q in split["eval"]:
        d = prep["questions"][q]
        corr = np.array([d["correct"][int(g)] for g in d["groups"]])
        if 0.25 <= corr.mean() <= 0.75:
            break
    pooled = np.concatenate(d["k_tok"]).astype(np.float64)
    lo, hi = np.quantile(pooled, [0.05, 0.95])
    picks = [i for i in range(32) if corr[i]][:2] + [i for i in range(32) if not corr[i]][:2]
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN - 0.05, 3.0), sharex=True, gridspec_kw={"hspace": 0.12})
    w = 15
    for i in picks:
        c = np.clip((d["k_tok"][i] - lo) / max(hi - lo, 1e-8), 0, 1)
        v = d["v_tok"][i]
        sm = lambda a: np.convolve(a, np.ones(w) / w, mode="valid") if len(a) >= w else a  # noqa: E731
        style = "-" if corr[i] else "--"
        col = "#2ca02c" if corr[i] else "#d62728"
        axes[0].plot(sm(c), style, color=col, lw=0.8, label=f"rollout {i} ({'correct' if corr[i] else 'wrong'})")
        axes[1].plot(sm(v), style, color=col, lw=0.8)
    axes[0].set_ylabel(r"confidence $c_t$")
    axes[1].set_ylabel(r"visual engagement $v_t$")
    axes[1].set_xlabel(f"generated token (moving average over {w})")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.2, top=0.92, left=0.1, right=0.99)
    axes[0].set_title(f"MathVista question {q}: two correct and two incorrect rollouts")
    save_fig(fig, OUT, "figA3_trajectories")
    return f"question {q}, rollouts {picks}"


# ============================================================================ main

def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--config", default="configs/task3/base.yaml")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    OUT = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
    rcparams()
    R = {ds: load(ds) for ds in DS}
    if (RES / "mathvista_lenient" / "summary.json").exists():
        R["mathvista_lenient"] = load("mathvista", "_lenient")
    done, failed, notes = [], [], {}
    steps = [("fig1_compute_curves", lambda: fig1_compute_curves(R)), ("tab1_methods", lambda: tab1_methods(R)),
             ("fig2_failures_signals", lambda: fig2_failures_signals(R)), ("tab2_signals", lambda: tab2_signals(R)),
             ("tab3_live", tab3_live), ("tab4_adaptive", lambda: tab4_adaptive(R)),
             ("tab5_examples", lambda: notes.__setitem__("examples", tab5_examples(R))),
             ("tabA1_all_curves", lambda: tabA1_all_curves(R)), ("tabA2_scorer_sensitivity", lambda: tabA2_scorer_sensitivity(R)),
             ("tabA3_bootstrap", lambda: tabA3_bootstrap(R)), ("tabA4_adaptive_grid", lambda: tabA4_adaptive_grid(R)),
             ("tabA5_data_scorer", lambda: tabA5_data_scorer(R)), ("figA1_all_methods", lambda: figA1_all_methods(R)),
             ("figA2_stopping_vs_correct", figA2_stopping_vs_correct),
             ("figA3_trajectories", lambda: notes.__setitem__("trajectories", figA3_trajectories(args.config)))]
    for name, fn in steps:
        try:
            fn()
            done.append(name)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    (OUT / "selection_notes.json").write_text(json.dumps(notes, indent=2, default=str))
    print(f"\nwrote to {OUT.relative_to(REPO_ROOT)}: {', '.join(done)}")
    print(f"trajectory figure: {notes.get('trajectories')}")
    if failed:
        print("FAILED:\n  " + "\n  ".join(failed))
    if args.push:
        try_push([OUT], "task3/report: figures and tables")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
