"""All Task 1 figures and tables for the report (CPU only; reads committed results only).

    python -m task1.make_report_assets
    python -m task1.make_report_assets --excerpt enough_of_my_husband:template --push

Outputs (default: report/figs/task1/ at the repo root; change with --fig-dir / --table-dir):
  fig1_training_tradeoff.{pdf,png}   (main)  SFT loss + LR | L2-SP penalty vs loss | adaptation-retention
  fig2_diagnostics.{pdf,png}         (main)  rho_k, eta, R(z) math, R(z) non-math by layer
  each table as .tex (booktabs, \\input-ready), .csv and .md
            main:     tab1_benchmarks, tab2_resources, tab3_annotation_summary, tab4_excerpts (needs --excerpt)
            appendix: tabA1_benchmarks_full, tabA2_boolq, tabA3_diagnostics, tabA4_verification,
                      tabA5_auto_flags, tabA6_data, tabA7_excerpts_all

Inputs: results/task1/{eval,diagnostics,qualitative,data,<run>/} and, for the annotation
tables, your file results/task1/qualitative/annotation_sheet_annotated.csv.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.config import REPO_ROOT  # noqa: E402
from common.publish import try_push  # noqa: E402
from task1.qualitative import MATH_PATTERNS, auto_flags, load_prompts  # noqa: E402

RES = REPO_ROOT / "results" / "task1"
DEFAULT_OUT = REPO_ROOT / "report" / "figs" / "task1"
FIG = TAB = DEFAULT_OUT  # overridable with --fig-dir / --table-dir

CONDS = ["baseline", "full_ft_midpoint", "full_ft_final", "l2sp_final", "lora4_final", "lora16_final", "oplora16_final"]
TRAINED = CONDS[1:]
RUNS = ["full_ft", "l2sp", "lora4", "lora16", "oplora16"]
LABEL = {"baseline": "Baseline", "full_ft_midpoint": "Full FT (mid)", "full_ft_final": "Full FT",
         "l2sp_final": "Full FT + L2-SP", "lora4_final": "LoRA-4", "lora16_final": "LoRA-16",
         "oplora16_final": "OPLoRA-16"}
RUN_LABEL = {"full_ft": "Full FT", "l2sp": "Full FT + L2-SP", "lora4": "LoRA-4", "lora16": "LoRA-16",
             "oplora16": "OPLoRA-16"}
COLOR = {"baseline": "#7f7f7f", "full_ft_midpoint": "#9ecae1", "full_ft_final": "#1f77b4",
         "l2sp_final": "#ff7f0e", "lora4_final": "#98df8a", "lora16_final": "#2ca02c", "oplora16_final": "#9467bd"}
RUN_COLOR = {r: COLOR[f"{r}_final"] for r in RUNS}
BENCH = [("gsm8k", "exact_match,flexible-extract", "GSM8K"), ("hellaswag", "acc_norm,none", "HellaSwag"),
         ("boolq", "acc,none", "BoolQ"), ("arc_easy", "acc_norm,none", "ARC-E")]
NONMATH = ["hellaswag", "boolq", "arc_easy"]
WIDTH = 5.5  # NeurIPS text width (inches)

from common.report_utils import FONT_PT, rcparams, write_table  # noqa: E402

rcparams()  # Times-style serif, >= 7 pt text at printed size


# ============================================================================ loading

def rjson(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def rjsonl(p: Path) -> list[dict]:
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def eval_metric(cond: str, bench: str, key: str):
    p = RES / "eval" / cond / f"{bench}.json"
    return rjson(p)["metrics"].get(key) if p.exists() else None


def eval_field(cond: str, bench: str, field: str):
    p = RES / "eval" / cond / f"{bench}.json"
    return rjson(p).get(field) if p.exists() else None


def run_log(run: str) -> list[dict]:
    return [r for r in rjsonl(RES / run / "train_log.jsonl") if r.get("event") == "update"]


def run_summary(run: str) -> dict:
    return rjson(RES / run / "summary.json")


def plain_log_ticks(ax) -> None:
    """Readable tick labels (0.2, 0.05, ...) instead of 2x10^-1 on log-scaled y axes."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def ema(x, span: int = 50):
    a, out, m = 2 / (span + 1), [], None
    for v in x:
        m = v if m is None else a * v + (1 - a) * m
        out.append(m)
    return np.array(out)


# ============================================================================ tables

# Unicode notation used in headers/captions -> LaTeX (applied before escaping, via placeholders)
TEX_TOKENS = [("‖θ−θ0‖₂", r"$\|\theta-\theta_0\|_2$"), ("‖ΔW‖F", r"$\|\Delta W\|_F$"),
              ("‖Uk^T B~‖", r"$\|U_k^\top \tilde B\|$"), ("‖A~ Vk‖", r"$\|\tilde A V_k\|$"),
              ("λ = 10⁻³", r"$\lambda = 10^{-3}$"), ("λΩ", r"$\lambda\Omega$"), ("ρk", r"$\rho_k$"),
              ("θ0", r"$\theta_0$"), ("η", r"$\eta$"), ("Δ", r"$\Delta$"), ("±", r"$\pm$"), ("→", r"$\to$")]


def tex_escape(s) -> str:
    s = str(s)
    holders = {}
    for i, (tok, tex) in enumerate(TEX_TOKENS):
        if tok in s:
            key = f"\x00{i}\x00"
            s = s.replace(tok, key)
            holders[key] = tex
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    for key, tex in holders.items():
        s = s.replace(key, tex)
    return s


def save_table(name: str, header: list[str], rows: list[list], caption: str, label: str, align: str | None = None,
               small: bool = True) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    flat = [str(h).replace("\n", " ") for h in header]
    with open(TAB / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(flat)
        w.writerows(rows)
    md = ["| " + " | ".join(flat) + " |", "|" + "---|" * len(flat)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    (TAB / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    write_table(TAB / f"{name}.tex", header, rows, caption, label, align or "l" + "r" * (len(header) - 1))


def pct(v, nd=1) -> str:
    return "–" if v is None else f"{100 * v:.{nd}f}"


def delta_cell(v, base, nd=1) -> str:
    if v is None:
        return "–"
    if base is None or v is base:
        return f"{100 * v:.{nd}f}"
    return f"{100 * v:.{nd}f} ({100 * (v - base):+.{nd}f})"


# ============================================================================ main-body tables

def scores() -> dict:
    return {c: {b: eval_metric(c, b, k) for b, k, _ in BENCH} for c in CONDS}


def mean_nonmath_delta(s: dict, cond: str):
    vals = [s[cond][b] - s["baseline"][b] for b in NONMATH if s[cond][b] is not None and s["baseline"][b] is not None]
    return 100 * sum(vals) / len(vals) if len(vals) == len(NONMATH) else None


def tab1_benchmarks(s: dict) -> None:
    header = ["Condition", *[f"{lab} (%)" for _, _, lab in BENCH], "Mean non-math Δpp", "GSM8K unextr. %",
              "GSM8K trunc. %"]
    rows = []
    for c in CONDS:
        mn = mean_nonmath_delta(s, c)
        rows.append([LABEL[c], *[delta_cell(s[c][b], None if c == "baseline" else s["baseline"][b]) for b, _, _ in BENCH],
                     "–" if c == "baseline" or mn is None else f"{mn:+.2f}",
                     pct(eval_field(c, "gsm8k", "unextractable_rate")), pct(eval_field(c, "gsm8k", "truncation_rate"))])
    save_table("tab1_benchmarks", header, rows,
               "Benchmark results (zero-shot). Primary metrics: GSM8K flexible-extract exact match; HellaSwag and "
               "ARC-E normalized accuracy; BoolQ accuracy. Parentheses: change from the shared baseline in percentage "
               "points (pp). Mean non-math Δpp averages HellaSwag, BoolQ and ARC-E. The last two columns separate "
               "GSM8K answer-extraction failures and 1,024-token truncation from incorrect answers.",
               "tab:t1-bench")


def _bytes(n) -> str:
    if n is None:
        return "–"
    return f"{n / 2**30:.2f} GB" if n >= 2**30 else f"{n / 2**20:.1f} MB"


def tab2_resources() -> None:
    header = ["Method", "Trainable\nparams", "Train\ntime (h)", "Peak GPU\nmem. (GB)", "Exported\nweights",
              "‖θ−θ0‖₂", "Method-specific overhead"]
    rows = []
    for run in RUNS:
        sm = run_summary(run)
        p = sm["params"]
        stage = sm["stages"].get("final", {})
        exported = sum((stage.get("bytes") or {}).values()) if stage.get("bytes") else None
        disp = stage.get("displacement_l2")
        overhead = "–"
        if run == "l2sp":
            l2 = sm["l2sp"]
            overhead = f"θ0 copy {_bytes(l2['ref_bytes'])} (CPU); penalty {l2['total_penalty_seconds'] / 60:.1f} min"
        ad = sm["model"].get("adapter")
        if ad and ad.get("protect_rank"):
            overhead = f"SVD+QR {ad['svd_seconds']:.0f} s; bases {_bytes(ad['basis_bytes_fp32'])}"
        rows.append([RUN_LABEL[run], f"{p['trainable'] / 1e6:.2f}M ({100 * p['trainable_fraction']:.2f}%)",
                     f"{sm['training']['train_seconds'] / 3600:.2f}", f"{sm['training']['peak_allocated_gb']:.2f}",
                     _bytes(exported), "–" if disp is None else f"{disp:.2f}", overhead])
    save_table("tab2_resources", header, rows,
               "Training cost (1,250 updates, 40,000 examples, one T4 per run; two runs shared each machine) and "
               "displacement ‖θ−θ0‖₂ over all unique parameters at the final checkpoint.", "tab:t1-resources",
               align="lrrrrrl")


ANNOT = RES / "qualitative" / "annotation_sheet_annotated.csv"


def load_annotations() -> list[dict] | None:
    if not ANNOT.exists():
        print(f"NOTE: {ANNOT.relative_to(REPO_ROOT)} not found; skipping annotation tables")
        return None
    with open(ANNOT, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def tab3_annotation_summary(ann: list[dict]) -> None:
    header = ["Condition", "Format: yes/partial/no", "Mean coherence (1–3)", "Repetition: some/severe",
              "Math framing", "Truncated", "Inappropriate"]
    rows = []
    for c in CONDS:
        rs = [r for r in ann if r["condition"] == c]
        fc = [r["format_compliance"].strip().lower() for r in rs]
        rep = [r["repetition"].strip().lower() for r in rs]
        coh = [float(r["coherence_relevance"]) for r in rs if r["coherence_relevance"].strip()]
        rows.append([LABEL[c], f"{fc.count('yes')}/{fc.count('partial')}/{fc.count('no')}",
                     f"{sum(coh) / len(coh):.2f}" if coh else "–", f"{rep.count('some')}/{rep.count('severe')}",
                     f"{sum(r['unnecessary_math_framing'].strip().lower() == 'yes' for r in rs)}/{len(rs)}",
                     f"{sum(r['truncated'].strip() == 'True' for r in rs)}/{len(rs)}",
                     f"{sum(r['inappropriate_content'].strip().lower() == 'yes' for r in rs)}/{len(rs)}"])
    save_table("tab3_annotation_summary", header, rows,
               "Manual annotation of the fixed 42-response subset (first and last plain questions × plain, JSON and "
               "template formats; one greedy response each, 512-token cap). Counts are out of 6 responses per "
               "condition.", "tab:t1-annotation")


def excerpt_rows(ann: list[dict], question: str, fmt: str) -> list[list]:
    rows = []
    for c in CONDS:
        r = next((x for x in ann if x["condition"] == c and x["question"] == question and x["format"] == fmt), None)
        if r:
            rows.append([LABEL[c], r["format_compliance"], r["coherence_relevance"], r["excerpt"].replace("\n", " ")])
    return rows


def tab4_excerpts(ann: list[dict], spec: str | None) -> None:
    if not spec:
        print("NOTE: pass --excerpt QUESTION:FORMAT to build the main-body excerpt table (tab4); "
              "all six are in tabA7")
        return
    q, f = spec.split(":")
    save_table("tab4_excerpts", ["Condition", "Format", "Coherence", "Excerpt"], excerpt_rows(ann, q, f),
               f"Aligned excerpts: “{q}” ({f} format), same prompt across all conditions.",
               "tab:t1-excerpts", align="lccp{7.2cm}")


# ============================================================================ figures

def fig1_training_tradeoff(s: dict) -> None:
    """Printed at 5.5 in x 2.3 in (scale 1): loss + LR | L2-SP penalty vs loss | adaptation vs retention."""
    fig = plt.figure(figsize=(WIDTH, 2.3))
    gs = fig.add_gridspec(1, 3, wspace=0.42, left=0.07, right=0.995, top=0.9, bottom=0.33)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))

    logs = {r: run_log(r) for r in RUNS}
    for r in RUNS:
        u = np.array([x["update"] for x in logs[r]])
        loss = np.array([x["loss"] for x in logs[r]])
        ax_a.plot(u, loss, color=RUN_COLOR[r], alpha=0.12, lw=0.4)
        ax_a.plot(u, ema(loss), color=RUN_COLOR[r], lw=1.1)
    ax_a.set_yscale("log")
    plain_log_ticks(ax_a)
    ax_a.set_xlabel("Optimizer update")
    ax_a.set_ylabel("SFT loss (nats/token)")
    ax_a.set_title("(a) Training loss")
    lr = [next(iter(x["lr"].values())) for x in logs["full_ft"]]
    twin = ax_a.twinx()
    twin.plot([x["update"] for x in logs["full_ft"]], np.array(lr) / max(lr), color="k", ls=":", lw=0.9)
    twin.set_ylim(0, 1.6)
    twin.set_yticks([])
    twin.spines["right"].set_visible(False)

    l2 = logs["l2sp"]
    u = np.array([x["update"] for x in l2])
    ax_b.plot(u, ema([x["loss"] for x in l2]), color=RUN_COLOR["l2sp"], lw=1.1, label="SFT loss")
    ax_b.plot(u, [x["lambda_omega"] for x in l2], color=RUN_COLOR["l2sp"], lw=1.1, ls="--", label="λΩ")
    st = run_summary("full_ft")["stages"]
    pts = [(rjson_stage("full_ft", k)["update"], 1e-3 * 0.5 * rjson_stage("full_ft", k)["displacement_l2"] ** 2)
           for k in ("midpoint", "final") if k in st]
    if pts:
        ax_b.plot(*zip(*pts), "o", color=RUN_COLOR["full_ft"], ms=4, label="λΩ of Full FT")
    ax_b.set_yscale("log")
    plain_log_ticks(ax_b)
    ax_b.set_xlabel("Optimizer update")
    ax_b.set_ylabel("Objective term (nats/token)")
    ax_b.set_title("(b) L2-SP penalty vs. loss")
    ax_b.legend(loc="lower right", frameon=False)

    ax_c.axhline(0, color="0.8", lw=0.6)
    ax_c.axvline(0, color="0.8", lw=0.6)
    xy = {}
    for c in CONDS:
        if s[c]["gsm8k"] is None or s["baseline"]["gsm8k"] is None:
            continue
        x = 100 * (s[c]["gsm8k"] - s["baseline"]["gsm8k"])
        y = mean_nonmath_delta(s, c) if c != "baseline" else 0.0
        if y is None:
            continue
        xy[c] = (x, y)
        ax_c.scatter([x], [y], s=24, color=COLOR[c], edgecolor="k", lw=0.4, zorder=3)
    if "full_ft_midpoint" in xy and "full_ft_final" in xy:
        ax_c.annotate("", xy=xy["full_ft_final"], xytext=xy["full_ft_midpoint"],
                      arrowprops={"arrowstyle": "->", "color": COLOR["full_ft_final"], "lw": 0.8})
    ax_c.set_xlabel("GSM8K change (pp)")
    ax_c.set_ylabel("Mean non-math change (pp)")
    ax_c.set_title("(c) Adaptation vs. retention")

    # one shared legend: condition colours (all panels) + the LR line of panel (a)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLOR[c], marker="o", ms=4, lw=1.1, mec="k", mew=0.3, label=LABEL[c])
               for c in CONDS]
    handles.append(Line2D([], [], color="k", ls=":", lw=0.9, label="LR / peak LR (a)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 0.0))
    save_fig(fig, "fig1_training_tradeoff")


def rjson_stage(run: str, stage: str) -> dict:
    return run_summary(run)["stages"][stage]


def fig2_diagnostics() -> dict:
    """Printed at 5.45 in x 2.25 in (scale 1), one row. Panels (c) and (d) share one y scale."""
    p = RES / "diagnostics" / "summary.json"
    d = rjson(p)
    layers = d["layers"]
    ws = {(r["condition"], r["layer"]): r for r in d["weight_space"]}
    act = defaultdict(lambda: defaultdict(list))
    with open(RES / "diagnostics" / "activation_R.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            act[(r["condition"], int(r["layer"]))][r["set"]].append(float(r["R"]))
    x = np.arange(len(layers))
    offs = np.linspace(-0.3, 0.3, len(TRAINED))
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH - 0.05, 2.25), sharex=True)
    a_rho, a_eta, a_rm, a_rn = axes
    a_rn.sharey(a_rm)
    for i, c in enumerate(TRAINED):
        kw = {"color": COLOR[c], "marker": "o", "ms": 3.5, "ls": "none", "label": LABEL[c]}
        a_rho.plot(x + offs[i], [ws[(c, l)]["rho_k"] for l in layers], **kw)
        a_eta.plot(x + offs[i], [ws[(c, l)]["eta"] for l in layers], **kw)
        for ax, setname in ((a_rm, "math"), (a_rn, "nonmath")):
            med = [np.median(act[(c, l)][setname]) for l in layers]
            lo = [np.percentile(act[(c, l)][setname], 25) for l in layers]
            hi = [np.percentile(act[(c, l)][setname], 75) for l in layers]
            ax.errorbar(x + offs[i], med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], capsize=1.5,
                        elinewidth=0.6, **kw)
    k, dout = d["k"], d["weight_space"][0]["d_out"]
    a_rho.axhline(k / dout, color="k", ls=":", lw=0.8)
    a_rho.text(len(layers) - 0.55, k / dout * 0.9, f"random ({k}/{dout})", va="top", ha="right",
               fontsize=FONT_PT["small"])
    a_rho.set_ylim(top=max(0.2, a_rho.get_ylim()[1] * 1.25))
    a_rho.set_ylabel(r"$\rho_k$ ($k$ = %d)" % k)
    a_rho.set_title(r"(a) Subspace energy $\rho_k$")
    a_eta.set_ylabel(r"$\eta$")
    a_eta.set_title(r"(b) Update norm $\eta$")
    a_rm.set_yscale("log")
    plain_log_ticks(a_rm)
    a_rm.set_ylabel(r"$R(z)$, median and IQR")
    a_rm.set_title(r"(c) $R(z)$, GSM8K")
    a_rn.set_title(r"(d) $R(z)$, HellaSwag")
    plt.setp(a_rn.get_yticklabels(), visible=False)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_xlabel("Layer")
    for ax in axes.flat:
        ax.set_xlim(-0.5, len(layers) - 0.5)
    h, lab = a_rho.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(TRAINED), frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=0.8)
    fig.subplots_adjust(wspace=0.55, bottom=0.33, left=0.075, right=0.995, top=0.9)
    save_fig(fig, "fig2_diagnostics")
    return d


def save_fig(fig, name: str) -> None:
    """Saved at its drawn size: include with \\includegraphics{...} (scale 1)."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


# ============================================================================ appendix tables

def tabA1_benchmarks_full(s: dict) -> None:
    header = ["Condition", "GSM8K flex", "GSM8K strict", "GSM8K mean tokens", "HellaSwag acc_norm", "HellaSwag acc",
              "BoolQ acc", "ARC-E acc_norm", "ARC-E acc"]
    keys = [("gsm8k", "exact_match,flexible-extract"), ("gsm8k", "exact_match,strict-match"), None,
            ("hellaswag", "acc_norm,none"), ("hellaswag", "acc,none"), ("boolq", "acc,none"),
            ("arc_easy", "acc_norm,none"), ("arc_easy", "acc,none")]
    rows = []
    for c in CONDS:
        row = [LABEL[c]]
        for k in keys:
            if k is None:
                v = eval_field(c, "gsm8k", "mean_response_tokens")
                row.append("–" if v is None else f"{v:.0f}")
                continue
            b, m = k
            v = eval_metric(c, b, m)
            se = eval_metric(c, b, m.replace(",", "_stderr,", 1))
            ok_se = isinstance(se, (int, float)) and not math.isnan(se)
            row.append("–" if v is None else (f"{100 * v:.1f} ± {100 * se:.1f}" if ok_se else f"{100 * v:.1f}"))
        rows.append(row)
    save_table("tabA1_benchmarks_full", header, rows,
               "All benchmark metrics (%, ± lm-eval standard error). Raw accuracy is reported alongside normalized "
               "accuracy; GSM8K strict-match requires the “The answer is” format.", "tab:t1-bench-full")


def tabA2_boolq() -> None:
    header = ["Condition", "Predicts “yes” (%)", "Acc. on gold-yes (%)", "Acc. on gold-no (%)", "Balanced acc. (%)",
              "Acc. (%)"]
    rows, gold_yes = [], None
    for c in CONDS:
        p = RES / "eval" / c / "boolq_samples.jsonl.gz"
        if not p.exists():
            continue
        rs = rjsonl(p)
        yes = [r for r in rs if int(r["gold"]) == 1]
        no = [r for r in rs if int(r["gold"]) == 0]
        gold_yes = len(yes) / len(rs)
        a_yes = sum(r["pred_raw"] == 1 for r in yes) / len(yes)
        a_no = sum(r["pred_raw"] == 0 for r in no) / len(no)
        rows.append([LABEL[c], pct(sum(r["pred_raw"] == 1 for r in rs) / len(rs)), pct(a_yes), pct(a_no),
                     pct((a_yes + a_no) / 2), pct(sum(r["pred_raw"] == int(r["gold"]) for r in rs) / len(rs))])
    save_table("tabA2_boolq", header, rows,
               f"BoolQ answer balance (validation, 3,270 questions; gold “yes” share {pct(gold_yes)}%). Predictions are "
               "the higher-likelihood choice; balanced accuracy averages accuracy over gold-yes and gold-no questions.",
               "tab:t1-boolq")


def tabA3_diagnostics(d: dict) -> None:
    header = ["Condition", "Layer", "ρk", "ρk (input)", "η", "‖ΔW‖F", "R math median", "R non-math median",
              "Median ratio"]
    summ = {(r["condition"], r["layer"]): r for r in d["activation_summary"]}
    rows = []
    for r in d["weight_space"]:
        a = summ[(r["condition"], r["layer"])]
        own = r.get("rho_k_own_bases")
        rows.append([LABEL[r["condition"]], r["layer"], f"{r['rho_k']:.4f}" + (f" ({own:.1e})" if own is not None else ""),
                     f"{r['rho_k_in']:.4f}", f"{r['eta']:.4f}", f"{r['dW_fro']:.3f}", f"{a['R_math_median']:.4f}",
                     f"{a['R_nonmath_median']:.4f}", f"{a['R_median_ratio_math_over_nonmath']:.2f}"])
    save_table("tabA3_diagnostics", header, rows,
               f"down_proj diagnostics (FP32; k = {d['k']}; random-direction reference ρk = k/d_out = "
               f"{d['k'] / d['weight_space'][0]['d_out']:.3f}). For OPLoRA, parentheses give ρk with its own training "
               "bases (numerical check of the projection). R over 32 GSM8K and 32 HellaSwag probe prompts.",
               "tab:t1-diag-full")


def tabA4_verification() -> None:
    header = ["Method", "Merged vs. unmerged: argmax agree (%)", "Mean |Δlogit| merge / adapter effect",
              "Max rel. ‖Uk^T B~‖", "Max rel. ‖A~ Vk‖", "Basis ortho. error (SVD → QR)"]
    rows = []
    for run in ("lora4", "lora16", "oplora16"):
        sm = run_summary(run)
        mc = sm.get("merge_check", {})
        pc = sm["stages"].get("final", {}).get("projection_check") or {}
        ad = sm["model"].get("adapter", {})
        rows.append([RUN_LABEL[run],
                     pct(mc.get("merged_vs_unmerged", {}).get("argmax_agreement"), 2),
                     f"{mc['relative_merge_error']:.4f}" if "relative_merge_error" in mc else "–",
                     f"{pc['max_rel_UkT_B']:.1e}" if pc else "–", f"{pc['max_rel_A_Vk']:.1e}" if pc else "–",
                     (f"{ad['max_ortho_error_svd']:.1e} → {ad['max_ortho_error_after_qr']:.1e}"
                      if "max_ortho_error_svd" in ad else "–")])
    save_table("tabA4_verification", header, rows,
               "Numerical checks (separate from benchmark evidence). Merge check on 4 training sequences: merged FP16 "
               "model vs. unmerged adapter model, with the merge error relative to the adapter's own effect "
               "(unmerged vs. base). OPLoRA constraint residuals of the exported projected factors.",
               "tab:t1-verify")


def _repeated_lines(resp: str) -> float:
    lines = [re.sub(r"\d+", "#", l.strip().lower()) for l in resp.splitlines() if l.strip()]
    seen, dup = set(), 0
    for l in lines:
        dup += l in seen
        seen.add(l)
    return dup / len(lines) if lines else 0.0


def tabA5_auto_flags() -> None:
    prompts = {p["id"]: p for p in load_prompts("assets/task1/first_plot_questions.yaml")}
    pats = MATH_PATTERNS + [r"####"]
    header = ["Condition", "Truncated /24", "JSON valid as-is /8", "JSON valid after unfencing /8",
              "Template keys complete /8", "With math markers /24", "Mean repeated-line fraction"]
    rows = []
    for c in CONDS:
        p = RES / "qualitative" / f"{c}.jsonl"
        if not p.exists():
            continue
        rs = rjsonl(p)
        flags = [{**r, **auto_flags(prompts[r["id"]], r["response"])} for r in rs]  # recomputed with current rules
        js = [f for f in flags if f["format"] == "json"]
        tp = [f for f in flags if f["format"] == "template"]
        rows.append([LABEL[c], sum(f["truncated"] for f in flags), sum(f["json_valid_as_is"] for f in js),
                     sum(f["json_valid_after_unfencing"] for f in js), sum(f["template_required_filled"] for f in tp),
                     sum(any(re.search(pt, f["response"]) for pt in pats) for f in flags),
                     f"{np.mean([_repeated_lines(f['response']) for f in flags]):.3f}"])
    save_table("tabA5_auto_flags", header, rows,
               "Automatic checks over all 24 qualitative prompts (8 questions × 3 formats) per condition. Math markers "
               "include LaTeX, equations, “The answer is” and “####”. Template keys complete: every non-optional key "
               "filled with a value other than the template's placeholder.", "tab:t1-autoflags")


def tabA6_data() -> None:
    st = rjson(RES / "data" / "selection_stats.json")
    rows = [["Rows scanned (source order)", f"{st['rows_scanned']:,}"],
            ["Selected examples", f"{st['n_selected']:,}"],
            *[[f"Excluded: {k.replace('_', ' ')}", f"{v:,}"] for k, v in st["excluded"].items()],
            ["GSM8K test questions checked (normalized)", f"{st['gsm8k_test_questions']:,}"],
            ["Responses with leading newlines stripped", f"{st['n_responses_with_leading_newlines_stripped']:,}"],
            ["Input tokens (total)", f"{st['total_input_tokens'] / 1e6:.2f}M"],
            ["Supervised tokens (total)", f"{st['total_supervised_tokens'] / 1e6:.2f}M"],
            ["Formatted length mean / p99 / max", f"{st['length_mean']:.0f} / {st['length_p99']} / {st['length_max']}"],
            ["Template render mismatches", f"{st['template_check']['n_render_mismatch']}"]]
    save_table("tabA6_data", ["Item", "Value"], rows,
               f"MetaMathQA selection. Normalization for decontamination: {st['normalization_rule']}.", "tab:t1-data",
               align="lr")


def tabA7_excerpts_all(ann: list[dict]) -> None:
    rows = []
    for q in ("what_is_your_wish", "enough_of_my_husband"):
        for f in ("plain", "json", "template"):
            for r in excerpt_rows(ann, q, f):
                rows.append([f"{q} / {f}", *r])
    save_table("tabA7_excerpts_all", ["Prompt", "Condition", "Format", "Coherence", "Excerpt"], rows,
               "All annotated excerpts (42 responses).", "tab:t1-excerpts-all", align="llccp{6cm}")


# ============================================================================ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--excerpt", default=None, help="QUESTION:FORMAT for the main-body excerpt table, "
                                                        "e.g. enough_of_my_husband:template")
    parser.add_argument("--fig-dir", default=str(DEFAULT_OUT.relative_to(REPO_ROOT)), help="figure output folder")
    parser.add_argument("--table-dir", default=str(DEFAULT_OUT.relative_to(REPO_ROOT)), help="table output folder")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    global FIG, TAB
    FIG = Path(args.fig_dir) if Path(args.fig_dir).is_absolute() else REPO_ROOT / args.fig_dir
    TAB = Path(args.table_dir) if Path(args.table_dir).is_absolute() else REPO_ROOT / args.table_dir

    s = scores()
    missing = [c for c in CONDS if any(v is None for v in s[c].values())]
    if missing:
        print(f"WARNING: incomplete benchmark results for {missing}")
    steps = [("tab1_benchmarks", lambda: tab1_benchmarks(s)), ("tab2_resources", tab2_resources),
             ("fig1_training_tradeoff", lambda: fig1_training_tradeoff(s)),
             ("tabA1_benchmarks_full", lambda: tabA1_benchmarks_full(s)), ("tabA2_boolq", tabA2_boolq),
             ("tabA4_verification", tabA4_verification), ("tabA5_auto_flags", tabA5_auto_flags),
             ("tabA6_data", tabA6_data)]
    done, failed = [], []
    for name, fn in steps:
        try:
            fn()
            done.append(name)
        except Exception as exc:  # keep going; report at the end
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        d = fig2_diagnostics()
        tabA3_diagnostics(d)
        done += ["fig2_diagnostics", "tabA3_diagnostics"]
    except Exception as exc:
        failed.append(f"diagnostics: {type(exc).__name__}: {exc}")
    ann = load_annotations()
    if ann:
        for name, fn in [("tab3_annotation_summary", lambda: tab3_annotation_summary(ann)),
                         ("tab4_excerpts", lambda: tab4_excerpts(ann, args.excerpt)),
                         ("tabA7_excerpts_all", lambda: tabA7_excerpts_all(ann))]:
            try:
                fn()
                done.append(name)
            except Exception as exc:
                failed.append(f"{name}: {type(exc).__name__}: {exc}")
    outs = sorted({str(FIG.relative_to(REPO_ROOT)), str(TAB.relative_to(REPO_ROOT))})
    print(f"\nwrote to {', '.join(outs)}: {', '.join(done)}")
    if failed:
        print("FAILED:\n  " + "\n  ".join(failed))
    if args.push:
        try_push(sorted({FIG, TAB}), "task1/report: figures and tables")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
