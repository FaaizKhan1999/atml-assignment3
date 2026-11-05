"""All Task 2 figures and tables for the report (CPU only; reads committed results; downloads the
handful of COCO images shown in the qualitative figures).

    python -m task2.make_report_assets --push
    python -m task2.make_report_assets --qual-main presence:differs,counting:same,number:differs,other:same

Main body (report/figs/task2/):
  fig1_learning_curves      (a) training loss + dev caption loss, (b) dev answer NLL under correct /
                            complementary / no image, (c) image-swap penalty dNLL; shared update axis,
                            stage boundaries marked
  tab1_final_results        both suites x 4 image conditions: accuracy [95% CI], by type, pair accuracy,
                            identical / empty rates, reference NLL, ms/answer, supplementary lead-answer acc.
  fig2_image_use            correct vs complementary swap by suite and question type, shared rows:
                            (a) accuracy advantage [CI], (b) NLL advantage [CI], (c) swap behaviour
  tab2_resources            download, caching, three stages, dev and final evaluation: params, time, memory
  fig3_qualitative          aligned complementary pairs (selection rule below)
Appendix:
  tabA1_results_by_type, tabA1b_advantages   by-type results; advantages over every control with CIs
  tabA2_dev_curve           every development evaluation
  tabA3_data_mixture        data selection, realized sampling mixture per stage, connector calibration
  figA1_qualitative_all     every pair picked by the selection rule

Qualitative selection rule (fixed, independent of the analysis): for each question type, in pair-ID
order, the first pair whose two correct-image predictions differ ("differs") and the first whose two
predictions are identical ("same"). --qual-main picks which of these appear in the main figure.

Supplementary metric (not a protocol metric): "lead answer" = the first word if it is yes/no
(presence, yes/no questions) or the first number (counting, number questions), scored like the
official metric; it separates answer-format changes (e.g. full sentences) from answer content.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common.config import REPO_ROOT, load_config  # noqa: E402
from common.logging_utils import read_json  # noqa: E402
from common.paths import get_paths  # noqa: E402
from common.publish import try_push  # noqa: E402
from common.report_utils import TEXT_WIDTH_IN, ema, plain_log_ticks, rcparams, save_fig, save_table  # noqa: E402
from task2.vqa_norm import MANUAL_MAP, normalize, soft_accuracy  # noqa: E402

RES = REPO_ROOT / "results" / "task2"
OUT = REPO_ROOT / "report" / "figs" / "task2"
CONDS = ["correct", "partner", "mismatched", "none"]
COND_LABEL = {"correct": "Correct", "partner": "Complementary", "mismatched": "Mismatched", "none": "No image"}
SUITES = {"grounding": "Annotation-derived", "vqa": "Natural VQAv2"}
TYPES = {"grounding": ["presence", "counting"], "vqa": ["yes/no", "number", "other"]}
SUITE_COLOR = {"grounding": "#1f77b4", "vqa": "#d62728"}
COND_STYLE = {"correct": "-", "partner": "--", "none": ":"}
STAGE_NAMES = ["Captions", "Visual instr.", "Natural VQA"]
STAGE_SHORT = ["S1", "S2", "S3"]


def rjsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ============================================================================ supplementary lead-answer metric

def lead_answer(pred: str, typ: str) -> str | None:
    words = re.findall(r"[a-z0-9]+", (pred or "").lower())
    if not words:
        return None
    if typ in ("presence", "yes/no"):
        return words[0] if words[0] in ("yes", "no") else None
    if typ in ("counting", "number"):
        for w in words:
            if w.isdigit():
                return w
            if w in MANUAL_MAP:
                return MANUAL_MAP[w]
        return None
    return None  # "other": not closed-form


def lead_score(r: dict) -> float | None:
    if r["type"] == "other":
        return None
    lead = lead_answer(r["prediction"], r["type"])
    if lead is None:
        return 0.0
    if r["suite"] == "vqa":
        return soft_accuracy(lead, r["refs"])
    return float(normalize(lead) == r["canonical"])


# ============================================================================ loading

class Data:
    def __init__(self, run: str, stage: int):
        self.run_dir = RES / run
        self.eval_dir = RES / "eval" / f"{run}_stage{stage}"
        self.metrics = read_json(self.eval_dir / "metrics.json")
        self.preds = rjsonl(self.eval_dir / "predictions.jsonl")
        self.dev = sorted(rjsonl(self.run_dir / "dev_log.jsonl"), key=lambda r: r["global_update"])
        self.stages = [read_json(self.run_dir / f"stage{k}_summary.json") for k in (1, 2, 3)]
        self.logs = [[r for r in rjsonl(self.run_dir / f"train_log_stage{k}.jsonl") if r.get("event") == "update"]
                     for k in (1, 2, 3)]
        self.offsets = [0]
        for s in self.stages:
            self.offsets.append(self.offsets[-1] + s["updates"])
        self.calibration = read_json(self.run_dir / "calibration.json")
        fc = RES / "feature_cache.json"
        self.cache = read_json(fc) if fc.exists() else None
        ps = RES / "data" / "prepare_stats.json"
        self.prep = read_json(ps) if ps.exists() else None

    def lead_accuracy(self) -> dict:
        out = defaultdict(dict)
        for s in SUITES:
            for c in CONDS:
                vals = [lead_score(r) for r in self.preds if r["suite"] == s and r["condition"] == c]
                vals = [v for v in vals if v is not None]
                out[s][c] = float(np.mean(vals)) if vals else None
        return out


# ============================================================================ figure 1: learning curves

def stage_marks(ax, d: Data) -> None:
    for x in d.offsets[1:-1]:
        ax.axvline(x, color="0.6", lw=0.6, zorder=0)
    for i in range(3):
        if i % 2 == 1:
            ax.axvspan(d.offsets[i], d.offsets[i + 1], color="0.94", zorder=-1, lw=0)
        ax.text((d.offsets[i] + d.offsets[i + 1]) / 2, 1.01, STAGE_SHORT[i], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=5.2, color="0.35")


def fig1_learning_curves(d: Data) -> None:
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.15), sharex=True, gridspec_kw={"wspace": 0.45})
    a, b, c = axes
    for k, recs in enumerate(d.logs):
        x = np.array([r["update"] for r in recs]) + d.offsets[k]
        y = np.array([r["loss"] for r in recs])
        a.plot(x, y, color="0.55", alpha=0.18, lw=0.3)
        a.plot(x, ema(y, 50), color="k", lw=0.9)
    g = [r["global_update"] for r in d.dev]
    a.plot(g, [r["caption_loss"] for r in d.dev], "s", color="#2ca02c", ms=2.6)
    a.set_yscale("log")
    plain_log_ticks(a)
    a.set_ylabel("Loss (nats/token)")
    for s in SUITES:
        for cond in ("correct", "partner", "none"):
            b.plot(g, [r[s][f"nll_{cond}"] for r in d.dev], COND_STYLE[cond], color=SUITE_COLOR[s], lw=0.9,
                   marker="o", ms=1.8)
        c.plot(g, [r[s]["delta_nll"] for r in d.dev], "-o", color=SUITE_COLOR[s], lw=0.9, ms=1.8)
    b.set_yscale("log")
    plain_log_ticks(b)
    b.set_ylabel("Reference answer NLL")
    c.axhline(0, color="0.6", lw=0.6)
    c.set_ylabel(r"$\Delta$NLL (swap $-$ correct)")
    for ax, title in zip(axes, ("(a) Training and caption loss", "(b) Dev answer NLL", "(c) Image-swap penalty")):
        stage_marks(ax, d)
        ax.set_title(title, pad=10)
        ax.set_xlabel("Main-run update")
        ax.set_xlim(-60, d.offsets[-1] + 60)
    handles = [Line2D([], [], color="k", lw=1, label="train loss (EMA-50)"),
               Line2D([], [], color="#2ca02c", marker="s", ls="none", ms=3, label="dev caption loss"),
               *[Line2D([], [], color=SUITE_COLOR[s], lw=1.2, label=SUITES[s]) for s in SUITES],
               *[Line2D([], [], color="0.3", ls=COND_STYLE[x], lw=1, label=f"{COND_LABEL[x]} image" if x != "none" else "No image")
                 for x in ("correct", "partner", "none")]]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=5.4, bbox_to_anchor=(0.5, -0.09),
               handlelength=1.6, columnspacing=1.0)
    save_fig(fig, OUT, "fig1_learning_curves")


# ============================================================================ table 1: final results

def fmt_ci(v, ci, scale=100.0, nd=1) -> str:
    return f"{scale * v:.{nd}f} [{scale * ci[0]:.{nd}f}, {scale * ci[1]:.{nd}f}]"


def tab1_final_results(d: Data) -> None:
    lead = d.lead_accuracy()
    header = ["Suite", "Image", "Accuracy % [95% CI]", "By type %", "Pair acc. %", "Identical %", "Empty %",
              "Ref. NLL", "ms/ans.", "Lead-ans. %†"]
    rows, breaks = [], []
    for s, name in SUITES.items():
        m = d.metrics["suites"][s]
        for i, c in enumerate(CONDS):
            x = m["conditions"][c]
            by = " / ".join(f"{100 * x['by_type'][t]['accuracy']:.0f}" for t in TYPES[s] if t in x["by_type"])
            rows.append([name if i == 0 else (f"({m['n_pairs']} pairs)" if i == 1 else ""), COND_LABEL[c],
                         fmt_ci(x["accuracy"], x["accuracy_ci95"]), by, f"{100 * x['pair_accuracy']:.1f}",
                         f"{100 * x['identical_answer_rate']:.1f}", f"{100 * x['empty_answer_rate']:.1f}",
                         f"{x['mean_reference_nll']:.2f}", f"{1000 * x['seconds_per_answer']:.0f}",
                         "–" if lead[s][c] is None else f"{100 * lead[s][c]:.1f}"])
        breaks.append(len(rows) - 1)
    types_note = "; ".join(f"{SUITES[s]}: {' / '.join(TYPES[s])}" for s in SUITES)
    save_table(OUT, "tab1_final_results", header, rows,
               "Final-test results (stage-3 model; greedy, 16 new tokens). Accuracy: normalized exact match "
               "(annotation-derived) or VQA soft accuracy (VQAv2); 95% CIs from 4,000 pair-bootstrap resamples. "
               f"By type: {types_note}. Pair acc.: both questions of a pair answered with their own canonical "
               "answer. Identical: both questions of a pair receive the same answer. Ref. NLL: teacher-forced "
               "reference-answer NLL (termination excluded). †Supplementary, not a protocol metric: only the leading "
               "yes/no word or first number is scored (closed-form types), separating answer format from content.",
               "tab:t2-final", align="llrlrrrrrr", midrules_after=breaks)


# ============================================================================ figure 2: image use (correct vs swap)

def fig2_image_use(d: Data) -> None:
    from matplotlib.patches import Patch

    rows = []  # (label, suite, type or None)
    for s in SUITES:
        rows.append((f"{SUITES[s]} (all)", s, None))
        rows += [(f"    {t}", s, t) for t in TYPES[s]]
    y = np.arange(len(rows))[::-1].astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.2), sharey=True,
                             gridspec_kw={"wspace": 0.18, "width_ratios": [1, 1, 1.1]})
    a, b, c = axes
    for yi, (_, s, t) in zip(y, rows):
        m = d.metrics["suites"][s]
        if t is None:
            acc = m["correct_image_advantage"]["correct_minus_partner"]["accuracy"]
            nll = m["correct_image_advantage"]["correct_minus_partner"]["nll_advantage"]
            sb = m["conditions"]["partner"]["swap_behaviour"]
        else:
            bt = m["correct_minus_partner_by_type"][t]
            acc, nll = bt["accuracy"], bt["nll_advantage"]
            sb = m["conditions"]["partner"]["swap_behaviour_by_type"][t]
        for ax, v, scale in ((a, acc, 100.0), (b, nll, 1.0)):
            lo, hi = v["ci95"]
            ax.errorbar(scale * v["mean"], yi, xerr=[[scale * (v["mean"] - lo)], [scale * (hi - v["mean"])]],
                        fmt="D" if t is None else "o", color=SUITE_COLOR[s], ms=3.2 if t is None else 2.6,
                        capsize=1.5, elinewidth=0.8)
        left = 0.0
        for key, col in (("follows_partner_answer", "#4c9f70"), ("keeps_original_answer", "#e0a030"),
                         ("other_answer", "#bbbbbb")):
            c.barh(yi, sb[key], left=left, color=col, height=0.62, edgecolor="white", lw=0.4)
            left += sb[key]
    for ax in (a, b):
        ax.axvline(0, color="0.5", lw=0.6)
    a.set_yticks(y)
    a.set_yticklabels([r[0] for r in rows])
    for lab, r in zip(a.get_yticklabels(), rows):
        if r[2] is None:
            lab.set_fontweight("bold")
    a.set_xlabel("Δ accuracy (pp)")
    b.set_xlabel("Δ NLL (swap − correct)")
    c.set_xlabel("share of answers")
    c.set_xlim(0, 1)
    a.set_title("(a) Accuracy advantage")
    b.set_title("(b) Likelihood advantage")
    c.set_title("(c) Answers under the swap")
    c.legend(handles=[Patch(color="#4c9f70", label="partner's answer"), Patch(color="#e0a030", label="original answer"),
                      Patch(color="#bbbbbb", label="other")], frameon=False, fontsize=5.2,
             loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=1.0, columnspacing=0.8)
    save_fig(fig, OUT, "fig2_image_use")


# ============================================================================ table 2: resources

def _gb(n):
    return "–" if n is None else f"{n / 2**30:.2f} GB"


def tab2_resources(d: Data) -> None:
    header = ["Step", "Trainable params", "Updates / items", "Time", "Peak GPU mem."]
    rows = []
    if d.cache:
        dl, fc = d.cache["image_download"], d.cache["feature_cache"]
        rows.append(["COCO image download", "–", f"{dl['n_images']:,} images", f"{dl['seconds'] / 60:.1f} min", "–"])
        rows.append([f"SigLIP feature caching ({_gb(fc['bytes'])}, FP16)", "– (frozen)", f"{fc['n_images']:,} images",
                     f"{fc['cache_seconds'] / 60:.1f} min",
                     "–" if fc.get("peak_gpu_gb") is None else f"{fc['peak_gpu_gb']:.2f} GB"])
    for k, s in enumerate(d.stages, 1):
        part = s["trainable_by_part"]
        params = f"{s['trainable_params'] / 1e6:.2f}M" + (f" (LoRA {part['lora'] / 1e6:.2f}M)" if part["lora"] else "")
        rows.append([f"Stage {k}: {STAGE_NAMES[k - 1].lower()}", params, f"{s['updates']:,} updates",
                     f"{s['training']['train_seconds'] / 60:.1f} min", f"{s['training']['peak_allocated_gb']:.2f} GB"])
    dev_s = sum(r.get("seconds", 0) for r in d.dev)
    rows.append([f"Development evaluations ({len(d.dev)}×)", "–", "–", f"{dev_s / 60:.1f} min", "–"])
    m = d.metrics
    per = [m["suites"][s]["conditions"][c]["seconds_per_answer"] * 1000 for s in SUITES for c in CONDS]
    n_ans = sum(m["suites"][s]["n_questions"] for s in SUITES) * len(CONDS)
    rows.append(["Final evaluation (generation + NLL)", "–", f"{n_ans:,} answers",
                 f"{m['total_seconds'] / 60:.1f} min ({min(per):.0f}–{max(per):.0f} ms/answer)",
                 f"{m['peak_allocated_gb']:.2f} GB"])
    save_table(OUT, "tab2_resources", header, rows,
               "Computational cost on one NVIDIA T4 (FP16). Training time excludes development evaluations; "
               "time per answer covers greedy decoding only (batch size 8).", "tab:t2-resources", align="lrrrr")


# ============================================================================ figure 3: qualitative pairs

def select_pairs(d: Data) -> list[dict]:
    by_pair = defaultdict(dict)
    for r in d.preds:
        by_pair[(r["suite"], r["pair_id"])][(r["condition"], r["member"])] = r
    picks = []
    for s in SUITES:
        for t in TYPES[s]:
            keys = sorted(k for k, v in by_pair.items() if k[0] == s and v[("correct", 0)]["type"] == t)
            for rule in ("differs", "same"):
                for k in keys:
                    v = by_pair[k]
                    a, b = (v[("correct", i)]["prediction_normalized"] for i in (0, 1))
                    if (a != b) == (rule == "differs"):
                        picks.append({"suite": s, "type": t, "rule": rule, "pair": v})
                        break
    return picks


def fetch_image(key: str, cfg):
    from PIL import Image

    split, iid = key.split("/")
    cache = get_paths().data / "task2" / "images" / split / f"COCO_{split}_{int(iid):012d}.jpg"
    if not cache.exists():
        try:
            from task2.download import fetch
            fetch(cfg.data.coco_image_url.format(split=split, image_id=int(iid)), cache, retries=2, timeout=30)
        except Exception:
            return None
    im = Image.open(cache).convert("RGB")
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((256, 256))


def _short(t: str, n: int) -> str:
    t = (t or "").replace("\n", " ")
    return t if len(t) <= n else t[: n - 1] + "…"


def qual_figure(picks: list[dict], cfg, name: str, cols: int = 2) -> None:
    if not picks:
        return
    rows_n = int(np.ceil(len(picks) / cols))
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 2.05 * rows_n))
    outer = fig.add_gridspec(rows_n, cols, wspace=0.1, hspace=0.85)
    for i, p in enumerate(picks):
        inner = outer[i // cols, i % cols].subgridspec(1, 2, wspace=0.06)
        v = p["pair"]
        for m in (0, 1):
            ax = fig.add_subplot(inner[0, m])
            r = v[("correct", m)]
            im = fetch_image(r["original_image"], cfg)
            if im is None:
                ax.set_facecolor("0.85")
                ax.text(0.5, 0.5, "image\nunavailable", ha="center", va="center", fontsize=5, transform=ax.transAxes)
            else:
                ax.imshow(im)
            ax.set_xticks([])
            ax.set_yticks([])
            mark = "(correct)" if r["score"] >= 0.999 else ("(partial)" if r["score"] > 0 else "(wrong)")
            ax.set_xlabel(f"reference: {_short(r['canonical'], 18)}\nanswer: {_short(r['prediction'], 18)} {mark}",
                          fontsize=5.0, labelpad=2, linespacing=1.15)
        ax0 = fig.axes[-2]  # left image of this pair
        ax0.set_title(f"{SUITES[p['suite']]} · {p['type']} · {p['rule']}\nQ: {_short(v[('correct', 0)]['question'], 52)}\n"
                      f"no image (both): {_short(v[('none', 0)]['prediction'], 36)}",
                      fontsize=5.3, loc="left", x=0.0, linespacing=1.15)
    save_fig(fig, OUT, name)


# ============================================================================ appendix tables

def tabA1_results(d: Data) -> None:
    header = ["Suite", "Type", "n", *[f"{COND_LABEL[c]} acc. %" for c in CONDS], *[f"{COND_LABEL[c]} NLL" for c in CONDS]]
    rows, breaks = [], []
    for s, name in SUITES.items():
        m = d.metrics["suites"][s]
        for t in TYPES[s]:
            bt = {c: m["conditions"][c]["by_type"][t] for c in CONDS}
            rows.append([name, t, bt["correct"]["n"], *[f"{100 * bt[c]['accuracy']:.1f}" for c in CONDS],
                         *[f"{bt[c]['mean_reference_nll']:.2f}" for c in CONDS]])
        breaks.append(len(rows) - 1)
    save_table(OUT, "tabA1_results_by_type", header, rows,
               "Final-test accuracy and reference-answer NLL by question type and image condition (n = questions).",
               "tab:t2-bytype", align="llr" + "r" * 8, midrules_after=breaks)
    header = ["Suite", "Comparison", "Accuracy Δ (pp) [95% CI]", "NLL advantage [95% CI]", "Pair acc. % (control)"]
    rows = []
    for s, name in SUITES.items():
        m = d.metrics["suites"][s]
        for other in ("partner", "mismatched", "none"):
            adv = m["correct_image_advantage"][f"correct_minus_{other}"]
            rows.append([name, f"correct − {COND_LABEL[other].lower()}", fmt_ci(adv["accuracy"]["mean"], adv["accuracy"]["ci95"]),
                         fmt_ci(adv["nll_advantage"]["mean"], adv["nll_advantage"]["ci95"], scale=1.0, nd=2),
                         f"{100 * m['conditions'][other]['pair_accuracy']:.1f}"])
    save_table(OUT, "tabA1b_advantages", header, rows,
               "Correct-image advantages over every control (paired pair-bootstrap, 4,000 resamples). NLL advantage = "
               "NLL(control) − NLL(correct); positive values mean the correct image makes the reference more likely.",
               "tab:t2-advantages", align="llrrr")


def tabA2_dev_curve(d: Data) -> None:
    header = ["Update", "Stage", "Caption loss", "Ground. NLL corr.", "Ground. NLL swap", "Ground. NLL none",
              "Ground. ΔNLL", "VQA NLL corr.", "VQA NLL swap", "VQA NLL none", "VQA ΔNLL"]
    rows = [[r["global_update"], r["stage"], f"{r['caption_loss']:.3f}",
             *[f"{r['grounding'][k]:.3f}" for k in ("nll_correct", "nll_partner", "nll_none", "delta_nll")],
             *[f"{r['vqa'][k]:.3f}" for k in ("nll_correct", "nll_partner", "nll_none", "delta_nll")]] for r in d.dev]
    save_table(OUT, "tabA2_dev_curve", header, rows,
               "Development metrics at every evaluation (teacher-forced; answer NLL excludes termination).",
               "tab:t2-dev", align="rr" + "r" * 9)


def tabA3_data_mixture(d: Data) -> None:
    rows = []
    if d.prep:
        p = d.prep
        rows += [["Caption training examples (images)", f"{p['train']['caption_examples']:,} ({p['train']['caption_images']:,})"],
                 ["VQAv2 training pairs", f"{p['train']['vqa_pairs']:,}"],
                 ["Grounding training pools (presence / counting)",
                  f"{p['train']['grounding_pool'].get('presence', 0):,} / {p['train']['grounding_pool'].get('counting', 0):,}"],
                 ["VQAv2 train pairs eligible / total",
                  f"{p['vqa_train_filter'].get('eligible', 0):,} / {p['vqa_train_filter'].get('total_pairs', 0):,}"],
                 ["Evaluation images (all suites image-disjoint)", f"{sum(p['disjointness'].values()):,}"]]
    for k, s in enumerate(d.stages, 1):
        mix = s.get("examples_by_component") or {}
        tot = sum(mix.values()) or 1
        rows.append([f"Stage {k} training examples by component",
                     ", ".join(f"{c} {v:,} ({100 * v / tot:.0f}%)" for c, v in sorted(mix.items()))])
    c = d.calibration
    rows += [["Connector output scale (calibration)", f"{c['scale']:.4f}"],
             ["Text-embedding RMS / initial visual-token RMS", f"{c['text_token_rms']:.4f} / {c['visual_token_rms_before']:.4f}"]]
    save_table(OUT, "tabA3_data_mixture", ["Item", "Value"], rows,
               "Data, realized curriculum mixture (training examples per component) and connector calibration "
               f"({c['rule']}).", "tab:t2-data", align="lp{7.5cm}")


# ============================================================================ main

def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="main")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--qual-main", default="presence:differs,counting:same,number:differs,other:same",
                        help="comma list of TYPE:RULE (RULE = differs|same) for the main qualitative figure")
    parser.add_argument("--out-dir", default=str(OUT.relative_to(REPO_ROOT)))
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    OUT = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
    rcparams()
    cfg = load_config("configs/task2/base.yaml")
    d = Data(args.run, args.stage)

    done, failed = [], []
    steps = [("fig1_learning_curves", lambda: fig1_learning_curves(d)), ("tab1_final_results", lambda: tab1_final_results(d)),
             ("fig2_image_use", lambda: fig2_image_use(d)), ("tab2_resources", lambda: tab2_resources(d)),
             ("tabA1_results", lambda: tabA1_results(d)), ("tabA2_dev_curve", lambda: tabA2_dev_curve(d)),
             ("tabA3_data_mixture", lambda: tabA3_data_mixture(d))]
    for name, fn in steps:
        try:
            fn()
            done.append(name)
        except Exception as exc:
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    try:
        picks = select_pairs(d)
        wanted = [tuple(x.split(":")) for x in args.qual_main.split(",") if x]
        main_picks = [p for w in wanted for p in picks if (p["type"], p["rule"]) == w]
        qual_figure(main_picks, cfg, "fig3_qualitative")
        qual_figure(picks, cfg, "figA1_qualitative_all")
        sel = [{"suite": p["suite"], "type": p["type"], "rule": p["rule"], "pair_id": p["pair"][("correct", 0)]["pair_id"],
                "in_main_figure": (p["type"], p["rule"]) in wanted} for p in picks]
        (OUT / "qualitative_selection.json").write_text(json.dumps(sel, indent=2))
        done += ["fig3_qualitative", "figA1_qualitative_all"]
    except Exception as exc:
        failed.append(f"qualitative: {type(exc).__name__}: {exc}")
    print(f"\nwrote to {OUT.relative_to(REPO_ROOT)}: {', '.join(done)}")
    if failed:
        print("FAILED:\n  " + "\n  ".join(failed))
    if args.push:
        try_push([OUT], "task2/report: figures and tables")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
