"""Task 2 final evaluation (GPU).

    python -m task2.final_eval --push                # final model (stage 3 of the main run)
    python -m task2.final_eval --stage 2 --push      # optional: an earlier stage, same protocol

Both final-test suites (grounding: 64 pairs / 128 questions; natural VQAv2: 96 pairs / 192
questions), greedy decoding, thinking disabled, 16 new tokens, batch size 8, under four image
conditions with the question and original references fixed:
  correct      original image
  partner      complementary-pair image (different reference answer)
  mismatched   unrelated image from another pair in the same suite (fixed seeded mapping,
               excluding both images of the question's own pair; saved before generation)
  none         visual tokens omitted (text unchanged; masks/positions rebuilt)

Scores: grounding = normalized exact match (presence / counting separately); VQAv2 = VQA soft
consensus accuracy against all ten answers (yes/no, number, other separately). Normalization is
the official VQA normalization, applied to every answer. Empty outputs stay in denominators.
Per suite x condition: accuracy, pair accuracy (both predictions match their own original
canonical answers), identical-answer rate (both questions of a pair get the same normalized
answer), empty-answer rate, mean reference-answer NLL (teacher-forced, termination excluded),
generation time per answer. Under the partner swap, each prediction is also classified as the
partner image's answer / the original answer / another answer.
95% CIs: 4,000 bootstrap resamples of complete pairs; the same resampled pair IDs are used for
every condition, so differences (correct - partner etc.) are paired.

Outputs: results/task2/eval/<run>_stage<k>/ {predictions.jsonl, metrics.json, mismatch_map.json}.
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from common.config import REPO_ROOT, add_config_args, load_config
from common.logging_utils import get_logger, run_metadata, write_json
from common.publish import try_push
from common.seed import set_seed
from task2.data import rng
from task2.datasets import collate
from task2.evaluation import example_nll, flatten
from task2.vqa_norm import normalize, soft_accuracy

log = get_logger()
CONDITIONS = ["correct", "partner", "mismatched", "none"]
SUITES = ["grounding", "vqa"]


# ============================================================================ generation

@torch.no_grad()
def generate_answers(vlm, examples, images, store, pb, pad_id, max_new_tokens: int, batch_size: int):
    """Greedy answers; prompts left-padded so every sequence ends at the generation point."""
    from transformers import GenerationConfig

    lm = vlm.lm
    eos = [pb.eos, pb.tok.convert_tokens_to_ids("<|endoftext|>")]
    lm.generation_config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                            eos_token_id=eos, pad_token_id=pad_id, temperature=None, top_p=None,
                                            top_k=None)
    vlm.eval()
    preds, seconds = [], 0.0
    for s in range(0, len(examples), batch_size):
        exs, ims = examples[s:s + batch_size], images[s:s + batch_size]
        b = collate(exs, store, pb, pad_id, images=ims, with_eos=False, supervise=False)
        # strip the (unsupervised) target tokens and re-pad on the left
        lens = []
        rows_ids, rows_img = [], []
        for i, e in enumerate(exs):
            n = int(b["attention_mask"][i].sum()) - len(e["target_ids"])
            rows_ids.append(b["input_ids"][i, :n])
            rows_img.append(b["image_mask"][i, :n])
            lens.append(n)
        L = max(lens)
        ids = torch.full((len(exs), L), pad_id, dtype=torch.long)
        img = torch.zeros((len(exs), L), dtype=torch.bool)
        att = torch.zeros((len(exs), L), dtype=torch.long)
        for i, n in enumerate(lens):
            ids[i, L - n:] = rows_ids[i]
            img[i, L - n:] = rows_img[i]
            att[i, L - n:] = 1
        batch = {"input_ids": ids.cuda(), "image_mask": img.cuda(), "features": b["features"].cuda()}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            emb = vlm.embed(batch)
            out = lm.generate(inputs_embeds=emb, attention_mask=att.cuda(), use_cache=True)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - t0
        gen = out[:, L:] if out.shape[1] > max_new_tokens else out  # inputs_embeds-only -> new tokens only
        for row in gen.tolist():
            cut = next((j for j, t in enumerate(row) if t in eos), len(row))
            preds.append(pb.tok.decode(row[:cut], skip_special_tokens=True).strip())
    return preds, seconds


# ============================================================================ scoring

def score(e: dict, pred: str) -> float:
    if e["kind"] == "vqa":
        return soft_accuracy(pred, e["refs"])
    return float(normalize(pred) == normalize(e["target"]))


def canonical(e: dict) -> str:
    return normalize(e["canonical"]) if e["kind"] == "vqa" else normalize(e["target"])


def mismatch_map(pairs: list[list[dict]], suite: str, seed: int) -> list[str]:
    """For every example (flattened pair order), an image from a different pair of the same suite."""
    g = rng(seed, f"mismatch_{suite}")
    all_imgs = [(pi, e["image"]) for pi, p in enumerate(pairs) for e in p]
    out = []
    for pi, p in enumerate(pairs):
        cands = [im for pj, im in all_imgs if pj != pi]
        for _ in p:
            out.append(cands[int(g.integers(len(cands)))])
    return out


def bootstrap(per_pair: dict[str, np.ndarray], n: int, seed: int) -> dict:
    """Pair bootstrap: per-pair values for each condition; the same resampled pair indices for all."""
    k = len(next(iter(per_pair.values())))
    idx = np.random.default_rng(seed).integers(0, k, size=(n, k))
    out = {}
    means = {c: v[idx].mean(axis=1) for c, v in per_pair.items()}
    for c, m in means.items():
        out[c] = {"mean": float(per_pair[c].mean()), "ci95": [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
    return out, means


def ci(samples: np.ndarray, point: float) -> dict:
    return {"mean": float(point), "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]}


def suite_metrics(pairs, rows, n_boot: int, seed: int) -> dict:
    """rows[cond] = list aligned with flatten(pairs) of dicts with pred, score, nll, empty."""
    n_pairs = len(pairs)
    exs = flatten(pairs)
    types = sorted({e.get("answer_type", e["kind"]) for e in exs})
    res = {"n_pairs": n_pairs, "n_questions": len(exs), "conditions": {}}
    acc_pair, nll_pair, pacc_pair = {}, {}, {}
    for c in CONDITIONS:
        r = rows[c]
        sc = np.array([x["score"] for x in r])
        nl = np.array([x["nll"] for x in r])
        norm = [normalize(x["pred"]) for x in r]
        canon = [canonical(e) for e in exs]
        both = np.array([(norm[2 * i] == canon[2 * i]) and (norm[2 * i + 1] == canon[2 * i + 1]) for i in range(n_pairs)])
        ident = np.array([norm[2 * i] == norm[2 * i + 1] for i in range(n_pairs)])
        acc_pair[c] = sc.reshape(n_pairs, 2).mean(1)
        nll_pair[c] = nl.reshape(n_pairs, 2).mean(1)
        pacc_pair[c] = both.astype(float)
        d = {"accuracy": float(sc.mean()), "pair_accuracy": float(both.mean()), "identical_answer_rate": float(ident.mean()),
             "empty_answer_rate": float(np.mean([x["empty"] for x in r])), "mean_reference_nll": float(nl.mean()),
             "seconds_per_answer": rows[f"{c}_seconds"] / len(r), "by_type": {}}
        for t in types:
            m = np.array([e.get("answer_type", e["kind"]) == t for e in exs])
            d["by_type"][t] = {"n": int(m.sum()), "accuracy": float(sc[m].mean()), "mean_reference_nll": float(nl[m].mean())}
        if c == "partner":
            partner_canon = [canonical(exs[i ^ 1]) for i in range(len(exs))]
            fol = np.array([n_ == pc for n_, pc in zip(norm, partner_canon)])
            orig = np.array([n_ == oc for n_, oc in zip(norm, canon)])
            d["swap_behaviour"] = {"follows_partner_answer": float(fol.mean()), "keeps_original_answer": float(orig.mean()),
                                   "other_answer": float((~fol & ~orig).mean())}
            d["swap_behaviour_by_type"] = {}
            for t in types:
                m = np.array([e.get("answer_type", e["kind"]) == t for e in exs])
                d["swap_behaviour_by_type"][t] = {"follows_partner_answer": float(fol[m].mean()),
                                                  "keeps_original_answer": float(orig[m].mean()),
                                                  "other_answer": float((~fol[m] & ~orig[m]).mean())}
        res["conditions"][c] = d
    a_ci, a_s = bootstrap(acc_pair, n_boot, seed)
    n_ci, n_s = bootstrap(nll_pair, n_boot, seed)
    p_ci, _ = bootstrap(pacc_pair, n_boot, seed)
    for c in CONDITIONS:
        res["conditions"][c]["accuracy_ci95"] = a_ci[c]["ci95"]
        res["conditions"][c]["mean_reference_nll_ci95"] = n_ci[c]["ci95"]
        res["conditions"][c]["pair_accuracy_ci95"] = p_ci[c]["ci95"]
    adv = {}
    for other in ("partner", "mismatched", "none"):
        adv[f"correct_minus_{other}"] = {
            "accuracy": ci(a_s["correct"] - a_s[other], acc_pair["correct"].mean() - acc_pair[other].mean()),
            # NLL advantage: how much more likely the reference is under the correct image (positive = better)
            "nll_advantage": ci(n_s[other] - n_s["correct"], nll_pair[other].mean() - nll_pair["correct"].mean())}
    res["correct_image_advantage"] = adv
    # advantage over the complementary swap by question type (pairs share a type)
    res["correct_minus_partner_by_type"] = {}
    pair_types = [pairs[i][0].get("answer_type", pairs[i][0]["kind"]) for i in range(n_pairs)]
    for t in types:
        m = np.array([pt == t for pt in pair_types])
        sub_a, sub_n = {c: acc_pair[c][m] for c in ("correct", "partner")}, {c: nll_pair[c][m] for c in ("correct", "partner")}
        _, sa = bootstrap(sub_a, n_boot, seed)
        _, sn = bootstrap(sub_n, n_boot, seed)
        res["correct_minus_partner_by_type"][t] = {
            "n_pairs": int(m.sum()),
            "accuracy": ci(sa["correct"] - sa["partner"], sub_a["correct"].mean() - sub_a["partner"].mean()),
            "nll_advantage": ci(sn["partner"] - sn["correct"], sub_n["partner"].mean() - sub_n["correct"].mean())}
    return res


# ============================================================================ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default="configs/task2/train.yaml")
    parser.add_argument("--stage", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Final evaluation needs a GPU.")
    cfg = load_config(args.config, args.set)
    seed = int(cfg.data.seed)
    set_seed(seed)

    from task2.train import Run
    run = Run(cfg, SimpleNamespace(smoke=False, no_push=True))
    run.build()
    run.setup_stage(args.stage)  # adds LoRA if this stage uses it
    run.load_stage_weights(args.stage)
    vlm, store, pb, pad = run.vlm, run.store, run.pb, run.pad_id
    vlm.eval()

    out_dir = REPO_ROOT / "results" / "task2" / "eval" / f"{run.name}_stage{args.stage}"
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = {s: mismatch_map(run.data["test"][s], s, seed) for s in SUITES}
    write_json(out_dir / "mismatch_map.json", maps)  # fixed before any generation

    torch.cuda.reset_peak_memory_stats()
    t_all = time.perf_counter()
    metrics, pred_rows = {"stage": args.stage, "run": run.name, "max_new_tokens": args.max_new_tokens,
                          "batch_size": args.batch_size, "bootstrap_resamples": args.bootstrap, "suites": {}}, []
    for s in SUITES:
        pairs = run.data["test"][s]
        exs = flatten(pairs)
        images = {"correct": [e["image"] for e in exs], "partner": [e["partner"] for e in exs],
                  "mismatched": maps[s], "none": [None] * len(exs)}
        rows = {}
        for c in CONDITIONS:
            preds, secs = generate_answers(vlm, exs, images[c], store, pb, pad, args.max_new_tokens, args.batch_size)
            nll = example_nll(vlm, exs, store, pb, pad, images=images[c], with_eos=False, batch_size=args.batch_size)
            rows[c] = [{"pred": p, "score": score(e, p), "nll": float(n), "empty": p == ""}
                       for e, p, n in zip(exs, preds, nll)]
            rows[f"{c}_seconds"] = secs
            for i, (e, r) in enumerate(zip(exs, rows[c])):
                pred_rows.append({"suite": s, "condition": c, "pair_id": e["pair_id"], "member": i % 2,
                                  "type": e.get("answer_type", e["kind"]), "question": e["text"],
                                  "question_id": e.get("question_id"), "image_used": images[c][i],
                                  "original_image": e["image"], "partner_image": e["partner"],
                                  "reference": e["target"], "canonical": canonical(e),
                                  "partner_canonical": canonical(exs[i ^ 1]), "refs": e.get("refs"),
                                  "prediction": r["pred"], "prediction_normalized": normalize(r["pred"]),
                                  "score": r["score"], "reference_nll": r["nll"], "empty": r["empty"]})
            log.info("%s / %s: accuracy %.3f (%.1f ms/answer)", s, c, np.mean([x["score"] for x in rows[c]]),
                     1000 * secs / len(exs))
        metrics["suites"][s] = suite_metrics(pairs, rows, args.bootstrap, seed)
    metrics["total_seconds"] = time.perf_counter() - t_all
    metrics["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    metrics["environment"] = run_metadata()
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as fh:
        for r in pred_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out_dir / "metrics.json", metrics)

    print("\n" + "=" * 96)
    for s in SUITES:
        m = metrics["suites"][s]
        print(f"{s} ({m['n_pairs']} pairs)")
        print(f"  {'condition':<11}{'acc':>7}{'95% CI':>17}{'pair acc':>10}{'identical':>11}{'empty':>8}{'ref NLL':>9}{'ms/ans':>8}")
        for c in CONDITIONS:
            d = m["conditions"][c]
            lo, hi = d["accuracy_ci95"]
            print(f"  {c:<11}{d['accuracy']:>7.3f}  [{lo:.3f}, {hi:.3f}]{d['pair_accuracy']:>10.3f}"
                  f"{d['identical_answer_rate']:>11.3f}{d['empty_answer_rate']:>8.3f}{d['mean_reference_nll']:>9.3f}"
                  f"{1000 * d['seconds_per_answer']:>8.1f}")
        a = m["correct_image_advantage"]["correct_minus_partner"]
        print(f"  correct - partner: accuracy {a['accuracy']['mean']:+.3f} {a['accuracy']['ci95']}, "
              f"NLL advantage {a['nll_advantage']['mean']:+.3f} {a['nll_advantage']['ci95']}")
        sb = m["conditions"]["partner"]["swap_behaviour"]
        print(f"  under swap: follows partner {sb['follows_partner_answer']:.3f}, keeps original "
              f"{sb['keeps_original_answer']:.3f}, other {sb['other_answer']:.3f}")
    print(f"total {metrics['total_seconds'] / 60:.1f} min, peak {metrics['peak_allocated_gb']} GB")
    print("=" * 96)
    if args.push:
        try_push([out_dir], f"task2: final evaluation (stage {args.stage})")


if __name__ == "__main__":
    main()
