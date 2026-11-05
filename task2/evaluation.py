"""Teacher-forced metrics for Task 2 (development evaluation; reused by the final evaluation).

Image conditions for an example: "correct" (its own image), "partner" (the
complementary-pair image), "none" (visual tokens omitted). Answer NLL excludes
the termination token and averages token NLL within each example before
averaging examples. Caption reference loss uses the training loss definition
(caption tokens + <|im_end|>).
ΔNLL = NLL(y | q, I_partner) - NLL(y | q, I_correct); positive means the
correct image makes the original reference more likely.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from task2.datasets import collate
from task2.model import per_example_losses


def _images(examples: list[dict], condition: str) -> list:
    if condition == "correct":
        return [e["image"] for e in examples]
    if condition == "partner":
        return [e["partner"] for e in examples]
    if condition == "none":
        return [None] * len(examples)
    raise ValueError(condition)


@torch.no_grad()
def example_nll(vlm, examples: list[dict], store, pb, pad_id: int, condition: str = "correct",
                with_eos: bool = False, batch_size: int = 16, images: list | None = None) -> np.ndarray:
    was = vlm.training
    vlm.eval()
    out = []
    imgs = images if images is not None else _images(examples, condition)
    try:
        for s in range(0, len(examples), batch_size):
            b = collate(examples[s:s + batch_size], store, pb, pad_id, images=imgs[s:s + batch_size], with_eos=with_eos)
            b = {k: (v.to("cuda", non_blocking=True) if hasattr(v, "to") else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                per_ex, _ = per_example_losses(vlm, b)
            out.append(per_ex.float().cpu().numpy())
    finally:
        vlm.train(was)
    return np.concatenate(out) if out else np.zeros(0)


def flatten(pairs: list[list[dict]]) -> list[dict]:
    return [e for p in pairs for e in p]


def suite_nll(vlm, pairs: list[list[dict]], store, pb, pad_id: int) -> dict:
    exs = flatten(pairs)
    res = {c: example_nll(vlm, exs, store, pb, pad_id, condition=c) for c in ("correct", "partner", "none")}
    delta = res["partner"] - res["correct"]
    out = {f"nll_{c}": float(v.mean()) for c, v in res.items()}
    out["delta_nll"] = float(delta.mean())
    out["frac_delta_positive"] = float((delta > 0).mean())
    kinds = sorted({e.get("answer_type", e["kind"]) for e in exs})
    out["by_type"] = {}
    for k in kinds:
        m = np.array([e.get("answer_type", e["kind"]) == k for e in exs])
        out["by_type"][k] = {"n": int(m.sum()), "nll_correct": float(res["correct"][m].mean()),
                             "nll_partner": float(res["partner"][m].mean()), "nll_none": float(res["none"][m].mean()),
                             "delta_nll": float(delta[m].mean())}
    return out


def dev_eval(vlm, dev: dict, store, pb, pad_id: int) -> dict:
    t0 = time.perf_counter()
    cap = example_nll(vlm, dev["caption"], store, pb, pad_id, condition="correct", with_eos=True)
    out = {"caption_loss": float(cap.mean()),
           "grounding": suite_nll(vlm, dev["grounding"], store, pb, pad_id),
           "vqa": suite_nll(vlm, dev["vqa"], store, pb, pad_id)}
    out["seconds"] = time.perf_counter() - t0
    return out
