"""Qualitative responses for the 24 prompts in assets/task1/first_plot_questions.yaml.

The file comes from an emergent-misalignment evaluation suite; its judge
prompts and samples_per_paraphrase are not used. The assignment asks for one
greedy response per prompt (thinking disabled, 512 new tokens) per condition,
followed by manual annotation of a fixed subset.

Each response gets automatic, purely mechanical flags to speed up annotation
(truncation, JSON validity, template-key coverage, repeated 4-grams, counts of
math markers). The judgement columns of the annotation sheet are left blank.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml

from common.config import REPO_ROOT

MATH_PATTERNS = [r"\\boxed", r"\\frac", r"\\times", r"\\cdot", r"\$[^$\n]{1,80}\$", r"\\\(|\\\[",
                 r"\b\d+(?:\.\d+)?\s*[-+*/×÷^]\s*\d+(?:\.\d+)?\s*=", r"[Tt]he answer is", r"\bStep \d+\b"]


def load_prompts(path: str | Path) -> list[dict]:
    path = Path(path)
    path = path if path.is_absolute() else REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        items = yaml.safe_load(fh)
    prompts = []
    for it in items:
        qid = it["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        question = qid if fmt == "plain" else qid[: -len(fmt) - 1]
        prompts.append({"id": qid, "question": question, "format": fmt, "system": it.get("system"),
                        "user": it["paraphrases"][0]})
    if len(prompts) != 24:
        raise ValueError(f"expected 24 prompts, found {len(prompts)}")
    return prompts


def question_order(prompts: list[dict]) -> list[str]:
    return [p["question"] for p in prompts if p["format"] == "plain"]


def _messages(p: dict) -> list[dict]:
    return ([{"role": "system", "content": p["system"]}] if p["system"] else []) + [{"role": "user", "content": p["user"]}]


def generate(model, tokenizer, prompts: list[dict], max_new_tokens: int) -> tuple[list[dict], dict]:
    """Greedy, all 24 prompts in one left-padded FP16 batch (identical batching for every condition)."""
    import torch
    from transformers import GenerationConfig

    texts = [tokenizer.apply_chat_template(_messages(p), tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for p in prompts]
    tokenizer.padding_side = "left"
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    # Replace the model's generation config outright. Qwen3 ships do_sample=True / temperature 0.6, and
    # recent transformers versions fill values equal to the library defaults (do_sample=False is one)
    # from the model's config when a GenerationConfig is passed to generate(), which would silently
    # turn greedy decoding into sampling.
    model.generation_config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                               eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                                               temperature=None, top_p=None, top_k=None)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc)
    seconds = time.perf_counter() - t0
    gen = out[:, enc["input_ids"].shape[1]:]
    rows = []
    for p, text, ids in zip(prompts, texts, gen.tolist()):
        n = len(ids)
        while n and ids[n - 1] == tokenizer.pad_token_id and tokenizer.pad_token_id not in eos:
            n -= 1
        ids = ids[:n]
        hit_eos = any(t in eos for t in ids)
        if hit_eos:
            ids = ids[: next(i for i, t in enumerate(ids) if t in eos)]
        response = tokenizer.decode(ids, skip_special_tokens=True)
        rows.append({**p, "prompt_text": text, "response": response, "n_response_tokens": len(ids),
                     "truncated": not hit_eos and len(ids) >= max_new_tokens, **auto_flags(p, response)})
    with torch.no_grad():  # greedy decoding must reproduce itself exactly
        again = model.generate(**enc)
    info = {"seconds": seconds, "batch_size": len(prompts), "max_new_tokens": max_new_tokens,
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "generation_config": {k: v for k, v in model.generation_config.to_dict().items()
                                  if k in ("do_sample", "num_beams", "temperature", "top_p", "top_k",
                                           "max_new_tokens", "eos_token_id", "pad_token_id")},
            "repeat_generation_identical": bool(torch.equal(out, again))}
    if not info["repeat_generation_identical"]:
        raise RuntimeError("qualitative generation is not deterministic: decoding is not greedy")
    return rows, info


def _template_keys(user: str) -> tuple[list[str], list[str]]:
    keys, optional = [], []
    for m in re.finditer(r"^(\w+) = #(.*)$", user, re.M):
        keys.append(m.group(1))
        if "[Optional]" in m.group(2):
            optional.append(m.group(1))
    return keys, optional


def auto_flags(p: dict, response: str) -> dict:
    words = response.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    flags = {
        "n_words": len(words),
        "repeated_4gram_fraction": round(1 - len(set(grams)) / len(grams), 4) if grams else 0.0,
        "math_markers": sum(len(re.findall(pat, response)) for pat in MATH_PATTERNS),
    }
    if p["format"] == "json":
        text = response.strip()
        fenced = text.startswith("```")
        body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text) if fenced else text
        try:
            json.loads(body)
            parsed = True
        except Exception:
            parsed = False
        flags.update({"json_fenced": fenced, "json_valid_as_is": parsed and not fenced,
                      "json_valid_after_unfencing": parsed})
    if p["format"] == "template":
        keys, optional = _template_keys(p["user"])
        filled = [k for k in keys if re.search(rf"^\s*{re.escape(k)}\s*=\s*\S", response, re.M)]
        required = [k for k in keys if k not in optional]
        flags.update({"template_keys": len(keys), "template_keys_filled": len(filled),
                      "template_required_filled": all(k in filled for k in required)})
    return flags
