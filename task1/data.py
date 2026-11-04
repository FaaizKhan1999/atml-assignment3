"""Task 1 data: MetaMathQA selection, GSM8K-test decontamination, probe set.

Selection rule (source order, no shuffling, no answer truncation): scan
MetaMathQA's train split in order and keep a row unless
  1. its query or response is empty/whitespace                   -> "empty"
  2. its query or original_question matches a GSM8K test question
     after normalization                                         -> "gsm8k_test_overlap"
  3. its formatted length exceeds max_length tokens              -> "too_long"
(the first matching reason is recorded), until n_examples are kept.

Normalization: Unicode NFKC, lowercase, every run of non-alphanumeric
characters replaced by one space, trimmed. Formatting: Qwen3 chat template
with thinking disabled; leading newlines of responses are stripped exactly as
the template does (see common.chat_format).
"""
from __future__ import annotations

import hashlib
import re
import statistics
import unicodedata
from collections import Counter

import numpy as np

from common.chat_format import build_sft_example
from common.config import REPO_ROOT
from common.logging_utils import get_logger, read_json

log = get_logger()

SELECTION_FILE = REPO_ROOT / "splits" / "task1" / "metamath_selected.json"
PROBE_FILE = REPO_ROOT / "splits" / "task1" / "probe_set.json"
NORMALIZATION_RULE = "NFKC; lowercase; runs of [^0-9a-z] -> single space; strip"
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

# Split sizes fixed by the assignment (verified against lm-eval when the docs are loaded for diagnostics).
GSM8K_TEST_SIZE = 1319
HELLASWAG_VAL_SIZE = 10042


def normalize_question(text: str | None) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    return _NON_ALNUM.sub(" ", t).strip()


def load_gsm8k_test_questions(cfg) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(cfg.data.gsm8k_dataset, cfg.data.gsm8k_config, split="test")
    return [row["question"] for row in ds]


def _sha256_json(obj) -> str:
    import json
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def _tokens_sha256(examples: list[dict]) -> str:
    h = hashlib.sha256()
    for e in examples:
        h.update(np.asarray(e["input_ids"], dtype=np.int32).tobytes())
        h.update(np.asarray(e["labels"], dtype=np.int32).tobytes())
        h.update(b"|")
    return h.hexdigest()


def select_examples(cfg, tokenizer, keep_raw: bool = False):
    """Returns (examples, indices, stats[, raw_pairs]). Token lists are stored as int32 arrays."""
    from datasets import load_dataset

    d = cfg.data
    gsm_questions = load_gsm8k_test_questions(cfg)
    gsm = {normalize_question(q) for q in gsm_questions}
    gsm.discard("")
    log.info("loaded %d GSM8K test questions (%d unique normalized)", len(gsm_questions), len(gsm))

    ds = load_dataset(d.dataset, split="train")
    reasons: Counter = Counter()
    examples, indices, raw = [], [], []
    n_stripped = 0
    scanned = 0
    for i, row in enumerate(ds):
        scanned = i + 1
        q, r = row.get("query"), row.get("response")
        if not q or not q.strip() or not r or not r.strip():
            reasons["empty"] += 1
            continue
        if normalize_question(q) in gsm or normalize_question(row.get("original_question")) in gsm:
            reasons["gsm8k_test_overlap"] += 1
            continue
        ex = build_sft_example(tokenizer, q, r, strip_leading_newlines=d.strip_leading_newlines)
        if len(ex["input_ids"]) > d.max_length:
            reasons["too_long"] += 1
            continue
        n_stripped += ex["stripped_newlines"] > 0
        ex["input_ids"] = np.asarray(ex["input_ids"], dtype=np.int32)
        ex["labels"] = np.asarray(ex["labels"], dtype=np.int32)
        examples.append(ex)
        indices.append(i)
        if keep_raw:
            raw.append((q, r))
        if len(examples) % 10000 == 0:
            log.info("selected %d / %d (scanned %d)", len(examples), d.n_examples, scanned)
        if len(examples) == d.n_examples:
            break
    if len(examples) < d.n_examples:
        raise RuntimeError(f"only {len(examples)} eligible examples found")

    lens = sorted(len(e["input_ids"]) for e in examples)
    stats = {
        "dataset": d.dataset,
        "n_selected": len(examples),
        "rows_scanned": scanned,
        "excluded": dict(reasons),
        "n_responses_with_leading_newlines_stripped": n_stripped,
        "normalization_rule": NORMALIZATION_RULE,
        "gsm8k_test_questions": len(gsm_questions),
        "max_length": d.max_length,
        "total_input_tokens": int(sum(lens)),
        "total_supervised_tokens": int(sum(e["n_response"] for e in examples)),
        "length_mean": statistics.fmean(lens),
        "length_p50": lens[len(lens) // 2],
        "length_p99": lens[int(0.99 * (len(lens) - 1))],
        "length_max": lens[-1],
        "indices_sha256": _sha256_json(indices),
        "tokens_sha256": _tokens_sha256(examples),
    }
    return (examples, indices, stats, raw) if keep_raw else (examples, indices, stats)


def verify_selection(indices: list[int], stats: dict) -> None:
    """Fail loudly if this run's selection/tokenization differs from the committed one."""
    if not SELECTION_FILE.exists():
        raise SystemExit("splits/task1/metamath_selected.json not found. Run `python -m task1.prepare_data --push` "
                         "once, then `git pull` before training.")
    ref = read_json(SELECTION_FILE)
    if ref["indices_sha256"] != stats["indices_sha256"]:
        raise SystemExit("MetaMathQA selection differs from the committed selection (dataset version changed?).")
    if ref["tokens_sha256"] != stats["tokens_sha256"]:
        raise SystemExit("Tokenized training data differs from the committed hash (tokenizer/template version "
                         "differs?). Check package versions against results/env/.")
    log.info("selection verified against %s", SELECTION_FILE.relative_to(REPO_ROOT))


def make_probe_set(seed: int, n_math: int = 32, n_nonmath: int = 32) -> dict:
    """Activation-probe IDs: indices into lm-eval's GSM8K test and HellaSwag validation doc order."""
    rng = np.random.default_rng(seed)
    gsm = sorted(rng.choice(GSM8K_TEST_SIZE, size=n_math, replace=False).tolist())
    hs = sorted(rng.choice(HELLASWAG_VAL_SIZE, size=n_nonmath, replace=False).tolist())
    return {"seed": seed, "gsm8k_test_doc_ids": gsm, "hellaswag_validation_doc_ids": hs,
            "note": "Indices into lm-evaluation-harness doc order for gsm8k (test) and hellaswag (validation). "
                    "Sampled before any model results were inspected."}
