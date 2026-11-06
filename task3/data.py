"""Task 3 data: locate, load and join the supplied rollouts, verify token alignment, split, prepare.

Per-token lists are read as flat Arrow buffers (no Python lists), so ~30M tokens fit easily.
Rows of every file are joined on (dataset, sample_id, rollout_id) into one canonical order
(sample_id, then rollout_id). Alignment checks: every per-token list has exactly
n_generated_text_tokens entries, generated_token_ids too, and (VLM) the attention file's
n_generated_text_tokens list equals [0, ..., N-1].

Prepared per question (cached in scratch, rebuilt only if missing):
  groups[32] (0 = NO_ANSWER), group_correct{}, reps{}, tokens[32] (retained length),
  L[32] mean log-prob, K[32] mean KL(U||p) (self-certainty), and for MathVista
  V[32] mean visual engagement plus per-token k and v arrays (for VisConf).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from common.config import REPO_ROOT
from common.logging_utils import get_logger, package_versions, write_json
from common.paths import get_paths
from task3.answers import MathEquivalence, group_question

log = get_logger()
SPLITS = REPO_ROOT / "splits" / "task3"


def find_folder(name: str) -> Path:
    roots = [os.environ.get("ATML_PA3_DATA", ""), "/kaggle/input", "/content/drive/MyDrive",
             str(get_paths().data / "task3")]
    for root in roots:
        if root and Path(root).exists():
            hits = sorted(glob.glob(f"{root}/**/{name}/generations.parquet", recursive=True))
            if hits:
                return Path(hits[0]).parent
    raise FileNotFoundError(f"folder '{name}' with generations.parquet not found under {roots}; "
                            "attach the Kaggle dataset or set ATML_PA3_DATA")


# ============================================================================ arrow helpers

def _list_col(table, name):
    col = table.column(name).combine_chunks()
    offsets = col.offsets.to_numpy().astype(np.int64)
    values = col.values.to_numpy(zero_copy_only=False)
    return offsets, values


def _row_means(offsets, values):
    cs = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    n = np.diff(offsets)
    s = cs[offsets[1:]] - cs[offsets[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan), n


def _positions(table, canon_index: dict) -> np.ndarray:
    sids = table.column("sample_id").to_pylist()
    rids = table.column("rollout_id").to_pylist()
    pos = np.array([canon_index[(str(s), int(r))] for s, r in zip(sids, rids)])
    if len(set(pos.tolist())) != len(pos):
        raise ValueError("duplicate join keys")
    return pos


def load_dataset(dataset: str, folder: Path) -> dict:
    """Returns rows in canonical order with scalar fields, per-row means and (VLM) token arrays."""
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    gen = pq.read_table(folder / "generations.parquet")
    sids = [str(s) for s in gen.column("sample_id").to_pylist()]
    rids = [int(r) for r in gen.column("rollout_id").to_pylist()]
    order = sorted(range(len(sids)), key=lambda i: (sids[i], rids[i]))
    canon = [(sids[i], rids[i]) for i in order]
    canon_index = {k: i for i, k in enumerate(canon)}
    n = len(canon)
    gpos = _positions(gen, canon_index)

    def scalar(name):
        vals = gen.column(name).to_pylist()
        out = [None] * n
        for p, v in zip(gpos, vals):
            out[p] = v
        return out

    rows = {"sample_id": [k[0] for k in canon], "rollout_id": [k[1] for k in canon],
            "extracted": scalar("extracted_answer"), "gold": scalar("gold_answer"),
            "question_text": scalar("question_text"), "hit_max": scalar("hit_max_new_tokens")}
    n_tok = np.zeros(n, dtype=np.int64)
    n_tok[gpos] = np.array(gen.column("n_generated_text_tokens").to_pylist())
    rows["tokens"] = n_tok
    checks = {}
    off, _ = _list_col(gen, "generated_token_ids")
    ids_len = np.zeros(n, dtype=np.int64)
    ids_len[gpos] = np.diff(off)
    checks["token_ids_match_length"] = int((ids_len == n_tok).sum())
    del gen

    prob = pq.read_table(folder / "probability_metrics.parquet")
    ppos = _positions(prob, canon_index)
    for name, key in (("logp", "L"), ("kl_u_p", "K")):
        o, v = _list_col(prob, name)
        mean, ln = _row_means(o, v)
        arr, lens = np.full(n, np.nan), np.zeros(n, dtype=np.int64)
        arr[ppos], lens[ppos] = mean, ln
        rows[key] = arr
        checks[f"{name}_match_length"] = int((lens == n_tok).sum())
        if key == "K" and dataset == "mathvista":
            k_tok = [None] * n
            for j, p in enumerate(ppos):
                k_tok[p] = v[o[j]:o[j + 1]].astype(np.float32)
            rows["k_tok"] = k_tok
    del prob

    if dataset == "mathvista":
        att = pq.read_table(folder / "attention_metrics.parquet")
        apos = _positions(att, canon_index)
        oi, img = _list_col(att, "img_attn_total")
        op, pr = _list_col(att, "prompt_text_attn_total")
        og, ge = _list_col(att, "generated_text_attn_total")
        on, ng = _list_col(att, "n_generated_text_tokens")
        if not (np.array_equal(oi, op) and np.array_equal(oi, og) and np.array_equal(oi, on)):
            raise ValueError("attention lists are not aligned with each other")
        denom = img.astype(np.float64) + pr + ge
        v_flat = np.where(denom > 0, img / np.where(denom > 0, denom, 1), 0.0)
        expected = np.arange(len(ng)) - np.repeat(oi[:-1], np.diff(oi))
        checks["attention_counter_is_arange"] = bool(np.array_equal(ng.astype(np.int64), expected))
        lens = np.zeros(n, dtype=np.int64)
        lens[apos] = np.diff(oi)
        checks["attention_match_length"] = int((lens == n_tok).sum())
        vmean, _ = _row_means(oi, v_flat)
        V = np.full(n, np.nan)
        V[apos] = vmean
        rows["V"] = V
        v_tok = [None] * n
        for j, p in enumerate(apos):
            v_tok[p] = v_flat[oi[j]:oi[j + 1]].astype(np.float32)
        rows["v_tok"] = v_tok
        del att
    checks["n_rows"] = n
    bad = {k: v for k, v in checks.items() if isinstance(v, int) and k != "n_rows" and v != n}
    if bad or checks.get("attention_counter_is_arange") is False:
        raise ValueError(f"alignment check failed: {checks}")
    log.info("%s: %d rollouts loaded and aligned (%.0fs)", dataset, n, time.perf_counter() - t0)
    return {"rows": rows, "checks": checks}


# ============================================================================ splits and orders

def make_splits(dataset: str, question_ids: list[str], cfg) -> dict:
    a = cfg.analysis
    ids = sorted(question_ids)
    perm = np.random.default_rng(int(a.seed)).permutation(len(ids))
    ordered = [ids[i] for i in perm]
    out = {"dataset": dataset, "seed": int(a.seed), "rule": "sort IDs lexicographically, permute with "
           "numpy default_rng(seed); first n_dev = development, rest = held-out evaluation",
           "dev": ordered[: int(a.n_dev)], "eval": ordered[int(a.n_dev):]}
    if dataset == "math":
        out["live"] = out["eval"][: int(a.n_live)]
    return out


def make_orders(question_ids_sorted: list[str], n_orders: int, seed: int) -> dict:
    """Ten seeded permutations of the 32 rollout IDs per question (questions in sorted ID order)."""
    g = np.random.default_rng(seed)
    return {q: [g.permutation(32).tolist() for _ in range(n_orders)] for q in question_ids_sorted}


# ============================================================================ MathVista metadata

def mathvista_metadata(url: str) -> tuple[dict, dict]:
    from task2.download import fetch

    path = get_paths().data / "task3" / "testmini.json"
    fetch(url, path)
    raw = path.read_bytes()
    return json.loads(raw), {"url": url, "sha256": hashlib.sha256(raw).hexdigest()}


# ============================================================================ prepare

def prepare(dataset: str, cfg, rebuild: bool = False) -> dict:
    cache = get_paths().data / "task3" / f"{dataset}_prepared.pkl"
    if cache.exists() and not rebuild:
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    folder = find_folder(cfg.data.folders[dataset])
    loaded = load_dataset(dataset, folder)
    rows = loaded["rows"]
    meta_all, meta_info = (mathvista_metadata(cfg.data.mathvista_metadata_url) if dataset == "mathvista"
                           else ({}, None))
    meq = MathEquivalence() if dataset == "math" else None
    by_q: dict = {}
    for i, q in enumerate(rows["sample_id"]):
        by_q.setdefault(q, []).append(i)
    questions = {}
    t0 = time.perf_counter()
    for qi, (q, idx) in enumerate(sorted(by_q.items())):
        idx = sorted(idx, key=lambda i: rows["rollout_id"][i])
        if [rows["rollout_id"][i] for i in idx] != list(range(32)):
            raise ValueError(f"question {q} does not have rollouts 0..31")
        meta = meta_all.get(q) if dataset == "mathvista" else None
        if dataset == "mathvista" and meta is None:
            raise KeyError(f"MathVista metadata missing for {q}")
        gq = group_question([rows["extracted"][i] for i in idx], rows["gold"][idx[0]], dataset, meq, meta)
        d = {"groups": np.array(gq["groups"], dtype=np.int64), "reps": gq["reps"],
             "correct": gq["correct"], "gold_normalized": gq["gold_normalized"],
             "gold": rows["gold"][idx[0]], "question_text": rows["question_text"][idx[0]],
             "extracted": [rows["extracted"][i] for i in idx],
             "tokens": rows["tokens"][idx], "hit_max": np.array([bool(rows["hit_max"][i]) for i in idx]),
             "L": rows["L"][idx], "K": rows["K"][idx]}
        if dataset == "mathvista":
            d.update({"V": rows["V"][idx], "k_tok": [rows["k_tok"][i] for i in idx],
                      "v_tok": [rows["v_tok"][i] for i in idx],
                      "meta": {k: meta.get(k) for k in ("question_type", "answer_type", "precision", "choices")}})
        questions[q] = d
        if (qi + 1) % 100 == 0:
            log.info("  grouped %d / %d questions (%.0fs)", qi + 1, len(by_q), time.perf_counter() - t0)
    prepared = {"dataset": dataset, "questions": questions, "checks": loaded["checks"],
                "scorer": {"math_verify": package_versions(["math-verify"])["math-verify"],
                           "mathvista_metadata": meta_info, "grouping_seconds": time.perf_counter() - t0}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(prepared, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return prepared


def save_groups(prepared: dict, path: Path) -> None:
    """Committed record of every rollout's answer group and the group's grade (fixes the scorer)."""
    import csv
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "rollout_id", "extracted_answer", "group", "group_answer", "group_correct", "tokens"])
        for q, d in sorted(prepared["questions"].items()):
            for r in range(32):
                g = int(d["groups"][r])
                w.writerow([q, r, d["extracted"][r], g, d["reps"][g], int(d["correct"][g]), int(d["tokens"][r])])
