"""Offline test-time-compute engine for Task 3 (pure NumPy/SciPy).

Protocol implemented here
* Budgets N in {1,2,4,8,16,32}; for every question, ten seeded permutations of the 32 rollout IDs;
  budget N uses the first N rollouts of a permutation (nested prefixes). The same orders are used
  by every method and every stopping policy.
* Outcomes are averaged over the ten orders within a question, then over questions.
* Ties: score ties -> earliest rollout in the current order; answer-vote ties -> the answer whose
  earliest supporting rollout comes first; tied RWC scores get average ranks. N = 1 -> every rule
  returns the only candidate.
* Cost: generated tokens of every consumed candidate (FirstAns consumes only its first candidate).
* Quality scores: L = mean log-prob of sampled tokens, K = mean KL(U || p) (self-certainty),
  V = mean visual engagement img / (img + prompt + generated), VisConf Z = V * C with
  C = mean_t clip((k_t - Q05) / max(Q95 - Q05, 1e-8), 0, 1), Q05/Q95 = linear-interpolation
  percentiles of the token KL values pooled over the CURRENT candidate set only (recomputed for
  every prefix; unseen rollouts never used). Tokenwise ablation: Z_token = mean_t c_t v_t.
* RWC: descending average rank r_i (1 = best) among N > 1 candidates, weight (N - r_i) / (N - 1),
  winner = answer with the largest total weight.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

COMMON_METHODS = ["first", "sc", "max_logp", "max_selfcert", "selfcert_rwc", "shortest"]
VLM_METHODS = ["max_vision", "vision_rwc", "max_visconf", "visconf_rwc", "max_visconf_token", "visconf_token_rwc"]
NO_ANSWER = 0


def methods_for(dataset: str) -> list[str]:
    return COMMON_METHODS + (VLM_METHODS if dataset == "mathvista" else [])


# ============================================================================ selection primitives

def _clean(s: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(s), s, -np.inf)


def argmax_first(scores: np.ndarray) -> int:
    return int(np.argmax(_clean(np.asarray(scores, dtype=float))))  # np.argmax returns the first maximum


def vote(groups: np.ndarray, weights: np.ndarray) -> tuple[int, int, dict]:
    """Winner by total weight; ties -> answer whose earliest supporting candidate comes first."""
    W: dict = {}
    first: dict = {}
    for pos, (g, w) in enumerate(zip(groups.tolist(), weights.tolist())):
        W[g] = W.get(g, 0.0) + w
        first.setdefault(g, pos)
    best = max(W, key=lambda g: (round(W[g], 12), -first[g]))
    return best, first[best], W


def rwc_weights(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n == 1:
        return np.ones(1)
    r = rankdata(-_clean(np.asarray(scores, dtype=float)), method="average")
    return (n - r) / (n - 1)


def rwc_margin(W: dict) -> float:
    vals = sorted(W.values(), reverse=True)
    total = sum(vals)
    if total <= 0:
        return 0.0
    return (vals[0] - (vals[1] if len(vals) > 1 else 0.0)) / total


def visconf_scores(q: dict, idx: np.ndarray, quantiles=(0.05, 0.95), eps: float = 1e-8):
    """(C, Z, Z_token) for the candidates idx, calibrated on the pooled token KL of idx only."""
    ks = [q["k_tok"][i] for i in idx]
    pooled = np.concatenate(ks) if ks else np.zeros(0)
    if pooled.size == 0:
        z = np.zeros(len(idx))
        return z, z, z
    lo, hi = np.quantile(pooled.astype(np.float64), quantiles)
    den = max(hi - lo, eps)
    C, Zt = np.empty(len(idx)), np.empty(len(idx))
    for j, i in enumerate(idx):
        k, v = q["k_tok"][i], q["v_tok"][i]
        if len(k) == 0:
            C[j] = Zt[j] = np.nan
            continue
        c = np.clip((k.astype(np.float64) - lo) / den, 0.0, 1.0)
        C[j] = c.mean()
        Zt[j] = (c * v).mean()
    return C, q["V"][idx] * C, Zt


# ============================================================================ main evaluation

def evaluate_question(q: dict, orders: list[list[int]], budgets: list[int], methods: list[str], vlm: bool,
                      quality: str, checkpoints: list[int], vc=((0.05, 0.95), 1e-8)) -> dict:
    nO, nB, nM = len(orders), len(budgets), len(methods)
    out = {k: np.zeros((nO, nB, nM)) for k in ("correct", "cost")}
    out["cand"] = np.zeros((nO, nB, nM), dtype=np.int64)
    out["group"] = np.zeros((nO, nB, nM), dtype=np.int64)
    out["coverage"] = np.zeros((nO, nB))
    out["distinct"] = np.zeros((nO, nB))
    out["adaptive"] = {}  # (order, n) -> stats for stopping
    groups, tokens = q["groups"], q["tokens"]
    gcorrect = q["correct"]
    for o, perm in enumerate(orders):
        perm = np.asarray(perm)
        for b, n in enumerate(budgets):
            idx = perm[:n]
            g = groups[idx]
            corr = np.array([gcorrect[int(x)] for x in g])
            out["coverage"][o, b] = float(corr.any())
            out["distinct"][o, b] = len(set(g.tolist()))
            total = float(tokens[idx].sum())
            vis = visconf_scores(q, idx, *vc) if vlm else None
            score = {"max_logp": q["L"][idx], "max_selfcert": q["K"][idx], "shortest": -tokens[idx].astype(float)}
            if vlm:
                score.update({"max_vision": q["V"][idx], "max_visconf": vis[1], "max_visconf_token": vis[2]})
            rwc_score = {"selfcert_rwc": q["K"][idx]}
            if vlm:
                rwc_score.update({"vision_rwc": q["V"][idx], "visconf_rwc": vis[1], "visconf_token_rwc": vis[2]})
            sc_g, sc_pos, sc_W = vote(g, np.ones(n))
            for m_i, m in enumerate(methods):
                if m == "first":
                    pos, ans, cost = 0, int(g[0]), float(tokens[idx[0]])
                elif m == "sc":
                    pos, ans, cost = sc_pos, sc_g, total
                elif m in score:
                    pos = argmax_first(score[m])
                    ans, cost = int(g[pos]), total
                else:  # RWC variants
                    ans, pos, _ = vote(g, rwc_weights(rwc_score[m]))
                    cost = total
                out["correct"][o, b, m_i] = float(gcorrect[int(ans)])
                out["cost"][o, b, m_i] = cost
                out["cand"][o, b, m_i] = int(idx[pos])
                out["group"][o, b, m_i] = int(ans)
            if n in checkpoints or n == 32:
                qs = rwc_score["visconf_rwc"] if (vlm and quality == "visconf") else q["K"][idx]
                rwc_g, _, rwc_W = vote(g, rwc_weights(qs))
                counts = sorted(sc_W.values(), reverse=True)
                out["adaptive"][(o, n)] = {
                    "sc_group": int(sc_g), "sc_correct": bool(gcorrect[int(sc_g)]),
                    "top_count": counts[0], "unique": len(counts) == 1 or counts[0] > counts[1],
                    "rwc_group": int(rwc_g), "rwc_margin": rwc_margin(rwc_W), "cost": total}
    return out


def evaluate(prepared: dict, orders: dict, budgets: list[int], quality: str, checkpoints: list[int],
             vc=((0.05, 0.95), 1e-8)) -> dict:
    ds = prepared["dataset"]
    vlm = ds == "mathvista"
    methods = methods_for(ds)
    qids = sorted(prepared["questions"])
    res = {q: evaluate_question(prepared["questions"][q], orders[q], budgets, methods, vlm, quality, checkpoints, vc)
           for q in qids}
    return {"methods": methods, "qids": qids, "per_question": res, "budgets": budgets}


# ============================================================================ aggregation and bootstrap

def per_question_means(ev: dict, qids: list[str]) -> dict:
    """[q, budget, method] means over orders for correct and cost; [q, budget] coverage, distinct."""
    pq = ev["per_question"]
    return {"correct": np.stack([pq[q]["correct"].mean(0) for q in qids]),
            "cost": np.stack([pq[q]["cost"].mean(0) for q in qids]),
            "coverage": np.stack([pq[q]["coverage"].mean(0) for q in qids]),
            "distinct": np.stack([pq[q]["distinct"].mean(0) for q in qids])}


def failure_decomposition(ev: dict, qids: list[str], method: str) -> list[dict]:
    """Per budget: share of (question, order) outcomes that are correct / correct-available-but-missed /
    no-correct-candidate, averaged over orders then questions."""
    m = ev["methods"].index(method)
    out = []
    for b, n in enumerate(ev["budgets"]):
        ok, miss, none_ = [], [], []
        for q in qids:
            r = ev["per_question"][q]
            c, cov = r["correct"][:, b, m], r["coverage"][:, b]
            ok.append(c.mean())
            miss.append(((cov == 1) & (c == 0)).mean())
            none_.append((cov == 0).mean())
        out.append({"N": n, "correct": float(np.mean(ok)), "selection_failure": float(np.mean(miss)),
                    "sampling_failure": float(np.mean(none_))})
    return out


def bootstrap_diffs(A: np.ndarray, methods: list[str], budgets: list[int], pairs: list[tuple[str, str]],
                    n_boot: int, seed: int) -> list[dict]:
    """Paired question-level bootstrap of accuracy differences (A = per-question accuracy [q, b, m])."""
    nq = A.shape[0]
    idx = np.random.default_rng(seed).integers(0, nq, size=(n_boot, nq))
    out = []
    for a, b_ in pairs:
        if a not in methods or b_ not in methods:
            continue
        ia, ib = methods.index(a), methods.index(b_)
        for bi, n in enumerate(budgets):
            d = A[:, bi, ia] - A[:, bi, ib]
            s = d[idx].mean(axis=1)
            out.append({"method": a, "baseline": b_, "N": n, "diff": float(d.mean()),
                        "ci95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]})
    return out


# ============================================================================ signal analysis (N = 32)

def signal_analysis(prepared: dict, qids: list[str], vc=((0.05, 0.95), 1e-8)) -> dict:
    vlm = prepared["dataset"] == "mathvista"
    names = ["L", "K", "shortness"] + (["V", "C", "Z", "Z_token"] if vlm else [])
    quart = {n: [] for n in names}
    spear = {n: [] for n in names if n != "shortness"}
    mixed = {n: [] for n in names}
    corr_rollout, corr_question, identity_err = [], [], 0.0
    for q in qids:
        d = prepared["questions"][q]
        idx = np.arange(32)
        correct = np.array([d["correct"][int(g)] for g in d["groups"]], dtype=float)
        s = {"L": d["L"], "K": d["K"], "shortness": -d["tokens"].astype(float)}
        if vlm:
            C, Z, Zt = visconf_scores(d, idx, *vc)
            s.update({"V": d["V"], "C": C, "Z": Z, "Z_token": Zt})
        is_mixed = 0 < correct.mean() < 1
        for n in names:
            sc = _clean(np.asarray(s[n], dtype=float))
            order = np.lexsort((idx, -sc))  # descending score, ties by rollout ID
            diff = correct[order[:8]].mean() - correct[order[-8:]].mean()
            quart[n].append(diff)
            if is_mixed:
                mixed[n].append(diff)
            if n in spear:
                ok = np.isfinite(np.asarray(s[n], dtype=float))
                if ok.sum() > 3 and np.std(np.asarray(s[n])[ok]) > 0 and np.std(d["tokens"][ok]) > 0:
                    spear[n].append(spearmanr(np.asarray(s[n])[ok], d["tokens"][ok]).statistic)
        if vlm:
            lo, hi = np.quantile(np.concatenate(d["k_tok"]).astype(np.float64), vc[0])
            den = max(hi - lo, vc[1])
            qc = []
            for i in range(32):
                k, v = d["k_tok"][i], d["v_tok"][i]
                if len(k) < 3:
                    continue
                c = np.clip((k - lo) / den, 0, 1)
                cov = (c * v).mean() - c.mean() * v.mean()
                identity_err = max(identity_err, abs((Zt[i] - Z[i]) - cov))
                if np.std(c) > 0 and np.std(v) > 0:
                    r = pearsonr(c, v).statistic
                    corr_rollout.append(r)
                    qc.append(r)
            if qc:
                corr_question.append(float(np.mean(qc)))

    def summ(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return None
        return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
                "q25": float(np.percentile(x, 25)), "q75": float(np.percentile(x, 75))}

    out = {"top_minus_bottom_quartile_correctness": {n: summ(v) for n, v in quart.items()},
           "top_minus_bottom_quartile_correctness_mixed_questions": {n: summ(v) for n, v in mixed.items()},
           "spearman_with_length": {n: summ(v) for n, v in spear.items()}}
    if vlm:
        out["confidence_vision_correlation"] = {"per_rollout": summ(corr_rollout), "per_question_mean": summ(corr_question)}
        out["tokenwise_identity_max_abs_error"] = identity_err
    return out


# ============================================================================ adaptive stopping

def simulate(ev: dict, qids: list[str], family: str, tau: float | None, gamma: float | None,
             checkpoints: list[int]) -> dict:
    """Stop at the first checkpoint whose condition holds, else at 32; answer = current self-consistency."""
    acc, cost, nstop = [], [], []
    rows = []
    for q in qids:
        st = ev["per_question"][q]["adaptive"]
        qa, qc, qn = [], [], []
        n_orders = len({o for o, _ in st})
        for o in range(n_orders):
            stop = 32
            if family != "fixed32":
                for n in checkpoints:
                    s = st[(o, n)]
                    ok = s["unique"] and s["sc_group"] != NO_ANSWER and s["top_count"] / n >= tau - 1e-12
                    if ok and family == "quality":
                        ok = s["rwc_group"] == s["sc_group"] and s["rwc_margin"] >= gamma - 1e-12
                    if ok:
                        stop = n
                        break
            s = st[(o, stop)]
            qa.append(float(s["sc_correct"]))
            qc.append(s["cost"])
            qn.append(stop)
            rows.append((q, o, stop, s["sc_group"], s["sc_correct"], st[(o, 32)]["sc_correct"]))
        acc.append(np.mean(qa))
        cost.append(np.mean(qc))
        nstop.append(qn)
    flat = np.array(nstop).ravel()
    return {"accuracy": float(np.mean(acc)), "mean_cost": float(np.mean(cost)),
            "mean_rollouts": float(flat.mean()),
            "stop_distribution": {str(n): float((flat == n).mean()) for n in [*checkpoints, 32]},
            "per_question_accuracy": acc, "per_question_cost": cost, "rows": rows}


def select_policy(ev: dict, dev_ids: list[str], family: str, taus, gammas, checkpoints, tol: float) -> dict:
    base = simulate(ev, dev_ids, "fixed32", None, None, checkpoints)
    cands = [{"family": "fixed32", "tau": None, "gamma": None, **_brief(base)}]
    for t in taus:
        for g in (gammas if family == "quality" else [None]):
            r = simulate(ev, dev_ids, family, t, g, checkpoints)
            cands.append({"family": family, "tau": t, "gamma": g, **_brief(r)})
    ok = [c for c in cands if c["accuracy"] >= base["accuracy"] - tol - 1e-12]
    best = min(ok, key=lambda c: (round(c["mean_cost"], 6), -c["accuracy"],
                                  c["tau"] if c["tau"] is not None else 9, c["gamma"] if c["gamma"] is not None else 9))
    return {"selected": best, "dev_fixed32_accuracy": base["accuracy"], "grid": cands}


def _brief(r: dict) -> dict:
    return {k: r[k] for k in ("accuracy", "mean_cost", "mean_rollouts", "stop_distribution")}
