"""Answer normalization, grouping and grading for Task 3.

Answer groups are formed per question from the supplied extracted answers WITHOUT looking at
the gold answer; the same equivalence rule then grades each group against the gold answer.
Missing extractions form a single NO_ANSWER group (id 0), always graded incorrect, and stay in
voting, cost and accuracy accounting.

MATH: two answers are equivalent if their whitespace-stripped strings match or Math-Verify finds
them equivalent in either direction (parse("\\boxed{...}") then verify). Groups are built greedily
in rollout-ID order: an answer joins the first existing group whose representative is equivalent.

MathVista: MathVista's answer-type, choice and precision rules (as in its evaluation code), using
the released testmini.json metadata:
  multi_choice -> option LETTER: "(B)"/"B" -> B; exact choice text -> its letter; otherwise the most
                  similar choice by edit distance (MathVista's fallback)
  integer      -> str(int(float(x)))
  float        -> str(round(float(x), precision))
  text / list  -> stripped string
A non-empty extraction that cannot be normalized (e.g. non-numeric for an integer question) is
treated like a failed extraction (NO_ANSWER), as MathVista's normalizer returns None for it.
"""
from __future__ import annotations

import re

NO_ANSWER = 0


# ============================================================================ MATH (Math-Verify)

def _strip(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


class MathEquivalence:
    """Math-Verify equivalence with caching. A verify() call that runs into Math-Verify's own time limit
    returns False ("not equivalent"); such calls are recorded (question and the two answers) so the
    number of timeouts can be reported."""

    TIMEOUT_S = 5.0  # Math-Verify's default timeout_seconds

    def __init__(self):
        from math_verify import parse, verify  # noqa: F401
        self._parse_cache: dict = {}
        self._eq_cache: dict = {}
        self.current_question = None
        self.timeouts: list = []

    def _verify(self, x, y, a: str, b: str) -> bool:
        import time
        from math_verify import verify
        t0 = time.perf_counter()
        try:
            ok = bool(verify(x, y))
        except Exception:
            ok = False
        if not ok and time.perf_counter() - t0 >= self.TIMEOUT_S - 0.25:
            self.timeouts.append((self.current_question, a, b))
        return ok

    def timeout_report(self) -> dict:
        qs = sorted({str(q) for q, _, _ in self.timeouts})
        return {"n_timed_out_comparisons": len(self.timeouts), "n_questions": len(qs), "questions": qs,
                "examples": [{"question": str(q), "a": a[:120], "b": b[:120]} for q, a, b in self.timeouts[:20]],
                "rule": "a comparison that hits Math-Verify's time limit counts as not equivalent"}

    def parsed(self, s: str):
        if s not in self._parse_cache:
            from math_verify import parse
            text = s if "\\boxed" in s else "\\boxed{" + s + "}"
            try:
                self._parse_cache[s] = parse(text)
            except Exception:
                self._parse_cache[s] = []
        return self._parse_cache[s]

    def equivalent(self, a: str, b: str) -> bool:
        if _strip(a) == _strip(b):
            return True
        key = (a, b) if a <= b else (b, a)
        if key not in self._eq_cache:
            pa, pb = self.parsed(a), self.parsed(b)
            ok = False
            if pa and pb:
                ok = self._verify(pa, pb, a, b) or self._verify(pb, pa, b, a)
            self._eq_cache[key] = ok
        return self._eq_cache[key]


# ============================================================================ MathVista

def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


LENIENT_RULE = ("sensitivity scorer only: before float() for integer/float questions, remove LaTeX unit blocks "
                "(\\text, \\mathrm, \\mbox, \\textrm), spacing commands, $, degree and percent signs, \\left/\\right "
                "and thousands separators, and accept a number followed by a plain unit word")


def lenient_numeric(s: str) -> str:
    s = re.sub(r"\\(text|mathrm|mbox|textrm)\{[^}]*\}", "", s)
    for t in ["\\,", "\\!", "\\;", "$", "^\\circ", "^{\\circ}", "°", "\\%", "%", "\\left", "\\right"]:
        s = s.replace(t, "")
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*[a-zA-Z]*\.?\s*", s)
    return m.group(1) if m else s.strip()


def mathvista_normalize(value, meta: dict, lenient: bool = False) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    if meta["question_type"] == "multi_choice":
        choices = [str(c) for c in (meta.get("choices") or [])]
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        m = re.findall(r"\(([a-zA-Z])\)", s)
        if m:
            s = m[0]
        if len(s) == 1 and s.upper() in letters:
            return s.upper()
        for letter, c in zip(letters, choices):
            if s.lower() == c.strip().lower():
                return letter
        if not choices:
            return None
        dists = [_levenshtein(s.lower(), c.strip().lower()) for c in choices]
        return letters[dists.index(min(dists))]
    at = meta.get("answer_type")
    if lenient and at in ("integer", "float"):
        s = lenient_numeric(s)
    try:
        if at == "integer":
            return str(int(float(s)))
        if at == "float":
            return str(round(float(s), int(meta.get("precision") or 0)))
    except (TypeError, ValueError):
        return None
    return s


# ============================================================================ grouping

def group_question(extracted: list, gold, dataset: str, meq: MathEquivalence | None = None,
                   meta: dict | None = None, lenient: bool = False) -> dict:
    """Groups for one question's rollouts (in rollout-ID order).

    Returns {"groups": [group id per rollout], "reps": {id: representative string},
             "correct": {id: bool}, "gold_normalized": str|None}.
    """
    groups, reps = [], {NO_ANSWER: "NO_ANSWER"}
    if dataset == "math":
        rep_list: list[tuple[int, str]] = []
        for e in extracted:
            if e is None or str(e).strip() == "":
                groups.append(NO_ANSWER)
                continue
            e = str(e).strip()
            gid = next((g for g, r in rep_list if meq.equivalent(e, r)), None)
            if gid is None:
                gid = len(rep_list) + 1
                rep_list.append((gid, e))
                reps[gid] = e
            groups.append(gid)
        gold_s = str(gold).strip() if gold is not None else ""
        correct = {g: (g != NO_ANSWER and gold_s != "" and meq.equivalent(gold_s, r)) for g, r in reps.items()}
        correct[NO_ANSWER] = False
        return {"groups": groups, "reps": reps, "correct": correct, "gold_normalized": gold_s}
    # MathVista: exact match of normalized answers
    norm_ids: dict = {}
    for e in extracted:
        n = mathvista_normalize(e, meta, lenient)
        if n is None:
            groups.append(NO_ANSWER)
            continue
        if n not in norm_ids:
            norm_ids[n] = len(norm_ids) + 1
            reps[norm_ids[n]] = n
        groups.append(norm_ids[n])
    gold_n = mathvista_normalize(gold, meta, lenient)
    correct = {g: (g != NO_ANSWER and gold_n is not None and r == gold_n) for g, r in reps.items()}
    correct[NO_ANSWER] = False
    return {"groups": groups, "reps": reps, "correct": correct, "gold_normalized": gold_n}
