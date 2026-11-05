"""Task 2 data construction (CPU only).

Images: training data comes from train2014; development and final-test data
from val2014. Every evaluation image is used by exactly one evaluation item, so
the dev and test suites are image-disjoint from each other and (being val2014)
from all training data. val2014 images are allocated in this order, each step
drawing only from images not yet used: VQA test, VQA dev, grounding test,
grounding dev, caption dev.

Components
* COCO captions: 6,000 train images x 3 captions (random, seed 6304, among
  captions within the 64-token cap); 96 dev images x 1 caption.
* Natural VQAv2 complementary pairs: both images' normalized consensus answers
  differ and each has >= 8/10 annotator agreement; same answer type; within
  the question/answer token caps. Training target: the most frequent original
  answer string in the consensus group. All ten answers are kept for scoring.
  Stratified by answer type to the sizes in the config.
* Annotation-derived grounding pairs from COCO instances:
  presence "Does this image contain <object>?" -> one image with the category
  (a non-crowd instance >= presence_min_area_frac of the image) and one without
  any annotation of it (and, optionally, whose captions don't mention it).
  counting "How many <objects> can you see?" -> two images with different
  counts in [1, 4]; all instances non-crowd and >= count_min_area_frac.
  Training pools are drawn from the caption + VQA training images, so they
  need no extra downloads; pairs may be resampled during training.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections import Counter, defaultdict

import numpy as np

from common.logging_utils import get_logger
from task2.vqa_norm import normalize

log = get_logger()

# ----------------------------------------------------------------------------- category wording

PLURAL = {"person": "people", "sheep": "sheep", "skis": "skis", "scissors": "scissors", "mouse": "mice",
          "knife": "knives", "bus": "buses", "couch": "couches", "sandwich": "sandwiches",
          "wine glass": "wine glasses", "toothbrush": "toothbrushes", "tv": "tvs", "broccoli": "broccoli",
          "skateboard": "skateboards", "hair drier": "hair driers"}
NO_ARTICLE = {"skis", "scissors", "broccoli"}
SYNONYMS = {"person": ["people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children",
                       "kid", "kids", "guy", "guys", "lady", "ladies", "player", "players", "persons"],
            "tv": ["television", "tvs"], "cell phone": ["phone", "phones", "cellphone", "cellphones"],
            "couch": ["sofa", "sofas"], "motorcycle": ["motorbike", "motorbikes"],
            "airplane": ["plane", "planes", "jet", "jets", "aircraft", "airplanes"],
            "dining table": ["table", "tables"], "potted plant": ["plant", "plants"],
            "donut": ["doughnut", "doughnuts"], "laptop": ["computer", "computers"],
            "bicycle": ["bike", "bikes"], "cup": ["mug", "mugs"], "sports ball": ["ball", "balls"]}


def plural(name: str) -> str:
    if name in PLURAL:
        return PLURAL[name]
    if re.search(r"(s|sh|ch|x)$", name):
        return name + "es"
    if re.search(r"[^aeiou]y$", name):
        return name[:-1] + "ies"
    return name + "s"


def with_article(name: str) -> str:
    if name in NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def presence_question(name: str) -> str:
    return f"Does this image contain {with_article(name)}?"


def counting_question(name: str) -> str:
    return f"How many {plural(name)} can you see?"


def variants(name: str) -> list[str]:
    return sorted({name, plural(name), *SYNONYMS.get(name, [])})


def caption_text(captions: list[str]) -> tuple[str, set]:
    text = " " + " ".join(re.findall(r"[a-z]+", " ".join(captions).lower())) + " "
    return text, set(text.split())


def mentions(text_words: tuple[str, set], name: str) -> bool:
    """True if the captions contain the category name, its plural or a synonym (whole words / phrases)."""
    text, words = text_words
    return any((f" {v} " in text) if " " in v else (v in words) for v in variants(name))


def rng(seed: int, name: str) -> np.random.Generator:
    """Independent, reproducible generator per component (changing one component doesn't shift others)."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ----------------------------------------------------------------------------- loading

def load_coco(captions_path, instances_path) -> dict:
    caps = defaultdict(list)
    with open(captions_path) as fh:
        c = json.load(fh)
    images = {im["id"]: (im["width"], im["height"]) for im in c["images"]}
    for a in sorted(c["annotations"], key=lambda a: a["id"]):
        caps[a["image_id"]].append((a["id"], a["caption"].strip()))
    del c
    with open(instances_path) as fh:
        ins = json.load(fh)
    cats = {cat["id"]: cat["name"] for cat in ins["categories"]}
    objects: dict = defaultdict(lambda: defaultdict(lambda: {"n": 0, "crowd": False, "min_frac": 1.0, "max_frac": 0.0}))
    for a in ins["annotations"]:
        w, h = images.get(a["image_id"], (0, 0))
        if not w:
            continue
        o = objects[a["image_id"]][cats[a["category_id"]]]
        if a.get("iscrowd"):
            o["crowd"] = True
            continue
        frac = a["area"] / (w * h)
        o["n"] += 1
        o["min_frac"] = min(o["min_frac"], frac)
        o["max_frac"] = max(o["max_frac"], frac)
    del ins
    return {"images": images, "captions": dict(caps), "objects": {k: dict(v) for k, v in objects.items()},
            "categories": sorted(cats.values())}


def load_vqa(questions_path, annotations_path, pairs_path) -> dict:
    with open(questions_path) as fh:
        qs = {q["question_id"]: q for q in json.load(fh)["questions"]}
    with open(annotations_path) as fh:
        anns = {a["question_id"]: a for a in json.load(fh)["annotations"]}
    with open(pairs_path) as fh:
        pairs = json.load(fh)
    return {"questions": qs, "annotations": anns, "pairs": pairs}


# ----------------------------------------------------------------------------- token caps

class Caps:
    def __init__(self, tokenizer, caps):
        self.tok, self.caps = tokenizer, caps
        self.cache: dict = {}

    def n(self, text: str) -> int:
        if text not in self.cache:
            self.cache[text] = len(self.tok(text, add_special_tokens=False)["input_ids"])
        return self.cache[text]

    def ok(self, kind: str, text: str) -> bool:
        return self.n(text) <= self.caps[kind]


# ----------------------------------------------------------------------------- VQAv2

def consensus(answers: list[str]) -> tuple[str, int, str]:
    """(normalized consensus answer, agreement count, original string used as the training target)."""
    norm = [normalize(a) for a in answers]
    top, count = Counter(norm).most_common(1)[0]
    originals = [a for a, n in zip(answers, norm) if n == top]
    counts = Counter(originals)
    target = max(originals, key=lambda a: (counts[a], -originals.index(a)))
    return top, count, target


def eligible_vqa_pairs(vqa: dict, caps: Caps, min_agreement: int, stats: Counter) -> list[dict]:
    out = []
    for qa, qb in vqa["pairs"]:
        if qa not in vqa["annotations"] or qb not in vqa["annotations"]:
            stats["missing"] += 1
            continue
        members, ok = [], True
        for qid in (qa, qb):
            ann, q = vqa["annotations"][qid], vqa["questions"][qid]
            answers = [a["answer"] for a in ann["answers"]]
            cons, agree, target = consensus(answers)
            if agree < min_agreement:
                stats["low_agreement"] += 1
                ok = False
                break
            members.append({"question_id": qid, "image_id": q["image_id"], "question": q["question"],
                            "answer_type": ann["answer_type"], "consensus": cons, "agreement": agree,
                            "target": target, "answers": answers})
        if not ok:
            continue
        a, b = members
        if a["consensus"] == b["consensus"]:
            stats["same_answer"] += 1
            continue
        if a["answer_type"] != b["answer_type"]:
            stats["type_mismatch"] += 1
            continue
        if a["image_id"] == b["image_id"]:
            stats["same_image"] += 1
            continue
        if not caps.ok("question", a["question"]) or not caps.ok("question", b["question"]):
            stats["question_too_long"] += 1
            continue
        if not caps.ok("answer", a["target"]) or not caps.ok("answer", b["target"]):
            stats["answer_too_long"] += 1
            continue
        out.append({"pair_id": f"vqa_{qa}_{qb}", "answer_type": a["answer_type"],
                    "question": a["question"], "members": members})
    stats["eligible"] = len(out)
    return out


def select_vqa(eligible: list[dict], quotas: dict, g: np.random.Generator, used: set | None) -> list[dict]:
    """Stratified sample; with ``used`` (evaluation), both images must be unused and are then reserved."""
    by_type = defaultdict(list)
    for p in eligible:
        by_type[p["answer_type"]].append(p)
    chosen = []
    for t, n in quotas.items():
        pool = by_type[t]
        order = g.permutation(len(pool))
        got = 0
        for i in order:
            p = pool[i]
            ims = [m["image_id"] for m in p["members"]]
            if used is not None and any(im in used for im in ims):
                continue
            chosen.append(p)
            if used is not None:
                used.update(ims)
            got += 1
            if got == n:
                break
        if got < n:
            raise RuntimeError(f"only {got} eligible '{t}' VQA pairs, need {n}")
    return chosen


# ----------------------------------------------------------------------------- captions

def select_captions(coco: dict, caps: Caps, n_images: int, per_image: int, g: np.random.Generator,
                    used: set | None, candidates: list[int] | None = None) -> list[dict]:
    ids = sorted(candidates if candidates is not None else coco["captions"])
    out, n = [], 0
    for i in g.permutation(len(ids)):
        im = ids[i]
        if used is not None and im in used:
            continue
        valid = [(cid, c) for cid, c in coco["captions"].get(im, []) if caps.ok("caption", c)]
        if len(valid) < per_image:
            continue
        for j in sorted(g.choice(len(valid), size=per_image, replace=False)):
            out.append({"image_id": im, "caption_id": valid[j][0], "caption": valid[j][1]})
        if used is not None:
            used.add(im)
        n += 1
        if n == n_images:
            return out
    raise RuntimeError(f"only {n} caption images available, need {n_images}")


# ----------------------------------------------------------------------------- grounding

def grounding_index(coco: dict, image_ids, gcfg) -> dict:
    """Per category: positive images, negative images, and images by count (label-quality rules applied)."""
    pos, neg, counts = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    lo, hi = gcfg.count_range
    excl = set(gcfg.exclude_counting)
    for im in sorted(image_ids):
        objs = coco["objects"].get(im, {})
        tw = caption_text([c for _, c in coco["captions"].get(im, [])]) if gcfg.negative_caption_check else None
        for name in coco["categories"]:
            o = objs.get(name)
            if o is None:
                if tw is None or not mentions(tw, name):
                    neg[name].append(im)
                continue
            if o["n"] >= 1 and o["max_frac"] >= gcfg.presence_min_area_frac:
                pos[name].append(im)
            if (name not in excl and not o["crowd"] and lo <= o["n"] <= hi
                    and o["min_frac"] >= gcfg.count_min_area_frac):
                counts[name][o["n"]].append(im)
    return {"pos": pos, "neg": neg, "counts": counts}


def _presence_pair(name, a, b, idx):
    return {"pair_id": f"presence_{idx}", "kind": "presence", "category": name,
            "question": presence_question(name),
            "members": [{"image_id": a, "answer": "yes"}, {"image_id": b, "answer": "no"}]}


def _counting_pair(name, a, ca, b, cb, idx):
    return {"pair_id": f"counting_{idx}", "kind": "counting", "category": name,
            "question": counting_question(name),
            "members": [{"image_id": a, "answer": str(ca)}, {"image_id": b, "answer": str(cb)}]}


def sample_grounding(index: dict, n_presence: int, n_counting: int, g: np.random.Generator,
                     used: set | None, max_tries: int = 200000) -> list[dict]:
    """Category chosen uniformly; with ``used`` (evaluation) every image is used at most once."""
    def free(lst):
        return [x for x in lst if used is None or x not in used]

    pairs, seen = [], set()
    pres_cats = sorted(c for c in index["pos"] if index["pos"][c] and index["neg"][c])
    cnt_cats = sorted(c for c, d in index["counts"].items() if sum(1 for v in d.values() if v) >= 2)
    tries = 0
    while sum(p["kind"] == "presence" for p in pairs) < n_presence:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not sample enough presence pairs")
        name = pres_cats[g.integers(len(pres_cats))]
        P, N = free(index["pos"][name]), free(index["neg"][name])
        if not P or not N:
            continue
        a, b = int(P[g.integers(len(P))]), int(N[g.integers(len(N))])
        if a == b or ("p", name, a, b) in seen:
            continue
        seen.add(("p", name, a, b))
        pairs.append(_presence_pair(name, a, b, len(pairs)))
        if used is not None:
            used.update((a, b))
    tries = 0
    while sum(p["kind"] == "counting" for p in pairs) < n_counting:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not sample enough counting pairs")
        name = cnt_cats[g.integers(len(cnt_cats))]
        avail = {k: free(v) for k, v in index["counts"][name].items()}
        ks = sorted(k for k, v in avail.items() if v)
        if len(ks) < 2:
            continue
        ca, cb = (int(x) for x in g.choice(ks, size=2, replace=False))
        a = int(avail[ca][g.integers(len(avail[ca]))])
        b = int(avail[cb][g.integers(len(avail[cb]))])
        if a == b or ("c", name, a, b) in seen:
            continue
        seen.add(("c", name, a, b))
        pairs.append(_counting_pair(name, a, ca, b, cb, len(pairs)))
        if used is not None:
            used.update((a, b))
    return pairs


# ----------------------------------------------------------------------------- checks

def pair_images(pairs: list[dict]) -> list[int]:
    return [m["image_id"] for p in pairs for m in p["members"]]


def check_disjoint(suites: dict[str, list[int]]) -> dict:
    """Every val2014 evaluation image used once across all suites."""
    names = list(suites)
    overlaps = {}
    for i, a in enumerate(names):
        if len(set(suites[a])) != len(suites[a]):
            overlaps[f"{a} (internal)"] = len(suites[a]) - len(set(suites[a]))
        for b in names[i + 1:]:
            n = len(set(suites[a]) & set(suites[b]))
            if n:
                overlaps[f"{a} & {b}"] = n
    if overlaps:
        raise RuntimeError(f"evaluation suites are not image-disjoint: {overlaps}")
    return {k: len(v) for k, v in suites.items()}
