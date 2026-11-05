"""Task 2 examples, collation and curriculum samplers.

Sequence layout (Qwen chat template, thinking disabled):
    <|im_start|>user\\n [196 visual tokens] \\n<question or instruction><|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n <target><|im_end|>
The text around the visual tokens is rendered once with a placeholder and
split, so the chat structure is exactly the template's. Without an image the
visual tokens are simply omitted (the text is unchanged; positions and masks
are rebuilt for the shorter sequence).

Loss labels cover the target and <|im_end|> (training); evaluation NLL uses the
target tokens only (termination excluded), as the assignment specifies.
"""
from __future__ import annotations

import zlib
from collections import Counter

import numpy as np

from common.config import REPO_ROOT
from common.logging_utils import read_json
from common.losses import IGNORE_INDEX

SPLITS = REPO_ROOT / "splits" / "task2"
PLACEHOLDER = "<image>"


def image_key(split: str, image_id: int) -> str:
    return f"{split}/{image_id}"


# ============================================================================ features

class FeatureStore:
    """Memory-mapped SigLIP features ([N, 196, 768] FP16) addressed by "train2014/<id>" keys."""

    def __init__(self, folder):
        self.arr = np.load(folder / "features.npy", mmap_mode="r")
        self.index = read_json(folder / "index.json")
        self.n_tokens, self.dim = self.arr.shape[1], self.arr.shape[2]

    def get(self, keys: list[str]):
        import torch
        if not keys:
            return torch.zeros(0, self.n_tokens, self.dim, dtype=torch.float16)
        return torch.from_numpy(np.stack([self.arr[self.index[k]] for k in keys]))

    def __contains__(self, key: str) -> bool:
        return key in self.index


def ensure_features():
    """Local folder with features.npy/index.json; downloaded from the HF weights repo if needed."""
    from common.paths import get_paths
    from common.sync import hf_download_dir, hf_repo
    from task2.cache_features import HF_PATH, feature_dir

    for folder in (feature_dir(), get_paths().downloads / HF_PATH):
        if (folder / "features.npy").exists() and (folder / "index.json").exists():
            return folder
    return hf_download_dir(hf_repo("weights"), HF_PATH, get_paths().downloads)


# ============================================================================ prompts

class PromptBuilder:
    def __init__(self, tokenizer, caption_instruction: str):
        from common.chat_format import end_of_turn_id, generation_prompt

        self.tok = tokenizer
        self.caption_instruction = caption_instruction
        self.eos = end_of_turn_id(tokenizer)
        full = generation_prompt(tokenizer, f"{PLACEHOLDER}\nX")
        before, after = full.split(PLACEHOLDER)
        self.before_ids = self._ids(before)
        self._after_tmpl = after  # "\nX<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._cache: dict = {}

    def _ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def after_ids(self, text: str) -> list[int]:
        if text not in self._cache:
            self._cache[text] = self._ids(self._after_tmpl.replace("\nX", "\n" + text, 1))
        return self._cache[text]

    def target_ids(self, text: str) -> list[int]:
        return self._ids(text)


def _example(pb: PromptBuilder, kind: str, text: str, target: str, image: str, partner: str | None = None,
             refs: list[str] | None = None, meta: dict | None = None) -> dict:
    return {"kind": kind, "text": text, "target": target, "image": image, "partner": partner,
            "after_ids": pb.after_ids(text), "target_ids": pb.target_ids(target), "refs": refs, **(meta or {})}


def caption_examples(pb: PromptBuilder, rows: list[dict], split: str) -> list[dict]:
    return [_example(pb, "caption", pb.caption_instruction, r["caption"], image_key(split, r["image_id"]),
                     meta={"image_id": r["image_id"], "caption_id": r["caption_id"]}) for r in rows]


def pair_examples(pb: PromptBuilder, pairs: list[dict], split: str) -> list[list[dict]]:
    """One [example, example] list per pair; each example knows its partner image."""
    out = []
    for p in pairs:
        kind = p.get("kind", "vqa")
        ms = p["members"]
        keys = [image_key(split, m["image_id"]) for m in ms]
        pair = []
        for i, m in enumerate(ms):
            target = m.get("target", m.get("answer"))
            pair.append(_example(pb, kind, p["question"], target, keys[i], partner=keys[1 - i],
                                 refs=m.get("answers"),
                                 meta={"pair_id": p["pair_id"], "answer_type": p.get("answer_type", kind),
                                       "question_id": m.get("question_id"), "image_id": m["image_id"],
                                       "canonical": m.get("consensus", target)}))
        out.append(pair)
    return out


def load_split_examples(pb: PromptBuilder) -> dict:
    j = lambda name: read_json(SPLITS / name)  # noqa: E731
    pools = j("grounding_train_pools.json")
    return {
        "train": {
            "caption": caption_examples(pb, j("captions_train.json"), "train2014"),
            "presence": pair_examples(pb, [p for p in pools if p["kind"] == "presence"], "train2014"),
            "counting": pair_examples(pb, [p for p in pools if p["kind"] == "counting"], "train2014"),
            "vqa": pair_examples(pb, j("vqa_train.json"), "train2014"),
        },
        "dev": {
            "caption": caption_examples(pb, j("captions_dev.json"), "val2014"),
            "grounding": pair_examples(pb, j("grounding_dev.json"), "val2014"),
            "vqa": pair_examples(pb, j("vqa_dev.json"), "val2014"),
        },
        "test": {
            "grounding": pair_examples(pb, j("grounding_test.json"), "val2014"),
            "vqa": pair_examples(pb, j("vqa_test.json"), "val2014"),
        },
    }


# ============================================================================ collation

def collate(examples: list[dict], store: FeatureStore, pb: PromptBuilder, pad_id: int, images: list | None = None,
            with_eos: bool = True, supervise: bool = True) -> dict:
    """Right-padded batch. ``images`` overrides each example's image key (None = no image)."""
    import torch

    keys = [e["image"] for e in examples] if images is None else images
    n_vis = store.n_tokens
    seqs, labs, masks = [], [], []
    for e, k in zip(examples, keys):
        tgt = e["target_ids"] + ([pb.eos] if with_eos else [])
        vis = n_vis if k is not None else 0
        ids = pb.before_ids + [pad_id] * vis + e["after_ids"] + tgt
        lab = [IGNORE_INDEX] * (len(ids) - len(tgt)) + (tgt if supervise else [IGNORE_INDEX] * len(tgt))
        img = [0] * len(pb.before_ids) + [1] * vis + [0] * (len(ids) - len(pb.before_ids) - vis)
        seqs.append(ids)
        labs.append(lab)
        masks.append(img)
    L = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    image_mask = torch.zeros((B, L), dtype=torch.bool)
    for i, (s, l, m) in enumerate(zip(seqs, labs, masks)):
        input_ids[i, :len(s)] = torch.tensor(s)
        labels[i, :len(s)] = torch.tensor(l)
        attn[i, :len(s)] = 1
        image_mask[i, :len(s)] = torch.tensor(m, dtype=torch.bool)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn, "image_mask": image_mask,
            "features": store.get([k for k in keys if k is not None]), "kinds": [e["kind"] for e in examples]}


# ============================================================================ samplers (stateless per update -> exact resumption)

def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


class _Base:
    def __init__(self, store, pb, pad_id, microbatch: int, accum: int, seed: int, stage: int):
        self.store, self.pb, self.pad_id = store, pb, pad_id
        self.microbatch, self.accum, self.seed, self.stage = microbatch, accum, seed, stage
        self.window_size = microbatch * accum
        self.consumed = 0
        self.last_counts: Counter = Counter()

    def _collate_window(self, exs: list[dict]) -> list[dict]:
        self.last_counts = Counter(e["kind"] for e in exs)
        return [collate(exs[i:i + self.microbatch], self.store, self.pb, self.pad_id)
                for i in range(0, len(exs), self.microbatch)]

    def state_dict(self) -> dict:
        return {"examples_consumed": self.consumed, "stage": self.stage}

    def load_state_dict(self, state: dict) -> None:
        self.consumed = int(state.get("examples_consumed", 0))


class EpochSampler(_Base):
    """Stage 1: passes over the caption examples, each pass in its own seeded random order."""

    def __init__(self, examples: list[dict], **kw):
        super().__init__(**kw)
        self.examples = examples
        self._perm: dict = {}

    def _order(self, epoch: int):
        if epoch not in self._perm:
            self._perm[epoch] = _rng(self.seed, f"stage{self.stage}_pass{epoch}").permutation(len(self.examples))
        return self._perm[epoch]

    def get_window(self, update: int) -> list[dict]:
        n = len(self.examples)
        idx = [(update - 1) * self.window_size + j for j in range(self.window_size)]
        exs = [self.examples[self._order(i // n)[i % n]] for i in idx]
        self.consumed = update * self.window_size
        return self._collate_window(exs)


class MixtureSampler(_Base):
    """Stages 2-3: each draw picks a component by probability; a draw is one complete question pair
    (two examples) or, for caption replay, two random caption examples. No exact quotas."""

    def __init__(self, pools: dict[str, list], probs: dict[str, float], **kw):
        super().__init__(**kw)
        self.pools = pools
        self.names = list(probs)
        p = np.array([probs[k] for k in self.names], dtype=float)
        self.p = p / p.sum()

    def get_window(self, update: int) -> list[dict]:
        g = _rng(self.seed, f"stage{self.stage}_update{update}")
        exs = []
        for _ in range(self.window_size // 2):
            name = self.names[g.choice(len(self.names), p=self.p)]
            pool = self.pools[name]
            if name == "caption":
                exs += [pool[int(i)] for i in g.integers(len(pool), size=2)]
            else:
                exs += pool[int(g.integers(len(pool)))]
        self.consumed = update * self.window_size
        return self._collate_window(exs)
