"""Collation and deterministic, resumable batch windows."""
from __future__ import annotations

from typing import Callable

from .losses import IGNORE_INDEX


def collate_causal(examples: list[dict], pad_id: int) -> dict:
    """Right-pad to the longest example in the microbatch."""
    import torch

    length = max(len(e["input_ids"]) for e in examples)
    bsz = len(examples)
    input_ids = torch.full((bsz, length), pad_id, dtype=torch.long)
    labels = torch.full((bsz, length), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((bsz, length), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.as_tensor(e["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.as_tensor(e["labels"], dtype=torch.long)
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def to_device(batch: dict, device: str = "cuda") -> dict:
    return {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


class SequentialWindows:
    """Fixed-order data source: update u (1-based) consumes examples [(u-1)*E, u*E).

    E = microbatch * accum. Order never changes, so resuming only needs the
    update index; the examples-consumed count is still saved as data position.
    """

    def __init__(self, examples: list[dict], microbatch: int, accum: int, collate_fn: Callable):
        self.examples = examples
        self.microbatch = microbatch
        self.accum = accum
        self.collate_fn = collate_fn
        self.window_size = microbatch * accum
        self.consumed = 0

    @property
    def num_windows(self) -> int:
        return len(self.examples) // self.window_size

    def get_window(self, update: int) -> list[dict]:
        start = (update - 1) * self.window_size
        chunk = self.examples[start:start + self.window_size]
        if len(chunk) < self.window_size:
            raise IndexError(f"update {update} needs examples up to {start + self.window_size}, "
                             f"only {len(self.examples)} available")
        self.consumed = start + self.window_size
        return [self.collate_fn(chunk[i:i + self.microbatch]) for i in range(0, self.window_size, self.microbatch)]

    def state_dict(self) -> dict:
        return {"examples_consumed": self.consumed}

    def load_state_dict(self, state: dict) -> None:
        self.consumed = int(state.get("examples_consumed", 0))


def load_metamath_rows(n: int | None = None, name: str = "meta-math/MetaMathQA"):
    """Stream MetaMathQA rows in source order (first n if given)."""
    from datasets import load_dataset

    ds = load_dataset(name, split="train", streaming=True)
    rows = []
    for row in ds:
        rows.append({k: row.get(k) for k in ("query", "response", "type", "original_question")})
        if n is not None and len(rows) >= n:
            break
    return rows
