"""Response-only causal LM losses.

The LM head is applied only at positions whose next token is supervised,
in chunks under activation checkpointing. This gives exactly the same loss
as projecting every position onto the 151k-token vocabulary, but avoids
materializing [batch, seq, vocab] logits (~2.5 GB in FP32 at 2x2048 tokens),
which is what makes full fine-tuning fit on a 16 GB T4.

Two objectives are provided:
* TokenNormalizedObjective   - Task 1: sum of token losses / supervised tokens
                               in the whole effective batch
* ExampleNormalizedObjective - Task 2: mean token loss within each example,
                               then mean over examples in the effective batch
"""
from __future__ import annotations

from functools import partial

IGNORE_INDEX = -100


def _ce_chunk(lm_head, hidden, targets):
    import torch.nn.functional as F
    logits = lm_head(hidden)
    return F.cross_entropy(logits.float(), targets, reduction="none")


def response_token_losses(model, batch: dict, chunk_size: int = 1024):
    """Per-token CE at supervised positions.

    Returns (losses [N] float32, example_index [N] long). ``batch`` holds
    ``labels`` and ``attention_mask`` plus ``input_ids`` or ``inputs_embeds``
    (and optionally ``position_ids``). Labels are aligned with inputs; the
    shift happens here.
    """
    import torch
    from torch.utils.checkpoint import checkpoint

    decoder = model.get_decoder()
    out = decoder(input_ids=batch.get("input_ids"), inputs_embeds=batch.get("inputs_embeds"),
                  attention_mask=batch["attention_mask"], position_ids=batch.get("position_ids"),
                  use_cache=False)
    hidden = out.last_hidden_state[:, :-1]
    labels = batch["labels"][:, 1:]
    mask = labels != IGNORE_INDEX
    h = hidden[mask]
    targets = labels[mask]
    example_index = mask.nonzero(as_tuple=True)[0]
    lm_head = model.get_output_embeddings()
    fn = partial(_ce_chunk, lm_head)
    pieces = []
    for start in range(0, h.shape[0], chunk_size):
        hc, tc = h[start:start + chunk_size], targets[start:start + chunk_size]
        if torch.is_grad_enabled() and hc.requires_grad:
            pieces.append(checkpoint(fn, hc, tc, use_reentrant=False))
        else:
            pieces.append(fn(hc, tc))
    losses = torch.cat(pieces) if pieces else hidden.new_zeros(0, dtype=torch.float32)
    return losses, example_index


def supervised_token_count(batch: dict) -> int:
    return int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())


class TokenNormalizedObjective:
    """Sum of token losses divided by all supervised tokens in the effective batch."""

    def prepare_window(self, window: list[dict]) -> dict:
        return {"n_tokens": sum(supervised_token_count(mb) for mb in window)}

    def __call__(self, model, batch: dict, ctx: dict):
        losses, _ = response_token_losses(model, batch)
        loss = losses.sum() / max(ctx["n_tokens"], 1)
        return loss, {"loss_sum": losses.detach().sum(), "n_tokens": losses.numel()}

    def summarize(self, stats: list[dict]) -> dict:
        import torch
        total = torch.stack([s["loss_sum"] for s in stats]).sum().item()
        n = sum(s["n_tokens"] for s in stats)
        return {"loss": total / max(n, 1), "supervised_tokens": n}


class ExampleNormalizedObjective:
    """Mean token loss per example, averaged over examples in the effective batch."""

    def prepare_window(self, window: list[dict]) -> dict:
        return {"n_examples": sum(int(mb["labels"].shape[0]) for mb in window)}

    def __call__(self, model, batch: dict, ctx: dict):
        import torch

        losses, ex = response_token_losses(model, batch)
        bsz = batch["labels"].shape[0]
        sums = torch.zeros(bsz, device=losses.device, dtype=losses.dtype).index_add_(0, ex, losses)
        counts = torch.bincount(ex, minlength=bsz).to(losses.dtype)
        if bool((counts == 0).any()):
            raise ValueError("an example in the batch has no supervised tokens")
        per_example = sums / counts
        loss = per_example.sum() / max(ctx["n_examples"], 1)
        return loss, {"example_loss_sum": per_example.detach().sum(), "n_examples": bsz,
                      "n_tokens": losses.numel()}

    def summarize(self, stats: list[dict]) -> dict:
        import torch
        total = torch.stack([s["example_loss_sum"] for s in stats]).sum().item()
        n = sum(s["n_examples"] for s in stats)
        return {"loss": total / max(n, 1), "examples": n,
                "supervised_tokens": sum(s["n_tokens"] for s in stats)}
