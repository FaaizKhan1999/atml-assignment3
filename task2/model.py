"""SigLIP features -> MLP connector -> Qwen3-0.6B.

Connector: Linear(768, 1024) -> GELU -> Linear(1024, 1024), FP32 parameters,
followed by a fixed output scale set once at initialization (calibration, see
``calibrate``). Visual tokens are spliced into the text embeddings with
``masked_scatter`` (differentiable w.r.t. the connector output), so the frozen
Qwen forward pass still carries gradients back to the connector.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.losses import response_token_losses


class Connector(nn.Module):
    def __init__(self, d_in: int = 768, hidden: int = 1024, d_out: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_out)
        self.register_buffer("out_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))) * self.out_scale


class VLM(nn.Module):
    def __init__(self, lm, connector: Connector):
        super().__init__()
        self.lm = lm
        self.connector = connector

    # delegation used by common.losses.response_token_losses
    def get_decoder(self):
        return self.lm.get_decoder()

    def get_output_embeddings(self):
        return self.lm.get_output_embeddings()

    def embed(self, batch: dict) -> torch.Tensor:
        emb = self.lm.get_input_embeddings()(batch["input_ids"])
        mask = batch["image_mask"]
        if bool(mask.any()):
            vis = self.connector(batch["features"].to(emb.device))
            emb = emb.masked_scatter(mask.unsqueeze(-1).expand_as(emb), vis.to(emb.dtype))
        return emb

    def lm_inputs(self, batch: dict) -> dict:
        return {"inputs_embeds": self.embed(batch), "attention_mask": batch["attention_mask"],
                "labels": batch["labels"]}

    def logits(self, batch: dict) -> torch.Tensor:
        return self.lm(inputs_embeds=self.embed(batch), attention_mask=batch["attention_mask"], use_cache=False).logits


def per_example_losses(vlm: VLM, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(mean token loss per example [B], supervised token count per example [B])."""
    losses, ex = response_token_losses(vlm, vlm.lm_inputs(batch))
    bsz = batch["labels"].shape[0]
    sums = torch.zeros(bsz, device=losses.device, dtype=losses.dtype).index_add_(0, ex, losses)
    counts = torch.bincount(ex, minlength=bsz).to(losses.dtype)
    return sums / counts.clamp(min=1), counts


class VLMObjective:
    """Assignment loss: mean token loss within each example, then mean over the effective batch."""

    def prepare_window(self, window: list[dict]) -> dict:
        return {"n_examples": sum(int(mb["labels"].shape[0]) for mb in window)}

    def __call__(self, model: VLM, batch: dict, ctx: dict):
        per_ex, counts = per_example_losses(model, batch)
        if bool((counts == 0).any()):
            raise ValueError("example without supervised tokens")
        return per_ex.sum() / ctx["n_examples"], {"example_loss_sum": per_ex.detach().sum(),
                                                  "n_examples": int(per_ex.numel()),
                                                  "kinds": batch.get("kinds", [])}

    def summarize(self, stats: list[dict]) -> dict:
        from collections import Counter
        total = torch.stack([s["example_loss_sum"] for s in stats]).sum().item()
        n = sum(s["n_examples"] for s in stats)
        kinds = Counter(k for s in stats for k in s["kinds"])
        return {"loss": total / max(n, 1), "examples": n, "mixture": dict(kinds)}


@torch.no_grad()
def calibrate(vlm: VLM, store, texts_ids: list[list[int]], image_keys: list[str], device: str = "cuda") -> dict:
    """Set connector.out_scale so the mean per-token RMS of the initial visual tokens equals the mean
    per-token RMS of Qwen input embeddings over training text tokens (training data only)."""
    emb = vlm.lm.get_input_embeddings().weight
    ids = torch.tensor([t for seq in texts_ids for t in seq], device=emb.device)
    text_rms = emb[ids].float().pow(2).mean(-1).sqrt().mean().item()
    vlm.connector.out_scale.fill_(1.0)
    vis = vlm.connector(store.get(image_keys).to(device).float())
    vis_rms = vis.pow(2).mean(-1).sqrt().mean().item()
    scale = text_rms / vis_rms
    vlm.connector.out_scale.fill_(scale)
    after = vlm.connector(store.get(image_keys).to(device).float()).pow(2).mean(-1).sqrt().mean().item()
    return {"rule": "out_scale = mean per-token RMS of Qwen input embeddings over training text tokens / "
                    "mean per-token RMS of the initialized connector outputs on training images",
            "text_token_rms": text_rms, "visual_token_rms_before": vis_rms, "scale": scale,
            "visual_token_rms_after": after, "n_text_tokens": int(ids.numel()), "n_images": len(image_keys)}
