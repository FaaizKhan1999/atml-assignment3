"""Shared, resumable training loop (Tasks 1 and 2).

One optimizer update:
  1. set the learning rate for update u (warmup + cosine, see ``lr_factor``)
  2. for each microbatch: FP16 autocast forward, loss normalized over the whole
     effective batch (the objective's job), scaled backward
  3. unscale gradients
  4. optional L2-SP: add lambda * (theta - theta0) to the unscaled gradients
     (identical to adding lambda * Omega to the loss), record Omega
  5. clip the global gradient norm
  6. GradScaler step/update; a decrease of the loss scale means the update was
     skipped because of inf/NaN gradients, which is logged

Every update is logged to JSONL. Checkpoints are written at the requested
updates and the loop resumes from the newest local (or fetched) checkpoint.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

from .checkpoint import find_latest, load_checkpoint, prune_checkpoints, save_checkpoint
from .data_utils import to_device
from .logging_utils import JsonlLogger, get_logger
from .profiling import cuda_sync, peak_memory, reset_peak_memory

log = get_logger()


def lr_factor(update: int, warmup: int, total: int, min_ratio: float = 0.0) -> float:
    """LR multiplier for 1-based update ``update``.

    Linear warmup reaches the peak at update ``warmup``; cosine decay then starts
    at the peak on update ``warmup + 1`` and approaches ``min_ratio`` at the end.
    No update is spent at learning rate zero.
    """
    if warmup > 0 and update <= warmup:
        return update / warmup
    progress = min(1.0, (update - warmup - 1) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


class L2SPRegularizer:
    """Starting-point L2 penalty Omega = 0.5 * sum_j (theta_j - theta0_j)^2 over all unique parameters.

    theta0 is a frozen FP32 copy kept on the GPU (``ref_device='cuda'``) or in
    pinned CPU memory (``'cpu'``, streamed in chunks each update). Work is done
    in 2^24-element chunks to keep temporary memory small.
    """

    CHUNK = 1 << 24

    def __init__(self, model, lam: float, ref_device: str = "cuda"):
        import torch

        self.lam = float(lam)
        self.ref_device = ref_device
        self.params = [p for _, p in model.named_parameters()]  # unique; tied weights once
        pin = ref_device == "cpu" and torch.cuda.is_available()
        self.refs = []
        with torch.no_grad():
            for p in self.params:
                r = p.detach().to(device=ref_device, dtype=torch.float32, copy=True)
                self.refs.append(r.pin_memory() if pin else r)
        self.ref_bytes = sum(r.numel() * r.element_size() for r in self.refs)
        self.n_params = sum(p.numel() for p in self.params)

    def apply_(self) -> tuple[float, float]:
        """Add lambda*(theta-theta0) to unscaled grads. Returns (Omega before the update, seconds)."""
        import torch

        cuda_sync()
        t0 = time.perf_counter()
        sq = []
        with torch.no_grad():
            for p, r in zip(self.params, self.refs):
                pf = p.detach().view(-1)
                rf = r.view(-1)
                g = None
                if p.grad is not None:
                    if not p.grad.is_contiguous():
                        raise RuntimeError("non-contiguous gradient")
                    g = p.grad.view(-1)
                for s in range(0, pf.numel(), self.CHUNK):
                    ref = rf[s:s + self.CHUNK]
                    if ref.device != pf.device:
                        ref = ref.to(pf.device, non_blocking=True)
                    d = pf[s:s + self.CHUNK].float() - ref
                    sq.append(torch.dot(d, d))
                    if g is not None:
                        g[s:s + self.CHUNK].add_(d.to(g.dtype), alpha=self.lam)
            omega = 0.5 * torch.stack(sq).sum().item()
        cuda_sync()
        return omega, time.perf_counter() - t0

    def displacement(self) -> float:
        """||theta - theta0||_2 over all unique parameters."""
        import torch

        sq = []
        with torch.no_grad():
            for p, r in zip(self.params, self.refs):
                pf, rf = p.detach().view(-1), r.view(-1)
                for s in range(0, pf.numel(), self.CHUNK):
                    d = pf[s:s + self.CHUNK].float() - rf[s:s + self.CHUNK].to(pf.device, non_blocking=True)
                    sq.append(torch.dot(d, d))
            return math.sqrt(torch.stack(sq).sum().item())


def build_adamw(param_groups: list[dict], betas=(0.9, 0.999), eps: float = 1e-8, fused: bool = True):
    import torch

    if fused:
        try:
            return torch.optim.AdamW(param_groups, betas=tuple(betas), eps=eps, fused=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            log.warning("fused AdamW unavailable (%s); using foreach", exc)
    return torch.optim.AdamW(param_groups, betas=tuple(betas), eps=eps, foreach=True)


class Trainer:
    """Generic loop. ``data`` must provide get_window(update), state_dict(), load_state_dict().
    ``objective`` must provide prepare_window(window), __call__(model, mb, ctx) -> (loss, stats),
    and summarize(stats_list) -> dict with a 'loss' key."""

    def __init__(self, *, model, param_groups: list[dict], objective, data, total_updates: int,
                 warmup_updates: int, ckpt_dir: str | Path, log_path: str | Path,
                 min_lr_ratio: float = 0.0, grad_clip: float = 1.0, betas=(0.9, 0.999), eps: float = 1e-8,
                 fused_adam: bool = True, l2sp: L2SPRegularizer | None = None,
                 checkpoint_at: tuple | list = (), checkpoint_final: bool = True, keep_last: int = 1,
                 on_checkpoint: Callable | None = None, device: str = "cuda"):
        import torch

        self.model = model
        self.objective = objective
        self.data = data
        self.total_updates = int(total_updates)
        self.warmup_updates = int(warmup_updates)
        self.min_lr_ratio = float(min_lr_ratio)
        self.grad_clip = float(grad_clip)
        self.l2sp = l2sp
        self.ckpt_dir = Path(ckpt_dir)
        self.checkpoint_at = set(int(u) for u in checkpoint_at)
        if checkpoint_final:
            self.checkpoint_at.add(self.total_updates)
        self.keep_last = keep_last
        self.on_checkpoint = on_checkpoint
        self.device = device

        self.trainable = [p for g in param_groups for p in g["params"]]
        self.optimizer = build_adamw(param_groups, betas=betas, eps=eps, fused=fused_adam)
        for g in self.optimizer.param_groups:
            g.setdefault("base_lr", g["lr"])
        self.scaler = torch.amp.GradScaler("cuda")
        self.logger = JsonlLogger(log_path)

        self.update = 0
        self.train_seconds = 0.0
        self.n_skipped = 0
        self.peak_allocated_gb = 0.0

    # ------------------------------------------------------------------ resume
    def resume(self, fallback_fetch: Callable[[], Path | None] | None = None) -> bool:
        path = find_latest(self.ckpt_dir)
        if path is None and fallback_fetch is not None:
            path = fallback_fetch()
        if path is None:
            return False
        state = load_checkpoint(path, self.model, self.optimizer, self.scaler)
        self.update = int(state["update"])
        self.data.load_state_dict(state["data"])
        extra = state.get("extra", {})
        self.train_seconds = float(extra.get("train_seconds", 0.0))
        self.n_skipped = int(extra.get("n_skipped", 0))
        self.peak_allocated_gb = float(extra.get("peak_allocated_gb", 0.0))
        self.logger.truncate_after("update", self.update)
        log.info("resumed from %s at update %d", path, self.update)
        return True

    # ------------------------------------------------------------------ one update
    def _set_lr(self, update: int) -> float:
        f = lr_factor(update, self.warmup_updates, self.total_updates, self.min_lr_ratio)
        for g in self.optimizer.param_groups:
            g["lr"] = g["base_lr"] * f
        return f

    def _train_update(self, update: int) -> dict:
        import torch

        self._set_lr(update)
        window = self.data.get_window(update)
        ctx = self.objective.prepare_window(window)
        self.model.train()
        stats, input_tokens = [], 0
        for mb in window:
            input_tokens += int(mb["attention_mask"].sum())
            mb = to_device(mb, self.device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, st = self.objective(self.model, mb, ctx)
            self.scaler.scale(loss).backward()
            stats.append(st)
        self.scaler.unscale_(self.optimizer)
        rec: dict = {}
        if self.l2sp is not None:
            omega, l2sp_seconds = self.l2sp.apply_()
            rec.update({"omega": omega, "lambda_omega": self.l2sp.lam * omega, "l2sp_seconds": l2sp_seconds})
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, self.grad_clip)
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        skipped = self.scaler.get_scale() < scale_before
        self.optimizer.zero_grad(set_to_none=True)

        summary = self.objective.summarize(stats)
        rec.update({
            "event": "update",
            "update": update,
            "lr": {g.get("name", str(i)): g["lr"] for i, g in enumerate(self.optimizer.param_groups)},
            "grad_norm": float(grad_norm),
            "loss_scale": scale_before,
            "skipped": bool(skipped),
            "input_tokens": input_tokens,
            **summary,
        })
        if self.l2sp is not None:
            rec["total_loss"] = rec["loss"] + rec["lambda_omega"]
        return rec

    # ------------------------------------------------------------------ checkpoint
    def checkpoint(self, update: int) -> tuple[Path, dict]:
        t0 = time.perf_counter()
        extra = {"train_seconds": self.train_seconds, "n_skipped": self.n_skipped,
                 "peak_allocated_gb": self.peak_allocated_gb}
        path, sizes = save_checkpoint(self.ckpt_dir, update, self.model, self.optimizer, self.scaler,
                                      self.data.state_dict(), extra)
        sizes["save_seconds"] = time.perf_counter() - t0
        self.logger.log({"event": "checkpoint", "update": update, "path": str(path), **sizes})
        prune_checkpoints(self.ckpt_dir, keep_last=self.keep_last)
        if self.on_checkpoint is not None:
            self.on_checkpoint(update, path, sizes)
        return path, sizes

    # ------------------------------------------------------------------ main loop
    def train(self) -> dict:
        import torch

        reset_peak_memory()
        self.optimizer.zero_grad(set_to_none=True)
        while self.update < self.total_updates:
            u = self.update + 1
            cuda_sync()
            t0 = time.perf_counter()
            rec = self._train_update(u)
            cuda_sync()
            dt = time.perf_counter() - t0
            self.train_seconds += dt
            self.update = u
            self.n_skipped += int(rec["skipped"])
            self.peak_allocated_gb = max(self.peak_allocated_gb, torch.cuda.max_memory_allocated() / 2**30)
            rec.update({"update_seconds": dt, "train_seconds": self.train_seconds})
            self.logger.log(rec)
            if rec["skipped"]:
                log.warning("update %d skipped (inf/NaN gradients); loss scale -> %.0f", u, self.scaler.get_scale())
            if u % 25 == 0 or u == 1 or u == self.total_updates:
                log.info("update %d/%d loss %.4f lr %.2e gnorm %.3f %.2fs", u, self.total_updates, rec["loss"],
                         next(iter(rec["lr"].values())), rec["grad_norm"], dt)
            if u in self.checkpoint_at:
                self.checkpoint(u)
        return self.summary()

    def summary(self) -> dict:
        return {"updates": self.update, "train_seconds": self.train_seconds, "skipped_updates": self.n_skipped,
                "peak_allocated_gb": round(self.peak_allocated_gb, 3), **peak_memory()}
