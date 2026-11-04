"""Model/tokenizer loading, parameter accounting, optimizer groups, FP16 checks."""
from __future__ import annotations

from typing import Iterable


def _dtype_kwarg() -> str:
    import transformers
    from packaging.version import Version
    return "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"


def load_tokenizer(name: str, revision: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(name: str, *, dtype, revision: str | None = None, device: str = "cuda",
                   attn_implementation: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, attn_implementation=attn_implementation, **{_dtype_kwarg(): dtype})
    return model.to(device)


def enable_gradient_checkpointing(model) -> None:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False


def count_parameters(model) -> dict:
    """Unique parameter counts (tied weights counted once)."""
    seen, total, trainable = set(), 0, 0
    for p in model.parameters():  # parameters() already de-duplicates shared tensors
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return {"total": total, "trainable": trainable, "trainable_fraction": trainable / max(total, 1)}


def build_param_groups(named_params: Iterable, *, lr: float, weight_decay: float, name: str = "params") -> list[dict]:
    """AdamW groups: weight decay on >=2-D non-bias tensors, none on biases and <2-D tensors."""
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or n.endswith(".bias") else decay).append(p)
    groups = []
    if decay:
        groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay, "name": f"{name}_decay"})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{name}_no_decay"})
    return groups


def fp16_cast_report(model) -> dict:
    """How well the loaded weights survive a cast to FP16 (overflow, underflow, max magnitude)."""
    import torch

    n = n_inf = n_zeroed = 0
    max_abs = 0.0
    max_rel_err = 0.0
    with torch.no_grad():
        for p in model.parameters():
            x = p.detach().float()
            h = x.half().float()
            n += x.numel()
            n_inf += int(torch.isinf(h).sum())
            n_zeroed += int(((x != 0) & (h == 0)).sum())
            max_abs = max(max_abs, float(x.abs().max()))
            nz = (x != 0) & torch.isfinite(h) & (h != 0)
            if nz.any():
                max_rel_err = max(max_rel_err, float(((h - x).abs()[nz] / x.abs()[nz]).max()))
    return {"n_params": n, "n_overflow_to_inf": n_inf, "n_underflow_to_zero": n_zeroed,
            "max_abs_weight": max_abs, "max_rel_cast_error_nonzero": max_rel_err}


def activation_range_check(model, batch: dict) -> dict:
    """Forward one batch under FP16 autocast; report max |hidden| per decoder layer and non-finite values."""
    import torch

    layers = model.get_decoder().layers
    maxima: list[float] = [0.0] * len(layers)
    nonfinite: list[bool] = [False] * len(layers)
    handles = []

    def make_hook(i):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            maxima[i] = float(h.detach().abs().float().max())
            nonfinite[i] = bool(~torch.isfinite(h.detach()).all())
        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        logits_finite = bool(torch.isfinite(out.logits).all())
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return {"max_abs_hidden_per_layer": maxima, "max_abs_hidden": max(maxima),
            "any_nonfinite_hidden": any(nonfinite), "logits_finite": logits_finite,
            "fp16_max": 65504.0}
