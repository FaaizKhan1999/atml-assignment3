"""LoRA and OPLoRA adapters (custom, shared implementation).

LoRA:    W = W0 + (alpha/r) B A,  A in R^{r x d_in} (random), B in R^{d_out x r} (zero)
OPLoRA:  same factors, but the forward pass uses projected effective factors
             B~ = B - U_k (U_k^T B),   A~ = A - (A V_k) V_k^T
         where U_k, V_k are the leading k singular vectors of the starting
         weight W0 = U S V^T, computed once in FP32 and frozen. Projections are
         applied to the factors, never as dense d x d projection matrices.

Both methods draw A from the same seeded CPU generator in the same module
order, so LoRA-16 and OPLoRA-16 start from identical unprojected factors.
A uses PEFT's default initialization (kaiming_uniform with a=sqrt(5)), i.e.
U(-1/sqrt(d_in), 1/sqrt(d_in)).
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _autocast_on(device_type: str) -> bool:
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:  # older torch: no device argument
        return torch.is_autocast_enabled()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float, A_init: torch.Tensor,
                 U_k: torch.Tensor | None = None, V_k: torch.Tensor | None = None):
        super().__init__()
        if base.bias is not None:
            base.bias.requires_grad_(False)
        base.weight.requires_grad_(False)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        device = base.weight.device
        self.lora_A = nn.Parameter(A_init.to(device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=device, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.protected = U_k is not None
        if self.protected:
            self.register_buffer("U_k", U_k.to(device=device, dtype=torch.float32).contiguous(), persistent=False)
            self.register_buffer("V_k", V_k.to(device=device, dtype=torch.float32).contiguous(), persistent=False)
        else:
            self.U_k = None
            self.V_k = None

    def effective_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.protected:
            return self.lora_A, self.lora_B
        with torch.autocast(device_type=self.lora_A.device.type, enabled=False):
            A = self.lora_A - (self.lora_A @ self.V_k) @ self.V_k.T
            B = self.lora_B - self.U_k @ (self.U_k.T @ self.lora_B)
        return A, B

    def delta_weight(self) -> torch.Tensor:
        """Dense FP32 correction (alpha/r) B~ A~ (for diagnostics/merging only)."""
        with torch.no_grad():
            A, B = self.effective_factors()
            return self.scaling * (B.float() @ A.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        A, B = self.effective_factors()
        if not _autocast_on(x.device.type):
            A, B = A.to(x.dtype), B.to(x.dtype)
        delta = F.linear(F.linear(self.lora_dropout(x), A), B)
        return out + delta * self.scaling

    def extra_repr(self) -> str:
        return f"r={self.r}, alpha={self.alpha}, protected={self.protected}"


def _init_A(r: int, d_in: int, gen: torch.Generator) -> torch.Tensor:
    bound = 1.0 / math.sqrt(d_in)
    return torch.rand(r, d_in, generator=gen, dtype=torch.float32) * (2 * bound) - bound


def inject_adapters(model, *, targets=DEFAULT_TARGETS, r: int, alpha: float, dropout: float, seed: int,
                    protect_rank: int | None = None) -> tuple[dict[str, LoRALinear], dict]:
    """Freeze the model and replace target nn.Linear layers with LoRALinear (in place).

    Returns ({module_path: LoRALinear}, info) where info has adapter parameter
    count, SVD preprocessing time and basis storage.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    targets = set(targets)
    found = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if child_name in targets and isinstance(child, nn.Linear):
                path = f"{parent_name}.{child_name}" if parent_name else child_name
                found.append((path, parent, child_name, child))
    if not found:
        raise ValueError(f"no target modules {sorted(targets)} found")

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    modules: dict[str, LoRALinear] = {}
    svd_seconds, basis_bytes = 0.0, 0
    for path, parent, child_name, lin in found:
        A0 = _init_A(r, lin.in_features, gen)
        U_k = V_k = None
        if protect_rank:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            W0 = lin.weight.detach().float()
            U, _, Vh = torch.linalg.svd(W0, full_matrices=False)
            U_k = U[:, :protect_rank].contiguous()
            V_k = Vh[:protect_rank].T.contiguous()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            svd_seconds += time.perf_counter() - t0
            basis_bytes += (U_k.numel() + V_k.numel()) * 4
        new = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout, A_init=A0, U_k=U_k, V_k=V_k)
        setattr(parent, child_name, new)
        modules[path] = new

    n_adapter = sum(m.lora_A.numel() + m.lora_B.numel() for m in modules.values())
    info = {"n_modules": len(modules), "adapter_params": n_adapter, "r": r, "alpha": alpha,
            "protect_rank": protect_rank, "svd_seconds": svd_seconds, "basis_bytes_fp32": basis_bytes}
    return modules, info


@torch.no_grad()
def export_adapters(modules: dict[str, LoRALinear]) -> tuple[dict[str, torch.Tensor], dict]:
    """Effective (projected for OPLoRA) FP32 factors, keyed '<module>.lora_A' / '<module>.lora_B'."""
    out = {}
    for name, m in modules.items():
        A, B = m.effective_factors()
        out[f"{name}.lora_A"] = A.detach().float().cpu().contiguous()
        out[f"{name}.lora_B"] = B.detach().float().cpu().contiguous()
    first = next(iter(modules.values()))
    meta = {"r": first.r, "alpha": first.alpha, "scaling": first.scaling, "projected": first.protected}
    return out, meta


@torch.no_grad()
def export_bases(modules: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    out = {}
    for name, m in modules.items():
        if m.protected:
            out[f"{name}.U_k"] = m.U_k.cpu().contiguous()
            out[f"{name}.V_k"] = m.V_k.cpu().contiguous()
    return out


@torch.no_grad()
def merge_exported_into(model, factors: dict[str, torch.Tensor], scaling: float) -> int:
    """Add scaling * B @ A to the matching nn.Linear weights of a fresh model (FP32 math). Returns #merged."""
    names = sorted({k.rsplit(".", 1)[0] for k in factors})
    for name in names:
        lin = model.get_submodule(name)
        W = lin.weight
        A = factors[f"{name}.lora_A"].to(W.device)
        B = factors[f"{name}.lora_B"].to(W.device)
        W.copy_((W.float() + scaling * (B @ A)).to(W.dtype))
    return len(names)


@torch.no_grad()
def projection_residuals(modules: dict[str, LoRALinear]) -> dict:
    """Numerical check of OPLoRA constraints: ||U_k^T B~||/||B~|| and ||A~ V_k||/||A~||."""
    rb, ra = [], []
    for m in modules.values():
        if not m.protected:
            continue
        A, B = m.effective_factors()
        nb, na = B.norm(), A.norm()
        rb.append(float((m.U_k.T @ B).norm() / nb) if nb > 0 else 0.0)
        ra.append(float((A @ m.V_k).norm() / na) if na > 0 else 0.0)
    if not rb:
        return {}
    return {"max_rel_UkT_B": max(rb), "mean_rel_UkT_B": sum(rb) / len(rb),
            "max_rel_A_Vk": max(ra), "mean_rel_A_Vk": sum(ra) / len(ra)}
