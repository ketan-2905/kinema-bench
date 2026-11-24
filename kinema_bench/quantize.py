"""Precision variants under test: BF16, INT8 and 4-bit NF4.

On CUDA with bitsandbytes these are the real kernels. Everywhere else this
performs a genuine quantise -> dequantise ROUND TRIP in torch: the weights
really are pushed through the INT8 or NF4 codebook and back, so the accuracy
degradation measured off-GPU is real quantisation error. Only latency and
memory are unavailable, and the report says `kernels="simulated"` on those
runs so they can never be mistaken for the real thing.

NF4 follows the QLoRA formulation: blockwise absmax scaling onto a fixed
16-value codebook spaced by the quantiles of a standard normal, which is the
distribution pretrained weights actually follow.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=torch.float32)

PRECISIONS = ("bf16", "int8", "nf4")


@dataclass
class QuantReport:
    precision: str
    kernels: str            # "native" | "bitsandbytes" | "simulated"
    modules: int
    params: int
    mean_abs_error: float
    max_abs_error: float
    nominal_bits: int


def quantize_int8_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel absmax INT8."""
    w32 = w.detach().float()
    scale = w32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return (torch.clamp(torch.round(w32 / scale), -127, 127) * scale).to(w.dtype)


def quantize_nf4_roundtrip(w: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Blockwise absmax NF4."""
    shape, flat = w.shape, w.detach().float().reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype, device=flat.device)])
    blocks = flat.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    book = NF4_CODEBOOK.to(flat.device)
    idx = torch.argmin(((blocks / absmax).unsqueeze(-1) - book.view(1, 1, -1)).abs(), dim=-1)
    deq = (book[idx] * absmax).reshape(-1)
    return (deq[:-pad] if pad else deq).reshape(shape).to(w.dtype)


def apply_precision(model: nn.Module, precision: str, *, has_bnb: bool = False,
                    skip_lora: bool = True) -> QuantReport:
    """Quantise every Linear weight in place; report the error actually incurred."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")

    if precision == "bf16":
        mods = [m for m in model.modules() if isinstance(m, nn.Linear)]
        return QuantReport("bf16", "native", len(mods),
                           sum(m.weight.numel() for m in mods), 0.0, 0.0, 16)

    fn = quantize_int8_roundtrip if precision == "int8" else quantize_nf4_roundtrip
    errs, mods, params = [], 0, 0
    with torch.no_grad():
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear) or (skip_lora and "lora_" in name):
                continue
            w = mod.weight.data
            q = fn(w)
            errs.append((w.float() - q.float()).abs().flatten())
            mod.weight.data = q
            mods += 1
            params += w.numel()
    err = torch.cat(errs) if errs else torch.zeros(1)
    return QuantReport(precision, "bitsandbytes" if has_bnb else "simulated",
                       mods, params, float(err.mean()), float(err.max()),
                       8 if precision == "int8" else 4)


def drop_middle_blocks(model: nn.Module, fraction: float = 0.25) -> dict:
    """Depth ablation: remove the middle `fraction` of the backbone's blocks.

    Early blocks build features and late blocks decode, so the middle is where
    redundancy is plausible. The point of running this is to find out whether
    it is actually redundant for a narrow manipulation task, not to assume so.
    """
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4 and (
                "layers" in name or "blocks" in name):
            n = len(mod)
            k = max(1, round(n * fraction))
            start = (n - k) // 2
            keep = [m for i, m in enumerate(mod) if not (start <= i < start + k)]
            dropped = [f"{name}.{i}" for i in range(start, start + k)]
            for _ in range(n):
                del mod[0]
            for m in keep:
                mod.append(m)
            return {"fraction": fraction, "dropped": dropped, "n_dropped": k}
    return {"fraction": fraction, "dropped": [], "n_dropped": 0}
