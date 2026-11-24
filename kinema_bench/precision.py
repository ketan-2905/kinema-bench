"""fp32 master weights for trainable parameters under bf16 training.

The failure this guards against: bf16 keeps 8 bits of mantissa. If a trainable
parameter is stored in bf16 and the optimiser applies an update smaller than
roughly 2^-8 of the parameter's own magnitude, the addition rounds straight
back to the original value. Nothing errors. The loss curve still looks
plausible, because the larger updates early in training do land. The adapter
just quietly stops learning, and at low LoRA rank, where updates are small to
begin with, that can erase most of a run.

The fix is ordinary mixed precision done properly: the frozen backbone stays
in bf16, where the memory saving is, while every trainable tensor is held in
fp32 and the optimiser steps on the fp32 copy.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def enforce_fp32_trainables(model: nn.Module, *, base_dtype=torch.bfloat16) -> dict:
    """Cast the frozen backbone to `base_dtype`; keep trainables in fp32."""
    promoted: list[str] = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.dtype is not torch.float32:
                p.data = p.data.to(torch.float32)
                promoted.append(name)
        elif p.is_floating_point() and p.dtype is not base_dtype:
            p.data = p.data.to(base_dtype)
    # Norm layers stay fp32 regardless: small, and numerically touchy.
    for mod in model.modules():
        if isinstance(mod, (nn.LayerNorm, nn.GroupNorm)):
            mod.float()
    return {"base_dtype": str(base_dtype), "promoted_to_fp32": promoted}


def assert_no_silent_erasure(model: nn.Module, *, rel_step: float = 1e-4,
                             max_erased_fraction: float = 0.5) -> dict:
    """Apply a realistically small update to each trainable tensor, confirm it sticks.

    Measured per ELEMENT, not per tensor. A weight matrix always holds a few
    near-zero entries where even a tiny absolute update lands, so asking
    whether the tensor changed at all would let wholesale erasure pass
    whenever one element happened to survive. What matters is the fraction of
    the tensor the optimiser can still move.

    Raises RuntimeError naming the offending tensors if more than
    `max_erased_fraction` of their elements rounded the update away, which is
    exactly what happens when a trainable was left in bf16. Call this after
    enforce_fp32_trainables() and before building the optimiser.
    """
    offenders: list[tuple[str, float]] = []
    checked = 0
    worst = 0.0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not (p.requires_grad and p.is_floating_point()):
                continue
            checked += 1
            before = p.detach().clone()
            scale = float(before.abs().mean()) or 1.0
            p.add_(torch.full_like(p, rel_step * scale))
            erased = float((p == before).to(torch.float32).mean())
            p.copy_(before)
            worst = max(worst, erased)
            if erased > max_erased_fraction:
                offenders.append((name, erased))
    if offenders:
        detail = ", ".join(f"{n} ({f:.0%} erased)" for n, f in offenders[:6])
        raise RuntimeError(
            f"silent update erasure: an update of {rel_step:g} x |w| vanished on "
            f"most of {len(offenders)} trainable tensor(s): {detail}"
            f"{'...' if len(offenders) > 6 else ''}. These are almost certainly "
            "still in bf16/fp16; call enforce_fp32_trainables() first."
        )
    return {"checked": checked, "rel_step": rel_step,
            "worst_erased_fraction": worst, "offenders": []}
