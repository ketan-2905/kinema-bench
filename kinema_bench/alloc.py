"""Budget-matched variable-rank LoRA allocation.

A LoRA adapter on a (d_out, d_in) weight at rank r adds r*(d_in + d_out)
trainable parameters.

The experiment only means something if the variable-rank configuration is
compared against uniform rank-32 at EQUAL trainable-parameter count.
Otherwise a win is just "more capacity wins", which nobody needs an ablation
to learn. So the uniform budget is a hard ceiling here, and sensitivity only
decides how that fixed budget is distributed.
"""
from __future__ import annotations

from .regions import REGION_ORDER


def lora_params(shape: tuple[int, int], rank: int) -> int:
    d_out, d_in = shape
    return rank * (d_in + d_out)


def uniform_budget(shapes: dict[str, tuple[int, int]], uniform_rank: int) -> int:
    return sum(lora_params(s, uniform_rank) for s in shapes.values())


def _cost_per_rank(shapes, region_of) -> dict[str, int]:
    """Parameters added per +1 of rank, per region."""
    cost = {r: 0 for r in REGION_ORDER}
    for name, shape in shapes.items():
        if (region := region_of.get(name)) is not None:
            cost[region] += shape[0] + shape[1]
    return cost


def allocate_variable_rank(
    *, shapes: dict[str, tuple[int, int]], region_of: dict[str, str],
    sensitivity: dict[str, float], budget: int,
    rank_min: int = 4, rank_max: int = 128, rank_multiple: int = 4,
    sensitivity_floor: float = 0.05,
) -> dict:
    """Spend `budget` trainable parameters across regions, sensitivity-weighted.

    Guarantees on return:
      * allocated_params <= budget            (never buys extra capacity)
      * rank_min <= rank <= rank_max          for every live region
      * every rank is a multiple of rank_multiple
    """
    cost = _cost_per_rank(shapes, region_of)
    live = [r for r in REGION_ORDER if cost[r] > 0]
    if not live:
        raise ValueError("no LoRA target modules matched any functional region")

    raw = {r: max(float(sensitivity.get(r, 0.0)), sensitivity_floor) for r in live}
    total = sum(raw.values())
    share = {r: raw[r] / total for r in live}

    def clamp(x: float) -> int:
        r = int(round(x / rank_multiple)) * rank_multiple
        return max(rank_min, min(rank_max, r))

    rank = {r: clamp(budget * share[r] / cost[r]) for r in live}
    spent = lambda: sum(rank[r] * cost[r] for r in live)

    # Shed from the least sensitive region that can afford it.
    while spent() > budget:
        options = [r for r in live if rank[r] - rank_multiple >= rank_min]
        if not options:
            break
        rank[min(options, key=lambda r: share[r])] -= rank_multiple

    # Hand the leftover to the most sensitive region that can take it.
    while True:
        options = [r for r in live if rank[r] + rank_multiple <= rank_max
                   and spent() + rank_multiple * cost[r] <= budget]
        if not options:
            break
        rank[max(options, key=lambda r: (share[r], -rank[r]))] += rank_multiple

    by_region = {r: int(rank.get(r, 0)) for r in REGION_ORDER}
    by_module = {n: by_region[region_of[n]] for n in shapes if n in region_of}
    allocated = sum(lora_params(shapes[n], by_module[n]) for n in by_module)
    assert allocated <= budget, f"allocator overspent: {allocated} > {budget}"

    return {
        "rank_by_region": by_region,
        "rank_by_module": by_module,
        "sensitivity_share": {r: float(share.get(r, 0.0)) for r in REGION_ORDER},
        "budget": int(budget),
        "allocated_params": int(allocated),
        "budget_utilisation": float(allocated / budget) if budget else 0.0,
    }


def uniform_allocation(shapes, region_of, uniform_rank: int) -> dict:
    """The control arm: same modules, same budget, one rank everywhere."""
    budget = uniform_budget(shapes, uniform_rank)
    return {
        "rank_by_region": {r: uniform_rank for r in REGION_ORDER},
        "rank_by_module": {n: uniform_rank for n in shapes},
        "budget": budget, "allocated_params": budget, "budget_utilisation": 1.0,
    }
