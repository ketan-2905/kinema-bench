"""Automated failure-mode attribution for manipulation rollouts.

A success rate tells you a policy got worse under 4-bit. It does not tell you
HOW, and the how is the whole diagnostic value. "NF4 costs 6 points of success
rate" is a fact; "NF4 costs 6 points, 80% of it zero-action collapse on
long-horizon tasks" is something a person can act on.

Three mutually exclusive labels, checked in order of how cheap they are to
rule out.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SUCCESS = "SUCCESS"
ZERO_ACTION_COLLAPSE = "ZERO_ACTION_COLLAPSE"
INSTRUCTION_SWAP_GROUNDING_FAILURE = "INSTRUCTION_SWAP_GROUNDING_FAILURE"
SPATIAL_MISS = "SPATIAL_MISS"
UNATTRIBUTED = "UNATTRIBUTED"

LABELS = (SUCCESS, ZERO_ACTION_COLLAPSE, INSTRUCTION_SWAP_GROUNDING_FAILURE,
          SPATIAL_MISS, UNATTRIBUTED)


@dataclass
class Thresholds:
    zero_action_rms: float = 0.02          # below this the arm is effectively still
    instruction_sensitivity: float = 0.05  # trajectory delta under a swapped instruction
    spatial_tolerance: float = 1.5


@dataclass
class Attribution:
    label: str
    action_rms: float
    instruction_delta: float | None
    final_distance: float
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


def attribute(rollout: dict, *, swapped_trajectory=None, thresholds: Thresholds | None = None):
    """Label one failed rollout.

    ZERO_ACTION_COLLAPSE
        The policy stopped committing; the arm barely moves. Typical of
        aggressive quantisation shrinking the action head's output
        distribution toward its mean.

    INSTRUCTION_SWAP_GROUNDING_FAILURE
        The policy acted confidently on the wrong instruction. Detected by
        replaying the SAME observations under a different task's instruction:
        if the trajectory barely changes, the model is not reading the
        language at all, it is pattern-matching the scene.

    SPATIAL_MISS
        Grounded the instruction, moved decisively, still ended up elsewhere.
        Residual category once the two above are ruled out.
    """
    th = thresholds or Thresholds()
    traj = np.asarray(rollout["trajectory"], dtype=np.float64)
    rms = float(np.sqrt((traj ** 2).mean()))
    dist = float(rollout["final_distance"])

    if rollout["success"]:
        return Attribution(SUCCESS, rms, None, dist, "task criterion met")

    if rms < th.zero_action_rms:
        return Attribution(ZERO_ACTION_COLLAPSE, rms, None, dist,
                           f"action RMS {rms:.4f} below floor {th.zero_action_rms}")

    delta = None
    if swapped_trajectory is not None:
        sw = np.asarray(swapped_trajectory, dtype=np.float64)
        n = min(len(sw), len(traj))
        delta = float(np.abs(traj[:n] - sw[:n]).mean()
                      / max(float(np.abs(traj[:n]).mean()), 1e-8))
        if delta < th.instruction_sensitivity:
            return Attribution(INSTRUCTION_SWAP_GROUNDING_FAILURE, rms, delta, dist,
                               f"trajectory moved only {delta:.3%} under a swapped instruction")

    if dist >= th.spatial_tolerance:
        return Attribution(SPATIAL_MISS, rms, delta, dist,
                           f"grounded and moved decisively, ended {dist:.3f} away")

    return Attribution(UNATTRIBUTED, rms, delta, dist,
                       "failed the task criterion but matched no failure signature")


def summarise(attributions: list[Attribution]) -> dict:
    counts = {label: 0 for label in LABELS}
    for a in attributions:
        counts[a.label] = counts.get(a.label, 0) + 1
    total = max(len(attributions), 1)
    failures = total - counts[SUCCESS]
    return {
        "counts": counts,
        "n_episodes": len(attributions),
        "success_rate": counts[SUCCESS] / total,
        # Shares are over FAILURES, not over all episodes.
        "failure_share": {k: (v / failures if failures else 0.0)
                          for k, v in counts.items() if k != SUCCESS},
    }
