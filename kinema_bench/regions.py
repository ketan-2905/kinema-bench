"""The four functional regions a multimodal manipulation policy is cut into.

Rank does not need to be spread evenly across them: adapting to a new
embodiment stresses the action head and the projector far more than it
stresses a language backbone that already understands the instruction.
Membership is decided purely by parameter name, so the same code works on a
toy model and on a 7B checkpoint.
"""
from __future__ import annotations

import re

VISION = "vision_encoder"
PROJECTOR = "multimodal_projector"
LANGUAGE = "language_backbone"
ACTION = "action_head"

REGION_ORDER: tuple[str, ...] = (VISION, PROJECTOR, LANGUAGE, ACTION)

DEFAULT_PATTERNS: dict[str, list[str]] = {
    VISION:    [r"^vision_backbone\.", r"^vision_tower\.", r"\bvision_encoder\b"],
    PROJECTOR: [r"^projector\.", r"\bmulti_modal_projector\b", r"^proprio_projector\."],
    LANGUAGE:  [r"^language_model\.", r"^llm_backbone\.", r"^model\.layers\."],
    ACTION:    [r"^action_head\.", r"\bnoisy_action_projector\b", r"^lm_head\."],
}


class RegionMap:
    def __init__(self, patterns: dict[str, list[str]] | None = None) -> None:
        self.patterns = patterns or {k: list(v) for k, v in DEFAULT_PATTERNS.items()}
        self._compiled = {r: [re.compile(p) for p in ps] for r, ps in self.patterns.items()}

    def classify(self, param_name: str) -> str | None:
        """Which region does this parameter live in? None means 'not adapted'."""
        # ACTION is checked first so `lm_head` is not swallowed by LANGUAGE.
        for region in (ACTION, PROJECTOR, VISION, LANGUAGE):
            if any(p.search(param_name) for p in self._compiled.get(region, [])):
                return region
        return None


def is_lora_target(name: str, shape: tuple[int, ...], suffixes: tuple[str, ...]) -> bool:
    """LoRA attaches only to 2-D weights whose name ends in a target suffix."""
    if len(shape) != 2:
        return False
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    return any(stem.endswith(s) for s in suffixes)
