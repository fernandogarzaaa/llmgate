"""Adapter for the costroute router (https://github.com/fernandogarzaaa/costroute).

When the ``costroute`` package is installed, its ``load_router`` is used, so
every router it ships works (``heuristic``, ``lora`` with the trained LoRA
adapter, ``auto``, baselines). When it is not installed, llmgate falls back to
a vendored copy of costroute's dependency-free **heuristic** router so the
``auto`` alias keeps working; asking for ``lora`` without the package fails
fast with an install hint.
"""

from __future__ import annotations

import logging
import math
import re
from typing import List, Sequence, Tuple

log = logging.getLogger("llmgate.costroute")

# Vendored from costroute.routers.HeuristicRouter (same author, Apache-2.0), kept in sync by
# tests/unit/test_costroute_adapter.py when costroute is installed.
_HARD = re.compile(
    r"\b(code|function|python|javascript|java|c\+\+|sql|regex|algorithm|implement|debug|"
    r"prove|proof|derive|equation|integral|calculate|solve|math|step[- ]by[- ]step|"
    r"analy[sz]e|compare|design|architecture|optimi[sz]e|explain why)\b",
    re.IGNORECASE,
)


class VendoredHeuristicRouter:
    """score = 0.5*length + 0.35*keywords + 0.15*code; see costroute's README."""

    name = "heuristic(vendored)"

    def score(self, prompts: Sequence[str]) -> List[float]:
        out = []
        for p in prompts:
            n_words = len(p.split())
            length_part = min(math.log1p(n_words) / math.log1p(400), 1.0)
            kw = min(len(_HARD.findall(p)), 3) / 3.0
            code = 1.0 if "```" in p or re.search(r"[{};]\s*\n", p) else 0.0
            out.append(0.5 * length_part + 0.35 * kw + 0.15 * code)
        return out


def costroute_available() -> bool:
    try:
        import costroute.routers  # noqa: F401
    except ImportError:
        return False
    return True


def load_scorer(spec: str = "heuristic") -> Tuple[object, str]:
    """Return ``(scorer, source)`` where source is ``"costroute"`` or ``"vendored"``."""
    try:
        from costroute.routers import load_router
    except ImportError:
        if spec in ("heuristic", "auto"):
            log.warning("costroute is not installed; using the vendored heuristic router")
            return VendoredHeuristicRouter(), "vendored"
        raise ImportError(
            f"router {spec!r} needs the costroute package: "
            'pip install "costroute @ git+https://github.com/fernandogarzaaa/costroute.git"'
        ) from None
    return load_router(spec), "costroute"
