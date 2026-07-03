#!/usr/bin/env python3
"""Extra motion-scenery keywords to supplement the main batch.

Drop-in for run_vbench_dataset.py --keywords-module motion_scenery_keywords_extra.
Targets sum to ~50 selected clips to fill the gap toward 300 motion_ok.
"""
from __future__ import annotations

from vbench_keywords import Keyword

_CINEMATIC = "cinematic 4K"


def _kw(category: str, base: str, target: int = 5) -> Keyword:
    return Keyword(category=category, query=f"{base}, {_CINEMATIC}", target_count=target)


KEYWORDS: list[Keyword] = [
    # --- waves / ocean --------------------------------------------------------
    _kw("waves", "ocean waves washing onto a tropical beach at sunset", 5),
    _kw("waves", "big ocean swell breaking on a reef from above", 5),

    # --- river / stream -------------------------------------------------------
    _kw("river", "crystal clear river flowing over smooth stones", 5),
    _kw("river", "rapids flowing through a rocky canyon", 5),

    # --- waterfall ------------------------------------------------------------
    _kw("waterfall", "close up of a waterfall splashing on rocks", 5),

    # --- wind / land ------------------------------------------------------------
    _kw("wind", "sand blowing across desert dunes", 5),
    _kw("wind", "bamboo forest swaying in the wind", 5),

    # --- clouds / sky ---------------------------------------------------------
    _kw("clouds", "dramatic storm clouds moving across the sky", 5),
    _kw("clouds", "sun rays breaking through moving clouds over hills", 5),

    # --- snow / fog ------------------------------------------------------------
    _kw("snow_fog", "mist rising from a calm lake at dawn", 5),
    _kw("snow_fog", "heavy rain falling on a forest lake", 5),

    # --- other natural motion --------------------------------------------------
    _kw("other_motion", "volcanic steam rising from rocky terrain", 5),
    _kw("other_motion", "dust devils spinning across a dry plain", 5),
]


def categories() -> list[str]:
    seen: dict[str, None] = {}
    for kw in KEYWORDS:
        seen.setdefault(kw.category, None)
    return list(seen)


def total_target() -> int:
    return sum(kw.target_count for kw in KEYWORDS)


if __name__ == "__main__":
    print(f"{len(KEYWORDS)} queries across {len(categories())} categories")
    print(f"categories: {', '.join(categories())}")
    print(f"total target (pre-gate Top-K): {total_target()}")
