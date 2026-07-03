#!/usr/bin/env python3
"""Motion-scenery Pexels keyword matrix (drop-in for run_vbench_dataset.py).

Goal: a 'natural scenery WITH real motion' dataset for a Wan2.2 TI2V LoRA whose
biggest weakness is collapsed dynamic_degree. Every query is biased toward
*continuous physical motion* (water / wind / clouds / falling) and/or smooth
camera motion, while explicitly avoiding 'timelapse' (which the Tier-A
mean_delta band and the final RAFT gate would reject as fake/over motion).

Categories
----------
waves / river / waterfall / wind / clouds / snow_fog / other_motion
    -> the motion backbone (~80% of the pool)
geo_wonders
    -> iconic geological wonders & unique landforms, but written with forced
       motion (drifting clouds, crashing waves, sweeping wind, drone orbit) and
       still passed through the same RAFT gate, so static postcard shots drop
       out automatically. ~15-20% of the pool; accept a lower yield as the price
       of aesthetic + content diversity.

Selection downstream (run_vbench_dataset.py) over-downloads, Tier-A quality- and
mean_delta-band-filters, Tier-B CLIP-proxy scores, then keeps Top-K. The exact
RAFT motion gate (motion_mean in [33,80]) + aesthetic >=0.58 runs LAST via
motion_aesthetic_gate.py over the merged selected clips.

Tune target_count to retune yield. With ~45-55% gate yield on motion clips and
~35% on wonders, the targets below aim at ~100 net RAFT-passed clips.
"""

from __future__ import annotations

from vbench_keywords import Keyword  # reuse the same dataclass the driver expects

_CINEMATIC = "cinematic 4K"


def _kw(category: str, base: str, target: int = 10) -> Keyword:
    return Keyword(category=category, query=f"{base}, {_CINEMATIC}", target_count=target)


# Motion-backbone target per query and wonders target per query.
# Scaled up 2026-07-01 to net ~300 motion-OK clips after the final low-motion
# filter (observed ok yield ~34% from 198 pre-gate selected -> 68 ok).
_M = 45   # motion categories: ~45-55% gate yield
_W = 54   # wonders: lower (~35%) yield, so request a few more pre-gate

KEYWORDS: list[Keyword] = [
    # --- waves / ocean -----------------------------------------------------
    _kw("waves", "ocean waves crashing on a rocky shore, slow motion", _M),
    _kw("waves", "aerial view of turquoise sea waves rolling onto a sandy beach", _M),

    # --- river / stream ----------------------------------------------------
    _kw("river", "river flowing through a green forest canyon", _M),
    _kw("river", "mountain stream cascading over mossy rocks, slow motion", _M),

    # --- waterfall ---------------------------------------------------------
    _kw("waterfall", "powerful waterfall in a tropical rainforest", _M),
    _kw("waterfall", "wide waterfall with drifting mist, slow motion", _M),

    # --- wind --------------------------------------------------------------
    _kw("wind", "wind blowing through a golden wheat field at sunset", _M),
    _kw("wind", "tall grass swaying in the wind on a green hillside", _M),
    _kw("wind", "palm trees swaying in strong wind before a storm", _M),

    # --- clouds (NOT timelapse) -------------------------------------------
    _kw("clouds", "clouds drifting fast over a mountain ridge", _M),
    _kw("clouds", "low clouds rolling through a forested valley at dawn", _M),

    # --- snow / fog --------------------------------------------------------
    _kw("snow_fog", "snow falling gently in a pine forest", _M),
    _kw("snow_fog", "fog rolling over a green valley at sunrise", _M),

    # --- other natural motion ---------------------------------------------
    _kw("other_motion", "autumn leaves falling in a forest", _M),
    _kw("other_motion", "steam rising from a geyser hot spring", _M),

    # --- geological wonders / unique landforms (motion-biased) ------------
    _kw("geo_wonders", "clouds drifting over the grand canyon, drone orbit", _W),
    _kw("geo_wonders", "waves crashing on an iceland black sand beach, slow motion", _W),
    _kw("geo_wonders", "wind sweeping across uyuni salt flat reflections", _W),
    _kw("geo_wonders", "mist drifting through zhangjiajie stone pillars", _W),
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
