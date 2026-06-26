#!/usr/bin/env python3
"""VBench-aligned Pexels keyword matrix.

The goal is a *broad-coverage* training dataset whose content distribution
mirrors VBench's prompt taxonomy, so a Wan2.2 LoRA trained on it generalizes
across VBench's semantic dimensions (object class, scene, color, human action,
spatial relationship, appearance/temporal style).

VBench prompts roughly span these content categories; we cover each with a few
queries crossed with cinematic / lighting / motion descriptors that bias Pexels
toward clips with smooth, real motion (good Dynamic Degree) while staying
visually clean (good Aesthetic / Imaging / Consistency).

Each entry is a ``Keyword``:
    category      -- VBench-style content bucket (used for Top-K grouping)
    query         -- the Pexels search string
    target_count  -- how many finally-selected PASS clips we want for this query

Tune ``target_count`` per query to shape the dataset's category balance.  The
driver (run_vbench_dataset.py) over-downloads (a multiple of target_count),
quality-filters, VBench-proxy-scores, then keeps the Top-K by composite score.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Keyword:
    category: str
    query: str
    target_count: int = 30


# Shared cinematic descriptors appended to bias toward clean, well-shot footage.
# Kept short so Pexels search still returns enough candidates.
_CINEMATIC = "cinematic 4K"


def _kw(category: str, base: str, target: int = 30) -> Keyword:
    return Keyword(category=category, query=f"{base}, {_CINEMATIC}", target_count=target)


# ---------------------------------------------------------------------------
# The matrix.  ~8 categories x a few queries each.  Edit freely to rebalance.
# ---------------------------------------------------------------------------

KEYWORDS: list[Keyword] = [
    # --- animals -----------------------------------------------------------
    _kw("animals", "slow motion wild horses running across a grassland"),
    _kw("animals", "close up of a bird flying over the ocean at sunrise"),
    _kw("animals", "underwater shot of a sea turtle swimming in clear blue water"),

    # --- architecture ------------------------------------------------------
    _kw("architecture", "slow pan across a modern city skyline at blue hour"),
    _kw("architecture", "timelapse-free aerial orbit of a historic cathedral"),
    _kw("architecture", "tracking shot through a narrow old european street"),

    # --- food --------------------------------------------------------------
    _kw("food", "close up of coffee being poured into a cup, slow motion"),
    _kw("food", "chef plating a gourmet dish in a restaurant kitchen"),
    _kw("food", "fresh fruit splashing into water in slow motion"),

    # --- human (action) ----------------------------------------------------
    _kw("human", "a person running on a beach at golden hour, tracking shot"),
    _kw("human", "a woman dancing gracefully in a sunlit studio"),
    _kw("human", "a man riding a bicycle through a city street, smooth gimbal"),

    # --- lifestyle ---------------------------------------------------------
    _kw("lifestyle", "people walking in a busy market, smooth handheld"),
    _kw("lifestyle", "friends laughing around a campfire at night"),
    _kw("lifestyle", "barista making latte art in a cozy cafe"),

    # --- plant -------------------------------------------------------------
    _kw("plant", "close up of flowers swaying gently in the wind"),
    _kw("plant", "sunlight rays through a green forest canopy, slow dolly"),
    _kw("plant", "macro shot of dew drops on green leaves"),

    # --- scenery -----------------------------------------------------------
    _kw("scenery", "drone aerial view of mountain peaks at golden hour sunrise"),
    _kw("scenery", "ocean waves crashing on rocks at sunset, slow motion"),
    _kw("scenery", "drone aerial shot of a winding river through autumn forest"),

    # --- vehicles ----------------------------------------------------------
    _kw("vehicles", "sports car driving on a coastal road, tracking shot"),
    _kw("vehicles", "sailboat gliding across calm sea at sunset"),
    _kw("vehicles", "train moving through a green countryside, side tracking"),
]


def categories() -> list[str]:
    """Distinct categories, preserving first-seen order."""
    seen: dict[str, None] = {}
    for kw in KEYWORDS:
        seen.setdefault(kw.category, None)
    return list(seen)


def total_target() -> int:
    return sum(kw.target_count for kw in KEYWORDS)


if __name__ == "__main__":
    print(f"{len(KEYWORDS)} queries across {len(categories())} categories")
    print(f"categories: {', '.join(categories())}")
    print(f"total target PASS clips: {total_target()}")
