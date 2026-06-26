#!/usr/bin/env python3
"""Compose a VBench composite score per clip and select the Top-K.

Joins two CSVs produced earlier in the pipeline:

  * clean_dataset.py  -> quality_report.csv   (pixel-level metrics + grade)
  * score_vbench_proxy.py -> vbench_proxy.csv  (CLIP aesthetic + consistencies)

then normalizes each VBench-proxy metric to 0..1 against fixed reference ranges
(so scores are comparable across runs/keywords), computes a weighted composite,
applies hard floors that protect consistency, and keeps the Top-K clips.

Outputs:
  * vbench_scores.csv      -- every joined clip with per-dimension + composite
  * selected_manifest.jsonl-- Top-K kept clips ({id, video, prompt})

Dimension -> proxy mapping (see plan):
  aesthetic              avg_aesthetic               (higher better)
  imaging                sharpness_grad_mean + clip  (higher better)
  background_consistency background_consistency      (higher better)
  subject_consistency    subject_consistency         (higher better)
  dynamic                temporal_diff_mean          (BAND: not static, not chaotic)
  motion_smoothness      temporal_diff_std/mean + cuts(higher better = smoother)
  flicker                flicker_luma_std            (lower better)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


# --- default weights (sum need not be 1; we normalize at the end) ----------
DEFAULT_WEIGHTS: dict[str, float] = {
    "aesthetic": 0.20,
    "imaging": 0.15,
    "background_consistency": 0.15,
    "subject_consistency": 0.15,
    "dynamic": 0.15,
    "motion_smoothness": 0.10,
    "flicker": 0.10,
}

# --- fixed reference ranges for min-max normalization ----------------------
# Derived from README "ideal range" guidance; clamp then scale to 0..1.
AES_LO, AES_HI = 4.0, 7.0            # LAION aesthetic typical band
SHARP_LO, SHARP_HI = 0.01, 0.08      # sharpness_grad_mean
BG_LO, BG_HI = 0.70, 0.98            # CLIP consecutive-frame cosine
SUBJ_LO, SUBJ_HI = 0.65, 0.98        # CLIP first-vs-frame cosine
FLICK_LO, FLICK_HI = 0.0, 0.05       # flicker_luma_std (lower better)

# Motion smoothness: temporal_diff_std/mean ratio (lower = smoother).
# Calibrated to the observed ratio distribution on real Pexels clips
# (~1.05 min, 1.44 median, 2.34 max); lower ratio scores higher.
SMOOTH_LO, SMOOTH_HI = 1.0, 2.3

# Dynamic Degree band on temporal_diff_mean.
DYN_LOW = 0.010                      # below = too static
DYN_HIGH = 0.120                     # above = too chaotic
DYN_PLATEAU_LO = 0.025               # full score band start
DYN_PLATEAU_HI = 0.090               # full score band end


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _norm(x: float, lo: float, hi: float, invert: bool = False) -> float:
    if hi <= lo:
        return 0.0
    v = _clamp01((x - lo) / (hi - lo))
    return 1.0 - v if invert else v


def _dynamic_score(mean_delta: float) -> float:
    """Band-pass score: 0 at/below DYN_LOW and at/above DYN_HIGH, 1 on plateau."""
    if mean_delta <= DYN_LOW or mean_delta >= DYN_HIGH:
        return 0.0
    if mean_delta < DYN_PLATEAU_LO:
        return _clamp01((mean_delta - DYN_LOW) / (DYN_PLATEAU_LO - DYN_LOW))
    if mean_delta > DYN_PLATEAU_HI:
        return _clamp01((DYN_HIGH - mean_delta) / (DYN_HIGH - DYN_PLATEAU_HI))
    return 1.0


def _read_quality(path: Path) -> dict[str, dict[str, str]]:
    """quality_report.csv is keyed by the 'path' column."""
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[Path(row["path"]).name] = row
    return rows


def _read_proxy(path: Path) -> dict[str, dict[str, str]]:
    """vbench_proxy.csv is keyed by the 'video' column (relative path)."""
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[Path(row["video"]).name] = row
    return rows


def compose_row(q: dict[str, str], p: dict[str, str], weights: dict[str, float]) -> dict[str, Any]:
    aes = float(p.get("avg_aesthetic", 0) or 0)
    subj = float(p.get("subject_consistency", 0) or 0)
    bg = float(p.get("background_consistency", 0) or 0)

    mean_delta = float(q.get("temporal_diff_mean", 0) or 0)
    std_delta = float(q.get("temporal_diff_std", 0) or 0)
    sharp = float(q.get("sharpness_grad_mean", 0) or 0)
    flicker = float(q.get("flicker_luma_std", 0) or 0)
    black = float(q.get("black_pixel_ratio", 0) or 0)
    white = float(q.get("white_pixel_ratio", 0) or 0)
    scene_cuts = int(float(q.get("scene_cut_count", 0) or 0))

    dims: dict[str, float] = {}
    dims["aesthetic"] = _norm(aes, AES_LO, AES_HI)
    # imaging: sharpness up, clipping down.
    clipping_penalty = _clamp01((black + white) / 0.15)
    dims["imaging"] = _clamp01(_norm(sharp, SHARP_LO, SHARP_HI) * (1.0 - clipping_penalty))
    dims["background_consistency"] = _norm(bg, BG_LO, BG_HI)
    dims["subject_consistency"] = _norm(subj, SUBJ_LO, SUBJ_HI)
    dims["dynamic"] = _dynamic_score(mean_delta)
    # smoothness: lower std/mean ratio = smoother; no scene cuts.
    ratio = std_delta / mean_delta if mean_delta > 1e-6 else 1.0
    smooth = _norm(ratio, SMOOTH_LO, SMOOTH_HI, invert=True)
    if scene_cuts > 0:
        smooth = 0.0
    dims["motion_smoothness"] = smooth
    dims["flicker"] = _norm(flicker, FLICK_LO, FLICK_HI, invert=True)

    wsum = sum(weights.values()) or 1.0
    composite = sum(dims[k] * weights[k] for k in weights) / wsum

    return {
        "dims": dims,
        "composite": composite,
        "subj_raw": subj,
        "bg_raw": bg,
        "scene_cuts": scene_cuts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose VBench composite scores and select Top-K")
    parser.add_argument("--quality-csv", required=True, type=Path, help="clean_dataset quality_report.csv")
    parser.add_argument("--proxy-csv", required=True, type=Path, help="score_vbench_proxy.py CSV")
    parser.add_argument("--clips-rel", default="clips", help="Manifest 'video' prefix dir (default: clips)")
    parser.add_argument("--prompt", default="", help="Prompt to write into the selected manifest")
    parser.add_argument("--top-k", type=int, default=None, help="Keep only the Top-K by composite (default: all passing)")
    parser.add_argument("--min-subject", type=float, default=SUBJ_LO, help="Floor: drop clips below this subject_consistency")
    parser.add_argument("--min-background", type=float, default=BG_LO, help="Floor: drop clips below this background_consistency")
    parser.add_argument("--scores-out", type=Path, default=None, help="Output vbench_scores.csv")
    parser.add_argument("--manifest-out", type=Path, default=None, help="Output selected_manifest.jsonl")
    args = parser.parse_args()

    quality = _read_quality(args.quality_csv)
    proxy = _read_proxy(args.proxy_csv)

    joined_keys = sorted(set(quality) & set(proxy))
    print(f"quality rows={len(quality)} proxy rows={len(proxy)} joined={len(joined_keys)}")

    scored: list[dict[str, Any]] = []
    for key in joined_keys:
        q = quality[key]
        # Only consider clips that survived Tier A.
        if q.get("grade") == "FAIL":
            continue
        p = proxy[key]
        res = compose_row(q, p, DEFAULT_WEIGHTS)
        # Hard floors protecting consistency.
        if res["subj_raw"] < args.min_subject or res["bg_raw"] < args.min_background:
            continue
        if res["scene_cuts"] > 0:
            continue
        scored.append({"key": key, **res})

    scored.sort(key=lambda r: r["composite"], reverse=True)
    print(f"passing floors: {len(scored)}")

    selected = scored[: args.top_k] if args.top_k else scored
    print(f"selected: {len(selected)}")

    # vbench_scores.csv (all joined, ranked).
    scores_out = args.scores_out or args.quality_csv.parent / "vbench_scores.csv"
    dim_names = list(DEFAULT_WEIGHTS.keys())
    with scores_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "composite", *dim_names, "selected"])
        selected_keys = {r["key"] for r in selected}
        for r in scored:
            writer.writerow(
                [r["key"], f"{r['composite']:.4f}"]
                + [f"{r['dims'][d]:.4f}" for d in dim_names]
                + [int(r["key"] in selected_keys)]
            )
    print(f"scores written: {scores_out}")

    # selected_manifest.jsonl
    manifest_out = args.manifest_out or args.quality_csv.parent / "selected_manifest.jsonl"
    with manifest_out.open("w", encoding="utf-8") as f:
        for r in selected:
            name = r["key"]
            stem = Path(name).stem
            record = {
                "id": stem,
                "video": f"{args.clips_rel}/{name}",
                "prompt": args.prompt or f"a cinematic video, {stem}",
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"manifest written: {manifest_out} ({len(selected)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
