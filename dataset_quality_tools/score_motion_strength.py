#!/usr/bin/env python3
"""Continuous motion-strength scoring, built on VBench's exact RAFT pipeline.

VBench's dynamic_degree only emits a per-clip 0/1 (dynamic if enough frame-pairs
exceed a resolution-scaled threshold). That binary hides *how far above the bar*
a clip sits — and clips that barely pass teach an I2V model to regress to static.

This scorer replays VBench's DynamicDegree.infer but captures the per-frame-pair
`max_rad` scores (mean of the top-5% optical-flow magnitudes), so we can rank by
motion margin, not just pass/fail. It uses the same raft-things weights and the
same threshold math, so the recomputed `dynamic` column matches VBench exactly
(and our earlier vbench_aes_dyn.csv) — a built-in consistency check.

CSV columns:
    video, motion_mean, motion_median, motion_max, thres,
    moving_frames, total_pairs, moving_ratio, count_num, dynamic
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

import torch
from easydict import EasyDict as edict

CACHE = Path(os.environ.get("VBENCH_CACHE_DIR", Path.home() / ".cache" / "vbench"))
RAFT_CKPT = CACHE / "raft_model" / "models" / "raft-things.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from vbench.dynamic_degree import DynamicDegree
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


def score_one(dyn: DynamicDegree, path: str) -> dict:
    """Replicate DynamicDegree.infer but keep the continuous per-pair scores."""
    with torch.no_grad():
        frames = dyn.get_frames(path)
        dyn.set_params(frame=frames[0], count=len(frames))
        thres = dyn.params["thres"]
        count_num = dyn.params["count_num"]
        scores: list[float] = []
        for img1, img2 in zip(frames[:-1], frames[1:]):
            padder = InputPadder(img1.shape)
            i1, i2 = padder.pad(img1, img2)
            _, flow_up = dyn.model(i1, i2, iters=20, test_mode=True)
            scores.append(dyn.get_score(i1, flow_up))
    total = len(scores)
    moving = sum(1 for s in scores if s > thres)
    return {
        "motion_mean": sum(scores) / total if total else 0.0,
        "motion_median": statistics.median(scores) if scores else 0.0,
        "motion_max": max(scores) if scores else 0.0,
        "thres": thres,
        "moving_frames": moving,
        "total_pairs": total,
        "moving_ratio": moving / total if total else 0.0,
        "count_num": count_num,
        "dynamic": 1 if moving >= count_num else 0,
    }


def discover(root: Path, exts: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lstrip(".").lower() in exts)


FIELDS = ["video", "motion_mean", "motion_median", "motion_max", "thres",
          "moving_frames", "total_pairs", "moving_ratio", "count_num", "dynamic"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous motion-strength scoring (VBench RAFT)")
    ap.add_argument("--input", "-i", required=True, type=Path)
    ap.add_argument("--output", "-o", required=True, type=Path)
    ap.add_argument("--ext", default="mp4,mov,avi,webm,mkv")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    exts = set(args.ext.lower().split(","))
    videos = discover(args.input, exts)
    print(f"Found {len(videos)} videos under {args.input}")
    if not videos:
        return 1

    done: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r["video"])
        print(f"Resuming: {len(done)} already scored")
    todo = [v for v in videos if v.relative_to(args.input).as_posix() not in done]
    print(f"To score this run: {len(todo)}")

    print(f"Loading RAFT (raft-things) from {RAFT_CKPT}...")
    raft_args = edict({"model": str(RAFT_CKPT), "small": False,
                       "mixed_precision": False, "alternate_corr": False})
    dyn = DynamicDegree(raft_args, DEVICE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fresh = not (args.resume and args.output.exists())
    f = args.output.open("w" if fresh else "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if fresh:
        w.writeheader(); f.flush()

    for i, v in enumerate(todo, 1):
        rel = v.relative_to(args.input).as_posix()
        try:
            r = score_one(dyn, str(v))
        except Exception as exc:
            print(f"  [{i}/{len(todo)}] {v.name} ERROR: {exc}", file=sys.stderr)
            r = {k: 0 for k in FIELDS if k != "video"}
        row = {"video": rel, "motion_mean": f"{r['motion_mean']:.3f}",
               "motion_median": f"{r['motion_median']:.3f}", "motion_max": f"{r['motion_max']:.3f}",
               "thres": f"{r['thres']:.2f}", "moving_frames": r["moving_frames"],
               "total_pairs": r["total_pairs"], "moving_ratio": f"{r['moving_ratio']:.3f}",
               "count_num": r["count_num"], "dynamic": r["dynamic"]}
        w.writerow(row); f.flush()
        print(f"  [{i}/{len(todo)}] {v.name}  motion_mean={row['motion_mean']} thres={row['thres']} "
              f"moving={r['moving_frames']}/{r['total_pairs']} dyn={r['dynamic']}")
    f.close()

    rows = list(csv.DictReader(args.output.open(encoding="utf-8")))
    n = len(rows)
    mm = sorted(float(r["motion_mean"]) for r in rows)
    dynamic = sum(int(r["dynamic"]) for r in rows)
    print("\n" + "=" * 50)
    print(f"clips: {n}")
    print(f"motion_mean: min={mm[0]:.2f} median={mm[n//2]:.2f} max={mm[-1]:.2f}")
    print(f"dynamic=1: {dynamic} ({dynamic/n:.1%})   dynamic_degree={dynamic/n:.3f}")
    print(f"CSV: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
