#!/usr/bin/env python3
"""Exact-replica VBench scoring for aesthetic_quality + dynamic_degree.

Calls VBench's OWN implementations (vbench 0.1.5, installed --no-deps) with the
exact weights VBench uses, so the numbers are directly comparable to what
nexisgen's validator (rendixnetwork/vbench:latest) produces:

  aesthetic_quality : CLIP ViT-L/14 -> normalize -> Linear(768,1) -> /10,
                      averaged over frames  (vbench.aesthetic_quality.laion_aesthetic
                      + sa_0_4_vit_l_14_linear.pth)
  dynamic_degree    : RAFT (raft-things.pth) optical flow, per-clip binary
                      dynamic/static; the dataset aggregate is the fraction of
                      clips judged dynamic  (vbench.dynamic_degree.DynamicDegree)

Per-clip CSV columns:
    video,aesthetic_quality,dynamic
where `dynamic` is 1.0 (moving) or 0.0 (static).  The printed dynamic_degree is
mean(dynamic) over the directory — the VBench set-level number.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import clip
import torch
from easydict import EasyDict as edict

CACHE = Path(os.environ.get("VBENCH_CACHE_DIR", Path.home() / ".cache" / "vbench"))
AES_DIR = CACHE / "aesthetic_model" / "emb_reader"          # holds sa_0_4_vit_l_14_linear.pth
RAFT_CKPT = CACHE / "raft_model" / "models" / "raft-things.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from vbench.aesthetic_quality import get_aesthetic_model, laion_aesthetic
from vbench.dynamic_degree import DynamicDegree


def discover(root: Path, exts: set[str]) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lstrip(".").lower() in exts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Exact VBench aesthetic_quality + dynamic_degree")
    ap.add_argument("--input", "-i", required=True, type=Path, help="clips root dir")
    ap.add_argument("--output", "-o", required=True, type=Path, help="output CSV")
    ap.add_argument("--ext", default="mp4,mov,avi,webm,mkv")
    ap.add_argument("--resume", action="store_true", help="skip clips already in the output CSV")
    args = ap.parse_args()

    exts = set(args.ext.lower().split(","))
    videos = discover(args.input, exts)
    print(f"Found {len(videos)} videos under {args.input}")
    if not videos:
        return 1

    # Resume: skip clips already scored in the output CSV.
    done: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r["video"])
        print(f"Resuming: {len(done)} already scored")
    todo = [v for v in videos if v.relative_to(args.input).as_posix() not in done]
    print(f"To score this run: {len(todo)}")

    # --- aesthetic_quality (exact VBench), only for the not-yet-done clips ---
    print("Loading CLIP ViT-L/14 + VBench aesthetic linear head...")
    aes_model = get_aesthetic_model(str(AES_DIR)).to(DEVICE)
    clip_model, _ = clip.load("ViT-L/14", device=DEVICE)
    aes_map: dict[str, float] = {}
    if todo:
        _, aes_per_video = laion_aesthetic(aes_model, clip_model, [str(v) for v in todo], DEVICE)
        aes_map = {Path(d["video_path"]).name: float(d["video_results"]) for d in aes_per_video}

    # --- dynamic_degree (exact VBench RAFT) ---
    print(f"Loading RAFT (raft-things) from {RAFT_CKPT}...")
    raft_args = edict({"model": str(RAFT_CKPT), "small": False,
                       "mixed_precision": False, "alternate_corr": False})
    dyn = DynamicDegree(raft_args, DEVICE)

    # Open CSV for incremental append (write header only when starting fresh).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fresh = not (args.resume and args.output.exists())
    f = args.output.open("w" if fresh else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=["video", "aesthetic_quality", "dynamic"])
    if fresh:
        writer.writeheader()
        f.flush()

    for i, v in enumerate(todo, 1):
        name = v.name
        try:
            moving = float(dyn.infer(str(v)))
        except Exception as exc:
            print(f"  [{i}/{len(todo)}] {name} dynamic ERROR: {exc}", file=sys.stderr)
            moving = 0.0
        aes = aes_map.get(name, 0.0)
        writer.writerow({"video": v.relative_to(args.input).as_posix(),
                         "aesthetic_quality": f"{aes:.4f}", "dynamic": f"{moving:.0f}"})
        f.flush()
        print(f"  [{i}/{len(todo)}] {name}  aesthetic={aes:.4f}  dynamic={moving:.0f}")
    f.close()

    # Final summary over the WHOLE output CSV (resumed + this run).
    all_rows = list(csv.DictReader(args.output.open(encoding="utf-8")))
    n = len(all_rows)
    if n == 0:
        print("no rows scored")
        return 1
    mean_aes = sum(float(r["aesthetic_quality"]) for r in all_rows) / n
    dyn_degree = sum(float(r["dynamic"]) for r in all_rows) / n
    print("\n" + "=" * 50)
    print(f"clips: {n}")
    print(f"aesthetic_quality  mean = {mean_aes:.4f}   (VBench scale, /10)")
    print(f"dynamic_degree (fraction moving) = {dyn_degree:.4f}")
    print(f"CSV: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
