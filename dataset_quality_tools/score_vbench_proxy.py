#!/usr/bin/env python3
"""Tier B VBench-proxy scoring using CLIP (and the LAION aesthetic head).

For each clip we decode once and compute three CLIP-based proxies that mirror
VBench dimensions without downloading VBench's official model zoo:

    avg_aesthetic           -- LAION aesthetic predictor (VBench's exact head;
                               same weights file as score_videos_motion_aesthetic.py)
                               -> VBench "Aesthetic Quality"
    subject_consistency     -- mean cosine sim between the first frame's CLIP
                               feature and every sampled frame's feature
                               -> VBench "Subject Consistency" (VBench uses DINO;
                               CLIP is a cheap proxy)
    background_consistency  -- mean cosine sim between consecutive sampled
                               frames' CLIP features
                               -> VBench "Background Consistency" (VBench uses CLIP)

Output CSV columns:
    video,avg_aesthetic,subject_consistency,background_consistency

This reuses the model/decoding helpers from score_videos_motion_aesthetic.py so
there is a single source of truth for CLIP / aesthetic loading.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Reuse the existing aesthetic + CLIP loaders and frame decoder.
from score_videos_motion_aesthetic import (
    DEVICE,
    decode_frames,
    load_aesthetic_predictor,
    load_clip,
    sample_frame_indices,
)


def _clip_features(frames: list[np.ndarray], clip_model, preprocess) -> torch.Tensor:
    """Return L2-normalized CLIP image features for a list of HWC uint8 frames."""
    images = [preprocess(Image.fromarray(f)) for f in frames]
    batch = torch.stack(images).to(DEVICE)
    with torch.no_grad():
        feats = clip_model.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def score_clip(
    path: str,
    clip_model,
    preprocess,
    aes_mlp,
    num_frames: int = 8,
) -> tuple[float, float, float]:
    """Return (avg_aesthetic, subject_consistency, background_consistency)."""
    total = sum(1 for _ in decode_frames(path))
    if total == 0:
        return 0.0, 0.0, 0.0

    idx_set = set(sample_frame_indices(total, num_frames))
    frames: list[np.ndarray] = []
    for i, frame in enumerate(decode_frames(path)):
        if i in idx_set:
            frames.append(frame)
        if len(frames) == len(idx_set):
            break

    if not frames:
        return 0.0, 0.0, 0.0

    feats = _clip_features(frames, clip_model, preprocess)  # (N, D), L2-normed

    # Aesthetic: mean over sampled frames.
    with torch.no_grad():
        aes_scores = aes_mlp(feats).squeeze(-1).cpu().tolist()
    if isinstance(aes_scores, float):
        aes_scores = [aes_scores]
    avg_aesthetic = float(np.mean(aes_scores)) if aes_scores else 0.0

    # Subject consistency: first frame vs every frame (cosine = dot of normed).
    if feats.shape[0] >= 2:
        first = feats[0:1]
        subj = (feats @ first.T).squeeze(-1)  # (N,)
        subject_consistency = float(subj.mean().clamp(0.0, 1.0).item())

        # Background consistency: consecutive-frame cosine similarity.
        consec = (feats[1:] * feats[:-1]).sum(dim=-1)  # (N-1,)
        background_consistency = float(consec.mean().clamp(0.0, 1.0).item())
    else:
        subject_consistency = 1.0
        background_consistency = 1.0

    return avg_aesthetic, subject_consistency, background_consistency


def main() -> int:
    parser = argparse.ArgumentParser(description="VBench-proxy CLIP scoring for video clips")
    parser.add_argument("--input", "-i", required=True, help="Root directory containing clips")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--ext", default="mp4,mov,avi,webm,mkv", help="Video extensions")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames sampled per clip")
    parser.add_argument("--resume", action="store_true", help="Skip already-scored rows")
    args = parser.parse_args()

    root = Path(args.input)
    out_csv = Path(args.output)
    exts = set(args.ext.lower().split(","))

    videos = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lstrip(".").lower() in exts
    )
    print(f"Found {len(videos)} videos under {root}")

    done: set[str] = set()
    if args.resume and out_csv.exists():
        with out_csv.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["video"])
        print(f"Resuming: {len(done)} already scored")

    print("Loading CLIP ViT-L/14 and aesthetic predictor...")
    clip_model, preprocess = load_clip()
    aes_mlp = load_aesthetic_predictor()
    print(f"Using device: {DEVICE}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    append = args.resume and out_csv.exists()
    mode = "a" if append else "w"
    with out_csv.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(
                ["video", "avg_aesthetic", "subject_consistency", "background_consistency"]
            )

        for i, vp in enumerate(videos, 1):
            rel = vp.relative_to(root).as_posix()
            if rel in done:
                continue
            print(f"[{i}/{len(videos)}] {rel}")
            try:
                aes, subj, bg = score_clip(
                    str(vp), clip_model, preprocess, aes_mlp, args.num_frames
                )
                writer.writerow([rel, f"{aes:.3f}", f"{subj:.4f}", f"{bg:.4f}"])
                f.flush()
            except Exception as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                writer.writerow([rel, "0.000", "0.0000", "0.0000"])
                f.flush()

    print(f"Done. CSV written to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
