#!/usr/bin/env python3
"""Score videos for motion (mean_delta/max_delta) and aesthetics.

Output CSV columns match batch08_combined_scores.csv:
    video,mean_delta,max_delta,avg_aesthetic,frame_aesthetics

Aesthetic model: LAION aesthetic predictor (CLIP ViT-L/14 + linear MSE head).
Motion: mean/max of L1-normalized per-pixel frame differences.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

# CLIP / aesthetic predictor imports
import clip


AESTHETIC_WEIGHTS = Path("/workspace/sac+logos+ava1-l14-linearMSE.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_aesthetic_predictor():
    """Load the LAION aesthetic predictor MLP on top of CLIP L14.

    The checkpoint stores linear layers under keys like layers.0.weight,
    layers.2.weight, ..., layers.7.weight, with ReLU modules at the odd
    indices that have no parameters. We reconstruct the exact Sequential.
    """
    state = torch.load(AESTHETIC_WEIGHTS, map_location="cpu", weights_only=True)

    # Collect (checkpoint_index, in_dim, out_dim) for each linear layer
    linear_specs = []
    for k, v in state.items():
        if "weight" in k and v.ndim == 2:
            idx = int(k.split(".")[1])
            linear_specs.append((idx, v.shape[1], v.shape[0]))
    linear_specs.sort(key=lambda x: x[0])

    # Build an OrderedDict that places modules at the exact indices used by
    # the checkpoint.  ReLUs fill odd parameter-less slots between linears.
    from collections import OrderedDict
    modules: OrderedDict[str, torch.nn.Module] = OrderedDict()
    next_idx = 0
    for lin_idx, in_dim, out_dim in linear_specs:
        # Fill any gap before this linear with ReLUs (one per missing index)
        while next_idx < lin_idx:
            modules[str(next_idx)] = torch.nn.ReLU()
            next_idx += 1
        modules[str(lin_idx)] = torch.nn.Linear(in_dim, out_dim)
        next_idx = lin_idx + 1

    mlp = torch.nn.Sequential(modules).to(DEVICE).eval()
    # Checkpoint keys are prefixed with "layers."; strip it to match Sequential keys.
    new_state = {k.replace("layers.", ""): v for k, v in state.items()}
    mlp.load_state_dict(new_state, strict=True)
    return mlp


def load_clip():
    model, preprocess = clip.load("ViT-L/14", device=DEVICE)
    model.eval()
    return model, preprocess


def decode_frames(path: str, max_frames: int | None = None):
    """Decode RGB frames using PyAV, yielding numpy arrays HWC uint8."""
    container = av.open(path)
    stream = container.streams.video[0]
    count = 0
    for frame in container.decode(stream):
        yield frame.to_ndarray(format="rgb24")
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    container.close()


def compute_motion(path: str):
    """Return mean_delta and max_delta across consecutive frames."""
    prev: np.ndarray | None = None
    diffs = []
    for frame in decode_frames(path):
        # frame is uint8 HWC; normalize to [0,1]
        f = frame.astype(np.float32) / 255.0
        if prev is not None:
            diff = np.mean(np.abs(f - prev))
            diffs.append(float(diff))
        prev = f
    if not diffs:
        return 0.0, 0.0
    return float(np.mean(diffs)), float(np.max(diffs))


def sample_frame_indices(total: int, n: int = 3):
    """Return n evenly-spaced frame indices (0-based, inclusive edges)."""
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    # linspace from 0 to total-1, round to nearest int
    idx = np.linspace(0, total - 1, n).round().astype(int)
    # deduplicate while preserving order
    seen = set()
    out = []
    for i in idx:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def score_aesthetics(path: str, clip_model, preprocess, aes_mlp, num_frames: int = 3):
    """Sample num_frames evenly, predict aesthetic score for each, return list and mean."""
    # First pass: count frames
    total = sum(1 for _ in decode_frames(path))
    if total == 0:
        return [0.0] * num_frames, 0.0

    idx_set = set(sample_frame_indices(total, num_frames))
    frames = []
    for i, frame in enumerate(decode_frames(path)):
        if i in idx_set:
            frames.append(frame)
        if len(frames) == len(idx_set):
            break

    if not frames:
        return [0.0] * num_frames, 0.0

    images = [preprocess(Image.fromarray(f)) for f in frames]
    batch = torch.stack(images).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(batch)
        image_features = image_features.float()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        scores = aes_mlp(image_features).squeeze(-1).cpu().tolist()

    # If we got fewer frames than requested, pad with last score
    while len(scores) < num_frames:
        scores.append(scores[-1] if scores else 0.0)
    mean_score = float(np.mean(scores[:num_frames]))
    return scores[:num_frames], mean_score


def main():
    parser = argparse.ArgumentParser(description="Score videos for motion and aesthetics")
    parser.add_argument("--input", "-i", required=True, help="Root directory containing videos")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--ext", default="mp4,mov,avi,webm,mkv", help="Video extensions")
    parser.add_argument("--workers", "-j", type=int, default=1, help="Not used; single-process GPU")
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

    # Resume support
    done = set()
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
            writer.writerow(["video", "mean_delta", "max_delta", "avg_aesthetic", "frame_aesthetics"])

        for i, vp in enumerate(videos, 1):
            rel = vp.relative_to(root).as_posix()
            if rel in done:
                continue
            print(f"[{i}/{len(videos)}] {rel}")
            try:
                mean_d, max_d = compute_motion(str(vp))
                aes_scores, avg_aes = score_aesthetics(str(vp), clip_model, preprocess, aes_mlp)
                aes_str = ";".join(f"{s:.3f}" for s in aes_scores)
                writer.writerow([rel, f"{mean_d:.6f}", f"{max_d:.6f}", f"{avg_aes:.3f}", aes_str])
                f.flush()
            except Exception as exc:
                print(f"  ERROR: {exc}", file=sys.stderr)
                writer.writerow([rel, "0.000000", "0.000000", "0.000", "0.000;0.000;0.000"])
                f.flush()

    print(f"Done. CSV written to {out_csv}")


if __name__ == "__main__":
    main()
