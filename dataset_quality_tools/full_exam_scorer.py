#!/usr/bin/env python3
"""Full 'physical-exam' scorer for a bucket of clips.

Produces ONE row per clip combining every dimension we use to judge dataset
quality, so a bucket (medium / strong) can be reviewed from a single CSV.

Columns
-------
video
motion_mean_RAFT   : VBench RAFT continuous motion margin   (motion_strength.csv)
motion_median_RAFT : "
motion_max_RAFT     : "
moving_ratio        : fraction of frame-pairs above VBench thres
aesthetic_quality   : exact VBench LAION aesthetic (/10)     (vbench_aes_dyn.csv)
mean_delta          : mean |frame[t+1]-frame[t]| on [0,1]  (cheap pixel proxy)
p95_delta           : 95th-pctile of per-pair mean abs delta
subject_consistency : VBench DINO vitb16 cosine sim (higher=more static subject)
background_consistency : VBench CLIP ViT-B/32 cosine sim (higher=more static bg)
temporal_flickering : VBench MAE-based, (255-meanMAE)/255 (higher=less flicker)
sharpness           : mean Sobel-gradient magnitude on luma (higher=sharper)
flicker_luma_std    : std of per-frame mean luma (higher=more exposure flicker)
black_ratio         : fraction of near-black pixels (<=16/255)
white_ratio         : fraction of near-white pixels (>=239/255)

The first 5 columns are looked up from existing master CSVs (no recompute);
everything else is computed here.  Run per bucket; GPU-heavy (DINO+CLIP+RAFT-free).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import clip

from vbench.utils import load_video, dino_transform, clip_transform
from vbench.temporal_flickering import cal_score as flicker_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- master-CSV lookups (RAFT motion + aesthetic), keyed by basename ----------
def load_lookup(path: Path, key_col: str, val_cols: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[Path(r[key_col]).name] = {c: r.get(c, "") for c in val_cols}
    return out


# ---------- cheap pixel deltas ----------
def pixel_deltas(tchw: np.ndarray) -> tuple[float, float]:
    """tchw: (T,C,H,W) [0,255] -> (mean_delta, p95_delta) on [0,1] scale."""
    x = tchw.astype(np.float32) / 255.0
    seq = np.abs(x[1:] - x[:-1]).reshape(x.shape[0] - 1, -1).mean(axis=1)
    return float(seq.mean()), float(np.percentile(seq, 95))


# ---------- sharpness / flicker / clipping from luma ----------
def luma_stats(tchw: np.ndarray) -> tuple[float, float, float, float]:
    """tchw: (T,C,H,W) [0,255]."""
    f = tchw.astype(np.float32)
    luma = 0.299 * f[:, 0] + 0.587 * f[:, 1] + 0.114 * f[:, 2]  # (T,H,W)
    # sharpness: mean gradient magnitude (Sobel-ish via np.gradient), normalized /255
    gy, gx = np.gradient(luma, axis=(1, 2))
    sharp = float(np.sqrt(gx ** 2 + gy ** 2).mean() / 255.0)
    flicker = float(luma.mean(axis=(1, 2)).std() / 255.0)
    black = float((f <= 16).all(axis=1).mean())   # all channels near-black
    white = float((f >= 239).all(axis=1).mean())
    return sharp, flicker, black, white


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, type=Path, help="bucket clips dir")
    ap.add_argument("--output", "-o", required=True, type=Path)
    ap.add_argument("--motion-csv", type=Path,
                    default=Path("/workspace/merged_dataset_nexisgen/motion_strength.csv"))
    ap.add_argument("--aes-csv", type=Path,
                    default=Path("/workspace/merged_dataset_nexisgen/vbench_aes_dyn.csv"))
    ap.add_argument("--ext", default="mp4,mov,avi,webm,mkv")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    exts = set(args.ext.lower().split(","))
    videos = sorted(p for p in args.input.rglob("*")
                    if p.is_file() and p.suffix.lstrip(".").lower() in exts)
    print(f"Found {len(videos)} videos under {args.input}")
    if not videos:
        return 1

    motion = load_lookup(args.motion_csv, "video",
                         ["motion_mean", "motion_median", "motion_max", "moving_ratio"])
    aes = load_lookup(args.aes_csv, "video", ["aesthetic_quality"])
    print(f"motion lookup: {len(motion)} rows | aesthetic lookup: {len(aes)} rows")

    done: set[str] = set()
    if args.resume and args.output.exists():
        with args.output.open(encoding="utf-8") as f:
            done = {r["video"] for r in csv.DictReader(f)}
    todo = [v for v in videos if v.name not in done]
    print(f"To score this run: {len(todo)}")

    print("Loading DINO vitb16 ...")
    dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16",
                          verbose=False).to(DEVICE).eval()
    dino_tf = dino_transform(224)
    print("Loading CLIP ViT-B/32 ...")
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_tf = clip_transform(224)

    fields = ["video", "motion_mean_RAFT", "motion_median_RAFT", "motion_max_RAFT",
              "moving_ratio", "aesthetic_quality", "mean_delta", "p95_delta",
              "subject_consistency", "background_consistency", "temporal_flickering",
              "sharpness", "flicker_luma_std", "black_ratio", "white_ratio"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fresh = not (args.resume and args.output.exists())
    fout = args.output.open("w" if fresh else "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=fields)
    if fresh:
        w.writeheader(); fout.flush()

    for i, v in enumerate(todo, 1):
        name = v.name
        try:
            frames = load_video(str(v))                    # (T,C,H,W) float32 [0,255]
            tchw_t = frames if torch.is_tensor(frames) else torch.as_tensor(frames)
            fr = tchw_t.numpy()
            mean_delta, p95_delta = pixel_deltas(fr)
            sharp, flick, black, white = luma_stats(fr)

            # subject_consistency (DINO) — VBench formula on TCHW
            imgs = dino_tf(tchw_t)
            sc_sum, sc_cnt = 0.0, 0
            with torch.no_grad():
                first = prev = None
                for j in range(len(imgs)):
                    feat = F.normalize(dino(imgs[j].unsqueeze(0).to(DEVICE)), dim=-1, p=2)
                    if j == 0:
                        first = feat
                    else:
                        s = (max(0.0, F.cosine_similarity(prev, feat).item()) +
                             max(0.0, F.cosine_similarity(first, feat).item())) / 2
                        sc_sum += s; sc_cnt += 1
                    prev = feat
            subj = sc_sum / sc_cnt if sc_cnt else 0.0

            # background_consistency (CLIP) — VBench formula
            cimgs = clip_tf(tchw_t).to(DEVICE)
            with torch.no_grad():
                cfeat = F.normalize(clip_model.encode_image(cimgs), dim=-1, p=2)
            bc_sum, bc_cnt = 0.0, 0
            first = prev = None
            for j in range(len(cfeat)):
                fj = cfeat[j].unsqueeze(0)
                if j == 0:
                    first = fj
                else:
                    s = (max(0.0, F.cosine_similarity(prev, fj).item()) +
                         max(0.0, F.cosine_similarity(first, fj).item())) / 2
                    bc_sum += s; bc_cnt += 1
                prev = fj
            bg = bc_sum / bc_cnt if bc_cnt else 0.0

            tflick = float(flicker_score(str(v)))
        except Exception as exc:
            print(f"  [{i}/{len(todo)}] {name} ERROR: {exc}", file=sys.stderr)
            continue

        m = motion.get(name, {})
        a = aes.get(name, {})
        row = {
            "video": name,
            "motion_mean_RAFT": m.get("motion_mean", ""),
            "motion_median_RAFT": m.get("motion_median", ""),
            "motion_max_RAFT": m.get("motion_max", ""),
            "moving_ratio": m.get("moving_ratio", ""),
            "aesthetic_quality": a.get("aesthetic_quality", ""),
            "mean_delta": f"{mean_delta:.4f}",
            "p95_delta": f"{p95_delta:.4f}",
            "subject_consistency": f"{subj:.4f}",
            "background_consistency": f"{bg:.4f}",
            "temporal_flickering": f"{tflick:.4f}",
            "sharpness": f"{sharp:.4f}",
            "flicker_luma_std": f"{flick:.4f}",
            "black_ratio": f"{black:.4f}",
            "white_ratio": f"{white:.4f}",
        }
        w.writerow(row); fout.flush()
        print(f"  [{i}/{len(todo)}] {name}  aes={row['aesthetic_quality']} "
              f"mdelta={row['mean_delta']} subj={row['subject_consistency']} "
              f"bg={row['background_consistency']} sharp={row['sharpness']}")
    fout.close()
    print(f"\nDONE -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
