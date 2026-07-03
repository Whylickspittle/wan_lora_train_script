#!/usr/bin/env python3
"""Combine curated review clips from the original 400 dataset with the already
packaged Pexels datasets into a single Nexisgen-format dataset.

Input:
  - /workspace/merged_dataset_nexisgen/dataset.parquet (400 rows)
    keep only clips present in
      strong_motion_review/clips/  +  medium_motion_review/clips/
  - /workspace/upload/pexel_dataset_merged/dataset.parquet (269 rows)
  - /workspace/upload/pexel_dataset_batch2/dataset.parquet (16 rows)

Output:
  - /workspace/upload/motion_scenery_final_394/
      clips/          (symlink-free hard copies)
      frames/
      dataset.parquet
      manifest.json
      manifest.jsonl

Note: pexel_dataset_batch2's 16 clips are already contained in
pexel_dataset_merged (identical clip_id), so they are deduplicated.
The final unique count is 125 + 269 = 394.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_clip_ids(*folders: Path) -> set[str]:
    ids: set[str] = set()
    for folder in folders:
        for p in folder.glob("*.mp4"):
            ids.add(p.stem)
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/workspace/upload/motion_scenery_final_394"))
    args = ap.parse_args()

    out_dir = args.out_dir
    clips_out = out_dir / "clips"
    frames_out = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_out.mkdir(exist_ok=True)
    frames_out.mkdir(exist_ok=True)

    # 1. Gather the 125 review clips from original 400 dataset.
    #    The review folders contain renamed files like
    #    "05_drone_aerial_pass__clip_30625451_node0.88.mp4", while the parquet
    #    stores a shorter clip_id ("clip_30625451_node0.88") but the full name
    #    in clip_uri.  We therefore match by the basename of clip_uri.
    orig_base = Path("/workspace/merged_dataset_nexisgen")
    review_ids = collect_clip_ids(
        orig_base / "strong_motion_review" / "clips",
        orig_base / "medium_motion_review" / "clips",
    )
    print(f"Review clips selected: {len(review_ids)}")

    df_orig = pd.read_parquet(orig_base / "dataset.parquet")
    df_orig["_uri_stem"] = df_orig["clip_uri"].apply(lambda x: Path(x).stem)
    df_orig = df_orig[df_orig["_uri_stem"].isin(review_ids)].copy()
    df_orig["_src_base"] = str(orig_base)
    print(f"Rows matched in original 400 parquet: {len(df_orig)}")

    # 2. Load Pexels datasets
    pexel_merged_base = Path("/workspace/upload/pexel_dataset_merged")
    pexel_batch2_base = Path("/workspace/upload/pexel_dataset_batch2")

    df_pm = pd.read_parquet(pexel_merged_base / "dataset.parquet").copy()
    df_pm["_src_base"] = str(pexel_merged_base)

    df_b2 = pd.read_parquet(pexel_batch2_base / "dataset.parquet").copy()
    df_b2["_src_base"] = str(pexel_batch2_base)

    # 3. Concatenate and deduplicate.  The original 400 dataset contains some
    #    Pexels clips that also appear in pexel_dataset_merged; we dedupe by
    #    source_video_id + clip_start_sec so the same video segment is kept only
    #    once regardless of filename/clip_id differences.
    df_all = pd.concat([df_orig, df_pm, df_b2], ignore_index=True)
    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["source_video_id", "clip_start_sec"], keep="first").reset_index(drop=True)
    after = len(df_all)
    print(f"Before dedup: {before}, after dedup: {after} (removed {before - after} duplicates)")

    # 4. Copy clips + frames and recompute hashes
    rows = []
    manifest_rows = []
    for _, row in df_all.iterrows():
        src_base = Path(row["_src_base"])
        clip_id = row["clip_id"]
        src_clip = src_base / row["clip_uri"]
        src_frame = src_base / row["first_frame_uri"]

        if not src_clip.exists():
            print(f"WARNING: missing source clip {src_clip}")
            continue
        if not src_frame.exists():
            print(f"WARNING: missing source frame {src_frame}")
            continue

        dst_clip = clips_out / src_clip.name
        dst_frame = frames_out / src_frame.name

        if not dst_clip.exists():
            shutil.copy2(src_clip, dst_clip)
        if not dst_frame.exists():
            shutil.copy2(src_frame, dst_frame)

        clip_sha256 = sha256_file(dst_clip)
        frame_sha256 = sha256_file(dst_frame)

        rows.append({
            "clip_id": clip_id,
            "clip_uri": f"clips/{dst_clip.name}",
            "clip_sha256": clip_sha256,
            "first_frame_uri": f"frames/{dst_frame.name}",
            "first_frame_sha256": frame_sha256,
            "source_video_id": row["source_video_id"],
            "clip_start_sec": float(row["clip_start_sec"]),
            "duration_sec": float(row["duration_sec"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "fps": float(row["fps"]),
            "num_frames": int(row["num_frames"]),
            "source_video_url": row["source_video_url"],
            "caption": row.get("caption", ""),
        })

        manifest_rows.append({
            "id": clip_id,
            "video": f"clips/{dst_clip.name}",
            "prompt": row.get("caption", ""),
        })

    # 5. Write outputs
    df_out = pd.DataFrame(rows)
    df_out.to_parquet(out_dir / "dataset.parquet", index=False)

    manifest = {
        "protocol_version": "2.0.0",
        "schema_version": "2.0.0",
        "spec_id": "video_v1",
        "netuid": 70,
        "miner_hotkey": "test_miner",
        "interval_id": 1,
        "created_at": pd.Timestamp.now("UTC").isoformat() + "Z",
        "record_count": len(rows),
        "dataset_sha256": sha256_file(out_dir / "dataset.parquet"),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in manifest_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Created dataset at {out_dir}")
    print(f"  clips: {len(rows)}")
    print(f"  duration total: {sum(r['duration_sec'] for r in rows):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
