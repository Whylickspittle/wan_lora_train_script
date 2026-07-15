#!/usr/bin/env python3
"""Uniformly rename clips/frames and sync dataset.parquet + manifest.json.

Usage:
    cd /workspace/top400_combined_motion_captioned
    python rename_clips_uniform.py --prefix motion --start 1 --dry-run
    python rename_clips_uniform.py --prefix motion --start 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def new_name_for_index(prefix: str, index: int, width: int = 4) -> str:
    return f"{prefix}_{str(index).zfill(width)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniformly rename clips/frames and sync metadata.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Dataset directory containing dataset.parquet, clips/, frames/, manifest.json.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="New filename prefix, e.g. 'motion' -> motion_0001.mp4.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting index (default: 1).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=4,
        help="Zero-padded width for index (default: 4).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without renaming files.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_dir = args.dataset_dir.resolve()
    parquet_path = dataset_dir / "dataset.parquet"
    manifest_path = dataset_dir / "manifest.json"
    clips_dir = dataset_dir / "clips"
    frames_dir = dataset_dir / "frames"

    if not parquet_path.exists():
        raise SystemExit(f"dataset.parquet not found: {parquet_path}")
    if not clips_dir.is_dir():
        raise SystemExit(f"clips/ directory not found: {clips_dir}")
    if not frames_dir.is_dir():
        raise SystemExit(f"frames/ directory not found: {frames_dir}")
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found: {manifest_path}")

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from %s", len(df), parquet_path)

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    # Sort rows by current clip_id for deterministic ordering.
    df = df.sort_values("clip_id").reset_index(drop=True)

    rename_plan: list[tuple[Path, Path, Path, Path, str]] = []
    id_map: dict[str, str] = {}

    for idx, row in df.iterrows():
        new_id = new_name_for_index(args.prefix, args.start + idx, args.width)
        old_id = str(row["clip_id"])
        old_clip_uri = Path(str(row["clip_uri"]))
        old_frame_uri = Path(str(row["first_frame_uri"]))

        old_clip_abs = dataset_dir / old_clip_uri
        old_frame_abs = dataset_dir / old_frame_uri

        new_clip_name = f"{new_id}.mp4"
        new_frame_name = f"{new_id}.jpg"
        new_clip_abs = clips_dir / new_clip_name
        new_frame_abs = frames_dir / new_frame_name

        id_map[old_id] = new_id
        rename_plan.append((old_clip_abs, new_clip_abs, old_frame_abs, new_frame_abs, new_id))

    if args.dry_run:
        logger.info("DRY RUN: would rename %d clips with prefix '%s'", len(df), args.prefix)
        for old_clip, new_clip, old_frame, new_frame, new_id in rename_plan:
            logger.info(
                "DRY RUN: %s -> %s, %s -> %s (clip_id=%s)",
                old_clip.name,
                new_clip.name,
                old_frame.name,
                new_frame.name,
                new_id,
            )
        return

    # Perform renames.
    for old_clip, new_clip, old_frame, new_frame, new_id in rename_plan:
        if not old_clip.exists():
            logger.error("Source clip missing: %s", old_clip)
            continue

        if old_clip != new_clip:
            logger.info("Renaming clip: %s -> %s", old_clip.name, new_clip.name)
            shutil.move(str(old_clip), str(new_clip))

        if old_frame.exists() and old_frame != new_frame:
            logger.info("Renaming frame: %s -> %s", old_frame.name, new_frame.name)
            shutil.move(str(old_frame), str(new_frame))

    # Update parquet.
    df["clip_id"] = df["clip_id"].map(id_map)
    df["clip_uri"] = df["clip_id"].apply(lambda x: f"clips/{x}.mp4")
    df["first_frame_uri"] = df["clip_id"].apply(lambda x: f"frames/{x}.jpg")

    df.to_parquet(parquet_path, index=False)
    logger.info("Updated %s", parquet_path)

    # Update manifest.json dataset_sha256 (parquet content changed).
    manifest["dataset_sha256"] = sha256_file(parquet_path)
    manifest["record_count"] = len(df)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    logger.info("Updated %s", manifest_path)

    logger.info("Done. %d clips renamed with prefix '%s'.", len(df), args.prefix)


if __name__ == "__main__":
    main()
