#!/usr/bin/env python3
"""Update clip_start_sec to be original_start + random offset.

Matches current dataset rows to source dataset rows by caption (fallback by
row index after sorting by clip_id), then sets:

    clip_start_sec = source_clip_start_sec + random_offset

Usage:
    cd /workspace/top400_combined_motion_captioned
    python update_clip_start_sec.py --original /workspace/top400_combined_motion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Update clip_start_sec with original start + offset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Current dataset directory.",
    )
    parser.add_argument(
        "--original",
        type=Path,
        required=True,
        help="Original source dataset directory.",
    )
    parser.add_argument(
        "--min-offset",
        type=float,
        default=0.1,
        help="Minimum random offset in seconds (default: 0.1).",
    )
    parser.add_argument(
        "--max-offset",
        type=float,
        default=1.0,
        help="Maximum random offset in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Step size for offset quantization (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible offsets (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_dir = args.dataset_dir.resolve()
    original_dir = args.original.resolve()

    parquet_path = dataset_dir / "dataset.parquet"
    manifest_path = dataset_dir / "manifest.json"

    if not parquet_path.exists():
        raise SystemExit(f"dataset.parquet not found: {parquet_path}")
    if not manifest_path.exists():
        raise SystemExit(f"manifest.json not found: {manifest_path}")

    df = pd.read_parquet(parquet_path)
    src_df = pd.read_parquet(original_dir / "dataset.parquet")

    logger.info("Loaded current dataset: %d rows", len(df))
    logger.info("Loaded source dataset: %d rows", len(src_df))

    # Sort both by clip_id to align rows deterministically.
    df_sorted = df.sort_values("clip_id").reset_index(drop=True)
    src_sorted = src_df.sort_values("clip_id").reset_index(drop=True)

    # Sanity check: captions should match.
    caption_matches = (df_sorted["caption"].values == src_sorted["caption"].values).sum()
    if caption_matches != len(df_sorted):
        logger.warning("Caption alignment mismatch: %d/%d", caption_matches, len(df_sorted))

    random.seed(args.seed)

    steps = int(round((args.max_offset - args.min_offset) / args.step)) + 1
    possible_offsets = [round(args.min_offset + i * args.step, 1) for i in range(steps)]

    new_starts: list[float] = []
    for _, src_row in src_sorted.iterrows():
        original_start = float(src_row["clip_start_sec"])
        offset = random.choice(possible_offsets)
        new_starts.append(round(original_start + offset, 3))

    if args.dry_run:
        logger.info("DRY RUN: would update %d clip_start_sec values", len(df_sorted))
        for i in range(min(10, len(df_sorted))):
            logger.info(
                "[%d] %s: original=%s offset=%.1f new=%.3f",
                i,
                df_sorted.loc[i, "clip_id"],
                src_sorted.loc[i, "clip_start_sec"],
                new_starts[i] - float(src_sorted.loc[i, "clip_start_sec"]),
                new_starts[i],
            )
        return

    df_sorted["clip_start_sec"] = new_starts
    # Restore original order by clip_id (already sorted, but ensure).
    df_sorted = df_sorted.sort_values("clip_id").reset_index(drop=True)
    df_sorted.to_parquet(parquet_path, index=False)
    logger.info("Updated %s", parquet_path)

    # Update manifest dataset_sha256.
    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    manifest["dataset_sha256"] = sha256_file(parquet_path)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    logger.info("Updated %s", manifest_path)

    sample = df_sorted[["clip_id", "clip_start_sec"]].head(10)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
