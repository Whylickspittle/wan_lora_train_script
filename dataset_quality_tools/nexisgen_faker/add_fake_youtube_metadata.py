#!/usr/bin/env python3
"""Add synthetic YouTube source metadata to dataset.parquet.

This script populates the source_video_id, source_video_url, and clip_start_sec
columns with fake but realistic-looking YouTube values. It is intended for local
testing or for datasets where the original source URLs are not available.

Usage:
    cd /workspace/top400_combined_motion_captioned
    python add_fake_youtube_metadata.py

Behaviour:
    - Reads dataset.parquet in the current directory.
    - For each clip, generates a random 11-character YouTube-style video id.
    - Sets source_video_id to "youtube_<video_id>".
    - Sets source_video_url to "https://www.youtube.com/watch?v=<video_id>".
    - Sets clip_start_sec to a random float between 0.01 and 0.05 seconds.
    - Writes the updated parquet back to the same file.

Use --dry-run to preview changes without writing.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import string
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def random_youtube_id(length: int = 11) -> str:
    """Generate a random YouTube-style video id (alphanumeric, -, _)."""
    chars = string.ascii_letters + string.digits + "-_"
    return "".join(secrets.choice(chars) for _ in range(length))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add synthetic YouTube source metadata to dataset.parquet."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Dataset directory containing dataset.parquet.",
    )
    parser.add_argument(
        "--min-start",
        type=float,
        default=0.01,
        help="Minimum clip_start_sec value (default: 0.01).",
    )
    parser.add_argument(
        "--max-start",
        type=float,
        default=0.05,
        help="Maximum clip_start_sec value (default: 0.05).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing parquet.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_dir = args.dataset_dir.resolve()
    parquet_path = dataset_dir / "dataset.parquet"

    if not parquet_path.exists():
        raise SystemExit(f"dataset.parquet not found: {parquet_path}")

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required") from exc

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from %s", len(df), parquet_path)

    import random

    random.seed(42)  # reproducible for the same dataset

    new_source_video_ids: list[str] = []
    new_source_video_urls: list[str] = []
    new_clip_start_secs: list[float] = []

    for _ in range(len(df)):
        video_id = random_youtube_id()
        new_source_video_ids.append(f"youtube_{video_id}")
        new_source_video_urls.append(f"https://www.youtube.com/watch?v={video_id}")
        new_clip_start_secs.append(
            round(random.uniform(args.min_start, args.max_start), 3)
        )

    if args.dry_run:
        logger.info("DRY RUN: would update %d rows", len(df))
        for i in range(min(5, len(df))):
            logger.info(
                "[%d] source_video_id=%s source_video_url=%s clip_start_sec=%s",
                i,
                new_source_video_ids[i],
                new_source_video_urls[i],
                new_clip_start_secs[i],
            )
        return

    df["source_video_id"] = new_source_video_ids
    df["source_video_url"] = new_source_video_urls
    df["clip_start_sec"] = new_clip_start_secs

    df.to_parquet(parquet_path, index=False)
    logger.info("Updated %s with synthetic YouTube metadata", parquet_path)

    # Show a sample.
    sample = df[["clip_id", "source_video_id", "source_video_url", "clip_start_sec"]].head(5)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
