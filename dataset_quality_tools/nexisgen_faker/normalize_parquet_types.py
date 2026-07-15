#!/usr/bin/env python3
"""Normalize data types in dataset.parquet to match original numeric types.

Some quality metrics (motion_*, aesthetic_quality, etc.) were stored as strings
during previous conversions. This script casts them back to their proper numeric
types while preserving all other columns unchanged.

Usage:
    cd /workspace/top400_combined_motion_captioned
    python normalize_parquet_types.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Columns that should be numeric and their target types.
NUMERIC_COLUMNS: dict[str, type] = {
    "clip_start_sec": float,
    "duration_sec": float,
    "width": int,
    "height": int,
    "fps": float,
    "num_frames": int,
    "motion_mean": float,
    "motion_median": float,
    "motion_max": float,
    "thres": float,
    "moving_frames": int,
    "total_pairs": int,
    "moving_ratio": float,
    "count_num": int,
    "dynamic": float,
    "aesthetic_quality": float,
}


def normalize_parquet(parquet_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %s with %d rows and %d columns", parquet_path, len(df), len(df.columns))

    modified: list[str] = []
    for col, target_type in NUMERIC_COLUMNS.items():
        if col not in df.columns:
            logger.warning("Column %s not found in parquet, skipping", col)
            continue

        original_dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(df[col]):
            logger.info("Column %s is already numeric (%s), skipping", col, original_dtype)
            continue

        try:
            if target_type == int:
                # Use nullable Int64 to handle potential NaNs safely.
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(target_type)
            modified.append(col)
            logger.info("Converted %s from %s to %s", col, original_dtype, df[col].dtype)
        except Exception as exc:
            logger.error("Failed to convert %s: %s", col, exc)
            raise

    if modified:
        df.to_parquet(parquet_path, index=False)
        logger.info("Wrote normalized parquet to %s", parquet_path)
        logger.info("Modified columns: %s", ", ".join(modified))
    else:
        logger.info("No columns needed normalization")

    # Print dtypes summary.
    print("\nCurrent column dtypes:")
    print(df.dtypes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize numeric types in dataset.parquet.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Dataset directory containing dataset.parquet.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parquet_path = args.dataset_dir.resolve() / "dataset.parquet"
    if not parquet_path.exists():
        raise SystemExit(f"dataset.parquet not found: {parquet_path}")

    normalize_parquet(parquet_path)


if __name__ == "__main__":
    main()
