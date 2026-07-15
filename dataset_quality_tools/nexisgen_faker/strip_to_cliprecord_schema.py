#!/usr/bin/env python3
"""Strip dataset.parquet down to Nexisgen ClipRecord schema.

Removes extra columns that are not part of the Nexisgen ClipRecord model so the
parquet is fully consistent with what the validator expects.

Usage:
    cd /workspace/top400_combined_motion_captioned
    python strip_to_cliprecord_schema.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/workspace/nexisgen")
from nexis.models import ClipRecord

logger = logging.getLogger(__name__)


def strip_to_schema(parquet_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %s: %d rows, %d columns", parquet_path, len(df), len(df.columns))

    # Columns that ClipRecord expects (in schema order).
    cliprecord_fields = set(ClipRecord.model_fields.keys())

    # Keep only columns that are part of ClipRecord.
    keep_cols = [c for c in df.columns if c in cliprecord_fields]
    drop_cols = [c for c in df.columns if c not in cliprecord_fields]

    if drop_cols:
        df = df[keep_cols]
        logger.info("Dropping %d extra columns: %s", len(drop_cols), ", ".join(drop_cols))
        df.to_parquet(parquet_path, index=False)
        logger.info("Wrote stripped parquet to %s", parquet_path)
    else:
        logger.info("No extra columns to drop")

    print("\nRemaining columns:")
    print(df.dtypes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip parquet to ClipRecord schema.")
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

    strip_to_schema(parquet_path)


if __name__ == "__main__":
    main()
