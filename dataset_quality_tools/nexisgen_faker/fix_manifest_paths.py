#!/usr/bin/env python3
"""Rewrite absolute video paths in manifest.jsonl to match the current dataset directory.

When a dataset directory is copied, manifest.jsonl may still contain absolute
paths pointing to the original location. This script updates those paths so that
the 'video' field in each row points to the current directory's clips/ folder.

Usage:
    cd /workspace/top400_combined_motion_captioned_test
    python fix_manifest_paths.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line: %s (%s)", line, exc)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def fix_video_paths(rows: list[dict[str, Any]], dataset_dir: Path) -> tuple[list[dict[str, Any]], int, int]:
    """Return updated rows plus counts of (fixed, already_correct)."""
    clips_dir = dataset_dir / "clips"
    fixed = 0
    already_correct = 0
    updated: list[dict[str, Any]] = []

    for row in rows:
        new_row = dict(row)
        video_path = Path(str(new_row.get("video", "")))

        if not video_path.name:
            updated.append(new_row)
            continue

        expected = clips_dir / video_path.name
        if video_path.resolve() == expected.resolve():
            already_correct += 1
        else:
            new_row["video"] = str(expected)
            fixed += 1

        updated.append(new_row)

    return updated, fixed, already_correct


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix absolute video paths in manifest.jsonl.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Dataset directory containing manifest.jsonl and clips/.",
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
    manifest_path = dataset_dir / "manifest.jsonl"

    if not manifest_path.exists():
        raise SystemExit(f"manifest.jsonl not found: {manifest_path}")

    rows = read_jsonl(manifest_path)
    updated_rows, fixed, already_correct = fix_video_paths(rows, dataset_dir)

    logger.info("Loaded %d rows from %s", len(rows), manifest_path)
    logger.info("Already correct: %d", already_correct)
    logger.info("Paths to fix: %d", fixed)

    if fixed > 0:
        if args.dry_run:
            logger.info("DRY RUN: would rewrite %d paths", fixed)
            for old, new in zip(rows, updated_rows):
                if old.get("video") != new.get("video"):
                    logger.info("  %s -> %s", old.get("video"), new.get("video"))
        else:
            write_jsonl(manifest_path, updated_rows)
            logger.info("Updated %s", manifest_path)
    else:
        logger.info("No paths need fixing.")


if __name__ == "__main__":
    main()
