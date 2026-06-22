#!/usr/bin/env python3
"""Merge multiple Nexisgen-format datasets into one package.

Inputs are directories produced by ``convert_to_nexisgen_dataset.py``:

    /workspace/pexels_batch01/nexisgen_format
    /workspace/pexels_batch02/nexisgen_format
    ...

Output structure:

    output_dir/
    ├── clips/          merged clips
    ├── frames/         merged first frames
    ├── dataset.parquet merged ClipRecord table
    ├── manifest.json   interval manifest
    └── manifest.jsonl  plain manifest for caption annotator
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NEXIS_COLUMNS = [
    "clip_id",
    "clip_uri",
    "clip_sha256",
    "first_frame_uri",
    "first_frame_sha256",
    "source_video_id",
    "clip_start_sec",
    "duration_sec",
    "width",
    "height",
    "fps",
    "num_frames",
    "source_video_url",
    "caption",
]


def _import_pyarrow() -> Any:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        return pa, pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required. Install it with:\n"
            "  pip install pyarrow>=14.0.0"
        ) from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    pa, _ = _import_pyarrow()
    table = pa.parquet.read_table(path)
    columns = {col: table.column(col).to_pylist() for col in NEXIS_COLUMNS}
    return [
        {col: columns[col][i] for col in NEXIS_COLUMNS}
        for i in range(table.num_rows)
    ]


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    pa, pq = _import_pyarrow()
    arrays: dict[str, list[Any]] = {col: [] for col in NEXIS_COLUMNS}
    for record in records:
        for col in NEXIS_COLUMNS:
            arrays[col].append(record.get(col, ""))
    table = pa.table({col: arrays[col] for col in NEXIS_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge multiple Nexisgen-format datasets into one."
    )
    parser.add_argument(
        "--inputs",
        required=True,
        nargs="+",
        type=Path,
        help="List of Nexisgen-format dataset directories to merge.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for the merged dataset.",
    )
    parser.add_argument(
        "--manifest-json-only",
        action="store_true",
        help="Skip writing dataset.parquet (useful for inspection).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    output_clips_dir = args.output_dir / "clips"
    output_frames_dir = args.output_dir / "frames"
    output_clips_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    merged_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for input_dir in args.inputs:
        parquet_path = input_dir / "dataset.parquet"
        if not parquet_path.exists():
            logger.warning("skipping %s: dataset.parquet not found", input_dir)
            continue

        records = read_parquet_records(parquet_path)
        logger.info("reading %s: %d records", input_dir, len(records))

        for record in records:
            clip_id = str(record["clip_id"])
            original_id = clip_id

            # Resolve ID collisions by appending a counter.
            counter = 1
            while clip_id in seen_ids:
                clip_id = f"{original_id}_{counter}"
                counter += 1
            seen_ids.add(clip_id)

            src_clip = input_dir / str(record["clip_uri"])
            src_frame = input_dir / str(record["first_frame_uri"])

            dst_clip = output_clips_dir / f"{clip_id}.mp4"
            dst_frame = output_frames_dir / f"{clip_id}.jpg"

            if src_clip.exists():
                shutil.copy2(src_clip, dst_clip)
            else:
                logger.warning("clip missing: %s", src_clip)

            if src_frame.exists():
                shutil.copy2(src_frame, dst_frame)
            else:
                logger.warning("frame missing: %s", src_frame)

            record["clip_id"] = clip_id
            record["clip_uri"] = f"clips/{clip_id}.mp4"
            record["first_frame_uri"] = f"frames/{clip_id}.jpg"
            # Recompute sha256 for the copied files.
            if dst_clip.exists():
                record["clip_sha256"] = sha256_file(dst_clip)
            if dst_frame.exists():
                record["first_frame_sha256"] = sha256_file(dst_frame)

            merged_records.append(record)

    if not args.manifest_json_only:
        write_parquet(merged_records, args.output_dir / "dataset.parquet")

    # Interval manifest
    manifest = {
        "protocol_version": "2.0.0",
        "schema_version": "2.0.0",
        "spec_id": "video_v1",
        "netuid": 70,
        "miner_hotkey": "synthetic_local_test",
        "interval_id": 1,
        "created_at": "2026-06-22T00:00:00+00:00",
        "record_count": len(merged_records),
        "dataset_sha256": "0" * 64,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # Plain JSONL manifest for the caption annotator.
    with (args.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for record in merged_records:
            fh.write(
                json.dumps(
                    {"id": record["clip_id"], "video": record["clip_uri"]},
                    ensure_ascii=True,
                )
                + "\n"
            )

    logger.info(
        "merged dataset ready: %s (%d records)", args.output_dir, len(merged_records)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
