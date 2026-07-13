#!/usr/bin/env python3
"""One-clip-per-source deduplication for a Nexisgen-format dataset.

Reads a merged Nexisgen dataset plus the original Pexels quality reports,
groups clips by ``source_video_id``, and keeps exactly one clip per source
using this priority:

1. No errors: ``source_frames`` must be at least ``--min-frames``.
2. Low duplicate frames: ``duplicate_frame_ratio`` must be <= ``--max-dup-ratio``.
3. Earliest ``clip_start_sec``.

Clips that are not kept are written to a separate directory for review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyarrow is required. Install it with:\n  pip install pyarrow>=14.0.0"
    ) from exc

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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_quality_map(batch_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Map clip_id (without extension) -> quality report row."""
    quality: dict[str, dict[str, Any]] = {}
    for batch_dir in batch_dirs:
        report_path = batch_dir / "quality_report" / "quality_report.csv"
        if not report_path.exists():
            logger.warning("quality report not found: %s", report_path)
            continue
        with report_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                clip_name = Path(row.get("path", "")).stem
                if clip_name:
                    quality[clip_name] = row
    return quality


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    columns = {col: table.column(col).to_pylist() for col in NEXIS_COLUMNS}
    return [
        {col: columns[col][i] for col in NEXIS_COLUMNS}
        for i in range(table.num_rows)
    ]


def write_dataset(
    records: list[dict[str, Any]],
    output_dir: Path,
    src_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    frames_dir = output_dir / "frames"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, list[Any]] = {col: [] for col in NEXIS_COLUMNS}
    for record in records:
        src_clip = src_dir / str(record["clip_uri"])
        src_frame = src_dir / str(record["first_frame_uri"])
        dst_clip = clips_dir / Path(record["clip_uri"]).name
        dst_frame = frames_dir / Path(record["first_frame_uri"]).name
        if src_clip.exists():
            shutil.copy2(src_clip, dst_clip)
        if src_frame.exists():
            shutil.copy2(src_frame, dst_frame)

        if dst_clip.exists():
            record["clip_sha256"] = sha256_file(dst_clip)
        if dst_frame.exists():
            record["first_frame_sha256"] = sha256_file(dst_frame)

        for col in NEXIS_COLUMNS:
            arrays[col].append(record.get(col, ""))

    table = pa.table({col: arrays[col] for col in NEXIS_COLUMNS})
    pq.write_table(table, output_dir / "dataset.parquet")

    manifest = {
        "protocol_version": "2.0.0",
        "schema_version": "2.0.0",
        "spec_id": "video_v1",
        "netuid": 70,
        "miner_hotkey": "synthetic_local_test",
        "interval_id": 1,
        "created_at": "2026-06-22T00:00:00+00:00",
        "record_count": len(records),
        "dataset_sha256": "0" * 64,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(
                json.dumps(
                    {"id": record["clip_id"], "video": record["clip_uri"]},
                    ensure_ascii=True,
                )
                + "\n"
            )

    with (output_dir / "manifest_captioned.jsonl").open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(
                json.dumps(
                    {
                        "id": record["clip_id"],
                        "video": record["clip_uri"],
                        "prompt": record.get("caption", ""),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Keep one clip per source video from a Nexisgen dataset."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Input Nexisgen-format dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for the one-per-source dataset.",
    )
    parser.add_argument(
        "--tbd-dir",
        required=True,
        type=Path,
        help="Output directory for clips not selected.",
    )
    parser.add_argument(
        "--batch-dirs",
        required=True,
        nargs="+",
        type=Path,
        help="Original Pexels batch directories containing quality_report/quality_report.csv.",
    )
    parser.add_argument(
        "--max-dup-ratio",
        type=float,
        default=0.03,
        help="Maximum duplicate_frame_ratio for a clip to be eligible (default: 0.03).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=121,
        help="Minimum source_frames for a clip to be eligible (default: 121).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    quality = load_quality_map(args.batch_dirs)
    logger.info("loaded quality metrics for %d clips", len(quality))

    records = read_parquet_records(args.input_dir / "dataset.parquet")
    logger.info("read %d records from %s", len(records), args.input_dir)

    # Group by source_video_id.
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        src_id = str(record["source_video_id"])
        groups.setdefault(src_id, []).append(record)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for src_id in sorted(groups):
        group = groups[src_id]
        group.sort(key=lambda r: float(r["clip_start_sec"]))

        candidates: list[dict[str, Any]] = []
        for record in group:
            clip_id = record["clip_id"]
            q = quality.get(clip_id, {})
            try:
                source_frames = int(float(q.get("source_frames", 0)))
                dup_ratio = float(q.get("duplicate_frame_ratio", 1.0))
            except (TypeError, ValueError):
                logger.warning("[%s] bad quality metrics; skipping", clip_id)
                continue

            record["_source_frames"] = source_frames
            record["_dup_ratio"] = dup_ratio

            if source_frames < args.min_frames:
                record["_remove_reason"] = f"frames={source_frames}<{args.min_frames}"
                removed.append(record)
                continue
            if dup_ratio > args.max_dup_ratio:
                record["_remove_reason"] = f"dup_ratio={dup_ratio:.3f}>{args.max_dup_ratio}"
                removed.append(record)
                continue
            candidates.append(record)

        if not candidates:
            # No valid candidate; keep the earliest one anyway and log it.
            record = group[0]
            record["_remove_reason"] = "no_valid_candidate_kept_earliest"
            kept.append(record)
            logger.warning("[%s] no valid candidates; keeping earliest", record["clip_id"])
            continue

        # Keep earliest valid candidate.
        chosen = candidates[0]
        chosen["_remove_reason"] = "kept"
        kept.append(chosen)

        # Mark others as removed.
        for record in group:
            if record["clip_id"] != chosen["clip_id"] and "_remove_reason" not in record:
                record["_remove_reason"] = "later_from_same_source"
                removed.append(record)

    logger.info("kept %d clips (one per source)", len(kept))
    logger.info("removed %d clips", len(removed))

    write_dataset(kept, args.output_dir, args.input_dir)
    write_dataset(removed, args.tbd_dir, args.input_dir)

    logger.info("output: %s", args.output_dir)
    logger.info("tbd: %s", args.tbd_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
