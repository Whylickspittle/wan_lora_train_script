#!/usr/bin/env python3
"""Convert a Pexels-format clip dataset into a Nexisgen-compatible package.

The output structure matches what ``nexisgen`` expects for a miner interval:

    output_dir/
    ├── clips/          copied clips (renamed to clip_id.mp4)
    ├── frames/         first-frame JPEG for every clip
    ├── dataset.parquet ClipRecord table
    └── manifest.json   interval manifest (protocol v2.0.0)

Input
-----
A Pexels pipeline directory containing at least:

    clips/              *.mp4 clips
    manifest.jsonl      {"id": "...", "video": "clips/...", "prompt": "..."}
    quality_report/     quality_report.csv (from clean_dataset.py)

Synthetic source metadata
-------------------------
Pexels clips do not carry the original source URL in their filenames. This
script reconstructs synthetic Nexisgen fields from the clip id, e.g.
``clip_11675383_node1.41`` becomes:

* ``source_video_id``: ``pexels_11675383``
* ``clip_start_sec``: ``1.41``
* ``source_video_url``: ``https://www.pexels.com/video/11675383/``

This is enough for local training / caption testing. It is NOT suitable for
submitting to the live Nexisgen subnet, because the source URLs are not real
Pexels page URLs and the sample count is far below 400.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
TARGET_FPS = 24.0
TARGET_NUM_FRAMES = 121
CLIP_DURATION_SEC = TARGET_NUM_FRAMES / TARGET_FPS

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


def parse_clip_id(clip_id: str) -> tuple[str, float]:
    """Parse ``clip_<video_id>_node<start>`` into video_id and start seconds."""
    match = re.match(r"^clip_(\d+)_node([0-9.]+)$", clip_id)
    if not match:
        raise ValueError(f"cannot parse clip_id: {clip_id}")
    video_id = match.group(1)
    start_sec = float(match.group(2))
    return video_id, start_sec


def synthetic_source(video_id: str) -> tuple[str, str]:
    """Return synthetic (source_video_id, source_video_url) for a Pexels video id."""
    source_video_id = f"pexels_{video_id}"
    source_video_url = f"https://www.pexels.com/video/{video_id}/"
    return source_video_id, source_video_url


def extract_first_frame(video_path: Path, output_path: Path, timeout: int = 60) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "select=eq(n\\,0)",
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, timeout=timeout)


def read_quality_report(path: Path) -> dict[str, dict[str, Any]]:
    """Map clip basename -> quality report row."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clip_name = Path(row.get("path", "")).name
            if clip_name:
                rows[clip_name] = row
    return rows


def filter_clips(
    manifest_rows: list[dict[str, Any]],
    quality: dict[str, dict[str, Any]],
    allowed_grades: set[str],
    max_dup_ratio: float,
) -> list[dict[str, Any]]:
    """Drop FAIL clips and clips with too many duplicate frames."""
    kept: list[dict[str, Any]] = []
    for row in manifest_rows:
        clip_id = row.get("id", "")
        video_path = Path(row.get("video", ""))
        clip_name = video_path.name or f"{clip_id}.mp4"
        q = quality.get(clip_name, {})

        grade = q.get("grade", "PASS")
        if grade not in allowed_grades:
            logger.info("[%s] rejected grade=%s", clip_id, grade)
            continue

        dup_ratio = float(q.get("duplicate_frame_ratio", 0.0))
        if dup_ratio > max_dup_ratio:
            logger.info("[%s] rejected duplicates=%.2f", clip_id, dup_ratio)
            continue

        kept.append(row)

    return kept


def build_nexis_record(
    clip_id: str,
    clip_path: Path,
    frame_path: Path,
    caption: str = "",
) -> dict[str, Any]:
    video_id, start_sec = parse_clip_id(clip_id)
    source_video_id, source_video_url = synthetic_source(video_id)
    return {
        "clip_id": clip_id,
        "clip_uri": f"clips/{clip_path.name}",
        "clip_sha256": sha256_file(clip_path),
        "first_frame_uri": f"frames/{frame_path.name}",
        "first_frame_sha256": sha256_file(frame_path),
        "source_video_id": source_video_id,
        "clip_start_sec": start_sec,
        "duration_sec": CLIP_DURATION_SEC,
        "width": TARGET_WIDTH,
        "height": TARGET_HEIGHT,
        "fps": TARGET_FPS,
        "num_frames": TARGET_NUM_FRAMES,
        "source_video_url": source_video_url,
        "caption": caption,
    }


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    pa, pq = _import_pyarrow()
    arrays: dict[str, list[Any]] = {col: [] for col in NEXIS_COLUMNS}
    for record in records:
        for col in NEXIS_COLUMNS:
            arrays[col].append(record.get(col, ""))
    table = pa.table({col: arrays[col] for col in NEXIS_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def write_interval_manifest(path: Path, record_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_version": "2.0.0",
        "schema_version": "2.0.0",
        "spec_id": "video_v1",
        "netuid": 70,
        "miner_hotkey": "synthetic_local_test",
        "interval_id": 1,
        "created_at": "2026-06-22T00:00:00+00:00",
        "record_count": record_count,
        "dataset_sha256": "0" * 64,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Pexels-format clips to Nexisgen-compatible dataset."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Pexels pipeline directory (contains clips/ and manifest.jsonl).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for the Nexisgen-format package.",
    )
    parser.add_argument(
        "--grades",
        default="PASS",
        help="Comma-separated grades to keep (default: PASS).",
    )
    parser.add_argument(
        "--max-dup-ratio",
        type=float,
        default=0.03,
        help="Maximum duplicate_frame_ratio to keep a clip (default: 0.03).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit output to N clips (0 = no limit).",
    )
    parser.add_argument(
        "--skip-frames",
        action="store_true",
        help="Do not extract first frames.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    manifest_path = args.input_dir / "manifest.jsonl"
    if not manifest_path.exists():
        logger.error("manifest not found: %s", manifest_path)
        return 1

    quality_report_path = args.input_dir / "quality_report" / "quality_report.csv"
    quality = read_quality_report(quality_report_path)
    logger.info("loaded quality report: %d rows", len(quality))

    manifest_rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                manifest_rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.error("invalid JSON line %d: %s", line_no, exc)
                return 1

    allowed_grades = {g.strip().upper() for g in args.grades.split(",")}
    filtered = filter_clips(manifest_rows, quality, allowed_grades, args.max_dup_ratio)
    logger.info("kept %d / %d clips after filtering", len(filtered), len(manifest_rows))

    if args.max_samples > 0:
        filtered = filtered[: args.max_samples]
        logger.info("limited to %d clips for minimal test", len(filtered))

    output_clips_dir = args.output_dir / "clips"
    output_frames_dir = args.output_dir / "frames"
    output_clips_dir.mkdir(parents=True, exist_ok=True)
    output_frames_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for row in filtered:
        clip_id = str(row.get("id", ""))
        if not clip_id:
            logger.warning("skipping row without id")
            continue

        video_rel = Path(str(row.get("video", "")))
        src_video = (
            video_rel
            if video_rel.is_absolute()
            else (args.input_dir / video_rel)
        ).resolve()
        if not src_video.exists():
            logger.warning("clip not found: %s", src_video)
            continue

        dst_clip = output_clips_dir / f"{clip_id}.mp4"
        shutil.copy2(src_video, dst_clip)

        frame_path = output_frames_dir / f"{clip_id}.jpg"
        if not args.skip_frames:
            extract_first_frame(dst_clip, frame_path)

        records.append(
            build_nexis_record(
                clip_id=clip_id,
                clip_path=dst_clip,
                frame_path=frame_path,
                caption=str(row.get("prompt", "")),
            )
        )
        logger.info("converted %s", clip_id)

    write_parquet(records, args.output_dir / "dataset.parquet")
    write_interval_manifest(args.output_dir / "manifest.json", len(records))

    # Also emit a plain JSONL manifest for the standalone caption annotator.
    manifest_jsonl = args.output_dir / "manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(
                json.dumps(
                    {"id": record["clip_id"], "video": record["clip_uri"]},
                    ensure_ascii=True,
                )
                + "\n"
            )

    logger.info(
        "Nexisgen-format dataset ready: %s (%d records)",
        args.output_dir,
        len(records),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
