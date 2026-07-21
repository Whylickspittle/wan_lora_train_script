#!/usr/bin/env python3
"""Generate a minimal Nexisgen-format test dataset with just a few clips.

This is NOT a valid Nexisgen submission (protocol requires 400 samples), but it
produces the exact same file layout so the upload mechanics can be tested end-to-end
with only a handful of small files.

Output layout:
    interval_1/
    ├── dataset.parquet
    ├── manifest.json
    ├── clips/
    └── frames/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def random_youtube_id(length: int = 11) -> str:
    chars = string.ascii_letters + string.digits + "-_"
    return "".join(secrets.choice(chars) for _ in range(length))


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


def build_record(
    clip_id: str,
    clip_path: Path,
    frame_path: Path,
    source_video_id: str,
    source_video_url: str,
    start_sec: float,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "clip_uri": f"clips/{clip_id}.mp4",
        "clip_sha256": sha256_file(clip_path),
        "first_frame_uri": f"frames/{clip_id}.jpg",
        "first_frame_sha256": sha256_file(frame_path),
        "source_video_id": source_video_id,
        "clip_start_sec": start_sec,
        "duration_sec": 5.041666666666667,
        "width": 1280,
        "height": 704,
        "fps": 24.0,
        "num_frames": 121,
        "source_video_url": source_video_url,
        "caption": f"A generated test clip {clip_id}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate minimal Nexisgen test dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path("/workspace/nexisgen_test_dataset/interval_1"))
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--hotkey", type=str, default="5EJGfSvRcEGVQtqDuU7YYwuZRHmaktf6JEZDeFPyeXksiHrm")
    parser.add_argument("--test-record-count", type=int, default=3, help="Number of test clips (default 3)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    clips_dir = dataset_dir / "clips"
    frames_dir = dataset_dir / "frames"

    if not clips_dir.is_dir():
        raise SystemExit(f"clips/ directory not found: {clips_dir}")
    if not frames_dir.is_dir():
        raise SystemExit(f"frames/ directory not found: {frames_dir}")

    clip_files = sorted(clips_dir.glob("*.mp4"))
    if not clip_files:
        raise SystemExit(f"no .mp4 files found in {clips_dir}")

    records: list[dict[str, Any]] = []
    for idx, clip_path in enumerate(clip_files[: args.test_record_count], start=1):
        clip_id = clip_path.stem
        frame_path = frames_dir / f"{clip_id}.jpg"
        if not frame_path.exists():
            print(f"warning: frame missing for {clip_id}, skipping")
            continue

        video_id = random_youtube_id()
        source_video_id = f"youtube_{video_id}"
        source_video_url = f"https://www.youtube.com/watch?v={video_id}"
        start_sec = round(0.1 * idx, 3)  # 0.1, 0.2, 0.3 ... spaced apart

        records.append(
            build_record(
                clip_id=clip_id,
                clip_path=clip_path,
                frame_path=frame_path,
                source_video_id=source_video_id,
                source_video_url=source_video_url,
                start_sec=start_sec,
            )
        )
        print(f"record {idx}: {clip_id} -> {source_video_url}")

    # Write dataset.parquet
    arrays: dict[str, list[Any]] = {col: [] for col in NEXIS_COLUMNS}
    for record in records:
        for col in NEXIS_COLUMNS:
            arrays[col].append(record.get(col, ""))

    table = pa.table({col: arrays[col] for col in NEXIS_COLUMNS})
    parquet_path = dataset_dir / "dataset.parquet"
    pq.write_table(table, parquet_path)
    print(f"wrote {parquet_path} ({len(records)} records)")

    # Write manifest.json (relaxed test version, not enforcing 400 records)
    manifest = {
        "protocol_version": "2.0.0",
        "schema_version": "2.0.0",
        "spec_id": "video_v1",
        "netuid": 70,
        "miner_hotkey": args.hotkey,
        "interval_id": args.interval,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "record_count": len(records),
        "dataset_sha256": sha256_file(parquet_path),
    }
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
