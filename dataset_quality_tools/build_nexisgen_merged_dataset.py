#!/usr/bin/env python3
"""Build a Nexisgen-compatible dataset package from the cleaned merged manifest.

Output structure:
    output_dir/
    ├── clips/          copied clips (original filenames preserved)
    ├── frames/         first-frame JPEG for every clip
    ├── dataset.parquet ClipRecord table
    └── manifest.json   interval manifest (protocol v2.0.0)

Input sources:
    - /workspace/623_351_extracted/final_623_351/manifest_captioned_clean.jsonl
    - /workspace/review_batch08_top70_motion/manifest_captioned.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
TARGET_FPS = 24.0
TARGET_NUM_FRAMES = 121
CLIP_DURATION_SEC = TARGET_NUM_FRAMES / TARGET_FPS

OUTPUT_DIR = Path("/workspace/merged_dataset_nexisgen")
MANIFEST_623 = Path("/workspace/623_351_extracted/final_623_351/manifest_captioned_clean.jsonl")
MANIFEST_BATCH08 = Path("/workspace/review_batch08_top70_motion/manifest_captioned.jsonl")

# Default sources used when no --source is passed on the CLI: (manifest, root_dir).
DEFAULT_SOURCES: list[tuple[Path, Path]] = [
    (MANIFEST_623, Path("/workspace/623_351_extracted/final_623_351")),
    (MANIFEST_BATCH08, Path("/workspace/review_batch08_top70_motion")),
]


def parse_source(spec: str) -> tuple[Path, Path]:
    """Parse a --source spec 'MANIFEST' or 'MANIFEST::ROOT'.

    When ROOT is omitted it defaults to the manifest's parent directory, which
    matches the Pexels / selected_manifest.jsonl layout (video paths relative
    to the manifest dir).
    """
    if "::" in spec:
        manifest_str, root_str = spec.split("::", 1)
        return Path(manifest_str), Path(root_str)
    manifest = Path(spec)
    return manifest, manifest.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_first_frame(video_path: Path, output_path: Path, timeout: int = 60) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vf", "select=eq(n\\,0)",
        "-frames:v", "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, timeout=timeout)


def parse_youtube_id(stem: str) -> tuple[str, float] | None:
    """Parse YouTube-style '<video_id>_<start_seconds>' stem."""
    # YouTube video IDs are 11 chars of [A-Za-z0-9_-]
    match = re.match(r"^([A-Za-z0-9_-]{11})_(\d+)$", stem)
    if match:
        return match.group(1), float(match.group(2))
    return None


def parse_pexels_id(stem: str) -> tuple[str, float] | None:
    """Parse Pexels-style 'clip_<video_id>_node<start>' from a possibly prefixed stem."""
    match = re.search(r"clip_(\d+)_node([0-9.]+)$", stem)
    if match:
        return match.group(1), float(match.group(2))
    return None


def build_record(row: dict[str, Any], clip_path: Path, frame_path: Path) -> dict[str, Any]:
    stem = clip_path.stem
    caption = row.get("prompt", "") or ""

    yt = parse_youtube_id(stem)
    if yt:
        video_id, start_sec = yt
        source_video_id = f"youtube_{video_id}"
        source_video_url = f"https://www.youtube.com/watch?v={video_id}&t={int(start_sec)}s"
        clip_id = f"{video_id}_{int(start_sec):04d}"
    else:
        pex = parse_pexels_id(stem)
        if pex:
            video_id, start_sec = pex
            source_video_id = f"pexels_{video_id}"
            source_video_url = f"https://www.pexels.com/video/{video_id}/"
            clip_id = f"clip_{video_id}_node{start_sec:.2f}"
        else:
            # Fallback: treat whole stem as id
            video_id = stem
            start_sec = 0.0
            source_video_id = f"unknown_{video_id}"
            source_video_url = ""
            clip_id = stem

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Nexisgen dataset package (parquet + manifest) from one or more manifests."
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help=f"Output package directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--source", action="append", default=None, metavar="MANIFEST[::ROOT]",
        help="Input manifest, optionally with '::ROOT' for the clip base dir "
             "(repeatable). ROOT defaults to the manifest's parent dir. "
             "If omitted entirely, the two legacy default sources are used.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    sources = [parse_source(s) for s in args.source] if args.source else DEFAULT_SOURCES

    if output_dir.exists():
        shutil.rmtree(output_dir)
    clips_dir = output_dir / "clips"
    frames_dir = output_dir / "frames"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[dict[str, Any], Path]] = []
    for manifest_path, root_dir in sources:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append((json.loads(line), root_dir))

    print(f"Total manifest entries: {len(rows)} from {len(sources)} source(s)")

    records: list[dict[str, Any]] = []
    for idx, (row, root_dir) in enumerate(rows, 1):
        src_video = root_dir / row["video"]
        dst_name = Path(row["video"]).name
        dst_clip = clips_dir / dst_name

        # Handle filename collision by prefixing source
        if dst_clip.exists():
            stem = dst_name.replace(".mp4", "")
            dst_name = f"{stem}_{row.get('source', 'unknown')}.mp4"
            dst_clip = clips_dir / dst_name

        shutil.copy2(src_video, dst_clip)

        frame_name = dst_clip.stem + ".jpg"
        dst_frame = frames_dir / frame_name
        extract_first_frame(dst_clip, dst_frame)

        record = build_record(row, dst_clip, dst_frame)
        records.append(record)
        print(f"[{idx}/{len(rows)}] {record['clip_id']} -> {dst_name}")

    # Build parquet
    schema = pa.schema([
        ("clip_id", pa.string()),
        ("clip_uri", pa.string()),
        ("clip_sha256", pa.string()),
        ("first_frame_uri", pa.string()),
        ("first_frame_sha256", pa.string()),
        ("source_video_id", pa.string()),
        ("clip_start_sec", pa.float64()),
        ("duration_sec", pa.float64()),
        ("width", pa.int64()),
        ("height", pa.int64()),
        ("fps", pa.float64()),
        ("num_frames", pa.int64()),
        ("source_video_url", pa.string()),
        ("caption", pa.string()),
    ])

    arrays = {col: [r[col] for r in records] for col in [f.name for f in schema]}
    table = pa.table(arrays, schema=schema)
    parquet_path = output_dir / "dataset.parquet"
    pq.write_table(table, parquet_path)

    # Write manifest.json
    manifest = {
        "protocol_version": "2.0.0",
        "schema_version": "2.0.0",
        "spec_id": "video_v1",
        "netuid": 70,
        "miner_hotkey": "test_miner",
        "interval_id": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "record_count": len(records),
        "dataset_sha256": sha256_file(parquet_path),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Also keep a simple manifest.jsonl for training pipelines
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "id": r["clip_id"],
                "video": r["clip_uri"],
                "prompt": r["caption"],
            }, ensure_ascii=False) + "\n")

    print(f"\nDone. Output: {output_dir}")
    print(f"Records: {len(records)}")
    print(f"Clips: {len(list(clips_dir.glob('*.mp4')))}")
    print(f"Frames: {len(list(frames_dir.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
