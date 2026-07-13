#!/usr/bin/env python3
"""Perceptual-hash deduplication for a Nexisgen-format dataset.

Groups clips by ``source_video_id`` (i.e. the same Pexels/original video),
sorts by ``clip_start_sec``, and removes later clips that are too visually
similar to an earlier kept clip from the same source.

The removed clips are written to a separate TBD folder so they can be
reviewed or used later instead of being deleted.

Similarity is based on dHash (difference hash) of three frames per clip:
first, middle, and last. The average Hamming distance across the three frames
must exceed ``--threshold`` for a clip to be kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Pillow is required. Install it with:\n  pip install Pillow>=10.0.0"
    ) from exc

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


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute a 64-bit difference hash (dHash) for an image."""
    # Resize to (hash_size + 1) x hash_size and convert to grayscale.
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = _get_pixels(gray)
    width = hash_size + 1

    bits: list[str] = []
    for row in range(hash_size):
        row_start = row * width
        for col in range(hash_size):
            left = pixels[row_start + col]
            right = pixels[row_start + col + 1]
            bits.append("1" if left > right else "0")
    return int("".join(bits), 2)


def _get_pixels(gray: Image.Image) -> list[int]:
    """Flattened grayscale pixels, using Pillow's modern API when available."""
    getter = getattr(gray, "get_flattened_data", None)
    if getter is not None:
        return list(getter())
    return list(gray.getdata())


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def extract_frame(video_path: Path, frame_index: int, output_path: Path, timeout: int = 60) -> None:
    """Extract a single frame by index using ffmpeg."""
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
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, timeout=timeout)


def compute_clip_hashes(
    clip_path: Path,
    first_frame_path: Path,
    cache_dir: Path,
    hash_size: int = 8,
) -> tuple[int, int, int] | None:
    """Return (first_frame_hash, middle_frame_hash, last_frame_hash)."""
    if not clip_path.exists() or not first_frame_path.exists():
        return None

    clip_id = clip_path.stem
    middle_path = cache_dir / f"{clip_id}_middle.jpg"
    last_path = cache_dir / f"{clip_id}_last.jpg"

    # Middle and last frame indices for a 121-frame clip.
    if not middle_path.exists():
        extract_frame(clip_path, 60, middle_path)
    if not last_path.exists():
        extract_frame(clip_path, 120, last_path)

    try:
        first_hash = dhash(Image.open(first_frame_path), hash_size)
        middle_hash = dhash(Image.open(middle_path), hash_size)
        last_hash = dhash(Image.open(last_path), hash_size)
        return first_hash, middle_hash, last_hash
    except Exception as exc:
        logger.warning("failed to hash frames for %s: %s", clip_id, exc)
        return None


def average_hash_distance(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
) -> float:
    return (
        hamming_distance(a[0], b[0])
        + hamming_distance(a[1], b[1])
        + hamming_distance(a[2], b[2])
    ) / 3.0


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    columns = {col: table.column(col).to_pylist() for col in NEXIS_COLUMNS}
    return [
        {col: columns[col][i] for col in NEXIS_COLUMNS}
        for i in range(table.num_rows)
    ]


def write_dataset(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "clips"
    frames_dir = output_dir / "frames"
    clips_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, list[Any]] = {col: [] for col in NEXIS_COLUMNS}
    for record in records:
        # Copy clip and frame into output dir if not already there.
        src_clip = Path(record["_src_dir"]) / record["clip_uri"]
        src_frame = Path(record["_src_dir"]) / record["first_frame_uri"]
        dst_clip = clips_dir / Path(record["clip_uri"]).name
        dst_frame = frames_dir / Path(record["first_frame_uri"]).name
        if src_clip.exists():
            shutil.copy2(src_clip, dst_clip)
        if src_frame.exists():
            shutil.copy2(src_frame, dst_frame)

        # Update sha256 for copied files.
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

    # Also emit a captioned manifest (prompt field) when captions are present.
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
        description="Deduplicate a Nexisgen dataset by perceptual hash within each source video."
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
        help="Output directory for the refined (kept) dataset.",
    )
    parser.add_argument(
        "--tbd-dir",
        required=True,
        type=Path,
        help="Output directory for clips removed as duplicates.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.0,
        help="Max average Hamming distance (dHash) to consider two clips duplicates (default: 8).",
    )
    parser.add_argument(
        "--hash-size",
        type=int,
        default=8,
        help="dHash size; hash length = hash_size * hash_size bits (default: 8 = 64 bits).",
    )
    parser.add_argument(
        "--keep-tbd-captions",
        action="store_true",
        help="Keep caption field in TBD dataset (default: empty captions in TBD).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    records = read_parquet_records(args.input_dir / "dataset.parquet")
    logger.info("read %d records from %s", len(records), args.input_dir)

    # Group by source_video_id.
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        src_id = str(record["source_video_id"])
        groups.setdefault(src_id, []).append(record)

    # Sort each group by start time and add internal helper.
    for src_id in groups:
        groups[src_id].sort(key=lambda r: float(r["clip_start_sec"]))
        for record in groups[src_id]:
            record["_src_dir"] = str(args.input_dir)

    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="dedup_frames_") as tmp:
        cache_dir = Path(tmp)

        for src_id, group in groups.items():
            last_kept_hashes: tuple[int, int, int] | None = None
            for record in group:
                clip_path = args.input_dir / str(record["clip_uri"])
                frame_path = args.input_dir / str(record["first_frame_uri"])
                hashes = compute_clip_hashes(
                    clip_path, frame_path, cache_dir, hash_size=args.hash_size
                )
                if hashes is None:
                    logger.warning("[%s] could not compute hashes; keeping it", record["clip_id"])
                    kept.append(record)
                    last_kept_hashes = None
                    continue

                if last_kept_hashes is None:
                    kept.append(record)
                    last_kept_hashes = hashes
                    logger.info("[%s] kept (first from source %s)", record["clip_id"], src_id)
                    continue

                distance = average_hash_distance(hashes, last_kept_hashes)
                if distance <= args.threshold:
                    if not args.keep_tbd_captions:
                        record["caption"] = ""
                    removed.append(record)
                    logger.info(
                        "[%s] removed (distance=%.1f <= threshold=%.1f from source %s)",
                        record["clip_id"],
                        distance,
                        args.threshold,
                        src_id,
                    )
                else:
                    kept.append(record)
                    last_kept_hashes = hashes
                    logger.info(
                        "[%s] kept (distance=%.1f > threshold=%.1f from source %s)",
                        record["clip_id"],
                        distance,
                        args.threshold,
                        src_id,
                    )

    write_dataset(kept, args.output_dir)
    write_dataset(removed, args.tbd_dir)

    logger.info(
        "refined dataset: %s (%d kept)",
        args.output_dir,
        len(kept),
    )
    logger.info(
        "TBD dataset: %s (%d removed)",
        args.tbd_dir,
        len(removed),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
