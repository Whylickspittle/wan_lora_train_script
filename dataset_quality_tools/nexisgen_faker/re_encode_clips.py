#!/usr/bin/env python3
"""Re-encode clips in parallel to change file content (and thus SHA256).

Re-encodes every MP4 in clips/ with ffmpeg using 6 parallel workers,
regenerates first frames, recomputes SHA256 hashes, re-probes metadata,
and updates dataset.parquet.

Usage:
    cd /workspace/top400_combined_motion_captioned_test
    python re_encode_clips.py --dry-run          # preview only
    python re_encode_clips.py                    # process all clips (CRF=22)
    python re_encode_clips.py --crf 20 --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Thread-safe counter for progress logging.
_progress_lock = threading.Lock()
_progress_done = 0
_progress_total = 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    result = {
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "num_frames": 0,
        "duration_sec": 0.0,
    }
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            str(path),
        ]
        proc = subprocess.run(cmd, check=True, timeout=60, capture_output=True, text=True)
        data = json.loads(proc.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                result["width"] = int(stream.get("width", 0))
                result["height"] = int(stream.get("height", 0))
                duration = stream.get("duration") or data.get("format", {}).get("duration")
                if duration:
                    result["duration_sec"] = float(duration)

                fps_str = stream.get("r_frame_rate", "0/1")
                try:
                    num, den = fps_str.split("/")
                    result["fps"] = float(num) / float(den) if float(den) != 0 else 0.0
                except Exception:
                    result["fps"] = 0.0

                nb_frames = stream.get("nb_frames")
                if nb_frames:
                    result["num_frames"] = int(nb_frames)
                elif result["fps"] > 0 and result["duration_sec"] > 0:
                    result["num_frames"] = int(round(result["duration_sec"] * result["fps"]))
                break
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
    return result


def re_encode_clip(src: Path, dst: Path, *, crf: int = 22) -> bool:
    """Re-encode a clip to 1280x704 @ 24fps, 121 frames, H.264."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-r",
        "24",
        "-s",
        "1280x704",
        "-frames:v",
        "121",
        "-pix_fmt",
        "yuv420p",
        "-an",  # no audio
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
        return True
    except Exception as exc:
        logger.error("ffmpeg re-encode failed for %s: %s", src, exc)
        return False


def extract_first_frame(video: Path, frame: Path) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vf",
        "select=eq(n\\,0)",
        "-q:v",
        "2",
        str(frame),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60, capture_output=True)
        return True
    except Exception as exc:
        logger.error("ffmpeg frame extract failed for %s: %s", video, exc)
        return False


def process_one_clip(
    idx: int,
    row: pd.Series,
    dataset_dir: Path,
    crf: int,
    max_retries: int = 2,
) -> dict[str, Any] | None:
    """Process a single clip in a thread-safe way with retries."""
    global _progress_done, _progress_total

    clip_uri = row["clip_uri"]
    frame_uri = row["first_frame_uri"]
    src_clip = dataset_dir / clip_uri
    final_clip = dataset_dir / clip_uri
    final_frame = dataset_dir / frame_uri

    # Each worker gets its own temp directory to avoid conflicts.
    with tempfile.TemporaryDirectory(prefix=f"reencode_{idx}_") as tmpdir:
        dst_clip_tmp = Path(tmpdir) / Path(clip_uri).name

        # Retry re-encode on failure.
        re_encode_ok = False
        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(
                    "[%d] Re-encode retry %d/%d for %s",
                    idx,
                    attempt,
                    max_retries,
                    clip_uri,
                )
            if re_encode_clip(src_clip, dst_clip_tmp, crf=crf):
                re_encode_ok = True
                break

        if not re_encode_ok:
            logger.error("Re-encode failed after %d attempts for %s", max_retries + 1, src_clip)
            return None

        shutil.move(str(dst_clip_tmp), str(final_clip))

        # Retry frame extraction on failure.
        frame_ok = False
        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(
                    "[%d] Frame extract retry %d/%d for %s",
                    idx,
                    attempt,
                    max_retries,
                    clip_uri,
                )
            if extract_first_frame(final_clip, final_frame):
                frame_ok = True
                break

        if not frame_ok:
            logger.error("Frame extract failed after %d attempts for %s", max_retries + 1, final_clip)
            return None

        probe = probe_video(final_clip)

        updated_row = row.to_dict()
        updated_row["clip_sha256"] = sha256_file(final_clip)
        updated_row["first_frame_sha256"] = sha256_file(final_frame)
        updated_row["duration_sec"] = probe["duration_sec"]
        updated_row["width"] = probe["width"]
        updated_row["height"] = probe["height"]
        updated_row["fps"] = probe["fps"]
        updated_row["num_frames"] = probe["num_frames"]

    with _progress_lock:
        _progress_done += 1
        logger.info(
            "[%d/%d] Done: %s",
            _progress_done,
            _progress_total,
            clip_uri,
        )

    return updated_row


def main() -> None:
    global _progress_total

    parser = argparse.ArgumentParser(description="Re-encode clips in parallel to change SHA256.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Dataset directory containing dataset.parquet, clips/, frames/.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=22,
        help="H.264 CRF value (lower = higher quality, default 22).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel ffmpeg workers (default 6).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries per clip when ffmpeg fails (default 2).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without re-encoding.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N clips (for testing).",
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

    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from %s", len(df), parquet_path)

    rows_to_process = df if args.limit is None else df.head(args.limit)
    _progress_total = len(rows_to_process)

    if args.dry_run:
        logger.info(
            "DRY RUN: would re-encode %d clips with CRF=%d using %d workers",
            len(rows_to_process),
            args.crf,
            args.workers,
        )
        for _, row in rows_to_process.iterrows():
            logger.info("  %s", row["clip_uri"])
        return

    logger.info(
        "Starting parallel re-encode: %d clips, CRF=%d, workers=%d, max_retries=%d",
        len(rows_to_process),
        args.crf,
        args.workers,
        args.max_retries,
    )

    results: list[dict[str, Any] | None] = [None] * len(rows_to_process)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(process_one_clip, idx, row, dataset_dir, args.crf, args.max_retries): idx
            for idx, (_, row) in enumerate(rows_to_process.iterrows())
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result
            except Exception as exc:
                logger.exception("Worker failed for row %d: %s", idx, exc)

    failed = sum(1 for r in results if r is None)
    if failed > 0:
        raise SystemExit(f"{failed} clips failed to re-encode")

    updated_df = pd.DataFrame([r for r in results if r is not None])
    if args.limit is not None and args.limit < len(df):
        df.loc[updated_df.index, updated_df.columns] = updated_df.values
    else:
        df = updated_df

    df.to_parquet(parquet_path, index=False)
    logger.info("Updated %s", parquet_path)

    sample = df[["clip_id", "clip_sha256", "first_frame_sha256", "duration_sec", "width", "height", "fps", "num_frames"]].head(3)
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
