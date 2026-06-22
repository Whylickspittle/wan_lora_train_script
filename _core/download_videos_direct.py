#!/usr/bin/env python3
"""Direct Pexels video downloader helpers.

This module is the missing companion for
``dataset_quality_tools/download_pexels_quality_pipeline.py``. It talks to the
Pexels Video API, selects the best available file, downloads it, and slices
clips with ffmpeg.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

PEXELS_API_BASE = "https://api.pexels.com/videos"

# Target clip spec aligned with the training pipeline.
TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
TARGET_FPS = 24.0
TARGET_NUM_FRAMES = 121
CLIP_DURATION = TARGET_NUM_FRAMES / TARGET_FPS  # 5.04 seconds


class PexelsError(Exception):
    """Raised for Pexels API or download errors."""

    pass


def search_videos(
    api_key: str,
    query: str,
    page: int = 1,
    per_page: int = 80,
) -> dict[str, Any]:
    """Search Pexels videos and return the JSON response.

    Args:
        api_key: Pexels API key.
        query: Search query string.
        page: 1-based page number.
        per_page: Results per page (max 80).

    Returns:
        Parsed JSON response from Pexels.

    Raises:
        requests.HTTPError: on non-2xx API responses.
    """
    headers = {"Authorization": api_key}
    params = {
        "query": query,
        "page": page,
        "per_page": per_page,
        "orientation": "landscape",
    }
    resp = requests.get(
        f"{PEXELS_API_BASE}/search",
        headers=headers,
        params=params,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def pick_best_file(
    video: dict[str, Any],
    min_height: int,
    min_fps: int,
) -> dict[str, Any] | None:
    """Pick the best video file that satisfies min_height and min_fps.

    Args:
        video: A video object from Pexels search results.
        min_height: Minimum acceptable height in pixels.
        min_fps: Minimum acceptable frame rate.

    Returns:
        Dict with keys ``height``, ``width``, ``fps``, ``link``,
        ``file_type``, or ``None`` if no file qualifies.
    """
    files = video.get("video_files", [])
    if not files:
        return None

    candidates = []
    for f in files:
        height = f.get("height", 0) or 0
        width = f.get("width", 0) or 0
        fps = f.get("fps", 0) or 0
        link = f.get("link", "")
        ftype = f.get("file_type", "")

        if height < min_height:
            continue
        if fps > 0 and fps < min_fps:
            continue
        if not link:
            continue

        candidates.append(
            {
                "height": height,
                "width": width,
                "fps": fps,
                "link": link,
                "file_type": ftype,
            }
        )

    if not candidates:
        return None

    # Prefer highest resolution, then highest fps.
    candidates.sort(key=lambda x: (x["height"], x["width"], x["fps"]), reverse=True)
    return candidates[0]


def download_file(
    url: str,
    save_path: Path,
    session: requests.Session,
    chunk_size: int = 8192,
    timeout: int = 300,
) -> None:
    """Stream-download ``url`` to ``save_path``.

    Args:
        url: Direct download URL.
        save_path: Destination file path.
        session: Requests session to use (carries auth/headers if any).
        chunk_size: Download chunk size in bytes.
        timeout: Request timeout in seconds.

    Raises:
        requests.HTTPError: on non-2xx responses.
        OSError: on filesystem errors.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    resp = session.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    with save_path.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)


def ffprobe_duration(path: Path) -> float:
    """Return video duration in seconds using ffprobe.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        subprocess.CalledProcessError: if ffprobe fails.
        ValueError: if ffprobe returns an empty duration.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    duration_str = result.stdout.strip()
    if not duration_str:
        raise ValueError(f"ffprobe returned empty duration for {path}")
    return float(duration_str)


def slice_clip(
    src_path: Path,
    dst_path: Path,
    start_sec: float,
    duration: float = CLIP_DURATION,
    width: int = TARGET_WIDTH,
    height: int = TARGET_HEIGHT,
    fps: float = TARGET_FPS,
) -> None:
    """Extract a clip with ffmpeg and normalize to target format.

    The output is resized to ``width x height`` with padding to preserve aspect
    ratio, encoded as yuv420p H.264 at the target fps, and audio is stripped.

    Args:
        src_path: Source video path.
        dst_path: Destination clip path.
        start_sec: Start time in seconds.
        duration: Clip duration in seconds.
        width: Output width.
        height: Output height.
        fps: Output frame rate.
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg will be looked up on PATH; raise a clear error if missing.
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        str(src_path),
        "-t",
        str(duration),
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p,fps={fps}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-an",
        str(dst_path),
    ]
    subprocess.run(cmd, check=True, timeout=600)
