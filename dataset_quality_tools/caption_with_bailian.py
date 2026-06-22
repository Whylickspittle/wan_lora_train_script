#!/usr/bin/env python3
"""Standalone Bailian/DashScope caption annotator for video clip manifests.

Reads a ``manifest.jsonl`` (for example produced by
``download_pexels_quality_pipeline.py``), extracts the first frame of every
video, asks an Aliyun Bailian vision model for a short prompt-style caption,
and writes a new manifest with the caption stored in the ``prompt`` field.

The implementation mirrors ``nexisgen/nexis/miner/captioner.py``:
* it sends a single first-frame image to the model;
* it uses the same system prompt so generated captions are suitable for
  text-to-video training.

Environment variables
---------------------
DASHSCOPE_API_KEY
    Required unless ``--api-key`` is passed.
NEXIS_CAPTION_MODEL
    Default model name (default: ``qwen3.5-omni-plus``).
NEXIS_CAPTION_TIMEOUT_SEC
    API call timeout in seconds (default: 30).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "The 'openai' package is required. Install it with:\n"
        "  pip install openai>=1.40.0"
    ) from exc

# Load .env from the script's directory first, then the current working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(_SCRIPT_DIR / ".env", override=False)
    load_dotenv(override=False)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("NEXIS_CAPTION_MODEL", "qwen3.5-omni-plus")
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TIMEOUT_SEC = int(os.getenv("NEXIS_CAPTION_TIMEOUT_SEC", "30"))

_PROMPT = (
    "Describe this video frame in one short sentence (≤ 20 words) that would "
    "work as a text-to-video generation prompt. Focus on subject, setting, and "
    "motion cues. Do not add commentary."
)


def _b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def extract_first_frame(video_path: Path, output_path: Path, timeout: int = 60) -> None:
    """Extract the first frame of ``video_path`` to ``output_path``."""
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


class BailianCaptioner:
    """OpenAI-compatible client targeting Aliyun Bailian/DashScope."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        if not api_key.strip():
            raise ValueError("api_key is required")
        self.model = model
        self.timeout_sec = timeout_sec
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(timeout_sec),
        )

    def caption_frame(self, frame_path: Path) -> str:
        """Return a caption for a single image, or an empty string on failure."""
        if not frame_path.exists():
            logger.warning("frame not found: %s", frame_path)
            return ""
        try:
            data_url = f"data:image/jpeg;base64,{_b64_image(frame_path)}"
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                max_tokens=80,
            )
            return (resp.choices[0].message.content or "").strip()[:300]
        except Exception as exc:
            logger.warning("caption call failed frame=%s err=%s", frame_path, exc)
            return ""


def resolve_video_path(
    row: dict[str, Any],
    manifest_dir: Path,
    clips_dir: Path,
) -> Path:
    """Resolve the video path for a manifest row.

    Relative paths are interpreted relative to ``clips_dir`` (which defaults to
    the manifest directory, matching the Pexels pipeline layout).
    """
    video = str(row.get("video", "")).strip()
    if not video:
        raise ValueError("row has no 'video' field")
    video_path = Path(video)
    if not video_path.is_absolute():
        video_path = clips_dir / video_path
    return video_path.resolve()


def _caption_one(
    args: argparse.Namespace,
    captioner: BailianCaptioner,
    row: dict[str, Any],
    manifest_dir: Path,
    clips_dir: Path,
    frames_dir: Path,
) -> dict[str, Any]:
    """Caption a single row. Returns the (possibly updated) row."""
    try:
        video_path = resolve_video_path(row, manifest_dir, clips_dir)
    except ValueError as exc:
        logger.warning("skipping row: %s", exc)
        return row

    if not video_path.exists():
        logger.warning("video not found: %s", video_path)
        return row

    clip_id = row.get("id") or video_path.stem
    frame_path = (frames_dir / f"{clip_id}.jpg").resolve()

    if not frame_path.exists():
        try:
            extract_first_frame(video_path, frame_path)
        except Exception as exc:
            logger.warning("failed to extract frame for %s: %s", clip_id, exc)
            return row

    caption = captioner.caption_frame(frame_path)
    if caption:
        row["prompt"] = caption
        logger.info("[%s] %s", clip_id, caption)
    else:
        logger.warning("[%s] empty caption", clip_id)

    if args.sleep > 0:
        time.sleep(args.sleep)
    return row


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Caption video clips using Bailian/DashScope vision model."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Input manifest.jsonl (must contain a 'video' field per row).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output manifest.jsonl. Defaults to <manifest>.captioned.jsonl.",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=None,
        help="Directory containing clips. Defaults to the manifest directory.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory to cache first frames. Defaults to <manifest_dir>/frames.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("DASHSCOPE_API_KEY", ""),
        help="DashScope API key (or set DASHSCOPE_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Bailian model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="OpenAI-compatible base URL for Bailian.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="API call timeout in seconds.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum concurrent caption requests.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between requests (per-worker).",
    )
    parser.add_argument(
        "--update-in-place",
        action="store_true",
        help="Overwrite the input manifest instead of writing a new file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.api_key.strip():
        logger.error(
            "No API key provided. Set DASHSCOPE_API_KEY or pass --api-key."
        )
        return 1

    if not shutil.which("ffmpeg"):
        logger.error("ffmpeg is required but not found on PATH.")
        return 1

    manifest_dir = args.manifest.parent.resolve()
    clips_dir = (args.clips_dir or manifest_dir).resolve()
    frames_dir = (args.frames_dir or (manifest_dir / "frames")).resolve()
    output_path = (
        args.manifest
        if args.update_in_place
        else (args.output or args.manifest.with_suffix(".captioned.jsonl"))
    )

    captioner = BailianCaptioner(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout_sec=args.timeout,
    )

    rows = _read_manifest(args.manifest)
    logger.info("read %d rows from %s", len(rows), args.manifest)

    if args.max_workers <= 1:
        results = [
            _caption_one(
                args, captioner, row, manifest_dir, clips_dir, frames_dir
            )
            for row in rows
        ]
    else:
        results: list[dict[str, Any]] = list(rows)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_idx = {
                executor.submit(
                    _caption_one,
                    args,
                    captioner,
                    row,
                    manifest_dir,
                    clips_dir,
                    frames_dir,
                ): idx
                for idx, row in enumerate(rows)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.warning("caption task failed idx=%d err=%s", idx, exc)

    _write_manifest(output_path, results)
    logger.info("wrote %d rows to %s", len(results), output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
