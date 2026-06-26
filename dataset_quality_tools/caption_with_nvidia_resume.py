#!/usr/bin/env python3
"""Resume NVIDIA caption generation for missing prompts.

Reads an existing captioned manifest and re-calls the NVIDIA NIM API only for
rows whose ``prompt`` field is empty. This avoids re-billing for already-
captioned clips and keeps the output order stable.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from caption_with_nvidia import NvidiaCaptioner, extract_first_frame

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    _SCRIPT_DIR = Path(__file__).resolve().parent
    load_dotenv(_SCRIPT_DIR / ".env", override=False)
    load_dotenv(override=False)

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume NVIDIA captioning for missing prompts."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Original manifest.jsonl (used for ordering and frame paths).",
    )
    parser.add_argument(
        "--captioned-manifest",
        required=True,
        type=Path,
        help="Already-captioned manifest.jsonl (will be updated in place).",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="NVIDIA API key (or set NVIDIA_API_KEY env var).",
    )
    parser.add_argument(
        "--model",
        default="moonshotai/kimi-k2.6",
        help="NVIDIA NIM model name.",
    )
    parser.add_argument(
        "--base-url",
        default="https://integrate.api.nvidia.com/v1",
        help="OpenAI-compatible base URL for NVIDIA NIM.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="API call timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Seconds to sleep between requests.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    api_key = args.api_key or __import__("os").getenv("NVIDIA_API_KEY", "")
    if not api_key.strip():
        logger.error("No API key provided. Set NVIDIA_API_KEY or pass --api-key.")
        return 1

    base_dir = args.manifest.parent.resolve()
    frames_dir = base_dir / "frames"

    original_rows = _read_jsonl(args.manifest)
    existing_rows = _read_jsonl(args.captioned_manifest)
    existing_by_id = {r["id"]: r for r in existing_rows}

    captioner = NvidiaCaptioner(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        timeout_sec=args.timeout,
    )

    missing = [
        r
        for r in original_rows
        if not existing_by_id.get(r["id"], {}).get("prompt", "").strip()
    ]
    logger.info(
        "total=%d already_captioned=%d missing=%d",
        len(original_rows),
        len(original_rows) - len(missing),
        len(missing),
    )

    succeeded = 0
    for i, row in enumerate(missing, start=1):
        clip_id = row["id"]
        frame_path = (frames_dir / f"{clip_id}.jpg").resolve()
        if not frame_path.exists():
            video_path = base_dir / row.get("video", f"clips/{clip_id}.mp4")
            try:
                extract_first_frame(video_path, frame_path)
            except Exception as exc:
                logger.warning("[%s] frame extraction failed: %s", clip_id, exc)
                continue

        caption = captioner.caption_frame(frame_path)
        if caption:
            existing_by_id[clip_id]["prompt"] = caption
            logger.info("[%d/%d] %s: %s", i, len(missing), clip_id, caption)
            succeeded += 1
        else:
            logger.warning("[%d/%d] %s still empty", i, len(missing), clip_id)

        if args.sleep > 0 and i < len(missing):
            time.sleep(args.sleep)

    final_rows = [existing_by_id[r["id"]] for r in original_rows]
    _write_jsonl(args.captioned_manifest, final_rows)
    logger.info(
        "wrote %d rows to %s (new captions: %d/%d)",
        len(final_rows),
        args.captioned_manifest,
        succeeded,
        len(missing),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
