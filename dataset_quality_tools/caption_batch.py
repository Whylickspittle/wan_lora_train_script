#!/usr/bin/env python3
"""Batch caption a directory of video clips.

Outputs a JSONL file where each line contains the clip path and both the
single-frame and multi-frame captions, useful for comparing the two modes.

Example:

    python caption_batch.py \
        --input_dir /path/to/clips \
        --output captions.jsonl \
        --num_keyframes 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from video_captioner import Captioner, load_env

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _find_clips(input_dir: Path) -> list[Path]:
    clips: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        clips.extend(sorted(input_dir.glob(f"*{ext}")))
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch caption video clips.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing video clips.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--num_keyframes", type=int, default=4, help="Number of keyframes for multi-frame caption.")
    parser.add_argument("--workdir", type=Path, default=Path(".caption_workdir"), help="Working directory for extracted frames.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    env = load_env(".env")
    if "OPENAI_API_KEY" not in env:
        raise SystemExit("OPENAI_API_KEY not found in .env")

    captioner = Captioner(
        api_key=env["OPENAI_API_KEY"],
        base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=env.get("NEXIS_CAPTION_MODEL", "gpt-4o-mini"),
        timeout_sec=120,
    )

    clips = _find_clips(args.input_dir)
    if not clips:
        raise SystemExit(f"No video clips found in {args.input_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.workdir.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        for idx, clip_path in enumerate(clips, start=1):
            logger.info("[%d/%d] Captioning %s", idx, len(clips), clip_path.name)
            clip_workdir = args.workdir / clip_path.stem
            clip_workdir.mkdir(parents=True, exist_ok=True)

            single_caption = captioner.caption_video(
                clip_path,
                workdir=clip_workdir / "single",
                num_keyframes=1,
            )
            multi_caption = captioner.caption_video(
                clip_path,
                workdir=clip_workdir / "multi",
                num_keyframes=args.num_keyframes,
            )

            row: dict[str, Any] = {
                "video": str(clip_path),
                "single_frame_caption": single_caption,
                "multi_frame_caption": multi_caption,
                "num_keyframes": args.num_keyframes,
            }
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            fh.flush()

    logger.info("Wrote %d captions to %s", len(clips), args.output)


if __name__ == "__main__":
    main()
