#!/usr/bin/env python3
"""Resume NVIDIA caption generation with exponential backoff on 429.

Keeps already-captioned prompts and retries only missing ones. On rate-limit
(429) it backs off and retries the same clip up to ``--max-retries`` times.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'openai' package is required. Install it with:\n"
        "  pip install openai>=1.40.0"
    ) from exc

try:
    from openai import RateLimitError
except ImportError:
    RateLimitError = Exception  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

_PROMPT = (
    "Describe this video frame in one short sentence (≤ 20 words) that would "
    "work as a text-to-video generation prompt. Focus on subject, setting, and "
    "motion cues. Do not add commentary."
)


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


def _caption_one(client: OpenAI, model: str, frame_path: Path, timeout: int) -> str:
    b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    resp = client.chat.completions.create(
        model=model,
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
        timeout=timeout,
    )
    return (resp.choices[0].message.content or "").strip()[:300]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume NVIDIA captions with 429 backoff."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--captioned-manifest", required=True, type=Path)
    parser.add_argument("--api-key", default=os.getenv("NVIDIA_API_KEY", ""))
    parser.add_argument("--model", default="moonshotai/kimi-k2.6")
    parser.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-base", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    if not args.api_key.strip():
        logger.error("Set NVIDIA_API_KEY or pass --api-key.")
        return 1

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        max_retries=0,  # we handle retries manually for 429
    )

    base_dir = args.manifest.parent.resolve()
    frames_dir = base_dir / "frames"

    original_rows = _read_jsonl(args.manifest)
    existing_rows = _read_jsonl(args.captioned_manifest)
    existing_by_id = {r["id"]: r for r in existing_rows}

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
        frame_path = frames_dir / f"{clip_id}.jpg"
        if not frame_path.exists():
            logger.warning("[%d/%d] frame missing: %s", i, len(missing), clip_id)
            continue

        caption = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                caption = _caption_one(client, args.model, frame_path, args.timeout)
                if caption:
                    break
            except RateLimitError as exc:
                wait = args.backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "[%d/%d] 429 on %s (attempt %d/%d), sleeping %.0fs",
                    i,
                    len(missing),
                    clip_id,
                    attempt,
                    args.max_retries,
                    wait,
                )
                time.sleep(wait)
            except Exception as exc:
                logger.warning("[%d/%d] %s failed attempt %d: %s", i, len(missing), clip_id, attempt, exc)
                time.sleep(args.sleep)

        if caption:
            existing_by_id[clip_id]["prompt"] = caption
            logger.info("[%d/%d] %s: %s", i, len(missing), clip_id, caption)
            succeeded += 1
        else:
            logger.warning("[%d/%d] %s still empty after retries", i, len(missing), clip_id)

        # persist after every row so progress is not lost
        _write_jsonl(args.captioned_manifest, [existing_by_id[r["id"]] for r in original_rows])

        if args.sleep > 0 and i < len(missing):
            time.sleep(args.sleep)

    logger.info(
        "finished: %d new captions written (missing was %d)",
        succeeded,
        len(missing),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
