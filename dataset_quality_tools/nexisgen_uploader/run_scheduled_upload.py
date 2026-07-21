#!/usr/bin/env python3
"""Scheduled runner for recurring Nexisgen dataset generation + R2 upload.

This script is meant to be triggered by Linux cron or systemd timer. On each
invocation it reads a local state file, generates a new interval dataset,
uploads it to the configured R2 bucket, and increments the state.

Usage:
    # Production: run once, upload to hotkey-named bucket
    python run_scheduled_upload.py --hotkey 5xxx... --env-file /workspace/nexisgen/.env

    # Test: run twice then stop
    python run_scheduled_upload.py --hotkey 5xxx... --bucket test01 --max-runs 2

State file (.schedule_state.json):
    {
      "next_interval_id": 2,
      "run_count": 0,
      "last_run_at": "2026-07-20T08:28:00Z"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from setup_r2_bucket import ensure_bucket

logger = logging.getLogger("run_scheduled_upload")


def load_state(state_file: Path) -> dict[str, Any]:
    if not state_file.exists():
        return {"next_interval_id": 2, "run_count": 0, "last_run_at": ""}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("failed to read state file: %s", exc)
        return {"next_interval_id": 2, "run_count": 0, "last_run_at": ""}


def save_state(state: dict[str, Any], state_file: Path) -> None:
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def prepare_interval_dir(interval_id: int, source_interval_dir: Path, data_dir: Path) -> Path:
    """Copy clips/frames from source interval and return the new interval dir."""
    interval_dir = data_dir / f"interval_{interval_id}"
    clips_dir = interval_dir / "clips"
    frames_dir = interval_dir / "frames"

    if interval_dir.exists():
        shutil.rmtree(interval_dir)
    interval_dir.mkdir(parents=True)

    shutil.copytree(source_interval_dir / "clips", clips_dir)
    shutil.copytree(source_interval_dir / "frames", frames_dir)
    logger.info("prepared %s from %s", interval_dir, source_interval_dir)
    return interval_dir


def generate_dataset(
    interval_id: int,
    interval_dir: Path,
    hotkey: str,
    script_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        str(script_dir / "generate_test_dataset.py"),
        f"--interval={interval_id}",
        f"--dataset-dir={interval_dir}",
        f"--hotkey={hotkey}",
    ]
    logger.info("generating dataset: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def upload_dataset(
    interval_id: int,
    interval_dir: Path,
    hotkey: str,
    bucket: str,
    script_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        str(script_dir / "upload_test_miner_dataset.py"),
        f"--interval={interval_id}",
        f"--dataset-dir={interval_dir}",
        f"--hotkey={hotkey}",
        f"--bucket={bucket}",
        "--max-workers=4",
    ]
    logger.info("uploading dataset: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scheduled Nexisgen dataset upload runner")
    parser.add_argument("--hotkey", type=str, required=True, help="Miner hotkey SS58 address")
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="R2 bucket name (default: lowercase hotkey)",
    )
    parser.add_argument(
        "--source-interval-dir",
        type=Path,
        default=None,
        help="Directory containing source clips/frames to copy from (default: interval_1 next to this script)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory where interval_* folders are created (default: same dir as this script)",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Path to schedule state JSON (default: .schedule_state.json next to this script)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file with R2 credentials (default: .env in current directory)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Stop after N total runs (0 = unlimited, useful for testing)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir
    source_interval_dir = args.source_interval_dir or script_dir / "interval_1"
    state_file = args.state_file or script_dir / ".schedule_state.json"
    bucket = args.bucket if args.bucket else args.hotkey.strip().lower()

    if not source_interval_dir.exists():
        logger.error("source interval dir not found: %s", source_interval_dir)
        return 1

    state = load_state(state_file)
    logger.info("loaded state: %s", state)

    if args.max_runs > 0 and state.get("run_count", 0) >= args.max_runs:
        logger.info("max runs (%d) reached, exiting", args.max_runs)
        return 0

    interval_id = state.get("next_interval_id", 2)
    logger.info("starting upload for interval_id=%d", interval_id)

    try:
        # Make sure the target bucket exists before generating/uploading data.
        if not ensure_bucket(bucket, args.env_file):
            logger.error("bucket setup failed for %s", bucket)
            return 1

        interval_dir = prepare_interval_dir(interval_id, source_interval_dir, data_dir)
        generate_dataset(interval_id, interval_dir, args.hotkey, script_dir)
        upload_dataset(interval_id, interval_dir, args.hotkey, bucket, script_dir)

        # Update state
        state["next_interval_id"] = interval_id + 1
        state["run_count"] = state.get("run_count", 0) + 1
        state["last_run_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        save_state(state, state_file)

        logger.info(
            "completed interval_id=%d, next_interval_id=%d, run_count=%d",
            interval_id,
            state["next_interval_id"],
            state["run_count"],
        )
        return 0

    except Exception as exc:
        logger.exception("scheduled upload failed for interval_id=%d: %s", interval_id, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
