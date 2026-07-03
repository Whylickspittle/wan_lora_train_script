#!/usr/bin/env python3
"""Run the Pexels download pipeline and notify via Feishu/Lark.

This wrapper is intentionally thin: it builds the command, runs the pipeline
as a subprocess, and sends Feishu notifications on start / success / failure.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from feishu import lark_bot

API_KEY_PATH = Path("/workspace/pexel_api")
ENV_PATH = Path(os.environ.get("WORKSPACE", "/workspace")) / ".env"
OUTPUT = Path("./pexels_nature_batch07")
LOG_FILE = Path("./pexels_nature_batch07.log")


def load_pexels_api_key() -> str:
    """Load Pexels API key from .env (PEXELS_API_KEY) or fallback to legacy file."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "PEXELS_API_KEY":
                token = value.strip().strip('"\'')
                if token:
                    return token
    if API_KEY_PATH.exists():
        token = API_KEY_PATH.read_text(encoding="utf-8").strip()
        if token:
            return token
    raise RuntimeError(
        f"Pexels API key not found. Set PEXELS_API_KEY in {ENV_PATH} "
        f"or create {API_KEY_PATH}."
    )


def main() -> int:
    try:
        api_key = load_pexels_api_key()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "download_pexels_quality_pipeline.py",
        "--api-key",
        api_key,
        "--query",
        "4k nature scenery drone",
        "--count",
        "150",
        "--min-height",
        "2160",
        "--min-fps",
        "24",
        "--clips-per-video",
        "3",
        "--output",
        str(OUTPUT),
        "--history-file",
        "./pexels_download_history.json",
        "--quarantine",
        "--run-diagnostics",
    ]

    start_time = datetime.now().isoformat(timespec="seconds")
    lark_bot(
        "info",
        f"Pexels pipeline started\n"
        f"Query: 4k nature scenery drone\n"
        f"Target: 150 raw videos, 3 clips each\n"
        f"Output: {OUTPUT.resolve()}\n"
        f"Start: {start_time}",
    )

    with LOG_FILE.open("w", encoding="utf-8") as log_fh:
        try:
            result = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            msg = (
                f"Pexels pipeline failed with exit code {exc.returncode}\n"
                f"Output: {OUTPUT.resolve()}\n"
                f"Log: {LOG_FILE.resolve()}"
            )
            lark_bot("error", msg)
            print(msg, file=sys.stderr)
            return exc.returncode
        except Exception as exc:
            msg = (
                f"Pexels pipeline crashed: {exc}\n"
                f"Output: {OUTPUT.resolve()}\n"
                f"Log: {LOG_FILE.resolve()}"
            )
            lark_bot("error", msg)
            print(msg, file=sys.stderr)
            return 1
        else:
            end_time = datetime.now().isoformat(timespec="seconds")
            msg = (
                f"Pexels pipeline completed successfully\n"
                f"Output: {OUTPUT.resolve()}\n"
                f"End: {end_time}\n"
                f"Log: {LOG_FILE.resolve()}"
            )
            lark_bot("info", msg)
            print(msg)
            return result.returncode


if __name__ == "__main__":
    sys.exit(main())
