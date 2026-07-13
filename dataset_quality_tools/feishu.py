#!/usr/bin/env python3
"""Feishu/Lark bot notification helper.

Reads the webhook URL from ``dataset_quality_tools/.env`` (``LARK_URL``).
This keeps secrets out of source control.
"""

from __future__ import annotations

import json
import os
import pprint
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

_ENV_PATH = Path(__file__).resolve().parent / ".env"


def _load_env_file(path: Path) -> None:
    """Load a simple KEY=VALUE env file into os.environ."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file(_ENV_PATH)


def _get_lark_url() -> str:
    """Return the Lark/Feishu webhook URL from environment."""
    return os.environ.get("LARK_URL", os.environ.get("lark_url", ""))


def lark_bot(
    message_type: str,
    message: str | dict[Any, Any],
    lark_url: str | None = None,
    timeout: float = 30.0,
) -> None:
    """Send a notification to the configured Feishu/Lark bot.

    Args:
        message_type: One of ``info``, ``warning``, ``error``.
        message: Text or dict to send.
        lark_url: Optional webhook URL override. If omitted, read from env.
        timeout: HTTP POST timeout in seconds.
    """
    if lark_url is None:
        lark_url = _get_lark_url()

    if not lark_url:
        print("Warning: LARK_URL not configured, skipping Feishu notification.")
        return

    shanghai = timezone(timedelta(hours=8))
    current_time = datetime.now(shanghai).strftime("%Y-%m-%d %H:%M:%S %Z")

    if isinstance(message, dict):
        message = pprint.pformat(message)

    message = f"Time: {current_time}\nwebnx54 完成任务\n{message}"

    if message_type == "info":
        message = "=" * 10 + " Info " + "=" * 10 + "\n" + message
        data = {"msg_type": "text", "content": {"text": message}}
    elif message_type == "warning":
        message = message + '<at user_id="all">所有人</at>'
        message = "=" * 10 + " Warning! " + "=" * 10 + "\n" + message
        data = {"msg_type": "text", "content": {"text": message}}
    elif message_type == "error":
        message = message + '<at user_id="all">所有人</at>'
        message = "=" * 10 + " Error! " + "=" * 10 + "\n" + message
        data = {"msg_type": "text", "content": {"text": message}}
    else:
        raise ValueError(f"Unknown message_type: {message_type}")

    headers = {"Content-type": "application/json"}
    try:
        resp = requests.post(lark_url, headers=headers, data=json.dumps(data), timeout=timeout)
        resp.raise_for_status()
        print(f"Feishu notification sent: {message_type}")
    except Exception as exc:
        print(f"Failed to send Feishu notification: {exc}")


def notify_exception(message: str) -> None:
    """Convenience helper: notify an error with full traceback."""
    tb = traceback.format_exc()
    lark_bot("error", f"{message}\n\n{tb}")


if __name__ == "__main__":
    lark_bot("info", "Feishu bot test message")
