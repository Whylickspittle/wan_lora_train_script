from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    data["_config_path"] = str(config_path)
    return data


def namespace_from_config(config: dict[str, Any], defaults: argparse.Namespace | None = None) -> argparse.Namespace:
    values = vars(defaults).copy() if defaults is not None else {}
    values.update(config)
    return argparse.Namespace(**values)


def load_namespace(config_path: str | Path, defaults: argparse.Namespace | None = None) -> argparse.Namespace:
    return namespace_from_config(load_json_config(config_path), defaults)
