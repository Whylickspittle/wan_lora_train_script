from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download

from customer_workflow import CONFIG_PATH, load_customer_config


def save_config(config: dict) -> None:
    backup = CONFIG_PATH.with_suffix(".json.bak")
    if CONFIG_PATH.exists() and not backup.exists():
        backup.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    config = load_customer_config()
    download = config.get("model_download", {})
    source_model_id = download.get("source_model_id") or config.get("model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    local_dir = Path(download.get("local_dir", ROOT / "models" / "Wan2.2-TI2V-5B-Diffusers"))
    revision = download.get("revision", "main")
    allow_patterns = download.get("allow_patterns")
    ignore_patterns = download.get("ignore_patterns")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print("Downloading or verifying Wan2.2 model files.")
    print(f"Source: {source_model_id}")
    print(f"Local folder: {local_dir}")
    print("This may take a while the first time. Later runs reuse the local files.")

    resolved = snapshot_download(
        repo_id=source_model_id,
        revision=revision,
        local_dir=str(local_dir),
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )

    summary = {
        "source_model_id": source_model_id,
        "revision": revision,
        "local_dir": str(local_dir),
        "resolved_path": resolved,
        "used_token": bool(token),
    }
    summary_path = Path(__file__).resolve().parent / "model_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    config.setdefault("model_download", download)
    config["model_download"]["resolved_path"] = resolved
    if download.get("use_local_model_after_download", True):
        config["model_id"] = str(local_dir)
        save_config(config)
        print(f"Updated h100_dataset_training/config.json so model_id points to: {local_dir}")

    print(f"Model download summary written to: {summary_path}")


if __name__ == "__main__":
    main()
