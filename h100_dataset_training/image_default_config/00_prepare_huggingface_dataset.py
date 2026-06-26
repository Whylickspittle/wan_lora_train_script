from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from customer_workflow import CONFIG_PATH, load_customer_config


def save_config(config: dict) -> None:
    backup = CONFIG_PATH.with_suffix(".json.bak")
    if CONFIG_PATH.exists() and not backup.exists():
        backup.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def clean_id(value: Any, fallback: str) -> str:
    text = str(value if value not in (None, "") else fallback)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return safe.strip("_") or fallback


def extension_from_path(path: str, default: str = ".mp4") -> str:
    suffix = Path(urlparse(path).path).suffix.lower()
    return suffix if suffix else default


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def download_url(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with out_path.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)


def materialize_video(video_value: Any, out_path_base: Path, overwrite: bool) -> Path:
    if isinstance(video_value, dict):
        if video_value.get("path"):
            return materialize_video(video_value["path"], out_path_base, overwrite)
        if video_value.get("bytes"):
            out_path = out_path_base.with_suffix(".mp4")
            if overwrite or not out_path.exists():
                write_bytes(out_path, video_value["bytes"])
            return out_path
    if isinstance(video_value, bytes):
        out_path = out_path_base.with_suffix(".mp4")
        if overwrite or not out_path.exists():
            write_bytes(out_path, video_value)
        return out_path
    if isinstance(video_value, str):
        ext = extension_from_path(video_value)
        out_path = out_path_base.with_suffix(ext)
        if video_value.startswith("http://") or video_value.startswith("https://"):
            if overwrite or not out_path.exists():
                download_url(video_value, out_path)
            return out_path
        src = Path(video_value)
        if not src.exists():
            raise FileNotFoundError(f"Video path does not exist: {src}")
        if overwrite or not out_path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out_path)
        return out_path
    raise TypeError(f"Unsupported video value type: {type(video_value).__name__}")


def main() -> None:
    config = load_customer_config()
    hf = config.get("huggingface_dataset", {})
    if not hf.get("enabled", False):
        print("Hugging Face dataset preparation is disabled.")
        print("Set huggingface_dataset.enabled to true in h100_dataset_training/config.json to use this script.")
        return

    repo_id = hf["repo_id"]
    subset = hf.get("subset")
    split = hf.get("split", "train")
    video_column = hf.get("video_column", "video")
    prompt_column = hf.get("prompt_column", "prompt")
    id_column = hf.get("id_column", "id")
    max_items = hf.get("max_items")
    streaming = bool(hf.get("streaming", True))
    local_dataset_dir = Path(hf.get("local_dataset_dir", f"J:/datasets/{config['active_dataset']}"))
    videos_dir = local_dataset_dir / "videos"
    manifest_path = local_dataset_dir / "manifest.jsonl"
    overwrite = bool(hf.get("overwrite", False))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

    print("Preparing Hugging Face dataset for Wan2.2 training.")
    print(f"Dataset: {repo_id}")
    print(f"Split: {split}")
    print(f"Streaming: {streaming}")
    print(f"Output folder: {local_dataset_dir}")

    dataset = load_dataset(
        repo_id,
        subset,
        split=split,
        token=token,
        streaming=streaming,
    )
    if max_items is not None and not streaming:
        dataset = dataset.select(range(min(int(max_items), len(dataset))))

    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for index, item in enumerate(dataset):
        if max_items is not None and index >= int(max_items):
            break
        try:
            sample_id = clean_id(item.get(id_column), f"sample_{index:06d}") if isinstance(item, dict) else f"sample_{index:06d}"
            prompt = str(item[prompt_column]).strip()
            if not prompt:
                prompt = f"Video sample {sample_id}."
            video_path = materialize_video(item[video_column], videos_dir / sample_id, overwrite)
            rows.append({"id": sample_id, "video": str(video_path).replace("\\", "/"), "prompt": prompt})
        except Exception as exc:
            failures.append({"index": index, "error": str(exc)})

    with manifest_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "repo_id": repo_id,
        "subset": subset,
        "split": split,
        "video_column": video_column,
        "prompt_column": prompt_column,
        "id_column": id_column,
        "streaming": streaming,
        "local_dataset_dir": str(local_dataset_dir),
        "manifest": str(manifest_path),
        "rows_written": len(rows),
        "failures": failures,
        "used_token": bool(token),
    }
    summary_path = local_dataset_dir / "huggingface_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if hf.get("update_active_dataset_manifest", True):
        dataset_id = config["active_dataset"]
        config.setdefault("datasets", {}).setdefault(dataset_id, {})
        config["datasets"][dataset_id]["manifest"] = str(manifest_path).replace("\\", "/")
        config["datasets"][dataset_id].setdefault("display_name", dataset_id)
        config["datasets"][dataset_id]["source_huggingface_dataset"] = repo_id
        save_config(config)
        print(f"Updated active dataset manifest in h100_dataset_training/config.json: {manifest_path}")

    print(f"Manifest rows written: {len(rows)}")
    print(f"Failures: {len(failures)}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
