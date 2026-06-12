from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self.fh.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class TableLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", newline="", encoding="utf-8")
        self.fieldnames = fieldnames
        self.writer = csv.DictWriter(self.fh, fieldnames=fieldnames)
        if self.path.stat().st_size == 0:
            self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow({key: row.get(key) for key in self.fieldnames})
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class MetricsLoggingManager:
    def __init__(self, output_dir: str | Path, csv_fields: list[str] | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events = JsonlLogger(self.output_dir / "events.jsonl")
        self.metrics = JsonlLogger(self.output_dir / "metrics.jsonl")
        self.validation = JsonlLogger(self.output_dir / "validation_metrics.jsonl")
        self.quality = JsonlLogger(self.output_dir / "dataset_quality.jsonl")
        self.inference = JsonlLogger(self.output_dir / "inference_metrics.jsonl")
        self.csv = TableLogger(self.output_dir / "metrics.csv", csv_fields) if csv_fields else None
        self.timers: dict[str, float] = {}

    def event(self, name: str, **payload: Any) -> None:
        self.events.write({"time": time.time(), "event": name, **payload})

    def log_metrics(self, row: dict[str, Any]) -> None:
        self.metrics.write(row)
        if self.csv is not None:
            self.csv.write(row)

    def log_validation(self, row: dict[str, Any]) -> None:
        self.validation.write(row)

    def log_quality(self, row: dict[str, Any]) -> None:
        self.quality.write(row)

    def log_inference(self, row: dict[str, Any]) -> None:
        self.inference.write(row)

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timers[name] = time.perf_counter() - start

    def flush_summary(self) -> None:
        for filename in ["metrics.jsonl", "validation_metrics.jsonl", "dataset_quality.jsonl", "inference_metrics.jsonl"]:
            path = self.output_dir / filename
            if path.exists():
                summarize_jsonl(path, self.output_dir / filename.replace(".jsonl", "_summary.json"))

    def close(self) -> None:
        self.flush_summary()
        self.events.close()
        self.metrics.close()
        self.validation.close()
        self.quality.close()
        self.inference.close()
        if self.csv is not None:
            self.csv.close()


def summarize_jsonl(input_path: str | Path, output_path: str | Path) -> None:
    rows = []
    with Path(input_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    numeric: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                numeric[key].append(float(value))
    summary: dict[str, Any] = {"count": len(rows), "metrics": {}}
    for key, values in numeric.items():
        arr = np.asarray(values, dtype=np.float64)
        summary["metrics"][key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p01": float(np.quantile(arr, 0.01)),
            "p05": float(np.quantile(arr, 0.05)),
            "p25": float(np.quantile(arr, 0.25)),
            "p50": float(np.quantile(arr, 0.50)),
            "p75": float(np.quantile(arr, 0.75)),
            "p95": float(np.quantile(arr, 0.95)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(arr.max()),
        }
    Path(output_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def selected_gpu_metrics(device: torch.device | str | None = None) -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    cuda_device = torch.device(device or torch.cuda.current_device())
    index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
    return {
        "gpu_index": float(index),
        "gpu_mem_allocated_gb": torch.cuda.memory_allocated(index) / 1024**3,
        "gpu_mem_reserved_gb": torch.cuda.memory_reserved(index) / 1024**3,
        "gpu_max_mem_allocated_gb": torch.cuda.max_memory_allocated(index) / 1024**3,
    }
