from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from customer_workflow import active_dataset, load_customer_config, run_dir


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            values.append(float(value))
    return values


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"count": 0, "metrics": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_line_chart(path: Path, title: str, y_label: str, series: dict[str, list[float]]) -> None:
    plt.figure(figsize=(10, 5))
    for label, values in series.items():
        if values:
            plt.plot(values, label=label, linewidth=1.8)
    plt.title(title)
    plt.xlabel("Logged step")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.25)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_bar_chart(path: Path, title: str, values: dict[str, float], y_label: str = "Score") -> None:
    labels = list(values.keys())
    nums = [values[k] for k in labels]
    plt.figure(figsize=(10, 5))
    colors = ["#2f7f6f" if n >= 0.75 else "#c9832b" if n >= 0.5 else "#b54545" for n in nums]
    plt.bar(labels, nums, color=colors)
    plt.title(title)
    plt.ylabel(y_label)
    plt.ylim(0, max(1.0, max(nums) * 1.1 if nums else 1.0))
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def mean(summary: dict[str, Any], key: str, default: float = float("nan")) -> float:
    return float(summary.get("metrics", {}).get(key, {}).get("mean", default))


def grade_dataset(summary: dict[str, Any]) -> tuple[str, str]:
    conformance = mean(summary, "conformance_score", 0.0)
    duplicates = mean(summary, "duplicate_frame_ratio", 1.0)
    clipping = mean(summary, "black_pixel_ratio", 0.0) + mean(summary, "white_pixel_ratio", 0.0)
    if conformance >= 0.90 and duplicates < 0.03 and clipping < 0.08:
        return "PASS", "Dataset format and visual health look strong."
    if conformance >= 0.75 and duplicates < 0.10 and clipping < 0.15:
        return "REVIEW", "Dataset is usable, but review flagged metrics before ranking it highly."
    return "FAIL", "Dataset has significant format or visual-quality issues for this benchmark."


def latest_inference_dir(output_dir: Path) -> Path | None:
    root = output_dir / "inference"
    if not root.exists():
        return None
    dirs = sorted([p for p in root.glob("step_*") if p.is_dir()])
    return dirs[-1] if dirs else None


def make_report() -> None:
    config = load_customer_config()
    dataset_id, dataset = active_dataset(config)
    output = run_dir(config, dataset_id)
    report_dir = output / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(output / "metrics.jsonl")
    val_rows = read_jsonl(output / "validation_metrics.jsonl")
    quality_summary = load_summary(output / "dataset_quality_summary.json")
    infer_dir = latest_inference_dir(output)
    infer_rows = read_jsonl(infer_dir / "inference_metrics.jsonl") if infer_dir else []
    infer_summary = load_summary(infer_dir / "inference_metrics_summary.json") if infer_dir else {"count": 0, "metrics": {}}

    charts = []
    if train_rows:
        path = report_dir / "training_loss.png"
        save_line_chart(
            path,
            "Training Loss",
            "Loss",
            {
                "loss": metric_values(train_rows, "loss"),
                "loss_ema": metric_values(train_rows, "loss_ema"),
            },
        )
        charts.append(path)
        path = report_dir / "system_performance.png"
        save_line_chart(
            path,
            "Training Throughput and GPU Memory",
            "Value",
            {
                "frames_per_second": metric_values(train_rows, "frames_per_second"),
                "gpu_reserved_gb": metric_values(train_rows, "gpu_mem_reserved_gb"),
            },
        )
        charts.append(path)
    if val_rows:
        path = report_dir / "validation_loss.png"
        save_line_chart(path, "Validation Loss", "Loss", {"val_loss": metric_values(val_rows, "val_loss")})
        charts.append(path)
    dataset_bars = {
        "conformance": mean(quality_summary, "conformance_score", 0.0),
        "motion": min(mean(quality_summary, "temporal_diff_mean", 0.0) * 8.0, 1.0),
        "sharpness": min(mean(quality_summary, "sharpness_grad_mean", 0.0) * 12.0, 1.0),
        "entropy": mean(quality_summary, "entropy_mean", 0.0),
        "low_duplicates": max(0.0, 1.0 - mean(quality_summary, "duplicate_frame_ratio", 1.0)),
    }
    path = report_dir / "dataset_quality_overview.png"
    save_bar_chart(path, "Dataset Quality Overview", dataset_bars)
    charts.append(path)
    if infer_rows:
        inference_bars = {
            "first_frame_psnr/40": min(mean(infer_summary, "first_frame_psnr", 0.0) / 40.0, 1.0),
            "motion": min(mean(infer_summary, "gen_temporal_delta_p50", 0.0) * 10.0, 1.0),
            "low_blank": max(0.0, 1.0 - mean(infer_summary, "gen_blank_frame_ratio", 1.0)),
            "speed": min(mean(infer_summary, "generation_frames_per_second", 0.0) / 8.0, 1.0),
        }
        path = report_dir / "inference_quality_overview.png"
        save_bar_chart(path, "Inference Quality Overview", inference_bars)
        charts.append(path)

    grade, grade_text = grade_dataset(quality_summary)
    final_loss = metric_values(train_rows, "loss_ema")[-1] if metric_values(train_rows, "loss_ema") else float("nan")
    val_loss = metric_values(val_rows, "val_loss")[-1] if metric_values(val_rows, "val_loss") else float("nan")
    latest_ckpt = (output / "latest_checkpoint.txt").read_text(encoding="utf-8").strip() if (output / "latest_checkpoint.txt").exists() else "No checkpoint found"
    chart_html = "\n".join(
        f'<figure><img src="{html.escape(p.name)}" alt="{html.escape(p.stem)}"><figcaption>{html.escape(p.stem.replace("_", " ").title())}</figcaption></figure>'
        for p in charts
    )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Wan2.2 Dataset QA Report - {html.escape(dataset_id)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grade {{ display: inline-block; padding: 8px 14px; border-radius: 4px; background: #eef4f2; font-weight: bold; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ border: 1px solid #d9dee3; border-radius: 6px; padding: 12px; }}
    .label {{ color: #5d6975; font-size: 13px; }}
    .value {{ font-size: 22px; font-weight: bold; margin-top: 4px; }}
    img {{ max-width: 100%; border: 1px solid #d9dee3; border-radius: 6px; }}
    figure {{ margin: 24px 0; }}
    code {{ background: #f3f5f7; padding: 2px 4px; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>Wan2.2 Dataset QA Report</h1>
  <p><strong>Dataset:</strong> {html.escape(dataset.get("display_name", dataset_id))} ({html.escape(dataset_id)})</p>
  <p><strong>Status:</strong> <span class="grade">{grade}</span> {html.escape(grade_text)}</p>
  <div class="grid">
    <div class="metric"><div class="label">Dataset clips checked</div><div class="value">{quality_summary.get("count", 0)}</div></div>
    <div class="metric"><div class="label">Conformance score</div><div class="value">{mean(quality_summary, "conformance_score", 0.0):.3f}</div></div>
    <div class="metric"><div class="label">Final train loss EMA</div><div class="value">{final_loss:.4f}</div></div>
    <div class="metric"><div class="label">Latest validation loss</div><div class="value">{val_loss:.4f}</div></div>
  </div>
  <h2>Checkpoint Used For QA</h2>
  <p><code>{html.escape(latest_ckpt)}</code></p>
  <h2>Charts</h2>
  {chart_html}
  <h2>How To Read This</h2>
  <p>Use this report to compare datasets trained with the same settings. Higher conformance, lower duplicate-frame ratio, stable validation loss, and better inference quality metrics indicate a stronger dataset for this benchmark.</p>
</body>
</html>
"""
    (report_dir / "report.html").write_text(html_text, encoding="utf-8")
    summary = {
        "dataset_id": dataset_id,
        "grade": grade,
        "grade_text": grade_text,
        "report_html": str(report_dir / "report.html"),
        "latest_checkpoint": latest_ckpt,
    }
    (report_dir / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Report written to: {report_dir / 'report.html'}")


if __name__ == "__main__":
    make_report()
