#!/usr/bin/env python3
"""
Nexisgen Dataset Quality Analyzer

Compatible with the Wan2.2 TI2V training pipeline in ../_core/.

Analyzes video clips using the same decoding and core metric logic as
train_wan22_ti2v_lora.py, plus additional detection for timelapse,
static scenes, and scene cuts.

Outputs:
  - quality_report.csv      per-clip metrics and grade
  - summary.json            aggregate statistics
  - report.html             human-readable HTML report
  - quarantine/             (optional) isolated bad clips

Usage:
  python clean_dataset.py ./clips --output ./report --quarantine
  python clean_dataset.py --manifest ./manifest.jsonl --output ./report
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

TARGET_WIDTH = 1280
TARGET_HEIGHT = 704
TARGET_FPS = 24.0
TARGET_NUM_FRAMES = 121


# ---------------------------------------------------------------------------
# Video decoding (matches _core/train_wan22_ti2v_lora.py::read_video)
# ---------------------------------------------------------------------------


def read_video(path: Path) -> tuple[torch.Tensor, float]:
    """Decode video to TCHW tensor [0, 255] and return fps."""
    container = av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames = [
        torch.from_numpy(frame.to_ndarray(format="rgb24"))
        for frame in container.decode(video=0)
    ]
    container.close()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    video = torch.stack(frames)  # T, H, W, C
    video = video.permute(0, 3, 1, 2).contiguous()  # T, C, H, W
    return video, fps


# ---------------------------------------------------------------------------
# Metric computation (aligned with _core/train_wan22_ti2v_lora.py)
# ---------------------------------------------------------------------------


def rgb_to_luma(video_0_1: torch.Tensor) -> torch.Tensor:
    """BT.601 luma from normalized RGB video (T, C, H, W)."""
    r, g, b = video_0_1[:, 0], video_0_1[:, 1], video_0_1[:, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def compute_entropy(gray_0_1: torch.Tensor) -> float:
    vals = gray_0_1.flatten()
    hist = torch.histc(vals, bins=64, min=0.0, max=1.0)
    prob = hist / hist.sum().clamp_min(1.0)
    entropy = -(prob * (prob + 1e-12).log2()).sum()
    return float(entropy / math.log2(64))


def compute_metrics(
    video: torch.Tensor,
    fps: float,
    clip_path: Path,
) -> dict[str, Any]:
    """Compute quality metrics for one clip.

    The logic is intentionally aligned with
    _core/train_wan22_ti2v_lora.py::compute_quality_metrics.
    """
    source_frames, _, source_h, source_w = video.shape
    video_0_1 = video.float() / 255.0

    luma = rgb_to_luma(video_0_1)

    # Frame-to-frame absolute difference.
    if video.shape[0] > 1:
        frame_delta = (video_0_1[1:] - video_0_1[:-1]).abs()
        max_delta = frame_delta.flatten(1).mean(dim=1)
    else:
        max_delta = torch.tensor([0.0])

    duplicate_ratio = float((max_delta < 0.002).float().mean())
    temporal_diff_mean = float(frame_delta.mean()) if frame_delta.numel() else 0.0
    temporal_diff_std = float(frame_delta.std()) if frame_delta.numel() else 0.0
    motion_p95 = float(torch.quantile(max_delta.float(), 0.95)) if max_delta.numel() else 0.0

    # Luma flicker.
    luma_frame_mean = luma.flatten(1).mean(dim=1)
    flicker_luma_std = float(luma_frame_mean.std())

    # Sharpness (gradient magnitude).
    dx = video_0_1[:, :, :, 1:] - video_0_1[:, :, :, :-1]
    dy = video_0_1[:, :, 1:, :] - video_0_1[:, :, :-1, :]
    sharpness_grad_mean = float(dx.abs().mean() + dy.abs().mean())

    # Entropy.
    entropy_mean = compute_entropy(luma)

    # Black/white clipping.
    black_pixel_ratio = float((video_0_1 < 0.01).float().mean())
    white_pixel_ratio = float((video_0_1 > 0.99).float().mean())

    # Conformance score (same formula as training code).
    aspect_error = abs((source_w / max(source_h, 1)) - (TARGET_WIDTH / TARGET_HEIGHT))
    frame_error = source_frames - TARGET_NUM_FRAMES
    fps_error = fps - TARGET_FPS if fps > 0 else 0.0
    penalties = [
        min(abs(source_w - TARGET_WIDTH) / TARGET_WIDTH, 1.0),
        min(abs(source_h - TARGET_HEIGHT) / TARGET_HEIGHT, 1.0),
        min(abs(frame_error) / TARGET_NUM_FRAMES, 1.0),
        min(abs(fps_error) / TARGET_FPS, 1.0) if fps > 0 else 0.25,
        min(aspect_error, 1.0),
        duplicate_ratio,
    ]
    conformance_score = 1.0 - sum(penalties) / len(penalties)

    # Extra: timelapse heuristic.
    high_motion_ratio = float((max_delta > 0.12).float().mean()) if max_delta.numel() else 0.0
    timelapse_score = 0.0
    if (
        duplicate_ratio < 0.05
        and high_motion_ratio > 0.30
        and temporal_diff_mean > 0.0025
    ):
        stability = 1.0 - min(temporal_diff_std / max(temporal_diff_mean, 1e-6), 1.0)
        timelapse_score = min(1.0, high_motion_ratio * 0.7 + stability * 0.3)

    # Extra: scene cuts.
    scene_cut_count = int((max_delta > 0.20).float().sum().item()) if max_delta.numel() else 0

    return {
        "path": str(clip_path),
        "source_width": source_w,
        "source_height": source_h,
        "source_frames": source_frames,
        "source_fps": fps,
        "duration_seconds": source_frames / fps if fps > 0 else 0.0,
        "duplicate_frame_ratio": duplicate_ratio,
        "temporal_diff_mean": temporal_diff_mean,
        "temporal_diff_std": temporal_diff_std,
        "motion_p95": motion_p95,
        "flicker_luma_std": flicker_luma_std,
        "sharpness_grad_mean": sharpness_grad_mean,
        "entropy_mean": entropy_mean,
        "black_pixel_ratio": black_pixel_ratio,
        "white_pixel_ratio": white_pixel_ratio,
        "aspect_error": aspect_error,
        "frame_error": frame_error,
        "fps_error": fps_error,
        "conformance_score": conformance_score,
        "timelapse_score": timelapse_score,
        "high_motion_ratio": high_motion_ratio,
        "scene_cut_count": scene_cut_count,
    }


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def grade_clip(
    metrics: dict[str, Any],
    cfg: Any,
) -> tuple[str, str]:
    """Return (grade, reason) for a single clip.

    Grades follow the same thresholds as quality_report.py::grade_dataset:
      PASS   -> strong candidate
      REVIEW -> usable but flagged
      FAIL   -> significant issues
    """
    conformance = metrics["conformance_score"]
    duplicates = metrics["duplicate_frame_ratio"]
    timelapse = metrics["timelapse_score"]
    scene_cuts = metrics["scene_cut_count"]
    clipping = metrics["black_pixel_ratio"] + metrics["white_pixel_ratio"]

    reasons: list[str] = []

    if scene_cuts > cfg.scene_cut_max:
        reasons.append(f"scene_cuts={scene_cuts}")
    if duplicates > cfg.dup_ratio:
        reasons.append(f"duplicates={duplicates:.2%}")
    if timelapse > cfg.timelapse_score:
        reasons.append(f"timelapse={timelapse:.2f}")
    if metrics["temporal_diff_mean"] < getattr(cfg, "static_mean_delta", 0.005):
        reasons.append(f"static_mean_delta={metrics['temporal_diff_mean']:.4f}")
    max_mean_delta = getattr(cfg, "max_mean_delta", None)
    if max_mean_delta is not None and metrics["temporal_diff_mean"] > max_mean_delta:
        # Upper bound on motion: clips this chaotic hurt Subject/Background
        # Consistency and Motion Smoothness (VBench dimensions).
        reasons.append(f"chaotic_mean_delta={metrics['temporal_diff_mean']:.4f}")
    if clipping > cfg.clipping_ratio:
        reasons.append(f"clipping={clipping:.2%}")
    if conformance < cfg.min_conformance:
        reasons.append(f"conformance={conformance:.3f}")

    if reasons:
        return "FAIL", "; ".join(reasons)

    if (
        conformance >= 0.90
        and duplicates < 0.03
        and timelapse < 0.3
        and clipping < 0.08
    ):
        return "PASS", "passed all checks"

    return "REVIEW", "minor flags"


# ---------------------------------------------------------------------------
# Clip discovery
# ---------------------------------------------------------------------------


def read_manifest_rows(manifest_path: Path) -> list[Path]:
    rows: list[Path] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            video_value = row.get("video")
            if video_value:
                p = Path(video_value)
                if not p.is_absolute():
                    p = (manifest_path.parent / p).resolve()
                rows.append(p)
    return rows


def discover_clips(input_dir: Path, extensions: set[str]) -> list[Path]:
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in extensions])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class ReportSummary:
    total: int = 0
    pass_count: int = 0
    review_count: int = 0
    fail_count: int = 0
    mean_conformance: float = 0.0
    mean_duplicate_ratio: float = 0.0
    mean_timelapse_score: float = 0.0
    mean_temporal_diff: float = 0.0
    issues_by_category: dict[str, int] = field(default_factory=dict)


def write_csv_report(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary: ReportSummary, path: Path, cfg: Any) -> None:
    payload = {
        "total": summary.total,
        "pass": summary.pass_count,
        "review": summary.review_count,
        "fail": summary.fail_count,
        "pass_rate": summary.pass_count / max(summary.total, 1),
        "review_rate": summary.review_count / max(summary.total, 1),
        "fail_rate": summary.fail_count / max(summary.total, 1),
        "aggregate": {
            "mean_conformance_score": summary.mean_conformance,
            "mean_duplicate_frame_ratio": summary.mean_duplicate_ratio,
            "mean_timelapse_score": summary.mean_timelapse_score,
            "mean_temporal_diff_mean": summary.mean_temporal_diff,
        },
        "issues_by_category": summary.issues_by_category,
        "thresholds": {
            "dup_ratio": cfg.dup_ratio,
            "timelapse_score": cfg.timelapse_score,
            "clipping_ratio": cfg.clipping_ratio,
            "min_conformance": cfg.min_conformance,
            "scene_cut_max": cfg.scene_cut_max,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_html_report(
    rows: list[dict[str, Any]],
    summary: ReportSummary,
    path: Path,
) -> None:
    status = "PASS" if summary.fail_count == 0 else "FAIL"
    status_color = "#2f7f6f" if status == "PASS" else "#b54545"

    table_rows = []
    for r in rows:
        grade = r["grade"]
        color = "#2f7f6f" if grade == "PASS" else "#c9832b" if grade == "REVIEW" else "#b54545"
        cells = [
            f"<td>{html.escape(str(r['path']))}</td>",
            f"<td>{r['source_frames']}</td>",
            f"<td>{r['source_fps']:.2f}</td>",
            f"<td>{r['conformance_score']:.3f}</td>",
            f"<td>{r['duplicate_frame_ratio']:.3f}</td>",
            f"<td>{r['timelapse_score']:.3f}</td>",
            f"<td>{r['temporal_diff_mean']:.4f}</td>",
            f"<td style='color:{color};font-weight:bold'>{grade}</td>",
            f"<td>{html.escape(r['reason'])}</td>",
        ]
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dataset Quality Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d9dee3; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f5; text-align: left; }}
    .status {{ display:inline-block; padding:8px 12px; border-radius:4px; color:white; background:{status_color}; font-weight:bold; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ border: 1px solid #d9dee3; border-radius: 6px; padding: 12px; }}
    .label {{ color: #5d6975; font-size: 13px; }}
    .value {{ font-size: 22px; font-weight: bold; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Dataset Quality Report</h1>
  <p><span class="status">{status}</span></p>
  <div class="grid">
    <div class="metric"><div class="label">Total clips</div><div class="value">{summary.total}</div></div>
    <div class="metric"><div class="label">PASS</div><div class="value">{summary.pass_count} ({summary.pass_count/max(summary.total,1):.1%})</div></div>
    <div class="metric"><div class="label">REVIEW</div><div class="value">{summary.review_count} ({summary.review_count/max(summary.total,1):.1%})</div></div>
    <div class="metric"><div class="label">FAIL</div><div class="value">{summary.fail_count} ({summary.fail_count/max(summary.total,1):.1%})</div></div>
    <div class="metric"><div class="label">Mean conformance</div><div class="value">{summary.mean_conformance:.3f}</div></div>
    <div class="metric"><div class="label">Mean duplicate ratio</div><div class="value">{summary.mean_duplicate_ratio:.3f}</div></div>
    <div class="metric"><div class="label">Mean timelapse score</div><div class="value">{summary.mean_timelapse_score:.3f}</div></div>
    <div class="metric"><div class="label">Mean temporal diff</div><div class="value">{summary.mean_temporal_diff:.4f}</div></div>
  </div>
  <h2>Clip Details</h2>
  <table>
    <tr>
      <th>Path</th><th>Frames</th><th>FPS</th><th>Conformance</th>
      <th>Duplicate</th><th>Timelapse</th><th>Temp Diff</th>
      <th>Grade</th><th>Reason</th>
    </tr>
    {''.join(table_rows)}
  </table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def analyze_clip(clip_path: Path, cfg: Any) -> dict[str, Any] | None:
    try:
        video, fps = read_video(clip_path)
        metrics = compute_metrics(video, fps, clip_path)
        grade, reason = grade_clip(metrics, cfg)
        metrics["grade"] = grade
        metrics["reason"] = reason
        return metrics
    except Exception as exc:
        logger.error("Failed to analyze %s: %s", clip_path, exc)
        return None


def analyze_manifest(manifest_path: Path, cfg: Any) -> list[dict[str, Any]]:
    clip_paths = read_manifest_rows(manifest_path)
    results: list[dict[str, Any]] = []
    for path in tqdm(clip_paths, desc="Analyzing clips"):
        result = analyze_clip(path, cfg)
        if result is not None:
            results.append(result)
    return results


def analyze_directory(input_dir: Path, cfg: Any) -> list[dict[str, Any]]:
    extensions = {f".{ext.strip().lower()}" for ext in cfg.extensions.split(",")}
    clip_paths = discover_clips(input_dir, extensions)
    results: list[dict[str, Any]] = []
    for path in tqdm(clip_paths, desc="Analyzing clips"):
        result = analyze_clip(path, cfg)
        if result is not None:
            results.append(result)
    return results


def build_summary(rows: list[dict[str, Any]]) -> ReportSummary:
    summary = ReportSummary(total=len(rows))
    for r in rows:
        grade = r["grade"]
        if grade == "PASS":
            summary.pass_count += 1
        elif grade == "REVIEW":
            summary.review_count += 1
        else:
            summary.fail_count += 1

        reason = r.get("reason", "")
        if reason:
            # Take the first issue tag as the primary category.
            category = reason.split("=")[0].split(";")[0].strip()
            summary.issues_by_category[category] = summary.issues_by_category.get(category, 0) + 1

    if rows:
        summary.mean_conformance = float(np.mean([r["conformance_score"] for r in rows]))
        summary.mean_duplicate_ratio = float(np.mean([r["duplicate_frame_ratio"] for r in rows]))
        summary.mean_timelapse_score = float(np.mean([r["timelapse_score"] for r in rows]))
        summary.mean_temporal_diff = float(np.mean([r["temporal_diff_mean"] for r in rows]))
    return summary


def quarantine_clips(rows: list[dict[str, Any]], quarantine_dir: Path, copy: bool) -> None:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    operation = shutil.copy2 if copy else shutil.move
    op_name = "copied" if copy else "moved"
    for r in rows:
        if r["grade"] == "FAIL":
            src = Path(r["path"])
            dst = quarantine_dir / src.name
            try:
                operation(str(src), str(dst))
            except Exception as exc:
                logger.warning("Failed to %s %s: %s", op_name, src, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Nexisgen/Wan2.2 video clip quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a directory of clips
  python clean_dataset.py ./clips --output ./report

  # Analyze via manifest.jsonl
  python clean_dataset.py --manifest ./manifest.jsonl --output ./report

  # Isolate failed clips
  python clean_dataset.py ./clips --output ./report --quarantine
        """,
    )
    parser.add_argument("clips_dir", type=Path, nargs="?", help="Directory containing video clips")
    parser.add_argument("--manifest", type=Path, help="Path to manifest.jsonl")
    parser.add_argument("--output", type=Path, default=Path("./quality_report"), help="Output directory")
    parser.add_argument("--extensions", type=str, default="mp4,mov,avi,webm,mkv", help="Video extensions")

    # Thresholds
    parser.add_argument("--dup-ratio", type=float, default=0.05, help="Duplicate frame ratio threshold")
    parser.add_argument("--timelapse-score", type=float, default=0.50, help="Timelapse score threshold")
    parser.add_argument("--clipping-ratio", type=float, default=0.15, help="Black+white pixel ratio threshold")
    parser.add_argument("--min-conformance", type=float, default=0.50, help="Minimum conformance score")
    parser.add_argument("--scene-cut-max", type=int, default=0, help="Max scene cuts per clip")
    parser.add_argument("--static-mean-delta", type=float, default=0.005, help="Clips with temporal_diff_mean below this value are marked static/FAIL")
    parser.add_argument("--max-mean-delta", type=float, default=None, help="Clips with temporal_diff_mean above this value are marked chaotic/FAIL (protects consistency/smoothness)")

    # Actions
    parser.add_argument("--quarantine", action="store_true", help="Move failed clips to output/quarantine")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move when quarantining")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (1 = sequential)")

    args = parser.parse_args()

    if args.manifest:
        logger.info("Analyzing manifest: %s", args.manifest)
    elif args.clips_dir:
        if not args.clips_dir.is_dir():
            print(f"Error: clips directory not found: {args.clips_dir}", file=sys.stderr)
            return 1
    else:
        print("Error: provide either clips_dir or --manifest", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    if args.workers > 1:
        clip_paths = []
        if args.manifest:
            clip_paths = read_manifest_rows(args.manifest)
        else:
            extensions = {f".{ext.strip().lower()}" for ext in args.extensions.split(",")}
            clip_paths = discover_clips(args.clips_dir, extensions)

        rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(analyze_clip, path, args) for path in clip_paths]
            for future in tqdm(as_completed(futures), total=len(clip_paths), desc="Analyzing"):
                result = future.result()
                if result is not None:
                    rows.append(result)
        rows.sort(key=lambda r: r["path"])
    else:
        if args.manifest:
            rows = analyze_manifest(args.manifest, args)
        else:
            rows = analyze_directory(args.clips_dir, args)

    summary = build_summary(rows)

    write_csv_report(rows, args.output / "quality_report.csv")
    write_summary(summary, args.output / "summary.json", args)
    write_html_report(rows, summary, args.output / "report.html")

    if args.quarantine:
        quarantine_clips(rows, args.output / "quarantine", args.copy)

    print("\n" + "=" * 60)
    print("Dataset Quality Report Summary")
    print("=" * 60)
    print(f"Total clips:  {summary.total}")
    print(f"PASS:         {summary.pass_count} ({summary.pass_count / max(summary.total, 1):.1%})")
    print(f"REVIEW:       {summary.review_count} ({summary.review_count / max(summary.total, 1):.1%})")
    print(f"FAIL:         {summary.fail_count} ({summary.fail_count / max(summary.total, 1):.1%})")
    print(f"Mean conformance:      {summary.mean_conformance:.3f}")
    print(f"Mean duplicate ratio:  {summary.mean_duplicate_ratio:.3f}")
    print(f"Mean timelapse score:  {summary.mean_timelapse_score:.3f}")
    print(f"\nReports written to: {args.output}")
    if args.quarantine:
        action = "copied" if args.copy else "moved"
        print(f"Failed clips {action} to: {args.output / 'quarantine'}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
