from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import av as _av
import torch as _torch


def read_video(path, pts_unit="sec", output_format="TCHW"):
    container = _av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames = [_torch.from_numpy(f.to_ndarray(format="rgb24")) for f in container.decode(video=0)]
    container.close()
    video = _torch.stack(frames)  # (T, H, W, C)
    if output_format == "TCHW":
        video = video.permute(0, 3, 1, 2)
    return video, _torch.zeros(0), {"video_fps": fps}

from train_wan22_ti2v_lora import parse_resolution


@dataclass
class DiagnosticIssue:
    severity: str
    code: str
    message: str
    fix: str
    line: int | None = None
    sample_id: str | None = None
    video: str | None = None
    value: str | None = None


def issue(
    severity: str,
    code: str,
    message: str,
    fix: str,
    line: int | None = None,
    sample_id: str | None = None,
    video: str | None = None,
    value: Any = None,
) -> DiagnosticIssue:
    return DiagnosticIssue(
        severity=severity,
        code=code,
        message=message,
        fix=fix,
        line=line,
        sample_id=sample_id,
        video=video,
        value=None if value is None else str(value),
    )


def resolve_video_path(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def read_manifest_rows(manifest_path: Path) -> tuple[list[dict[str, Any]], list[DiagnosticIssue]]:
    issues: list[DiagnosticIssue] = []
    rows: list[dict[str, Any]] = []
    if not manifest_path.exists():
        return [], [
            issue(
                "ERROR",
                "MANIFEST_NOT_FOUND",
                f"Manifest file does not exist: {manifest_path}",
                "Set the dataset manifest path in config.json to an existing manifest.jsonl file.",
            )
        ]
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                issues.append(
                    issue(
                        "ERROR",
                        "MANIFEST_BAD_JSON",
                        f"Line {line_no} is not valid JSON.",
                        "Fix the JSON object on this line. Each line must be one complete JSON object.",
                        line=line_no,
                        value=exc,
                    )
                )
                continue
            if not isinstance(row, dict):
                issues.append(
                    issue(
                        "ERROR",
                        "MANIFEST_ROW_NOT_OBJECT",
                        f"Line {line_no} is not a JSON object.",
                        "Each manifest line must look like {\"id\":\"...\",\"video\":\"...\",\"prompt\":\"...\"}.",
                        line=line_no,
                    )
                )
                continue
            row["_line"] = line_no
            rows.append(row)
    if not rows:
        issues.append(
            issue(
                "ERROR",
                "MANIFEST_EMPTY",
                "Manifest contains no usable rows.",
                "Add at least one JSONL row with video and prompt fields.",
            )
        )
    return rows, issues


def motion_duplicate_ratio(video: Any) -> float:
    if video.shape[0] < 2:
        return 1.0
    arr = video.float() / 255.0
    delta = (arr[1:] - arr[:-1]).abs().flatten(1).mean(dim=1)
    return float((delta < 0.002).float().mean())


def diagnose_dataset(
    manifest: str | Path,
    output_dir: str | Path,
    resolution: str = "1280x704",
    num_frames: int = 121,
    fps: float = 24.0,
    fps_tolerance: float = 0.75,
    allow_resize: bool = True,
    train_batch_size: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    max_decode_failures_before_stop: int = 25,
) -> dict[str, Any]:
    manifest_path = Path(manifest)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target_w, target_h = parse_resolution(resolution)
    rows, issues = read_manifest_rows(manifest_path)
    seen_ids: Counter[str] = Counter()
    video_stats: list[dict[str, Any]] = []
    decode_failures = 0

    for index, row in enumerate(rows):
        line = int(row.get("_line", index + 1))
        sample_id = str(row.get("id") or Path(str(row.get("video", f"row_{line}"))).stem)
        seen_ids[sample_id] += 1
        video_value = row.get("video")
        prompt_value = row.get("prompt")

        if "id" not in row or not str(row.get("id", "")).strip():
            issues.append(
                issue(
                    "WARN",
                    "MISSING_ID",
                    f"Line {line} has no stable id.",
                    "Add an id field so reports and generated samples have stable names.",
                    line=line,
                    sample_id=sample_id,
                )
            )

        if not video_value or not str(video_value).strip():
            issues.append(
                issue(
                    "ERROR",
                    "MISSING_VIDEO_FIELD",
                    f"Line {line} has no video path.",
                    "Add a video field pointing to an mp4 or other readable video file.",
                    line=line,
                    sample_id=sample_id,
                )
            )
            continue

        if prompt_value is None or not str(prompt_value).strip():
            issues.append(
                issue(
                    "ERROR",
                    "MISSING_PROMPT_FIELD",
                    f"Line {line} has no prompt.",
                    "Add a prompt/caption describing subject, motion, camera, scene, and lighting.",
                    line=line,
                    sample_id=sample_id,
                    video=str(video_value),
                )
            )
        elif len(str(prompt_value).split()) < 6:
            issues.append(
                issue(
                    "WARN",
                    "PROMPT_TOO_SHORT",
                    f"Prompt on line {line} is very short.",
                    "Use a descriptive caption. Mention subject, motion, camera movement, setting, and lighting.",
                    line=line,
                    sample_id=sample_id,
                    video=str(video_value),
                    value=prompt_value,
                )
            )

        path = resolve_video_path(str(video_value), manifest_path)
        if path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
            issues.append(
                issue(
                    "WARN",
                    "UNUSUAL_VIDEO_EXTENSION",
                    f"Video extension looks unusual: {path.suffix}",
                    "Use mp4 when possible. If this file is valid, this warning can be ignored.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                )
            )
        if not path.exists():
            issues.append(
                issue(
                    "ERROR",
                    "VIDEO_NOT_FOUND",
                    f"Video file was not found: {path}",
                    "Fix the video path in manifest.jsonl. Relative paths are resolved from the manifest folder.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                )
            )
            continue
        file_size_mb = path.stat().st_size / (1024.0 * 1024.0)

        try:
            video, _, info = read_video(str(path), pts_unit="sec", output_format="TCHW")
        except Exception as exc:
            decode_failures += 1
            issues.append(
                issue(
                    "ERROR",
                    "VIDEO_DECODE_FAILED",
                    f"Could not decode video: {path}",
                    "Re-encode the file as a normal H.264 mp4, then update the manifest path if needed.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                    value=exc,
                )
            )
            if decode_failures >= max_decode_failures_before_stop:
                issues.append(
                    issue(
                        "ERROR",
                        "TOO_MANY_DECODE_FAILURES",
                        "Many videos failed decoding; stopping diagnostics early.",
                        "Fix the video paths/codecs before rerunning diagnostics.",
                    )
                )
                break
            continue

        if video.numel() == 0 or video.shape[0] == 0:
            issues.append(
                issue(
                    "ERROR",
                    "VIDEO_EMPTY",
                    f"Decoded video has no frames: {path}",
                    "Replace or re-export the source video.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                )
            )
            continue

        source_frames = int(video.shape[0])
        source_h = int(video.shape[2])
        source_w = int(video.shape[3])
        source_fps = float(info.get("video_fps", 0.0) or 0.0)
        aspect_error = abs((source_w / max(source_h, 1)) - (target_w / target_h))
        duplicate_ratio = motion_duplicate_ratio(video)
        video_stats.append(
            {
                "line": line,
                "id": sample_id,
                "video": str(path),
                "width": source_w,
                "height": source_h,
                "frames": source_frames,
                "fps": source_fps,
                "file_size_mb": file_size_mb,
                "duplicate_frame_ratio": duplicate_ratio,
                "aspect_error": aspect_error,
            }
        )

        if source_w != target_w:
            sev = "WARN" if allow_resize else "ERROR"
            issues.append(
                issue(
                    sev,
                    "WIDTH_MISMATCH",
                    f"{sample_id} width is {source_w}, expected {target_w}.",
                    "Export videos at the target width, or keep allow_resize enabled so the loader resizes consistently.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                    value=source_w,
                )
            )
        if source_h != target_h:
            sev = "WARN" if allow_resize else "ERROR"
            issues.append(
                issue(
                    sev,
                    "HEIGHT_MISMATCH",
                    f"{sample_id} height is {source_h}, expected {target_h}.",
                    "Export videos at the target height, or keep allow_resize enabled so the loader resizes consistently.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                    value=source_h,
                )
            )
        if aspect_error > 0.02:
            issues.append(
                issue(
                    "WARN",
                    "ASPECT_RATIO_MISMATCH",
                    f"{sample_id} aspect ratio differs from target.",
                    "Prefer source videos already matching 1280x704 aspect ratio to avoid stretching or cropping artifacts.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                    value=round(aspect_error, 4),
                )
            )
        if source_frames < num_frames:
            issues.append(
                issue(
                    "ERROR",
                    "NOT_ENOUGH_FRAMES",
                    f"{sample_id} has {source_frames} frames, expected at least {num_frames}.",
                    "Use clips with at least the target frame count, or lower num_frames for a smoke test only.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                    value=source_frames,
                )
            )
        elif source_frames != num_frames:
            issues.append(
                issue(
                    "WARN",
                    "FRAME_COUNT_MISMATCH",
                    f"{sample_id} has {source_frames} frames; loader will sample {num_frames}.",
                    "For clean comparisons, export every clip at exactly the target frame count.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                    value=source_frames,
                )
            )
        if source_fps <= 0:
            issues.append(
                issue(
                    "WARN",
                    "FPS_UNKNOWN",
                    f"{sample_id} FPS metadata is missing.",
                    "Re-encode with explicit 24fps metadata.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                )
            )
        elif abs(source_fps - fps) > fps_tolerance:
            issues.append(
                issue(
                    "WARN",
                    "FPS_MISMATCH",
                    f"{sample_id} FPS is {source_fps:.3f}, expected {fps:.3f}.",
                    "Re-encode videos at 24fps for consistent temporal motion.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                    value=source_fps,
                )
            )
        if duplicate_ratio > 0.10:
            issues.append(
                issue(
                    "WARN",
                    "HIGH_DUPLICATE_FRAME_RATIO",
                    f"{sample_id} appears to contain many repeated frames.",
                    "Remove frozen/duplicate clips, or re-export from the original source at the correct FPS.",
                    line=line,
                    sample_id=sample_id,
                    video=str(path),
                    value=round(duplicate_ratio, 4),
                )
            )

    for sample_id, count in seen_ids.items():
        if count > 1:
            issues.append(
                issue(
                    "ERROR",
                    "DUPLICATE_ID",
                    f"Sample id appears {count} times: {sample_id}",
                    "Make every manifest id unique so checkpoints, reports, and generated samples are unambiguous.",
                    sample_id=sample_id,
                    value=count,
                )
            )

    decoded_batch_float32_gb = train_batch_size * num_frames * target_h * target_w * 3 * 4 / (1024.0**3)
    resident_batches = max(1, int(num_workers) * 2) if int(num_workers) > 0 else 1
    estimated_loader_float32_gb = decoded_batch_float32_gb * resident_batches
    if int(train_batch_size) > 1:
        issues.append(
            issue(
                "WARN",
                "BATCH_SIZE_MEMORY_RISK",
                f"train_batch_size is {train_batch_size}; full-resolution video batches can become very large.",
                "Keep train_batch_size at 1 for Wan2.2 5B dataset comparisons unless memory testing proves a larger batch is stable.",
                value=train_batch_size,
            )
        )
    if int(num_workers) > 0:
        issues.append(
            issue(
                "WARN",
                "NUM_WORKERS_MEMORY_RISK",
                f"num_workers is {num_workers}; PyTorch may keep multiple decoded video batches in system RAM.",
                "Use num_workers=0 for the safest large-dataset workflow. Increase only after the first successful full pass.",
                value=num_workers,
            )
        )
    if bool(pin_memory):
        issues.append(
            issue(
                "WARN",
                "PIN_MEMORY_MEMORY_RISK",
                "pin_memory is enabled; decoded batches may be duplicated in page-locked system RAM.",
                "Leave pin_memory=false for large video datasets unless the machine has plenty of spare RAM.",
                value=pin_memory,
            )
        )

    issue_rows = [asdict(x) for x in issues]
    with (output / "diagnostics.jsonl").open("w", encoding="utf-8") as fh:
        for row in issue_rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
    with (output / "diagnostics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(DiagnosticIssue.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(issue_rows)
    with (output / "video_inventory.csv").open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "line",
            "id",
            "video",
            "width",
            "height",
            "frames",
            "fps",
            "file_size_mb",
            "duplicate_frame_ratio",
            "aspect_error",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(video_stats)

    by_severity = Counter(x.severity for x in issues)
    by_code = Counter(x.code for x in issues)
    summary = {
        "manifest": str(manifest_path),
        "target": {"width": target_w, "height": target_h, "frames": num_frames, "fps": fps},
        "loader_memory": {
            "train_batch_size": int(train_batch_size),
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
            "estimated_decoded_batch_float32_gb": round(decoded_batch_float32_gb, 3),
            "estimated_loader_prefetch_float32_gb": round(estimated_loader_float32_gb, 3),
            "note": "This estimates decoded video tensor RAM only. Model weights, optimizer state, latents, and activations are additional.",
        },
        "rows_seen": len(rows),
        "videos_decoded": len(video_stats),
        "error_count": int(by_severity.get("ERROR", 0)),
        "warning_count": int(by_severity.get("WARN", 0)),
        "issues_by_code": dict(sorted(by_code.items())),
        "ok_to_train": int(by_severity.get("ERROR", 0)) == 0,
    }
    if video_stats:
        total_video_mb = sum(float(x.get("file_size_mb") or 0.0) for x in video_stats)
        summary["storage"] = {
            "decoded_video_files_total_gb": round(total_video_mb / 1024.0, 3),
            "largest_decoded_video_file_mb": round(max(float(x.get("file_size_mb") or 0.0) for x in video_stats), 2),
        }
        for key in ["width", "height", "frames", "fps", "file_size_mb", "duplicate_frame_ratio", "aspect_error"]:
            vals = [float(x[key]) for x in video_stats if x.get(key) is not None and math.isfinite(float(x[key]))]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                summary[f"{key}_mean"] = float(arr.mean())
                summary[f"{key}_min"] = float(arr.min())
                summary[f"{key}_max"] = float(arr.max())
    (output / "diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_diagnostics_html(output / "diagnostics_report.html", summary, issues)
    return summary


def write_diagnostics_html(path: Path, summary: dict[str, Any], issues: list[DiagnosticIssue]) -> None:
    rows = []
    for item in issues:
        color = "#b54545" if item.severity == "ERROR" else "#b7791f"
        rows.append(
            "<tr>"
            f"<td style='color:{color};font-weight:bold'>{html.escape(item.severity)}</td>"
            f"<td><code>{html.escape(item.code)}</code></td>"
            f"<td>{'' if item.line is None else item.line}</td>"
            f"<td>{html.escape(item.sample_id or '')}</td>"
            f"<td>{html.escape(item.message)}</td>"
            f"<td>{html.escape(item.fix)}</td>"
            "</tr>"
        )
    status = "OK TO TRAIN" if summary["ok_to_train"] else "FIX ERRORS BEFORE TRAINING"
    status_color = "#2f7f6f" if summary["ok_to_train"] else "#b54545"
    loader_memory = summary.get("loader_memory", {})
    storage = summary.get("storage", {})
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Dataset Diagnostics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d9dee3; padding: 8px; vertical-align: top; }}
    th {{ background: #eef2f5; text-align: left; }}
    code {{ background: #f3f5f7; padding: 2px 4px; border-radius: 3px; }}
    .status {{ display:inline-block; padding:8px 12px; border-radius:4px; color:white; background:{status_color}; font-weight:bold; }}
  </style>
</head>
<body>
  <h1>Dataset Diagnostics</h1>
  <p><span class="status">{status}</span></p>
  <p><strong>Manifest:</strong> <code>{html.escape(summary['manifest'])}</code></p>
  <p><strong>Rows seen:</strong> {summary['rows_seen']} | <strong>Videos decoded:</strong> {summary['videos_decoded']} | <strong>Errors:</strong> {summary['error_count']} | <strong>Warnings:</strong> {summary['warning_count']}</p>
  <h2>Memory and Storage</h2>
  <p><strong>Training batch size:</strong> {loader_memory.get('train_batch_size', 1)} | <strong>Data loader workers:</strong> {loader_memory.get('num_workers', 0)} | <strong>Pin memory:</strong> {loader_memory.get('pin_memory', False)}</p>
  <p><strong>Estimated decoded batch RAM:</strong> {loader_memory.get('estimated_decoded_batch_float32_gb', 0)} GB | <strong>Estimated loader prefetch RAM:</strong> {loader_memory.get('estimated_loader_prefetch_float32_gb', 0)} GB</p>
  <p><strong>Decoded video storage scanned:</strong> {storage.get('decoded_video_files_total_gb', 0)} GB | <strong>Largest scanned video:</strong> {storage.get('largest_decoded_video_file_mb', 0)} MB</p>
  <p>{html.escape(loader_memory.get('note', ''))}</p>
  <h2>Issue List</h2>
  <table>
    <tr><th>Severity</th><th>Code</th><th>Line</th><th>Sample</th><th>Problem</th><th>Fix</th></tr>
    {''.join(rows) if rows else '<tr><td colspan="6">No issues found.</td></tr>'}
  </table>
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose a Wan2.2 training manifest before training.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", default="1280x704")
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true", default=False)
    args = parser.parse_args()
    result = diagnose_dataset(
        args.manifest,
        args.output_dir,
        args.resolution,
        args.num_frames,
        args.fps,
        train_batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    print(json.dumps(result, indent=2))
