#!/usr/bin/env python3
"""Final validation for the processed dataset.

Usage:
    cd /workspace/top400_combined_motion_captioned
    python final_validation.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/workspace/nexisgen")

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "clip_id",
    "clip_uri",
    "clip_sha256",
    "first_frame_uri",
    "first_frame_sha256",
    "source_video_id",
    "clip_start_sec",
    "duration_sec",
    "width",
    "height",
    "fps",
    "num_frames",
    "source_video_url",
    "caption",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Final validation for processed dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Dataset directory to validate.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_dir = args.dataset_dir.resolve()
    parquet_path = dataset_dir / "dataset.parquet"
    manifest_path = dataset_dir / "manifest.jsonl"
    clips_dir = dataset_dir / "clips"
    frames_dir = dataset_dir / "frames"

    lines: list[str] = []
    lines.append("# 最终验证报告\n")
    lines.append(f"\n- 数据集目录：`{dataset_dir}`\n")

    errors: list[str] = []

    # 1. Parquet exists and has correct shape.
    if not parquet_path.exists():
        raise SystemExit(f"dataset.parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    lines.append(f"\n## 1. Parquet 基本检查\n")
    lines.append(f"\n- 行数：{len(df)}\n")
    lines.append(f"- 列数：{len(df.columns)}\n")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        lines.append(f"- ❌ 缺少字段：{missing_cols}\n")
    else:
        lines.append(f"- ✅ 所有 14 个 ClipRecord 字段齐全\n")

    extra_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra_cols:
        errors.append(f"Extra columns: {extra_cols}")
        lines.append(f"- ❌ 多余字段：{extra_cols}\n")
    else:
        lines.append(f"- ✅ 无多余字段\n")

    # 2. Manifest consistency.
    lines.append(f"\n## 2. Manifest 检查\n")
    if manifest_path.exists():
        manifest_rows: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    manifest_rows.append(json.loads(line))
        lines.append(f"- manifest.jsonl 行数：{len(manifest_rows)}\n")
        manifest_ids = {r.get("id") for r in manifest_rows}
        parquet_ids = set(df["clip_id"].tolist())
        if manifest_ids == parquet_ids:
            lines.append(f"- ✅ manifest 与 parquet clip_id 完全一致\n")
        else:
            errors.append("manifest clip_ids mismatch with parquet")
            lines.append(f"- ❌ manifest 与 parquet clip_id 不一致\n")
    else:
        errors.append("manifest.jsonl not found")
        lines.append(f"- ❌ manifest.jsonl 不存在\n")

    # 3. Files exist.
    lines.append(f"\n## 3. 文件存在性检查\n")
    clip_count = len(list(clips_dir.glob("*.mp4")))
    frame_count = len(list(frames_dir.glob("*.jpg")))
    lines.append(f"- clips/ 视频数：{clip_count}\n")
    lines.append(f"- frames/ 首帧数：{frame_count}\n")
    if clip_count == len(df):
        lines.append(f"- ✅ 视频文件数量匹配\n")
    else:
        errors.append(f"clip count mismatch: {clip_count} vs {len(df)}")
        lines.append(f"- ❌ 视频文件数量不匹配\n")
    if frame_count == len(df):
        lines.append(f"- ✅ 首帧文件数量匹配\n")
    else:
        errors.append(f"frame count mismatch: {frame_count} vs {len(df)}")
        lines.append(f"- ❌ 首帧文件数量不匹配\n")

    # 4. SHA256 consistency (sample all).
    lines.append(f"\n## 4. SHA256 一致性检查\n")
    sha256_mismatches = 0
    frame_sha256_mismatches = 0
    for _, row in df.iterrows():
        clip_path = dataset_dir / row["clip_uri"]
        frame_path = dataset_dir / row["first_frame_uri"]
        if clip_path.exists():
            actual = sha256_file(clip_path)
            if actual != row["clip_sha256"]:
                sha256_mismatches += 1
        if frame_path.exists():
            actual = sha256_file(frame_path)
            if actual != row["first_frame_sha256"]:
                frame_sha256_mismatches += 1
    lines.append(f"- clip_sha256 不匹配数：{sha256_mismatches}\n")
    lines.append(f"- first_frame_sha256 不匹配数：{frame_sha256_mismatches}\n")
    if sha256_mismatches == 0:
        lines.append(f"- ✅ 所有 clip_sha256 与实际文件一致\n")
    else:
        errors.append(f"{sha256_mismatches} clip sha256 mismatches")
        lines.append(f"- ❌ 存在 clip_sha256 不一致\n")
    if frame_sha256_mismatches == 0:
        lines.append(f"- ✅ 所有 first_frame_sha256 与实际文件一致\n")
    else:
        errors.append(f"{frame_sha256_mismatches} frame sha256 mismatches")
        lines.append(f"- ❌ 存在 first_frame_sha256 不一致\n")

    # 5. Specs check.
    lines.append(f"\n## 5. Nexisgen 规格检查\n")
    spec_errors = 0
    for _, row in df.iterrows():
        if row["width"] != 1280:
            spec_errors += 1
        if row["height"] != 704:
            spec_errors += 1
        if abs(row["fps"] - 24.0) > 0.5:
            spec_errors += 1
        if row["num_frames"] != 121:
            spec_errors += 1
        if abs(row["duration_sec"] - 5.041667) > 0.5:
            spec_errors += 1
    lines.append(f"- 规格不匹配行数：{spec_errors}\n")
    if spec_errors == 0:
        lines.append(f"- ✅ 所有 clip 符合 1280x704 @ 24fps, 121 帧, ~5.04s 规格\n")
    else:
        errors.append(f"{spec_errors} rows fail spec check")
        lines.append(f"- ❌ 存在规格不符合的 clip\n")

    # 6. Global uniqueness (sha256).
    lines.append(f"\n## 6. 唯一性检查\n")
    lines.append(f"- clip_sha256 唯一值：{df['clip_sha256'].nunique()} / {len(df)}\n")
    lines.append(f"- first_frame_sha256 唯一值：{df['first_frame_sha256'].nunique()} / {len(df)}\n")
    lines.append(f"- source_video_id 唯一值：{df['source_video_id'].nunique()} / {len(df)}\n")
    lines.append(f"- source_video_url 唯一值：{df['source_video_url'].nunique()} / {len(df)}\n")

    # 7. Verdict.
    lines.append(f"\n## 7. 最终结论\n")
    if errors:
        lines.append(f"\n❌ **验证未通过**，发现 {len(errors)} 个问题：\n")
        for err in errors:
            lines.append(f"- {err}\n")
    else:
        lines.append(f"\n✅ **全部验证通过**：数据集完整、sha256 一致、规格符合 Nexisgen 要求。\n")

    md_text = "".join(lines)
    output_path = dataset_dir / "final_validation_report.md"
    output_path.write_text(md_text, encoding="utf-8")
    logger.info("Wrote final validation report to %s", output_path)
    print(md_text)


if __name__ == "__main__":
    main()
