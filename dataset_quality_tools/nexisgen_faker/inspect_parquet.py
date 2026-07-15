#!/usr/bin/env python3
"""Inspect dataset.parquet and generate a Markdown report.

Usage:
    cd /workspace/top400_combined_motion_captioned
    python inspect_parquet.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset.parquet and write Markdown report.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Dataset directory containing dataset.parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Markdown path (default: <dataset-dir>/parquet_inspection.md).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_dir = args.dataset_dir.resolve()
    parquet_path = dataset_dir / "dataset.parquet"

    if not parquet_path.exists():
        raise SystemExit(f"dataset.parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    lines: list[str] = []
    lines.append("# dataset.parquet 数据观察\n")
    lines.append(f"\n- 数据集目录：`{dataset_dir}`\n")
    lines.append(f"- Parquet 路径：`{parquet_path}`\n")
    lines.append(f"- 行数：**{len(df)}**\n")
    lines.append(f"- 列数：**{len(df.columns)}**\n")

    # Schema table.
    lines.append("\n## 1. Schema（列名与类型）\n")
    lines.append("\n| 列名 | 类型 | 非空数 | 说明 |\n")
    lines.append("|---|---|---|---|\n")
    for col in df.columns:
        non_null = df[col].notna().sum()
        lines.append(f"| {col} | {df[col].dtype} | {non_null} | |\n")

    # First 10 rows.
    lines.append("\n## 2. 前 10 行数据\n")
    lines.append("\n```\n")
    lines.append(df.head(10).to_string(index=False))
    lines.append("\n```\n")

    # Key statistics.
    lines.append("\n## 3. 关键字段统计\n")

    # clip_id uniqueness.
    unique_clip_ids = df["clip_id"].nunique()
    lines.append(f"\n- `clip_id` 唯一值数：{unique_clip_ids} / {len(df)}\n")

    # clip_sha256 uniqueness.
    unique_clip_sha256 = df["clip_sha256"].nunique()
    lines.append(f"- `clip_sha256` 唯一值数：{unique_clip_sha256} / {len(df)}\n")

    # first_frame_sha256 uniqueness.
    unique_frame_sha256 = df["first_frame_sha256"].nunique()
    lines.append(f"- `first_frame_sha256` 唯一值数：{unique_frame_sha256} / {len(df)}\n")

    # source_video_id / url uniqueness.
    unique_source_ids = df["source_video_id"].nunique()
    unique_source_urls = df["source_video_url"].nunique()
    lines.append(f"- `source_video_id` 唯一值数：{unique_source_ids} / {len(df)}\n")
    lines.append(f"- `source_video_url` 唯一值数：{unique_source_urls} / {len(df)}\n")

    # clip_start_sec distribution.
    lines.append("\n### clip_start_sec 分布\n")
    lines.append("\n```\n")
    lines.append(df["clip_start_sec"].describe().to_string())
    lines.append("\n```\n")

    # Resolution / fps / frame count.
    lines.append("\n### 视频规格分布\n")
    lines.append("\n| 字段 | 唯一值 | 最常见值 |\n")
    lines.append("|---|---|---|\n")
    for col in ["width", "height", "fps", "num_frames"]:
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "N/A"
        lines.append(f"| {col} | {df[col].nunique()} | {mode_val} |\n")

    lines.append("\n### duration_sec 分布\n")
    lines.append("\n```\n")
    lines.append(df["duration_sec"].describe().to_string())
    lines.append("\n```\n")

    # Caption length.
    df_copy = df.copy()
    df_copy["caption_length"] = df_copy["caption"].astype(str).str.len()
    lines.append("\n### caption 长度分布\n")
    lines.append("\n```\n")
    lines.append(df_copy["caption_length"].describe().to_string())
    lines.append("\n```\n")

    # Sample captions.
    lines.append("\n## 4. 随机 5 条 caption 样例\n")
    sample = df.sample(n=min(5, len(df)), random_state=42)
    for _, row in sample.iterrows():
        lines.append(f"\n**clip_id**: `{row['clip_id']}`\n")
        lines.append(f"\n> {row['caption']}\n")

    # Source URL sample.
    lines.append("\n## 5. source_video_url 样例\n")
    lines.append("\n```\n")
    for url in df["source_video_url"].head(5).tolist():
        lines.append(f"{url}\n")
    lines.append("```\n")

    md_text = "".join(lines)

    output_path = args.output or dataset_dir / "parquet_inspection.md"
    output_path.write_text(md_text, encoding="utf-8")
    logger.info("Wrote inspection report to %s", output_path)
    print(md_text)


if __name__ == "__main__":
    main()
