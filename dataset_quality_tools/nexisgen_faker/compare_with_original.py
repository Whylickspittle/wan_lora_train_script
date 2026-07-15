#!/usr/bin/env python3
"""Compare processed dataset with its original source.

Usage:
    cd /workspace/top400_combined_motion_captioned
    python compare_with_original.py --original /workspace/top400_combined_motion
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


COMPARE_FIELDS = [
    "clip_sha256",
    "first_frame_sha256",
    "source_video_id",
    "source_video_url",
    "clip_start_sec",
    "duration_sec",
    "width",
    "height",
    "fps",
    "num_frames",
    "caption",
]


def field_equal(a: Any, b: Any) -> bool:
    """Compare two field values, treating numeric types loosely."""
    if pd.isna(a) and pd.isna(b):
        return True
    try:
        return float(a) == float(b)
    except Exception:
        return str(a) == str(b)


def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare processed dataset with original.")
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("/workspace/top400_combined_motion"),
        help="Path to the original dataset directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Path to the processed dataset directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without writing files.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    original_dir = args.original.resolve()
    target_dir = args.dataset_dir.resolve()

    orig_parquet = original_dir / "dataset.parquet"
    target_parquet = target_dir / "dataset.parquet"

    if not orig_parquet.exists():
        raise SystemExit(f"Original parquet not found: {orig_parquet}")
    if not target_parquet.exists():
        raise SystemExit(f"Target parquet not found: {target_parquet}")

    orig_df = pd.read_parquet(orig_parquet)
    target_df = pd.read_parquet(target_parquet)

    logger.info("Loaded %d rows from original", len(orig_df))
    logger.info("Loaded %d rows from target", len(target_df))

    merged = orig_df.merge(
        target_df,
        on="clip_id",
        suffixes=("_orig", "_target"),
        how="outer",
    )

    # Build per-row comparison.
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        clip_id = row["clip_id"]
        clip_uri_orig = row.get("clip_uri_orig") or f"clips/{clip_id}.mp4"
        clip_uri_target = row.get("clip_uri_target") or f"clips/{clip_id}.mp4"
        size_orig = (original_dir / clip_uri_orig).stat().st_size
        size_target = (target_dir / clip_uri_target).stat().st_size
        size_ratio = size_target / size_orig if size_orig else 0.0

        record: dict[str, Any] = {
            "idx": len(rows),
            "clip_id": clip_id,
            "size_orig": size_orig,
            "size_target": size_target,
            "size_ratio": round(size_ratio, 3),
        }
        for field in COMPARE_FIELDS:
            val_orig = row.get(f"{field}_orig")
            val_target = row.get(f"{field}_target")
            same = field_equal(val_orig, val_target)
            record[f"{field}_same"] = same
            record[f"{field}_orig"] = val_orig
            record[f"{field}_target"] = val_target
        rows.append(record)

    comparison_df = pd.DataFrame(rows)

    # Summary counts.
    summary_lines: list[str] = []
    summary_lines.append("# 伪装数据集 vs 原始数据集对比报告\n")
    summary_lines.append(f"\n生成时间：2026-07-15\n")
    summary_lines.append(f"\n## 1. 整体统计\n")
    summary_lines.append(f"\n- 总 clip 数：{len(comparison_df)}\n")

    for field in COMPARE_FIELDS:
        same_count = int(comparison_df[f"{field}_same"].sum())
        summary_lines.append(f"- {field} 相同数：{same_count}\n")

    total_orig = comparison_df["size_orig"].sum()
    total_target = comparison_df["size_target"].sum()
    summary_lines.append(f"- 原始数据集 clips 总大小：{format_bytes(total_orig)}\n")
    summary_lines.append(f"- 伪装数据集 clips 总大小：{format_bytes(total_target)}\n")
    summary_lines.append(f"- 大小比例：{total_target / total_orig * 100:.1f}%\n")

    summary_lines.append(f"\n## 2. 字段差异表\n")
    summary_lines.append("\n| 字段 | 相同数 | 不同数 | 说明 |\n")
    summary_lines.append("|---|---|---|---|\n")
    for field in COMPARE_FIELDS:
        same_count = int(comparison_df[f"{field}_same"].sum())
        diff_count = len(comparison_df) - same_count
        summary_lines.append(f"| {field} | {same_count} | {diff_count} | |\n")

    summary_lines.append(f"\n## 3. 前 10 行详细对比\n")
    summary_lines.append("\n| idx | clip_id | clip_sha256_same | source_video_id_same | source_video_url_same | size_ratio |\n")
    summary_lines.append("|---|---|---|---|---|---|\n")
    for _, row in comparison_df.head(10).iterrows():
        summary_lines.append(
            f"| {int(row['idx'])} | {row['clip_id']} | {row['clip_sha256_same']} | "
            f"{row['source_video_id_same']} | {row['source_video_url_same']} | {row['size_ratio']} |\n"
        )

    summary_lines.append(f"\n## 4. 说明\n")
    summary_lines.append("\n1. **clip_sha256 / first_frame_sha256 不同**：重新编码的目标。\n")
    summary_lines.append("2. **source_video_id / source_video_url 不同**：已替换为伪造 YouTube 数据。\n")
    summary_lines.append("3. **duration_sec / width / height / fps / num_frames 相同**：保持 Nexisgen 规格不变。\n")
    summary_lines.append("4. **caption 相同**：只修改来源和编码，prompt 保持一致。\n")

    summary_text = "".join(summary_lines)

    if args.dry_run:
        logger.info("DRY RUN: would write comparison files")
        print(summary_text)
        return

    csv_path = target_dir / "comparison_with_original.csv"
    md_path = target_dir / "comparison_summary.md"

    # Drop raw *_orig/*_target columns from CSV for brevity; keep only _same flags and sizes.
    csv_cols = ["idx", "clip_id", "size_orig", "size_target", "size_ratio"] + [
        f"{field}_same" for field in COMPARE_FIELDS
    ]
    comparison_df[csv_cols].to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    md_path.write_text(summary_text, encoding="utf-8")
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
