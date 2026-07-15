#!/usr/bin/env python3
"""Check global/within-dataset overlap using Nexisgen validator logic.

Usage:
    cd /workspace/top400_combined_motion_captioned
    python check_global_overlap.py --original /workspace/top400_combined_motion
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

sys.path.insert(0, "/workspace/nexisgen")

logger = logging.getLogger(__name__)

# Mirrored from nexisgen/nexis/protocol.py to avoid full import.
OVERLAP_WINDOW_SEC = 4.5
GLOBAL_OVERLAP_REJECT_THRESHOLD = 100


def canonical_source_key(url: str) -> str:
    """Normalize a source URL to a canonical key for overlap detection."""
    parsed = urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if host == "youtu.be":
        video_id = parsed.path.strip("/")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    if host == "youtube.com" or host.endswith(".youtube.com"):
        query = parse_qs(parsed.query)
        values = query.get("v", [])
        if values and values[0].strip():
            return f"https://www.youtube.com/watch?v={values[0].strip()}"
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] in {"shorts", "embed", "v"} and parts[1].strip():
            return f"https://www.youtube.com/watch?v={parts[1].strip()}"
    return url.strip()


def build_overlap_index(df: pd.DataFrame) -> dict[str, list[float]]:
    """Build {canonical_source_key: [clip_start_sec, ...]} from a dataframe."""
    index: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        key = canonical_source_key(str(row["source_video_url"]))
        index.setdefault(key, []).append(float(row["clip_start_sec"]))
    return index


def within_dataset_overlap(index: dict[str, list[float]]) -> list[tuple[str, float, float]]:
    """Return list of (key, start_a, start_b) pairs that overlap within one dataset."""
    overlaps: list[tuple[str, float, float]] = []
    for key, positions in index.items():
        sorted_positions = sorted(positions)
        for i in range(len(sorted_positions)):
            for j in range(i + 1, len(sorted_positions)):
                if abs(sorted_positions[i] - sorted_positions[j]) < OVERLAP_WINDOW_SEC:
                    overlaps.append((key, sorted_positions[i], sorted_positions[j]))
    return overlaps


def count_index_overlap(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
) -> int:
    """Count positions in `a` that overlap with any position in `b`."""
    if not a or not b:
        return 0
    if sum(len(v) for v in a.values()) > sum(len(v) for v in b.values()):
        a, b = b, a
    count = 0
    for key, positions in a.items():
        other = b.get(key)
        if not other:
            continue
        for p in positions:
            if any(abs(p - q) < OVERLAP_WINDOW_SEC for q in other):
                count += 1
    return count


def overlap_details(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
) -> list[tuple[str, float, float]]:
    """Return detail tuples (key, start_a, start_b) for overlaps between a and b."""
    details: list[tuple[str, float, float]] = []
    for key, positions in a.items():
        other = b.get(key)
        if not other:
            continue
        for p in positions:
            for q in other:
                if abs(p - q) < OVERLAP_WINDOW_SEC:
                    details.append((key, p, q))
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description="Check global overlap using Nexisgen logic.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path.cwd(),
        help="Processed dataset directory.",
    )
    parser.add_argument(
        "--original",
        type=Path,
        default=Path("/workspace/top400_combined_motion"),
        help="Original dataset directory for cross-dataset overlap check.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    dataset_dir = args.dataset_dir.resolve()
    target_parquet = dataset_dir / "dataset.parquet"
    if not target_parquet.exists():
        raise SystemExit(f"Target parquet not found: {target_parquet}")

    target_df = pd.read_parquet(target_parquet)
    logger.info("Loaded target dataset: %d rows", len(target_df))

    lines: list[str] = []
    lines.append("# 全局视频链接去重检查报告\n")
    lines.append(f"\n- 目标数据集：`{dataset_dir}`\n")
    lines.append(f"- 原始数据集：`{args.original}`\n")
    lines.append(f"- 总 clip 数：{len(target_df)}\n")
    lines.append(f"- OVERLAP_WINDOW_SEC：`{OVERLAP_WINDOW_SEC}` 秒\n")
    lines.append(f"- GLOBAL_OVERLAP_REJECT_THRESHOLD：`{GLOBAL_OVERLAP_REJECT_THRESHOLD}`\n")

    # 1. Within-dataset overlap.
    lines.append("\n## 1. 去重规则说明\n")
    lines.append("\nNexisgen 的去重逻辑基于两个字段：\n")
    lines.append("\n1. **canonical_source_key(source_video_url)**：把同一个视频的不同 URL 格式归一化为同一个 key。\n")
    lines.append("   - 例如 YouTube 的 `youtu.be/ABC`、`youtube.com/watch?v=ABC`、`youtube.com/shorts/ABC` 都会被归一化为 `https://www.youtube.com/watch?v=ABC`。\n")
    lines.append("2. **clip_start_sec**：在该 canonical key 下，如果有两个 clip 的开始时间差小于 `OVERLAP_WINDOW_SEC`（4.5 秒），则视为重复。\n")
    lines.append("\n注意：**不使用 clip_sha256 判断重复**。SHA256 只用于文件完整性校验。\n")

    target_index = build_overlap_index(target_df)
    lines.append("\n## 2. 目标数据集内部去重检查\n")
    lines.append(f"\n- canonical key 数量：{len(target_index)}\n")
    lines.append(f"- 总 clip-start 位置数：{sum(len(v) for v in target_index.values())}\n")

    internal_overlaps = within_dataset_overlap(target_index)
    lines.append(f"- 内部重复对数：**{len(internal_overlaps)}**\n")
    if internal_overlaps:
        lines.append("\n重复明细：\n")
        lines.append("\n| canonical_key | start_a | start_b | diff |\n")
        lines.append("|---|---|---|---|\n")
        for key, a, b in internal_overlaps[:20]:
            lines.append(f"| {key} | {a} | {b} | {abs(a - b):.3f} |\n")
    else:
        lines.append("\n✅ 目标数据集内部无重复。\n")

    # 3. Cross-dataset overlap with original.
    lines.append("\n## 3. 与原始数据集的全局去重对比\n")
    orig_parquet = args.original / "dataset.parquet"
    if not orig_parquet.exists():
        lines.append(f"\n⚠️ 原始数据集 parquet 不存在：`{orig_parquet}`，跳过跨数据集对比。\n")
    else:
        orig_df = pd.read_parquet(orig_parquet)
        logger.info("Loaded original dataset: %d rows", len(orig_df))
        orig_index = build_overlap_index(orig_df)
        lines.append(f"\n- 原始数据集 clip 数：{len(orig_df)}\n")
        lines.append(f"- 原始数据集 canonical key 数量：{len(orig_index)}\n")

        cross_count = count_index_overlap(target_index, orig_index)
        lines.append(f"- 跨数据集重复数：**{cross_count}**\n")

        if cross_count:
            details = overlap_details(target_index, orig_index)
            lines.append("\n跨数据集重复明细（前 20）：\n")
            lines.append("\n| canonical_key | target_start | original_start | diff |\n")
            lines.append("|---|---|---|---|\n")
            for key, a, b in details[:20]:
                lines.append(f"| {key} | {a} | {b} | {abs(a - b):.3f} |\n")
        else:
            lines.append("\n✅ 目标数据集与原始数据集之间无全局重复。\n")

    # 4. URL normalization examples.
    lines.append("\n## 4. URL 归一化示例\n")
    lines.append("\n| 原始 URL | canonical_source_key |\n")
    lines.append("|---|---|\n")
    for url in target_df["source_video_url"].head(10).tolist():
        lines.append(f"| {url} | {canonical_source_key(url)} |\n")

    # 5. Verdict.
    lines.append("\n## 5. 结论\n")
    internal_ok = len(internal_overlaps) == 0
    cross_ok = "跨数据集重复数：**0**" in "".join(lines) or "跳过跨数据集对比" in "".join(lines)
    if internal_ok and cross_ok:
        lines.append("\n✅ **去重检查通过**：目标数据集内部无重复，与原始数据集也无全局重复。\n")
    else:
        lines.append("\n❌ **去重检查未通过**：\n")
        if not internal_ok:
            lines.append(f"- 内部重复对数：{len(internal_overlaps)}\n")
        if not cross_ok:
            # Extract cross_count from earlier line for completeness.
            lines.append("- 跨数据集重复数大于 0\n")

    md_text = "".join(lines)

    output_path = dataset_dir / "global_overlap_check.md"
    output_path.write_text(md_text, encoding="utf-8")
    logger.info("Wrote overlap check report to %s", output_path)
    print(md_text)


if __name__ == "__main__":
    main()
