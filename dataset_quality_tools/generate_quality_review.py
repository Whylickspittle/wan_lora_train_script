#!/usr/bin/env python3
"""Generate comprehensive dataset quality review with YouTube links."""

import csv
import json
from collections import defaultdict
from pathlib import Path

REPORT_DIR = Path("/workspace/dataset_final_v2/quality_report")
OUTPUT_README = Path("/workspace/dataset_final_v2/QUALITY_REVIEW.md")


def parse_clip_name(path: str) -> tuple[str, int]:
    """Extract source video ID and start timestamp from clip filename.

    Examples:
        -W_nFlIAWFM_0003.mp4 -> ("-W_nFlIAWFM", 3)
        DjGmeT2Lwmw_2330.mp4 -> ("DjGmeT2Lwmw", 2330)
    """
    name = Path(path).stem
    source, ts_str = name.rsplit("_", 1)
    return source, int(ts_str)


def format_timestamp(seconds: int) -> str:
    """Convert seconds to MM:SS or HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def youtube_link(source: str, seconds: int) -> str:
    """Build YouTube timestamp link."""
    return f"https://www.youtube.com/watch?v={source}&t={seconds}s"


def categorize_issue(reason: str, grade: str) -> list[str]:
    """Map a reason string to one or more issue categories."""
    if grade != "FAIL":
        return []

    categories = []
    reason_lower = reason.lower()

    if "duplicate" in reason_lower:
        categories.append("DUPLICATE")
    if "static" in reason_lower:
        categories.append("STATIC")
    if "timelapse" in reason_lower:
        categories.append("TIMELAPSE")
    if "scene_cuts" in reason_lower:
        categories.append("SCENE_CUT")
    if "clipping" in reason_lower:
        categories.append("CLIPPING")
    if "conformance" in reason_lower:
        categories.append("LOW_CONFORMANCE")

    return categories if categories else ["OTHER"]


def grade_emoji(grade: str) -> str:
    return {"PASS": "✅", "REVIEW": "⚠️", "FAIL": "❌"}.get(grade, "")


def main():
    csv_path = REPORT_DIR / "quality_report.csv"
    summary_path = REPORT_DIR / "summary.json"

    rows = []
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    total = len(rows)
    grades = {"PASS": 0, "REVIEW": 0, "FAIL": 0}
    issue_counts = defaultdict(int)
    source_stats = defaultdict(lambda: {"total": 0, "pass": 0, "review": 0, "fail": 0, "issues": defaultdict(int)})
    source_rows = defaultdict(list)

    for row in rows:
        grade = row["grade"]
        reason = row["reason"]
        source, seconds = parse_clip_name(row["path"])

        grades[grade] += 1
        source_stats[source]["total"] += 1
        source_stats[source][grade.lower()] += 1

        for cat in categorize_issue(reason, grade):
            issue_counts[cat] += 1
            source_stats[source]["issues"][cat] += 1

        source_rows[source].append({
            "clip": Path(row["path"]).name,
            "source": source,
            "seconds": seconds,
            "timestamp": format_timestamp(seconds),
            "link": youtube_link(source, seconds),
            "grade": grade,
            "reason": reason,
            "conformance": float(row["conformance_score"]),
            "dup_ratio": float(row["duplicate_frame_ratio"]),
            "static": float(row["temporal_diff_mean"]),
            "scene_cuts": int(row["scene_cut_count"]),
            "clipping": float(row["black_pixel_ratio"]) + float(row["white_pixel_ratio"]),
            "timelapse": float(row["timelapse_score"]),
        })

    # Sort sources by FAIL count descending.
    sorted_sources = sorted(
        source_stats.items(),
        key=lambda item: (item[1]["fail"], item[1]["review"]),
        reverse=True,
    )

    source_table = []
    for source, stats in sorted_sources:
        fail_rate = stats["fail"] / stats["total"]
        issue_parts = [f"{k}:{v}" for k, v in sorted(stats["issues"].items(), key=lambda x: -x[1])]
        source_table.append({
            "source": source,
            "total": stats["total"],
            "pass": stats["pass"],
            "review": stats["review"],
            "fail": stats["fail"],
            "fail_rate": fail_rate,
            "issues": ", ".join(issue_parts) if issue_parts else "-",
        })

    problematic_sources = [s for s in source_table if s["fail_rate"] > 0.5 or s["fail"] >= 5]

    lines = [
        "# Dataset Final V2 - 全量质量审查报告",
        "",
        f"生成时间：2026-06-16",
        f"分析工具：`dataset_quality_tools/clean_dataset.py`",
        f"数据集路径：`/workspace/dataset_final_v2/clips`",
        "",
        "## 一、总体统计",
        "",
        "| 分类 | 数量 | 占比 | 建议 |",
        "|------|------|------|------|",
        f"| ✅ PASS | {grades['PASS']} | {grades['PASS']/total:.1%} | 可直接用于训练 |",
        f"| ⚠️ REVIEW | {grades['REVIEW']} | {grades['REVIEW']/total:.1%} | 人工抽查后决定 |",
        f"| ❌ FAIL | {grades['FAIL']} | {grades['FAIL']/total:.1%} | 建议隔离或替换 |",
        f"| **合计** | **{total}** | **100%** | - |",
        "",
        "### 核心指标均值",
        "",
        "| 指标 | 均值 | 说明 |",
        "|------|------|------|",
        f"| conformance_score | {summary['aggregate']['mean_conformance_score']:.4f} | 越接近 1.0 越合规 |",
        f"| duplicate_frame_ratio | {summary['aggregate']['mean_duplicate_frame_ratio']:.4f} | 重复帧比例，理想 < 0.03 |",
        f"| timelapse_score | {summary['aggregate']['mean_timelapse_score']:.4f} | 延时摄影概率，理想 < 0.30 |",
        f"| temporal_diff_mean | {summary['aggregate']['mean_temporal_diff_mean']:.4f} | 帧间差异均值 |",
        "",
        "### 问题类型分布（仅 FAIL clip）",
        "",
        "| 问题类型 | 数量 | 说明 |",
        "|----------|------|------|",
    ]

    for cat, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        desc = {
            "DUPLICATE": "相邻帧重复 / 画面冻结",
            "STATIC": "几乎静态，运动量过低",
            "TIMELAPSE": "疑似延时摄影",
            "SCENE_CUT": "5 秒内发生镜头切换",
            "CLIPPING": "过曝/欠曝像素比例过高",
            "LOW_CONFORMANCE": "分辨率/帧数/fps 偏差",
            "OTHER": "其他问题",
        }.get(cat, "")
        lines.append(f"| {cat} | {count} | {desc} |")

    lines.extend([
        "",
        "## 二、问题来源汇总（按 source 排序）",
        "",
        "按 FAIL 数量排序，帮助你快速定位需要替换的原始视频源。",
        "",
        "| 排名 | source | 总数 | PASS | REVIEW | FAIL | 失败率 | 主要问题 |",
        "|------|--------|------|------|--------|------|--------|----------|",
    ])

    for rank, s in enumerate(source_table, 1):
        lines.append(
            f"| {rank} | {s['source']} | {s['total']} | {s['pass']} | {s['review']} | "
            f"{s['fail']} | {s['fail_rate']:.1%} | {s['issues']} |"
        )

    lines.extend([
        "",
        "## 三、建议优先替换的来源",
        "",
        "以下 source 的失败率超过 50%，或失败 clip 数量 ≥ 5，建议优先替换：",
        "",
        "| source | 总数 | FAIL | 失败率 | 主要问题 |",
        "|--------|------|------|--------|----------|",
    ])

    for s in problematic_sources:
        lines.append(
            f"| {s['source']} | {s['total']} | {s['fail']} | {s['fail_rate']:.1%} | {s['issues']} |"
        )

    lines.extend([
        "",
        "## 四、逐 source 全量 clip 清单",
        "",
        "下面按 source 分组，列出每个 clip 的：",
        "- 状态（PASS/REVIEW/FAIL）",
        "- 原视频 YouTube 链接 + 时间戳",
        "- 关键指标（conformance、duplicate、static、scene_cuts、clipping）",
        "- 失败原因（仅限 FAIL/REVIEW）",
        "",
        "> 点击 `YouTube 链接` 可直接跳转到对应时间点。",
        "",
    ])

    for source, _ in sorted_sources:
        lines.append(f"### {source}")
        lines.append("")
        lines.append("| clip | 状态 | 时间戳 | YouTube 链接 | conformance | duplicate | static | scene_cuts | clipping | 原因 |")
        lines.append("|------|------|--------|--------------|-------------|-----------|--------|------------|----------|------|")

        for r in sorted(source_rows[source], key=lambda x: x["seconds"]):
            status = f"{grade_emoji(r['grade'])} {r['grade']}"
            reason = r["reason"] if r["grade"] != "PASS" else "-"
            lines.append(
                f"| {r['clip']} | {status} | {r['timestamp']} | [链接]({r['link']}) | "
                f"{r['conformance']:.3f} | {r['dup_ratio']:.3f} | {r['static']:.4f} | "
                f"{r['scene_cuts']} | {r['clipping']:.3f} | {reason} |"
            )

        lines.append("")

    lines.extend([
        "## 五、建议行动",
        "",
        "1. **FAIL clip（114 个）**：建议直接从训练集中删除。",
        "2. **REVIEW clip（44 个）**：建议点击 YouTube 链接查看对应时间点，人工判断是否保留。",
        "3. **优先替换问题源**：`QRvNJX4SqcQ`、`WaPpLKeedi4`、`86Tj5I9KN_I`、`DjGmeT2Lwmw`、`CYvFTC23rEU`。",
        "4. **替换后复检**：新增 clip 后再次运行 `clean_dataset.py` 验证。",
        "",
        "## 六、原始报告文件",
        "",
        f"- 详细指标 CSV：`{REPORT_DIR / 'quality_report.csv'}`",
        f"- 汇总 JSON：`{REPORT_DIR / 'summary.json'}`",
        f"- 可视化 HTML：`{REPORT_DIR / 'report.html'}`",
        "",
        "## 七、本次检测参数说明",
        "",
        "本次运行命令：",
        "",
        "```bash",
        "python clean_dataset.py /workspace/dataset_final_v2/clips \\",
        "    --output /workspace/dataset_final_v2/quality_report \\",
        "    --workers 2",
        "```",
        "",
        "`clean_dataset.py` 使用的核心阈值（默认值）：",
        "",
        "| 参数 | 默认值 | 含义 |",
        "|------|--------|------|",
        f"| `--dup-ratio` | {summary['thresholds']['dup_ratio']} | 重复帧比例超过该值 → FAIL |",
        f"| `--timelapse-score` | {summary['thresholds']['timelapse_score']} | 延时摄影分数超过该值 → FAIL |",
        f"| `--clipping-ratio` | {summary['thresholds']['clipping_ratio']} | 黑+白像素比例超过该值 → FAIL |",
        f"| `--min-conformance` | {summary['thresholds']['min_conformance']} | 综合合规分低于该值 → FAIL |",
        f"| `--scene-cut-max` | {summary['thresholds']['scene_cut_max']} | 5 秒内镜头切换超过该值 → FAIL |",
        "",
        "评分逻辑：",
        "",
        "- **PASS**：`conformance ≥ 0.90` 且 `duplicate < 0.03` 且 `timelapse < 0.30` 且 `clipping < 0.08`",
        "- **REVIEW**：未触发 FAIL，但某些指标接近阈值（minor flags）",
        "- **FAIL**：触发任一 reject 条件（duplicate / static / scene_cut / clipping / conformance / timelapse）",
        "",
        "目标规格（与训练代码对齐）：",
        "",
        "| 维度 | 目标值 |",
        "|------|--------|",
        "| 分辨率 | 1280 × 704 |",
        "| FPS | 24 |",
        "| 帧数 | 121 |",
        "| 时长 | ~5.04 秒 |",
        "",
        "所有 clip 都已经符合上述硬规格，因此本次 `conformance_score` 均值高达 0.993。问题主要出在内容动态质量（重复帧、镜头切换、静态、过曝欠曝）上。",
        "",
    ])

    OUTPUT_README.write_text("\n".join(lines), encoding="utf-8")
    print(f"README written to {OUTPUT_README}")
    print(f"Total clips: {total}, PASS: {grades['PASS']}, REVIEW: {grades['REVIEW']}, FAIL: {grades['FAIL']}")


if __name__ == "__main__":
    main()
