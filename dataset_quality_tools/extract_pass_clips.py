#!/usr/bin/env python3
"""Extract PASS clips and generate urls-style list + filtered parquet."""

import csv
from pathlib import Path

import pandas as pd

REPORT_CSV = Path("/workspace/dataset_final_v2/quality_report/quality_report.csv")
INPUT_PARQUET = Path("/workspace/dataset_final_v2/dataset.parquet")
OUTPUT_PARQUET = Path("/workspace/dataset_final_v2/dataset_pass.parquet")
OUTPUT_URLS = Path("/workspace/dataset_final_v2/pass_urls.txt")


def clip_id_from_path(path: str) -> str:
    """clips/CYvFTC23rEU_0093.mp4 -> CYvFTC23rEU_0093"""
    return Path(path).stem


def format_timestamp(seconds: float) -> str:
    """Format seconds like urls.txt: SS, MM:SS, or H:MM:SS."""
    s = int(round(seconds))
    hours = s // 3600
    minutes = (s % 3600) // 60
    secs = s % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    if minutes > 0:
        return f"{minutes}:{secs:02d}"
    return f"{secs}"


def main():
    # 1. Collect PASS clip IDs from quality report.
    pass_clip_ids = set()
    with REPORT_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["grade"] == "PASS":
                pass_clip_ids.add(clip_id_from_path(row["path"]))

    print(f"PASS clips: {len(pass_clip_ids)}")

    # 2. Read parquet and filter.
    df = pd.read_parquet(INPUT_PARQUET)
    original_count = len(df)
    df_pass = df[df["clip_id"].isin(pass_clip_ids)].copy()
    pass_count = len(df_pass)

    print(f"Original parquet rows: {original_count}")
    print(f"PASS parquet rows: {pass_count}")

    # 3. Write filtered parquet.
    df_pass.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Written: {OUTPUT_PARQUET}")

    # 4. Generate urls-style list grouped by source video.
    grouped = df_pass.sort_values(["source_video_url", "clip_start_sec"]).groupby("source_video_url")

    lines = []
    lines.append(f"# PASS clips URL list")
    lines.append(f"# Generated from {INPUT_PARQUET.name}")
    lines.append(f"# Total PASS clips: {pass_count} / {original_count}")
    lines.append("")

    for url, group in grouped:
        timestamps = [format_timestamp(sec) for sec in group["clip_start_sec"].tolist()]
        lines.append(f"{url} | {', '.join(timestamps)}")

    OUTPUT_URLS.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {OUTPUT_URLS}")


if __name__ == "__main__":
    main()
