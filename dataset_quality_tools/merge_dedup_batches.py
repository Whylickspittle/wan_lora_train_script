#!/usr/bin/env python3
"""Merge existing final clips with new gate-kept clips and deduplicate by source video ID."""
import csv
import json
import shutil
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXIST_ROOT = HERE / "motion_scenery_final_merged"
NEW_ROOT = HERE / "motion_scenery_final_300"
OUT_ROOT = HERE / "motion_scenery_final_merged_300"

CLIP_RE = re.compile(r"clip_(\d+)_node[\d.]+\.mp4$")


def video_id(name: str) -> str | None:
    m = CLIP_RE.search(name)
    return m.group(1) if m else None


def category_from_staged(name: str) -> str:
    # e.g. clouds__clouds_drifting... -> clouds
    # geo_wonders__waves... -> geo_wonders
    # wind__wind... -> wind
    if "__" in name:
        return name.split("__", 1)[0]
    return "unknown"


def load_existing_scores() -> dict[str, dict]:
    scores: dict[str, dict] = {}
    csv_path = EXIST_ROOT / "final_scores_for_review.csv"
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = video_id(row["video"])
            if vid:
                scores[vid] = row
    return scores


def load_new_scores() -> dict[str, dict]:
    motion: dict[str, dict] = {}
    with (NEW_ROOT / "gate_motion.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            motion[Path(row["video"]).name] = row
    aes: dict[str, dict] = {}
    with (NEW_ROOT / "gate_aesthetic.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aes[Path(row["video"]).name] = row
    prompts: dict[str, str] = {}
    with (NEW_ROOT / "final_manifest.jsonl").open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompts[Path(rec["video"]).name] = rec.get("prompt", "")

    scores: dict[str, dict] = {}
    for name, mrec in motion.items():
        vid = video_id(name)
        if not vid:
            continue
        arec = aes.get(name, {})
        scores[vid] = {
            "category": category_from_staged(name),
            "video": name,
            "motion_mean": mrec["motion_mean"],
            "motion_median": mrec["motion_median"],
            "motion_max": mrec["motion_max"],
            "moving_ratio": mrec["moving_ratio"],
            "dynamic": mrec["dynamic"],
            "aesthetic_quality": arec.get("aesthetic_quality", ""),
            "composite_score": "",  # not computed for new batch
            "prompt": prompts.get(name, ""),
        }
    return scores


def copy_clip(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copy2(src, dst)


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    clips_out = OUT_ROOT / "clips"
    clips_out.mkdir(exist_ok=True)

    existing_scores = load_existing_scores()
    new_scores = load_new_scores()

    # Collect existing kept clips from motion_ok + aesthetic_exception dirs
    existing_sources: list[Path] = []
    for sub in ("clips_motion_ok", "clips_aesthetic_exception"):
        d = EXIST_ROOT / sub
        if d.exists():
            existing_sources.extend(sorted(d.glob("*.mp4")))

    kept_existing = 0
    kept_new = 0
    skipped_dup = 0
    seen_ids: set[str] = set()
    merged_rows: list[dict] = []

    # Add existing first (they are already vetted)
    for src in existing_sources:
        vid = video_id(src.name)
        if not vid or vid in seen_ids:
            continue
        seen_ids.add(vid)
        copy_clip(src, clips_out / src.name)
        row = existing_scores.get(vid)
        if row:
            merged_rows.append(row)
        kept_existing += 1

    # Add new gate-kept clips, skip duplicates
    for src in sorted((NEW_ROOT / "clips").glob("*.mp4")):
        vid = video_id(src.name)
        if not vid:
            continue
        if vid in seen_ids:
            skipped_dup += 1
            continue
        seen_ids.add(vid)
        copy_clip(src, clips_out / src.name)
        row = new_scores.get(vid)
        if row:
            merged_rows.append(row)
        kept_new += 1

    # Write merged final_scores_for_review.csv
    fieldnames = ["category", "video", "motion_mean", "motion_median", "motion_max",
                  "moving_ratio", "dynamic", "aesthetic_quality", "composite_score", "prompt"]
    out_csv = OUT_ROOT / "final_scores_for_review.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print("=" * 60)
    print(f"Existing kept clips copied: {kept_existing}")
    print(f"New gate-kept clips copied: {kept_new}")
    print(f"Duplicate new clips skipped: {skipped_dup}")
    print(f"Total merged clips: {len(merged_rows)}")
    print(f"Output: {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
