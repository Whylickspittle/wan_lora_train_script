#!/usr/bin/env python3
"""Gate the extra 50-batch and merge into motion_scenery_final_merged_300."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "motion_scenery_raw_50"
GATE_OUT = HERE / "motion_scenery_final_extra50"
FINAL_ROOT = HERE / "motion_scenery_final_merged_300"

CLIP_RE = __import__("re").compile(r"clip_(\d+)_node[\d.]+\.mp4$")


def video_id(name: str) -> str | None:
    m = CLIP_RE.search(name)
    return m.group(1) if m else None


def run_gate() -> int:
    if GATE_OUT.exists():
        shutil.rmtree(GATE_OUT)
    cmd = [
        sys.executable, str(HERE / "motion_aesthetic_gate.py"),
        "--dataset-root", str(RAW_DIR),
        "--output", str(GATE_OUT),
        "--motion-lo", "15",
        "--aesthetic-floor", "0.55",
    ]
    print("$", " ".join(cmd))
    return subprocess.run(cmd).returncode


def collect_existing_scores() -> dict[str, dict]:
    scores: dict[str, dict] = {}
    csv_path = FINAL_ROOT / "final_scores_for_review.csv"
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = video_id(row["video"])
            if vid:
                scores[vid] = row
    return scores


def collect_new_scores() -> dict[str, dict]:
    motion: dict[str, dict] = {}
    with (GATE_OUT / "gate_motion.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            motion[Path(row["video"]).name] = row
    aes: dict[str, dict] = {}
    with (GATE_OUT / "gate_aesthetic.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aes[Path(row["video"]).name] = row
    prompts: dict[str, str] = {}
    with (GATE_OUT / "final_manifest.jsonl").open(encoding="utf-8") as f:
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
            "category": name.split("__", 1)[0] if "__" in name else "unknown",
            "video": name,
            "motion_mean": mrec["motion_mean"],
            "motion_median": mrec["motion_median"],
            "motion_max": mrec["motion_max"],
            "moving_ratio": mrec["moving_ratio"],
            "dynamic": mrec["dynamic"],
            "aesthetic_quality": arec.get("aesthetic_quality", ""),
            "composite_score": "",
            "prompt": prompts.get(name, ""),
        }
    return scores


def merge() -> int:
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    clips_dir = FINAL_ROOT / "clips"
    clips_dir.mkdir(exist_ok=True)

    existing_scores = collect_existing_scores()
    new_scores = collect_new_scores()

    fieldnames = ["category", "video", "motion_mean", "motion_median", "motion_max",
                  "moving_ratio", "dynamic", "aesthetic_quality", "composite_score", "prompt"]

    # Read current rows
    current_rows: list[dict] = []
    with (FINAL_ROOT / "final_scores_for_review.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            current_rows.append(row)

    seen_ids = {video_id(Path(row["video"]).name) for row in current_rows if video_id(Path(row["video"]).name)}
    added = 0
    skipped = 0

    for src in sorted((GATE_OUT / "clips").glob("*.mp4")):
        vid = video_id(src.name)
        if not vid:
            continue
        if vid in seen_ids:
            skipped += 1
            print(f"skip duplicate source: {src.name}")
            continue
        seen_ids.add(vid)
        dst = clips_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        row = new_scores.get(vid)
        if row:
            current_rows.append(row)
            added += 1

    with (FINAL_ROOT / "final_scores_for_review.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(current_rows)

    print(f"\nAdded {added} new clips, skipped {skipped} duplicates")
    print(f"Total final_scores_for_review rows: {len(current_rows)}")
    return 0


def run_filter() -> int:
    cmd = [
        sys.executable, str(HERE / "filter_low_motion.py"),
        "--root", str(FINAL_ROOT),
    ]
    print("\n$", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main() -> int:
    print("=" * 60)
    print("Step 1/3: RAFT + aesthetic gate on extra batch")
    print("=" * 60)
    if run_gate() != 0:
        print("Gate failed", file=sys.stderr)
        return 1

    print("\n" + "=" * 60)
    print("Step 2/3: Merge into motion_scenery_final_merged_300")
    print("=" * 60)
    if merge() != 0:
        return 1

    print("\n" + "=" * 60)
    print("Step 3/3: Re-run low-motion filter")
    print("=" * 60)
    if run_filter() != 0:
        print("Filter failed", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
