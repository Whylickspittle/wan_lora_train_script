#!/usr/bin/env python3
"""Filter the merged motion-scenery clips by RAFT motion + aesthetic exception.

Rule:
- Drop clips where motion_mean < 20 AND motion_median < 20.
- BUT if such a clip has aesthetic_quality >= 0.70, keep it in a special
  "aesthetic_exception" folder instead of dropping.
- All other clips go to the "motion_ok" folder.

Outputs:
- motion_scenery_final_merged/clips_motion_ok/
- motion_scenery_final_merged/clips_aesthetic_exception/
- final_scores_motion_ok.csv
- final_scores_aesthetic_exception.csv
- manifest_motion_ok.jsonl
- manifest_aesthetic_exception.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

MOTION_LO = 20.0
AESTHETIC_EXCEPTION_FLOOR = 0.70


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter low-motion clips with aesthetic exception")
    parser.add_argument("--root", type=Path,
                        default=Path("./motion_scenery_final_merged"),
                        help="Root directory containing clips/ and final_scores_for_review.csv")
    args = parser.parse_args()

    root: Path = args.root
    src_clips = root / "clips"
    scores_csv = root / "final_scores_for_review.csv"

    if not scores_csv.exists():
        print(f"Scores file not found: {scores_csv}", file=__import__("sys").stderr)
        return 1
    if not src_clips.exists():
        print(f"Source clips dir not found: {src_clips}", file=__import__("sys").stderr)
        return 1

    ok_dir = root / "clips_motion_ok"
    aes_dir = root / "clips_aesthetic_exception"
    ok_dir.mkdir(parents=True, exist_ok=True)
    aes_dir.mkdir(parents=True, exist_ok=True)

    ok_rows: list[dict] = []
    aes_rows: list[dict] = []
    dropped_rows: list[dict] = []

    with scores_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            mean = float(row["motion_mean"])
            median = float(row["motion_median"])
            aes = float(row["aesthetic_quality"])
            low_motion = mean < MOTION_LO and median < MOTION_LO

            if low_motion and aes >= AESTHETIC_EXCEPTION_FLOOR:
                aes_rows.append(row)
            elif low_motion:
                dropped_rows.append(row)
            else:
                ok_rows.append(row)

    def copy_out(rows: list[dict], dst_dir: Path) -> int:
        copied = 0
        for row in rows:
            src = src_clips / row["video"]
            if not src.exists():
                # Try basename fallback for mixed naming
                src = src_clips / Path(row["video"]).name
            if not src.exists():
                print(f"WARN: source clip missing: {row['video']}", file=__import__("sys").stderr)
                continue
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
        return copied

    ok_copied = copy_out(ok_rows, ok_dir)
    aes_copied = copy_out(aes_rows, aes_dir)

    def write_csv(rows: list[dict], path: Path) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(ok_rows, root / "final_scores_motion_ok.csv")
    write_csv(aes_rows, root / "final_scores_aesthetic_exception.csv")
    write_csv(dropped_rows, root / "final_scores_dropped.csv")

    def write_manifest(rows: list[dict], path: Path, clip_prefix: str) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                video_name = Path(row["video"]).name
                clip_id = video_name[:-4] if video_name.endswith(".mp4") else video_name
                entry = {
                    "id": clip_id,
                    "video": f"{clip_prefix}/{video_name}",
                    "prompt": row["prompt"],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    write_manifest(ok_rows, root / "manifest_motion_ok.jsonl", "clips_motion_ok")
    write_manifest(aes_rows, root / "manifest_aesthetic_exception.jsonl", "clips_aesthetic_exception")

    print("=" * 60)
    print(f"Source clips : {len(ok_rows) + len(aes_rows) + len(dropped_rows)}")
    print(f"Motion OK    : {len(ok_rows)}  -> {ok_dir}")
    print(f"Aesthetic exc: {len(aes_rows)}  -> {aes_dir}")
    print(f"Dropped      : {len(dropped_rows)}")
    print("=" * 60)
    print(f"OK copied     : {ok_copied}")
    print(f"Exception copied: {aes_copied}")
    print(f"Score tables:")
    print(f"  {root / 'final_scores_motion_ok.csv'}")
    print(f"  {root / 'final_scores_aesthetic_exception.csv'}")
    print(f"  {root / 'final_scores_dropped.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
