#!/usr/bin/env python3
"""Package PASS-only dataset into a nexisgen-style folder and zip archive."""

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SRC_DIR = Path("/workspace/dataset_final_v2")
DST_DIR = Path("/workspace/dataset_final_v2_pass")
ZIP_PATH = Path("/workspace/dataset_final_v2_pass.zip")
REPORT_CSV = SRC_DIR / "quality_report" / "quality_report.csv"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def clip_id_from_path(path: str) -> str:
    return Path(path).stem


def main():
    # 1. Collect PASS clip IDs.
    pass_clip_ids = set()
    with REPORT_CSV.open("r", encoding="utf-8") as fh:
        import csv
        reader = csv.DictReader(fh)
        for row in reader:
            if row["grade"] == "PASS":
                pass_clip_ids.add(clip_id_from_path(row["path"]))

    print(f"PASS clips: {len(pass_clip_ids)}")

    # 2. Create output directories.
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    (DST_DIR / "clips").mkdir(parents=True, exist_ok=True)
    (DST_DIR / "frames").mkdir(parents=True, exist_ok=True)

    # 3. Read parquet and filter.
    df = pd.read_parquet(SRC_DIR / "dataset.parquet")
    df_pass = df[df["clip_id"].isin(pass_clip_ids)].copy()
    print(f"Parquet rows: {len(df)} -> {len(df_pass)}")

    # 4. Copy clips and frames.
    copied_clips = 0
    copied_frames = 0
    for clip_id in pass_clip_ids:
        src_clip = SRC_DIR / "clips" / f"{clip_id}.mp4"
        dst_clip = DST_DIR / "clips" / f"{clip_id}.mp4"
        if src_clip.exists():
            shutil.copy2(src_clip, dst_clip)
            copied_clips += 1

        src_frame = SRC_DIR / "frames" / f"{clip_id}.jpg"
        dst_frame = DST_DIR / "frames" / f"{clip_id}.jpg"
        if src_frame.exists():
            shutil.copy2(src_frame, dst_frame)
            copied_frames += 1

    print(f"Copied clips: {copied_clips}, frames: {copied_frames}")

    # 5. Write parquet.
    parquet_path = DST_DIR / "dataset.parquet"
    df_pass.to_parquet(parquet_path, index=False)

    # 6. Write manifest.json.
    manifest = {
        "protocol_version": "2.0.0",
        "schema_version": "2.0.0",
        "spec_id": "video_v1",
        "netuid": 70,
        "miner_hotkey": "test_miner",
        "interval_id": 1,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "record_count": len(df_pass),
        "dataset_sha256": sha256_file(parquet_path),
    }
    manifest_path = DST_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 7. Write README.md.
    readme = f"""# Dataset Final V2 - PASS Only

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
Source: `/workspace/dataset_final_v2`
Filter: `clean_dataset.py` grade == PASS

## Contents

| Item | Count |
|------|-------|
| Total clips in source | {len(df)} |
| PASS clips | {len(df_pass)} |
| FAIL clips | {len(df) - len(df_pass)} |

## Structure

```text
clips/         PASS-only video clips
frames/        First frame thumbnails for PASS clips
dataset.parquet   Parquet with PASS-only rows
manifest.json     Interval manifest
```

## Notes

- This package contains only clips that passed the quality checks in
  `dataset_quality_tools/clean_dataset.py`.
- For Nexisgen protocol submission, an interval must contain exactly 400 records.
  This PASS-only subset has {len(df_pass)} records and is intended as a cleaned
  dataset archive, not a direct submission package.
"""
    (DST_DIR / "README.md").write_text(readme, encoding="utf-8")

    # 8. Create zip archive.
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in DST_DIR.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(DST_DIR.parent))

    print(f"\nPackage created: {DST_DIR}")
    print(f"Zip archive: {ZIP_PATH}")
    print(f"Zip size: {ZIP_PATH.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
