#!/usr/bin/env python3
"""Collect nature clips that hit VBench aesthetic_quality + dynamic_degree bars.

For each category (scenery, plant) we:
  1. Over-gather Tier-A survivors using all of that category's matrix queries
     (run_vbench_dataset._tier_a: batched download + strict motion/quality gate).
  2. Score survivors with the EXACT VBench aesthetic_quality + dynamic_degree
     (vbench_exact_scorer.py).
  3. Keep clips with dynamic == 1 AND aesthetic_quality >= AES_THRESHOLD,
     ranked by aesthetic, capped at TARGET_KEEP. Non-kept clips are moved to
     <category>/below_threshold/ (reversible).

Outputs OUTPUT_ROOT/review_manifest.jsonl and OUTPUT_ROOT/vbench_scores.csv
(kept clips, ranked best-first), plus per-category vbench_exact.csv.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import run_vbench_dataset as R
from vbench_keywords import KEYWORDS, Keyword

CATEGORIES = ["scenery", "plant"]
TARGET_KEEP = 8
AES_THRESHOLD = 0.55
POOL_FACTOR = 3                      # gather ~3x survivors before the VBench filter
OUTPUT_ROOT = Path("./vbench_nature")

# Lean download knobs (same as the review80 collection).
R.OVERDOWNLOAD_FACTOR = 1.0
R.BATCH_SIZE = 10
R.CLIPS_PER_VIDEO = "2"
R.HISTORY_FILE = Path("./pexels_download_history.json")


def queries_for(category: str) -> list[str]:
    return [kw.query for kw in KEYWORDS if kw.category == category]


def gather_pool(api_key: str, category: str, out_dir: Path, log_file: Path) -> int:
    """Accumulate ~TARGET_KEEP*POOL_FACTOR Tier-A survivors across the 3 queries."""
    qs = queries_for(category)
    pool_target = TARGET_KEEP * POOL_FACTOR
    per = pool_target / max(len(qs), 1)
    survivors = 0
    for i, q in enumerate(qs, 1):
        # Cumulative target so survivors accumulate in the shared clips/ dir.
        cum = max(1, round(per * i))
        kw = Keyword(category, q, cum)
        survivors = R._tier_a(api_key, kw, out_dir, log_file)
        print(f"    [{category}] after query {i}/{len(qs)}: {survivors} survivors")
    return survivors


def score_and_filter(category: str, out_dir: Path, log_file: Path) -> list[dict]:
    clips = out_dir / "clips"
    csv_path = out_dir / "vbench_exact.csv"
    rc = R._run(
        [sys.executable, "vbench_exact_scorer.py", "-i", str(clips), "-o", str(csv_path)],
        log_file,
    )
    if rc != 0 or not csv_path.exists():
        print(f"    [{category}] scoring failed (rc={rc})", file=sys.stderr)
        return []

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    for r in rows:
        r["aes"] = float(r["aesthetic_quality"])
        r["dyn"] = float(r["dynamic"])
    passing = [r for r in rows if r["aes"] >= AES_THRESHOLD and r["dyn"] >= 1.0]
    passing.sort(key=lambda r: -r["aes"])
    kept = passing[:TARGET_KEEP]
    kept_names = {Path(r["video"]).name for r in kept}

    hold = out_dir / "below_threshold"
    hold.mkdir(exist_ok=True)
    moved = 0
    for p in clips.glob("*.mp4"):
        if p.name not in kept_names:
            shutil.move(str(p), str(hold / p.name))
            moved += 1
    print(f"    [{category}] scored={len(rows)} passing={len(passing)} kept={len(kept)} moved_out={moved}")
    return [{"category": category, **r} for r in kept]


def main() -> int:
    if not R.API_KEY_PATH.exists():
        print(f"API key not found: {R.API_KEY_PATH}", file=sys.stderr)
        return 1
    api_key = R.API_KEY_PATH.read_text(encoding="utf-8").strip()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    all_kept: list[dict] = []
    for category in CATEGORIES:
        out_dir = OUTPUT_ROOT / category
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{category}.log"
        print(f"\n=== [{category}] gathering pool ===")
        gather_pool(api_key, category, out_dir, log_file)
        print(f"=== [{category}] VBench scoring + filter (aes>={AES_THRESHOLD}, dynamic=1) ===")
        all_kept.extend(score_and_filter(category, out_dir, log_file))

    all_kept.sort(key=lambda r: -r["aes"])

    with (OUTPUT_ROOT / "vbench_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "clip", "aesthetic_quality", "dynamic"])
        for r in all_kept:
            w.writerow([r["category"], Path(r["video"]).name, f"{r['aes']:.4f}", f"{r['dyn']:.0f}"])

    with (OUTPUT_ROOT / "review_manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in all_kept:
            name = Path(r["video"]).name
            f.write(json.dumps({
                "id": Path(name).stem,
                "video": f"{r['category']}/clips/{name}",
                "category": r["category"],
                "aesthetic_quality": round(r["aes"], 4),
                "dynamic": int(r["dyn"]),
            }, ensure_ascii=True) + "\n")

    print(f"\nTOTAL kept: {len(all_kept)}")
    for category in CATEGORIES:
        n = sum(1 for r in all_kept if r["category"] == category)
        print(f"  {category}: {n}")
    print(f"manifest: {OUTPUT_ROOT/'review_manifest.jsonl'}")

    R.lark_bot(
        "info",
        "[VBench nature] collection complete\n"
        f"kept={len(all_kept)} (aes>={AES_THRESHOLD}, dynamic=1)\n"
        f"output: {OUTPUT_ROOT.resolve()}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
