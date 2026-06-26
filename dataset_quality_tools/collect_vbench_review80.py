#!/usr/bin/env python3
"""Collect ~N Tier-A PASS clips per VBench category for manual review.

Unlike run_vbench_dataset.py (which over-downloads then Top-K cuts), this keeps
EVERY Tier-A PASS clip so we reach the review target with minimal downloading,
then runs Tier-B proxy scoring + composite ranking over the whole pool so the
reviewer sees clips ordered best-first. Strict thresholds are kept.

Output:
    OUTPUT_ROOT/<category>/clips/                surviving PASS clips
    OUTPUT_ROOT/<category>/vbench_proxy.csv      Tier-B per-clip CLIP scores
    OUTPUT_ROOT/review_manifest.jsonl            all clips, ranked best-first
    OUTPUT_ROOT/review_scores.csv                composite + per-dimension, ranked
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import run_vbench_dataset as R
from vbench_keywords import KEYWORDS, Keyword

OUTPUT_ROOT = Path("./vbench_review80")
PASS_PER_CATEGORY = 10

# Lean download knobs (override the driver's Top-K-oriented defaults).
R.OVERDOWNLOAD_FACTOR = 1.0          # gather exactly the target, no extra
R.BATCH_SIZE = 10                    # small batches to avoid big overshoot
R.CLIPS_PER_VIDEO = "2"
# Use the shared production history so we never re-pull a known source.
R.HISTORY_FILE = Path("./pexels_download_history.json")


def representative_keywords() -> list[Keyword]:
    """First query of each category, with target = PASS_PER_CATEGORY."""
    seen: dict[str, Keyword] = {}
    for kw in KEYWORDS:
        if kw.category not in seen:
            seen[kw.category] = Keyword(kw.category, kw.query, PASS_PER_CATEGORY)
    return list(seen.values())


def tier_b(clips_dir: Path, proxy_csv: Path, log_file: Path) -> None:
    R._run(
        [sys.executable, "score_vbench_proxy.py",
         "-i", str(clips_dir), "-o", str(proxy_csv), "--resume"],
        log_file,
    )


def main() -> int:
    if not R.API_KEY_PATH.exists():
        print(f"API key not found: {R.API_KEY_PATH}", file=sys.stderr)
        return 1
    api_key = R.API_KEY_PATH.read_text(encoding="utf-8").strip()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    keywords = representative_keywords()
    print(f"Collecting {PASS_PER_CATEGORY} PASS each for {len(keywords)} categories "
          f"(target {PASS_PER_CATEGORY * len(keywords)} PASS)")

    review_rows: list[dict] = []
    total_pass = 0

    for kw in keywords:
        out_dir = OUTPUT_ROOT / kw.category
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{kw.category}.log"

        print(f"\n=== [{kw.category}] {kw.query} ===")
        survivors = R._tier_a(api_key, kw, out_dir, log_file)
        print(f"    Tier-A PASS: {survivors}")
        total_pass += survivors

        # Tier-B score this category's survivors.
        clips_dir = out_dir / "clips"
        proxy_csv = out_dir / "vbench_proxy.csv"
        if survivors > 0:
            tier_b(clips_dir, proxy_csv, log_file)
            quality_csv = out_dir / "quality_report" / "quality_report.csv"
            review_rows.extend(
                _join_category(kw, quality_csv, proxy_csv)
            )

    # Rank the whole pool best-first and write review artifacts.
    review_rows.sort(key=lambda r: r["composite"], reverse=True)
    _write_review(review_rows)

    print(f"\nTotal Tier-A PASS collected: {total_pass}")
    print(f"Ranked review rows: {len(review_rows)}")
    print(f"Review manifest: {OUTPUT_ROOT / 'review_manifest.jsonl'}")
    print(f"Review scores:   {OUTPUT_ROOT / 'review_scores.csv'}")

    R.lark_bot(
        "info",
        "[VBench review80] collection complete\n"
        f"Tier-A PASS: {total_pass}\n"
        f"Ranked rows: {len(review_rows)}\n"
        f"Output: {OUTPUT_ROOT.resolve()}",
    )
    return 0


def _join_category(kw: Keyword, quality_csv: Path, proxy_csv: Path) -> list[dict]:
    """Compute composite per clip for one category (no Top-K, no floors)."""
    import vbench_compose_select as V

    if not quality_csv.exists() or not proxy_csv.exists():
        return []
    quality = V._read_quality(quality_csv)
    proxy = V._read_proxy(proxy_csv)
    rows: list[dict] = []
    prompt = f"a cinematic 4k video of {' '.join(kw.query.split())}"
    for key in sorted(set(quality) & set(proxy)):
        q = quality[key]
        if q.get("grade") == "FAIL":
            continue
        res = V.compose_row(q, proxy[key], V.DEFAULT_WEIGHTS)
        rows.append({
            "category": kw.category,
            "clip": key,
            "video": f"{kw.category}/clips/{key}",
            "prompt": prompt,
            "composite": res["composite"],
            "dims": res["dims"],
        })
    return rows


def _write_review(rows: list[dict]) -> None:
    import json
    import vbench_compose_select as V

    dim_names = list(V.DEFAULT_WEIGHTS.keys())
    with (OUTPUT_ROOT / "review_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "clip", "composite", *dim_names])
        for r in rows:
            w.writerow([r["category"], r["clip"], f"{r['composite']:.4f}"]
                       + [f"{r['dims'][d]:.4f}" for d in dim_names])

    with (OUTPUT_ROOT / "review_manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({
                "id": Path(r["clip"]).stem,
                "video": r["video"],
                "prompt": r["prompt"],
                "category": r["category"],
                "composite": round(r["composite"], 4),
            }, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    sys.exit(main())
