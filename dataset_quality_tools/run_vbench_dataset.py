#!/usr/bin/env python3
"""VBench-aligned Pexels dataset driver.

For every keyword in ``vbench_keywords.KEYWORDS`` this runner:

  1. Over-downloads raw videos (a multiple of target_count) and slices + Tier-A
     quality-checks them via download_pexels_quality_pipeline.py
     (clean_dataset.py with static AND chaotic motion bounds + quarantine).
  2. Tier-B scores the surviving clips with CLIP proxies (score_vbench_proxy.py).
  3. Composes a VBench composite score and keeps the Top-K (= target_count) via
     vbench_compose_select.py, writing selected_manifest.jsonl + vbench_scores.csv.

Downloaded Pexels IDs are de-duplicated through pexels_download_history.json so
re-runs and other batches never fetch the same source twice. Feishu notifications
fire when a keyword finishes (reached target / exhausted) or the runner crashes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from feishu import lark_bot
from vbench_keywords import KEYWORDS, Keyword

API_KEY_PATH = Path("/workspace/pexel_api")
HISTORY_FILE = Path("./pexels_download_history.json")

# Per-clip Tier-A motion bounds (temporal_diff_mean).
STATIC_MEAN_DELTA = "0.02"   # below -> static / timelapse -> FAIL
MAX_MEAN_DELTA = "0.20"      # above -> chaotic -> FAIL (protects consistency)

CLIPS_PER_VIDEO = "3"
BATCH_SIZE = 30
OVERDOWNLOAD_FACTOR = 2.5    # gather ~2.5x target PASS clips before Top-K


def _safe_name(keyword: str) -> str:
    return keyword.replace(" ", "_").replace("/", "_").replace("\\", "_")[:120]


def _count_clips(clips_dir: Path) -> int:
    if not clips_dir.exists():
        return 0
    return sum(1 for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4")


def _delete_raw(output_dir: Path) -> None:
    raw_dir = output_dir / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir, ignore_errors=True)


def _run(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 60}\n{' '.join(cmd)}\n{'=' * 60}\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


def _base_pipeline_cmd(api_key: str, query: str, output_dir: Path) -> list[str]:
    return [
        sys.executable, "download_pexels_quality_pipeline.py",
        "--api-key", api_key,
        "--query", query,
        "--min-height", "2160",
        "--min-fps", "24",
        "--clips-per-video", CLIPS_PER_VIDEO,
        "--output", str(output_dir),
        "--history-file", str(HISTORY_FILE),
        "--quarantine",
        "--static-mean-delta", STATIC_MEAN_DELTA,
        "--max-mean-delta", MAX_MEAN_DELTA,
    ]


def _tier_a(api_key: str, kw: Keyword, output_dir: Path, log_file: Path) -> int:
    """Over-download + slice + Tier-A quality check. Returns surviving clip count."""
    base = _base_pipeline_cmd(api_key, kw.query, output_dir)
    overdownload_target = int(kw.target_count * OVERDOWNLOAD_FACTOR)
    survivors = _count_clips(output_dir / "clips")
    batch_no = 0

    while survivors < overdownload_target:
        batch_no += 1
        rc = _run(base + ["--count", str(BATCH_SIZE), "--download-only"], log_file)
        if rc != 0:
            print(f"[{kw.query}] download batch {batch_no} failed (rc={rc})", file=sys.stderr)
            break

        raw_dir = output_dir / "raw"
        if not raw_dir.exists() or not any(raw_dir.glob("*.mp4")):
            print(f"[{kw.query}] keyword exhausted at batch {batch_no}", file=sys.stderr)
            break

        rc = _run(base + ["--process-only"], log_file)
        if rc != 0:
            print(f"[{kw.query}] process batch {batch_no} failed (rc={rc})", file=sys.stderr)
            break

        _delete_raw(output_dir)
        survivors = _count_clips(output_dir / "clips")

    return survivors


def _tier_b_and_select(kw: Keyword, output_dir: Path, log_file: Path) -> int:
    """Tier-B proxy scoring + composite Top-K selection. Returns selected count."""
    clips_dir = output_dir / "clips"
    proxy_csv = output_dir / "vbench_proxy.csv"
    quality_csv = output_dir / "quality_report" / "quality_report.csv"

    rc = _run(
        [
            sys.executable, "score_vbench_proxy.py",
            "--input", str(clips_dir),
            "--output", str(proxy_csv),
            "--resume",
        ],
        log_file,
    )
    if rc != 0:
        print(f"[{kw.query}] Tier-B scoring failed (rc={rc})", file=sys.stderr)
        return 0

    rc = _run(
        [
            sys.executable, "vbench_compose_select.py",
            "--quality-csv", str(quality_csv),
            "--proxy-csv", str(proxy_csv),
            "--top-k", str(kw.target_count),
            "--prompt", f"a cinematic 4k video of {' '.join(kw.query.split())}",
            "--scores-out", str(output_dir / "vbench_scores.csv"),
            "--manifest-out", str(output_dir / "selected_manifest.jsonl"),
        ],
        log_file,
    )
    if rc != 0:
        print(f"[{kw.query}] composite selection failed (rc={rc})", file=sys.stderr)
        return 0

    manifest = output_dir / "selected_manifest.jsonl"
    if not manifest.exists():
        return 0
    return sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())


def run_keyword(api_key: str, kw: Keyword, output_root: Path, log_dir: Path) -> tuple[bool, int]:
    safe = f"{kw.category}__{_safe_name(kw.query)}"
    output_dir = output_root / safe
    log_file = log_dir / f"{safe}.log"
    output_dir.mkdir(parents=True, exist_ok=True)

    survivors = _tier_a(api_key, kw, output_dir, log_file)
    selected = _tier_b_and_select(kw, output_dir, log_file)
    reached = selected >= kw.target_count

    lark_bot(
        "info" if reached else "error",
        f"[VBench] keyword done: {kw.query}\n"
        f"category={kw.category}\n"
        f"Tier-A survivors: {survivors}\n"
        f"Selected (Top-K): {selected} / {kw.target_count}\n"
        f"Status: {'target reached' if reached else 'short'}",
    )
    return reached, selected


def main() -> int:
    parser = argparse.ArgumentParser(description="VBench-aligned Pexels dataset driver")
    parser.add_argument("--output", type=Path, default=Path("./vbench_dataset"))
    parser.add_argument("--log-dir", type=Path, default=Path("./vbench_dataset_logs"))
    parser.add_argument("--limit-keywords", type=int, default=None,
                        help="Only process the first N keywords (smoke testing)")
    parser.add_argument("--target-override", type=int, default=None,
                        help="Override every keyword's target_count (smoke testing)")
    args = parser.parse_args()

    if not API_KEY_PATH.exists():
        print(f"API key file not found: {API_KEY_PATH}", file=sys.stderr)
        return 1
    api_key = API_KEY_PATH.read_text(encoding="utf-8").strip()

    args.output.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    keywords = list(KEYWORDS)
    if args.limit_keywords:
        keywords = keywords[: args.limit_keywords]
    if args.target_override is not None:
        keywords = [Keyword(k.category, k.query, args.target_override) for k in keywords]

    total_selected = 0
    total_target = sum(k.target_count for k in keywords)
    shortfalls: list[str] = []

    try:
        for kw in keywords:
            reached, selected = run_keyword(api_key, kw, args.output, args.log_dir)
            total_selected += selected
            if not reached:
                shortfalls.append(kw.query)
    except Exception as exc:
        lark_bot("error", f"[VBench] driver crashed: {exc}")
        raise

    lark_bot(
        "info" if not shortfalls else "error",
        "[VBench] dataset run complete\n"
        f"Total selected: {total_selected} / {total_target}\n"
        f"Keywords short: {len(shortfalls)}\n"
        f"Output: {args.output.resolve()}\n"
        f"End: {datetime.now().isoformat(timespec='seconds')}",
    )
    return 0 if not shortfalls else 1


if __name__ == "__main__":
    sys.exit(main())
