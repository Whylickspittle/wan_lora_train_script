#!/usr/bin/env python3
"""Multi-keyword Pexels batch08 runner with batched download-process loops.

Each keyword is processed in small batches: download N raw videos, slice and
quality-check them, delete the raw videos, and repeat until the target number
of PASS clips is reached or the keyword is exhausted. Feishu notifications are
sent only when a keyword finishes (success or exhausted) or when the runner
crashes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from feishu import lark_bot

API_KEY_PATH = Path("/workspace/pexel_api")
OUTPUT_ROOT = Path("./pexels_nature_batch08_multi")
LOG_DIR = Path("./pexels_nature_batch08_multi_logs")
PID_FILE = Path("./pexels_nature_batch08_multi.pid")

KEYWORDS = [
    "drone aerial view of mountain peaks at golden hour sunrise, cinematic 4K",
    "drone aerial shot of misty forest with sunlight rays through trees, cinematic 4K",
    "drone aerial view over desert sand dunes at dramatic sunset, cinematic 4K",
    "drone aerial view of vast blue glacier ice landscape, cinematic 4K",
]

TARGET_PASS_PER_KEYWORD = 50
BATCH_SIZE = 30


def _safe_name(keyword: str) -> str:
    return keyword.replace(" ", "_").replace("/", "_").replace("\\", "_")


def _read_summary(output_dir: Path) -> dict | None:
    summary_path = output_dir / "quality_report" / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_clips(clips_dir: Path) -> int:
    if not clips_dir.exists():
        return 0
    return len([p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])


def _delete_raw(output_dir: Path) -> None:
    raw_dir = output_dir / "raw"
    if raw_dir.exists():
        try:
            shutil.rmtree(raw_dir)
        except Exception as exc:
            print(f"[cleanup] failed to remove {raw_dir}: {exc}", file=sys.stderr)


def _build_base_cmd(keyword: str, output_dir: Path) -> list[str]:
    api_key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    return [
        sys.executable,
        "download_pexels_quality_pipeline.py",
        "--api-key",
        api_key,
        "--query",
        keyword,
        "--min-height",
        "2160",
        "--min-fps",
        "24",
        "--clips-per-video",
        "3",
        "--output",
        str(output_dir),
        "--history-file",
        "./pexels_download_history.json",
        "--quarantine",
        "--static-mean-delta",
        "0.02",
    ]


def _run_subprocess(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"\n{'=' * 60}\n{' '.join(cmd)}\n{'=' * 60}\n")
        fh.flush()
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    return result.returncode


def _rebuild_manifest(output_dir: Path, keyword: str) -> int:
    """Regenerate manifest.jsonl from all surviving clips in output_dir/clips."""
    clips_dir = output_dir / "clips"
    manifest_path = output_dir / "manifest.jsonl"
    kept = 0
    with manifest_path.open("w", encoding="utf-8") as fh:
        for clip_path in sorted(clips_dir.glob("*.mp4")):
            record = {
                "id": clip_path.stem,
                "video": f"clips/{clip_path.name}",
                "prompt": f"a cinematic 4k video of {' '.join(keyword.split())}",
            }
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            kept += 1
    return kept


def run_keyword(keyword: str) -> tuple[int, int]:
    """Process one keyword in batches. Returns (returncode, cumulative_pass)."""
    safe_name = _safe_name(keyword)
    output_dir = OUTPUT_ROOT / safe_name
    log_file = LOG_DIR / f"{safe_name}.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if log_file.exists():
        log_file.write_text("", encoding="utf-8")

    base_cmd = _build_base_cmd(keyword, output_dir)
    cumulative_pass = _count_clips(output_dir / "clips")
    batch_no = 0

    while cumulative_pass < TARGET_PASS_PER_KEYWORD:
        batch_no += 1

        # Download batch.
        download_cmd = base_cmd + ["--count", str(BATCH_SIZE), "--download-only"]
        returncode = _run_subprocess(download_cmd, log_file)
        if returncode != 0:
            print(f"[keyword={keyword}] download batch {batch_no} failed", file=sys.stderr)
            break

        raw_dir = output_dir / "raw"
        if not raw_dir.exists() or not any(raw_dir.glob("*.mp4")):
            print(f"[keyword={keyword}] no new raw videos in batch {batch_no}", file=sys.stderr)
            break

        # Process batch.
        process_cmd = base_cmd + ["--process-only", "--run-diagnostics"]
        returncode = _run_subprocess(process_cmd, log_file)
        if returncode != 0:
            print(f"[keyword={keyword}] process batch {batch_no} failed", file=sys.stderr)
            break

        # Clean up raw videos (process phase usually does this, but be safe).
        _delete_raw(output_dir)

        # Update cumulative PASS count.
        cumulative_pass = _count_clips(output_dir / "clips")

    # Final manifest covering all batches for this keyword.
    final_kept = _rebuild_manifest(output_dir, keyword)
    reached = cumulative_pass >= TARGET_PASS_PER_KEYWORD
    lark_bot(
        "info" if reached else "error",
        f"Keyword finished: {keyword}\n"
        f"Cumulative PASS clips: {cumulative_pass} / {TARGET_PASS_PER_KEYWORD}\n"
        f"Final manifest: {final_kept} clips\n"
        f"Status: {'target reached' if reached else 'exhausted'}",
    )

    return 0 if reached else 1, cumulative_pass


def main() -> int:
    if not API_KEY_PATH.exists():
        print(f"API key file not found: {API_KEY_PATH}", file=sys.stderr)
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    PID_FILE.write_text(str(sys.argv[0]), encoding="utf-8")

    overall_pass = 0
    overall_failures: list[str] = []

    try:
        for keyword in KEYWORDS:
            safe_name = _safe_name(keyword)
            output_dir = OUTPUT_ROOT / safe_name
            summary_path = output_dir / "quality_report" / "summary.json"

            # Skip if this keyword already has a final summary from a previous run.
            if summary_path.exists():
                summary = _read_summary(output_dir)
                pass_count = summary.get("pass", 0) if summary else _count_clips(output_dir / "clips")
                overall_pass += pass_count
                continue

            returncode, cumulative_pass = run_keyword(keyword)
            overall_pass += cumulative_pass
            if returncode != 0:
                overall_failures.append(keyword)

    except Exception as exc:
        lark_bot("error", f"Pexels batch08 multi crashed: {exc}")
        raise
    finally:
        try:
            PID_FILE.unlink()
        except OSError:
            pass

    total_target = TARGET_PASS_PER_KEYWORD * len(KEYWORDS)
    lark_bot(
        "info" if overall_pass >= total_target else "error",
        "Pexels batch08 multi completed\n"
        f"Total PASS: {overall_pass} / {total_target}\n"
        f"Output: {OUTPUT_ROOT.resolve()}\n"
        f"End: {datetime.now().isoformat(timespec='seconds')}",
    )

    return 0 if overall_pass >= total_target else 1


if __name__ == "__main__":
    sys.exit(main())
