#!/usr/bin/env python3
"""
End-to-end Pexels dataset pipeline.

1. Search Pexels API for videos matching --query.
2. Filter by min-height / min-fps and download.
3. Slice into 5.04 s clips (1280x704, 24 fps, 121 frames).
4. Run dataset_quality_tools/clean_dataset.py analysis.
5. Quarantine FAIL clips and write a training manifest.jsonl.
6. Optionally run _core/dataset_diagnostics.py for a final pre-flight report.

The pipeline can be run in one pass (default) or split into two phases:

  # Phase 1: download 200 raw videos in one hour
  python download_pexels_quality_pipeline.py \
      --api-key "YOUR_API_KEY" \
      --query "4k nature scenery drone" \
      --count 200 \
      --min-height 2160 \
      --min-fps 24 \
      --output ./pexels_dataset \
      --download-only

  # Phase 2: slice, quality-check, and generate manifest
  python download_pexels_quality_pipeline.py \
      --output ./pexels_dataset \
      --clips-per-video 2 \
      --quarantine \
      --process-only
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "wan_lora_train_script" / "_core"))

import download_videos_direct

CLEAN_DATASET = None
_IMPORT_ERROR: Exception | None = None
try:
    import clean_dataset

    CLEAN_DATASET = clean_dataset
except Exception as exc:
    _IMPORT_ERROR = exc


def resolution_label(height: int) -> str:
    if height >= 2160:
        return "4k"
    if height >= 1440:
        return "2k"
    if height >= 1080:
        return "1080p"
    if height >= 720:
        return "720p"
    return f"{height}p"


def build_prompt(template: str, query: str, height: int) -> str:
    description = " ".join(query.split())
    return template.format(
        description=description,
        query=description,
        resolution=resolution_label(height),
        height=height,
    )


def build_quality_cfg(args: argparse.Namespace) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        dup_ratio=args.dup_ratio,
        timelapse_score=args.timelapse_score,
        clipping_ratio=args.clipping_ratio,
        min_conformance=args.min_conformance,
        scene_cut_max=args.scene_cut_max,
        extensions="mp4,mov,avi,webm,mkv",
        copy=args.copy_quarantine,
    )


def run_diagnostics(manifest_path: Path, output_dir: Path) -> dict:
    import dataset_diagnostics

    return dataset_diagnostics.diagnose_dataset(
        str(manifest_path),
        str(output_dir),
        resolution="1280x704",
        num_frames=121,
        fps=24.0,
        allow_resize=True,
        train_batch_size=1,
        num_workers=0,
        pin_memory=False,
    )


def download_with_retry(
    url: str,
    save_path: Path,
    session,
    max_retries: int,
) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            download_videos_direct.download_file(url, save_path, session)
            return True
        except Exception as exc:
            print(f"    download attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return False


def load_history(history_path: Path | None) -> set[int]:
    """Load previously downloaded Pexels video IDs from a JSON history file."""
    if history_path is None or not history_path.exists():
        return set()
    try:
        data = json.loads(history_path.read_text(encoding="utf-8"))
        ids = data.get("downloaded_ids", [])
        return {int(x) for x in ids}
    except Exception as exc:
        print(f"  warning: could not read history file: {exc}")
        return set()


def save_history(history_path: Path, seen_ids: set[int]) -> None:
    """Persist downloaded Pexels video IDs so future runs can skip them."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "downloaded_ids": sorted(seen_ids),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_phase(args: argparse.Namespace) -> list[Path]:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = download_videos_direct.requests.Session()
    session.headers.update({"Authorization": args.api_key})

    history_path: Path | None = args.history_file
    downloaded = 0
    seen_ids: set[int] = load_history(history_path)
    page = 1
    raw_paths: list[Path] = []

    print(f"[download phase] target: {args.count} videos")
    if seen_ids:
        print(f"  skipping {len(seen_ids)} video(s) from existing history")

    while downloaded < args.count:
        print(f"\n[page {page}] searching '{args.query}'...")
        try:
            data = download_videos_direct.search_videos(
                args.api_key, args.query, page=page, per_page=80
            )
        except Exception as exc:
            print(f"  search failed: {exc}")
            break

        videos = data.get("videos", [])
        if not videos:
            print("  no more videos found")
            break

        for video in videos:
            if downloaded >= args.count:
                break

            video_id = video.get("id")
            if video_id in seen_ids:
                continue
            seen_ids.add(video_id)

            best_file = download_videos_direct.pick_best_file(
                video, args.min_height, args.min_fps
            )
            if not best_file:
                continue

            height = best_file.get("height", 0)
            width = best_file.get("width", 0)
            fps = best_file.get("fps", 0)
            print(f"\n[{downloaded + 1}/{args.count}] ID {video_id}")
            print(f"  page: {video['url']}")
            print(f"  resolution: {width}x{height}, fps: {fps}")

            raw_path = raw_dir / f"video_{video_id}.mp4"
            if raw_path.exists():
                print("  already exists, skip download")
            else:
                ok = download_with_retry(
                    best_file["link"], raw_path, session, args.max_retries
                )
                if not ok:
                    print("  giving up on this video")
                    continue

            downloaded += 1
            raw_paths.append(raw_path)
            if history_path is not None:
                save_history(history_path, seen_ids)

        page += 1
        time.sleep(args.sleep)

    print(f"\n[download phase] saved {len(raw_paths)} raw videos to {raw_dir}")
    if history_path is not None:
        print(f"  history written to: {history_path}")
    return raw_paths


def process_phase(args: argparse.Namespace) -> list[Path]:
    output_dir = args.output
    raw_dir = output_dir / "raw"
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"[process phase] raw directory not found: {raw_dir}", file=sys.stderr)
        return []

    raw_paths = sorted(raw_dir.glob("*.mp4"))
    if not raw_paths:
        print(f"[process phase] no raw videos in {raw_dir}", file=sys.stderr)
        return []

    clip_paths: list[Path] = []
    print(f"[process phase] slicing {len(raw_paths)} raw videos...")

    for raw_path in raw_paths:
        try:
            video_id = raw_path.stem.split("_", 1)[1]
        except IndexError:
            print(f"  skip {raw_path.name}: cannot parse video id")
            continue

        print(f"\n  ID {video_id}: {raw_path.name}")

        try:
            duration = download_videos_direct.ffprobe_duration(raw_path)
        except Exception as exc:
            print(f"    ffprobe failed: {exc}, skip slicing")
            continue

        max_start = max(0.0, duration - download_videos_direct.CLIP_DURATION)
        if max_start < 0.01:
            print("    too short to slice")
            continue

        for _ in range(args.clips_per_video):
            node = round(random.uniform(0.0, max_start), 2)
            clip_path = clips_dir / f"clip_{video_id}_node{node:.2f}.mp4"
            if clip_path.exists():
                print(f"    clip already exists: {clip_path.name}")
                clip_paths.append(clip_path)
                continue
            try:
                download_videos_direct.slice_clip(raw_path, clip_path, node)
                print(f"    clip saved: {clip_path.name}")
                clip_paths.append(clip_path)
            except Exception as exc:
                print(f"    clip failed: {exc}")

        if not args.keep_raw:
            try:
                raw_path.unlink()
            except OSError:
                pass

    if not args.keep_raw:
        try:
            shutil.rmtree(raw_dir, ignore_errors=True)
        except OSError:
            pass

    print(f"\n[process phase] produced {len(clip_paths)} clips")
    return clip_paths


def run_quality_phase(args: argparse.Namespace) -> list[dict]:
    if CLEAN_DATASET is None:
        print(
            "Cannot run quality check: clean_dataset module is not available.\n"
            "Install dependencies: pip install torch numpy av",
            file=sys.stderr,
        )
        if _IMPORT_ERROR is not None:
            print(f"Original error: {_IMPORT_ERROR}", file=sys.stderr)
        sys.exit(1)

    clips_dir = args.output / "clips"
    report_dir = args.output / "quality_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_quality_cfg(args)
    rows = CLEAN_DATASET.analyze_directory(clips_dir, cfg)
    if not rows:
        print("No clips could be analyzed.")
        return rows

    summary = CLEAN_DATASET.build_summary(rows)
    CLEAN_DATASET.write_csv_report(rows, report_dir / "quality_report.csv")
    CLEAN_DATASET.write_summary(summary, report_dir / "summary.json", cfg)
    CLEAN_DATASET.write_html_report(rows, summary, report_dir / "report.html")

    if args.quarantine:
        CLEAN_DATASET.quarantine_clips(
            rows, args.output / "quarantine", cfg.copy
        )

    print("\n" + "=" * 60)
    print("Dataset Quality Report Summary")
    print("=" * 60)
    print(f"Total clips:  {summary.total}")
    print(f"PASS:         {summary.pass_count} ({summary.pass_count / max(summary.total, 1):.1%})")
    print(f"REVIEW:       {summary.review_count} ({summary.review_count / max(summary.total, 1):.1%})")
    print(f"FAIL:         {summary.fail_count} ({summary.fail_count / max(summary.total, 1):.1%})")
    print(f"\nReports written to: {report_dir}")
    if args.quarantine:
        op = "copied" if cfg.copy else "moved"
        print(f"Failed clips {op} to: {args.output / 'quarantine'}")

    return rows


def write_manifest(
    rows: list[dict],
    args: argparse.Namespace,
) -> Path:
    keep_grades = {"PASS", "REVIEW"} if args.allow_review else {"PASS"}
    manifest_path = args.output / "manifest.jsonl"
    kept = 0
    with manifest_path.open("w", encoding="utf-8") as fh:
        if rows:
            for row in rows:
                if row.get("grade") not in keep_grades:
                    continue
                clip_path = Path(row["path"])
                height = row.get("source_height", args.min_height)
                prompt = build_prompt(args.prompt_template, args.query, height)
                record = {
                    "id": clip_path.stem,
                    "video": f"clips/{clip_path.name}",
                    "prompt": prompt,
                }
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")
                kept += 1
        else:
            for clip_path in sorted((args.output / "clips").glob("*.mp4")):
                prompt = build_prompt(args.prompt_template, args.query, args.min_height)
                record = {
                    "id": clip_path.stem,
                    "video": f"clips/{clip_path.name}",
                    "prompt": prompt,
                }
                fh.write(json.dumps(record, ensure_ascii=True) + "\n")
                kept += 1

    print(f"\nmanifest written: {manifest_path} ({kept} clips)")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Pexels videos, slice clips, run quality checks, and write manifest.jsonl."
    )
    parser.add_argument("--api-key", help="Pexels API key (required for download)")
    parser.add_argument("--query", default="4k nature scenery drone", help="Pexels search query")
    parser.add_argument("--output", type=Path, default=Path("pexels_dataset"), help="output directory")
    parser.add_argument(
        "--history-file",
        type=Path,
        default=None,
        help="JSON file tracking downloaded Pexels video IDs (defaults to {output}/download_history.json)",
    )
    parser.add_argument("--count", type=int, default=10, help="how many videos to download")
    parser.add_argument("--min-height", type=int, default=2160, help="min video height")
    parser.add_argument("--min-fps", type=int, default=24, help="min fps")
    parser.add_argument("--clips-per-video", type=int, default=1, help="clips per video")
    parser.add_argument("--sleep", type=float, default=1.0, help="sleep between API pages")
    parser.add_argument("--max-retries", type=int, default=3, help="download retries per video")
    parser.add_argument("--keep-raw", action="store_true", help="keep full downloaded videos")
    parser.add_argument("--prompt-template", default="a cinematic {resolution} video of {description}", help="prompt template")
    parser.add_argument("--allow-review", action="store_true", help="include REVIEW clips in manifest")

    # Phase switches
    parser.add_argument("--download-only", action="store_true", help="only download raw videos")
    parser.add_argument("--process-only", action="store_true", help="only slice/quality-check existing raw videos")
    parser.add_argument("--skip-quality-check", action="store_true", help="skip clean_dataset analysis")
    parser.add_argument("--skip-diagnostics", action="store_true", help="skip final dataset_diagnostics")
    parser.add_argument("--run-diagnostics", action="store_true", help="run final dataset_diagnostics after manifest")

    # Quality thresholds
    parser.add_argument("--dup-ratio", type=float, default=0.05)
    parser.add_argument("--timelapse-score", type=float, default=0.50)
    parser.add_argument("--clipping-ratio", type=float, default=0.15)
    parser.add_argument("--min-conformance", type=float, default=0.50)
    parser.add_argument("--scene-cut-max", type=int, default=0)
    parser.add_argument("--quarantine", action="store_true", help="isolate FAIL clips")
    parser.add_argument("--copy-quarantine", action="store_true", help="copy instead of move when quarantining")

    args = parser.parse_args()

    if args.history_file is None:
        args.history_file = args.output / "download_history.json"

    if not args.process_only and not args.api_key:
        print("Error: --api-key is required unless using --process-only", file=sys.stderr)
        return 1

    if args.download_only and args.process_only:
        print("Error: --download-only and --process-only cannot be used together", file=sys.stderr)
        return 1

    missing: list[str] = []
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            missing.append(tool)
    if missing:
        print(f"Missing required tools: {', '.join(missing)}", file=sys.stderr)
        return 1

    if args.download_only:
        download_phase(args)
        print("\nDownload phase complete. Run with --process-only to slice and quality-check.")
        return 0

    clip_paths: list[Path] = []
    if args.process_only:
        clip_paths = process_phase(args)
    else:
        download_phase(args)
        clip_paths = process_phase(args)

    if not clip_paths:
        print("No clips were produced. Exiting.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    if not args.skip_quality_check:
        rows = run_quality_phase(args)
    else:
        print("Skipping quality check.")

    manifest_path = write_manifest(rows, args)

    if args.run_diagnostics and not args.skip_diagnostics:
        print("\nRunning final dataset diagnostics...")
        try:
            diag_summary = run_diagnostics(manifest_path, args.output / "diagnostics")
            print(f"Diagnostics report: {args.output / 'diagnostics' / 'diagnostics_report.html'}")
            print(f"ok_to_train: {diag_summary.get('ok_to_train', False)}")
        except Exception as exc:
            print(f"Diagnostics failed: {exc}", file=sys.stderr)

    print("\nPipeline complete!")
    print(f"Output directory: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
