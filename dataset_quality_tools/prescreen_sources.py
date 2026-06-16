#!/usr/bin/env python3
"""
Source URL Pre-screener for Nexisgen Dataset

Lightweight tool to filter candidate source URLs before downloading full videos.

Recommended mode:
  --metadata-only    Fast heuristic filtering using title/description/duration.

The full-sample mode downloads short segments from each URL for frame analysis.
It is slower and can be unreliable due to network/DRM issues, so it should be
used only for a small set of promising candidates.

Usage:
  python prescreen_sources.py candidate_urls.txt --output ./prescreen --metadata-only
  python prescreen_sources.py candidate_urls.txt --output ./prescreen --sample-duration 15
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata heuristics
# ---------------------------------------------------------------------------

REJECT_KEYWORDS = {
    "timelapse", "time-lapse", "time lapse", "hyperlapse",
    "relaxation", "meditation", "ambient", "calming", "sleep",
    "slideshow", "wallpaper", "screensaver", "asmr",
    "nature sounds", "4k scenery", "scenery", "landscape only",
    "meditative", "soothing", "zen", "fireplace", "rain sounds",
    "virtual walk", "static", "loop", "10 hours", "1 hour",
    "compilation", "best of", "top 10", "mix",
}

KEEP_KEYWORDS = {
    "documentary", "wildlife", "animals", "nature documentary",
    "gimbal", "stabilized", "follow shot", "tracking shot",
    "walking", "hiking", "running", "cycling", "skiing",
    "real time", "realtime", "live",
    "close up", "slow motion",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SourceResult:
    url: str
    title: str = ""
    duration: float = 0.0
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    metadata_score: float = 0.0
    metadata_reason: str = ""
    sample_duplicate_ratio: float | None = None
    sample_static_ratio: float | None = None
    sample_timelapse_score: float | None = None
    recommendation: str = "REVIEW"
    reason: str = ""


# ---------------------------------------------------------------------------
# Metadata fetching
# ---------------------------------------------------------------------------


def fetch_metadata(url: str) -> dict[str, Any]:
    cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        "--no-playlist",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {"error": str(exc)}

    if result.returncode != 0 or not result.stdout.strip():
        return {"error": result.stderr[:200]}

    last_line = ""
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line:
            last_line = line
            break

    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return {"error": "json_parse_failed"}


def score_metadata(info: dict[str, Any]) -> tuple[float, str]:
    if "error" in info:
        return 0.0, f"metadata_error:{info['error']}"

    title = str(info.get("title", "")).lower()
    desc = str(info.get("description", "")).lower()

    for kw in REJECT_KEYWORDS:
        if kw in title:
            return 0.0, f"reject_title:{kw}"
        if kw in desc:
            return 0.0, f"reject_desc:{kw}"

    score = 0.5
    reasons: list[str] = []

    for kw in KEEP_KEYWORDS:
        if kw in title:
            score += 0.15
            reasons.append(f"keep:{kw}")

    duration = float(info.get("duration") or 0)
    if duration < 30:
        score -= 0.3
        reasons.append("too_short")
    elif duration > 120:
        score += 0.1
        reasons.append("good_duration")

    width = info.get("width")
    height = info.get("height")
    if width and height:
        if height >= 1080 and width >= 1920:
            score += 0.1
            reasons.append("high_resolution")
        elif height < 704 or width < 1280:
            score -= 0.3
            reasons.append("low_resolution")

    fps = info.get("fps")
    if fps:
        fps_f = float(fps)
        if abs(fps_f - 24.0) < 0.1:
            score += 0.1
            reasons.append("native_24fps")
        elif fps_f < 20:
            score -= 0.2
            reasons.append("low_fps")

    return min(1.0, score), "; ".join(reasons) if reasons else "no_signals"


# ---------------------------------------------------------------------------
# Sample analysis (optional, slower)
# ---------------------------------------------------------------------------


def safe_mean_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a.float() - b.float()).abs()))


def analyze_video_sample(path: Path) -> dict[str, float] | None:
    try:
        container = av.open(str(path))
        stream = container.streams.video[0]
        frames = [
            torch.from_numpy(f.to_ndarray(format="rgb24"))
            for f in container.decode(video=0)
        ]
        container.close()
    except Exception as exc:
        logger.warning("Sample decode failed for %s: %s", path, exc)
        return None

    if len(frames) < 2:
        return None

    video = torch.stack(frames)
    deltas = []
    for i in range(1, len(video)):
        deltas.append(safe_mean_abs_delta(video[i], video[i - 1]))
    deltas_t = torch.tensor(deltas)

    duplicate_ratio = float((deltas_t < 0.002).float().mean())
    static_ratio = float((deltas_t < 0.02).float().mean())
    mean_delta = float(deltas_t.mean())
    std_delta = float(deltas_t.std())
    high_motion_ratio = float((deltas_t > 0.12).float().mean())

    timelapse_score = 0.0
    if duplicate_ratio < 0.05 and high_motion_ratio > 0.30 and mean_delta > 0.0025:
        stability = 1.0 - min(std_delta / max(mean_delta, 1e-6), 1.0)
        timelapse_score = min(1.0, high_motion_ratio * 0.7 + stability * 0.3)

    return {
        "duplicate_ratio": duplicate_ratio,
        "static_ratio": static_ratio,
        "timelapse_score": timelapse_score,
    }


def download_sample(url: str, output_path: Path, start: float, duration: float) -> bool:
    cmd = [
        "yt-dlp",
        "-f",
        "bestvideo[height>=704][ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--recode-video", "mp4",
        "--download-sections", f"*{start:.1f}-{start + duration:.1f}",
        "--no-playlist",
        "-o", str(output_path),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except Exception:
        return False
    return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 1024


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_source(url: str, cfg: argparse.Namespace, workdir: Path) -> SourceResult:
    result = SourceResult(url=url)
    info = fetch_metadata(url)

    result.title = str(info.get("title", ""))[:120]
    result.duration = float(info.get("duration") or 0)
    result.fps = float(info["fps"]) if info.get("fps") else None
    result.width = info.get("width")
    result.height = info.get("height")

    result.metadata_score, result.metadata_reason = score_metadata(info)

    if cfg.metadata_only:
        if result.metadata_score <= 0.0:
            result.recommendation = "REJECT"
            result.reason = result.metadata_reason
        elif result.metadata_score < 0.5:
            result.recommendation = "REVIEW"
            result.reason = "weak_metadata"
        else:
            result.recommendation = "KEEP"
            result.reason = result.metadata_reason
        return result

    # Full sample mode.
    if result.metadata_score <= 0.0:
        result.recommendation = "REJECT"
        result.reason = result.metadata_reason
        return result

    if result.duration <= 0:
        result.recommendation = "REVIEW"
        result.reason = "no_duration"
        return result

    video_id = re.sub(r"\W+", "_", url.split("v=")[-1][:20] if "v=" in url else url[-20:])
    sample_path = workdir / f"{video_id}_sample.mp4"
    sample_duration = min(cfg.sample_duration, result.duration * 0.25)
    start = min(result.duration * 0.4, max(result.duration * 0.1, result.duration / 2 - sample_duration / 2))

    if download_sample(url, sample_path, start, sample_duration):
        sample_metrics = analyze_video_sample(sample_path)
        try:
            sample_path.unlink()
        except OSError:
            pass

        if sample_metrics:
            result.sample_duplicate_ratio = sample_metrics["duplicate_ratio"]
            result.sample_static_ratio = sample_metrics["static_ratio"]
            result.sample_timelapse_score = sample_metrics["timelapse_score"]

            reasons: list[str] = []
            if result.sample_duplicate_ratio > cfg.dup_ratio:
                reasons.append(f"sample_dup={result.sample_duplicate_ratio:.2%}")
            if result.sample_static_ratio > cfg.static_ratio:
                reasons.append(f"sample_static={result.sample_static_ratio:.2%}")
            if result.sample_timelapse_score > cfg.timelapse_score:
                reasons.append(f"sample_timelapse={result.sample_timelapse_score:.2f}")

            if reasons:
                result.recommendation = "REJECT"
                result.reason = "; ".join(reasons)
            elif result.metadata_score < 0.5:
                result.recommendation = "REVIEW"
                result.reason = "weak_metadata"
            else:
                result.recommendation = "KEEP"
                result.reason = "sample_ok"
        else:
            result.recommendation = "REVIEW"
            result.reason = "sample_decode_failed"
    else:
        result.recommendation = "REVIEW"
        result.reason = "sample_download_failed"

    return result


def read_urls(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def write_reports(results: list[SourceResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "url", "title", "duration", "fps", "width", "height",
        "metadata_score", "metadata_reason",
        "sample_duplicate_ratio", "sample_static_ratio", "sample_timelapse_score",
        "recommendation", "reason",
    ]
    with open(output_dir / "prescreen_report.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "url": r.url,
                "title": r.title,
                "duration": r.duration,
                "fps": r.fps,
                "width": r.width,
                "height": r.height,
                "metadata_score": f"{r.metadata_score:.2f}",
                "metadata_reason": r.metadata_reason,
                "sample_duplicate_ratio": f"{r.sample_duplicate_ratio:.4f}" if r.sample_duplicate_ratio is not None else "",
                "sample_static_ratio": f"{r.sample_static_ratio:.4f}" if r.sample_static_ratio is not None else "",
                "sample_timelapse_score": f"{r.sample_timelapse_score:.4f}" if r.sample_timelapse_score is not None else "",
                "recommendation": r.recommendation,
                "reason": r.reason,
            })

    (output_dir / "keep_urls.txt").write_text(
        "\n".join(r.url for r in results if r.recommendation == "KEEP") + "\n",
        encoding="utf-8",
    )
    (output_dir / "review_urls.txt").write_text(
        "\n".join(r.url for r in results if r.recommendation == "REVIEW") + "\n",
        encoding="utf-8",
    )
    (output_dir / "reject_urls.txt").write_text(
        "\n".join(r.url for r in results if r.recommendation == "REJECT") + "\n",
        encoding="utf-8",
    )

    counts = {"KEEP": 0, "REVIEW": 0, "REJECT": 0}
    for r in results:
        counts[r.recommendation] += 1
    summary = {
        "total": len(results),
        **counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Source Pre-screen Summary")
    print("=" * 60)
    print(f"Total:  {summary['total']}")
    print(f"KEEP:   {counts['KEEP']} ({counts['KEEP']/max(summary['total'],1):.1%})")
    print(f"REVIEW: {counts['REVIEW']} ({counts['REVIEW']/max(summary['total'],1):.1%})")
    print(f"REJECT: {counts['REJECT']} ({counts['REJECT']/max(summary['total'],1):.1%})")
    print(f"\nReports written to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-screen candidate source URLs for Nexisgen dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast metadata-only pre-screen (recommended)
  python prescreen_sources.py urls.txt --output ./prescreen --metadata-only

  # Slower sample-based pre-screen
  python prescreen_sources.py urls.txt --output ./prescreen --sample-duration 15
        """,
    )
    parser.add_argument("urls_file", type=Path, help="File with one URL per line")
    parser.add_argument("--output", type=Path, default=Path("./prescreen_report"), help="Output directory")
    parser.add_argument("--metadata-only", action="store_true", help="Use only title/description/duration heuristics")
    parser.add_argument("--sample-duration", type=float, default=20.0, help="Sample duration in seconds")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")

    # Thresholds
    parser.add_argument("--dup-ratio", type=float, default=0.05, help="Sample duplicate ratio threshold")
    parser.add_argument("--static-ratio", type=float, default=0.50, help="Sample static ratio threshold")
    parser.add_argument("--timelapse-score", type=float, default=0.50, help="Sample timelapse score threshold")

    args = parser.parse_args()

    urls = read_urls(args.urls_file)
    if not urls:
        print("No URLs found.", file=sys.stderr)
        return 1

    print(f"Pre-screening {len(urls)} URLs (metadata_only={args.metadata_only})...")
    args.output.mkdir(parents=True, exist_ok=True)

    workdir = Path(tempfile.mkdtemp(prefix="prescreen_", dir=args.output))
    results: list[SourceResult] = []

    try:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_source, url, args, workdir) for url in urls]
                for future in tqdm(as_completed(futures), total=len(urls), desc="Pre-screening"):
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        logger.error("Worker failed: %s", exc)
        else:
            for url in tqdm(urls, desc="Pre-screening"):
                results.append(process_source(url, args, workdir))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    results.sort(key=lambda r: (r.recommendation != "KEEP", r.recommendation != "REVIEW", r.url))
    write_reports(results, args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(main())
