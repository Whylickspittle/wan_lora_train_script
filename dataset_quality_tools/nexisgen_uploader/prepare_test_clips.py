#!/usr/bin/env python3
"""Generate a small set of test clips and first frames for Nexisgen uploader testing.

Usage:
    python prepare_test_clips.py --output ./interval_1
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y"] + args
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_clip(output_path: Path, pattern: str) -> None:
    """Generate a 1280x704, 24fps, 121-frame H.264 MP4 test clip."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-f", "lavfi",
        "-i", pattern,
        "-r", "24",
        "-pix_fmt", "yuv420p",
        "-frames:v", "121",
        str(output_path),
    ])


def extract_first_frame(video_path: Path, frame_path: Path) -> None:
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg([
        "-i", str(video_path),
        "-vf", "select=eq(n\\,0)",
        "-q:v", "2",
        "-frames:v", "1",
        str(frame_path),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate test clips and frames")
    parser.add_argument("--output", type=Path, default=Path("./interval_1"), help="Output directory")
    args = parser.parse_args()

    clips_dir = args.output / "clips"
    frames_dir = args.output / "frames"

    patterns = [
        ("clip_0001.mp4", "testsrc=duration=5.0417:size=1280x704:rate=24"),
        ("clip_0002.mp4", "smptebars=duration=5.0417:size=1280x704:rate=24"),
        ("clip_0003.mp4", "color=c=blue:s=1280x704:d=5.0417"),
    ]

    for filename, pattern in patterns:
        clip_path = clips_dir / filename
        frame_path = frames_dir / (Path(filename).stem + ".jpg")
        print(f"generating {clip_path} ...")
        generate_clip(clip_path, pattern)
        print(f"extracting {frame_path} ...")
        extract_first_frame(clip_path, frame_path)

    print(f"done: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
