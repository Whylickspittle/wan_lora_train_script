#!/usr/bin/env python3
"""Convert curated Pexels clips into Nexisgen dataset format.

Reference: /workspace/merged_dataset_nexisgen/
Columns: clip_id, clip_uri, clip_sha256, first_frame_uri, first_frame_sha256,
         source_video_id, clip_start_sec, duration_sec, width, height, fps,
         num_frames, source_video_url, caption
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

CLIP_RE = re.compile(r"clip_(\d+)_node([\d.]+)\.mp4$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {r.stderr}")
    return json.loads(r.stdout)


def parse_fps(r_frame_rate: str) -> float:
    if "/" in r_frame_rate:
        num, den = r_frame_rate.split("/")
        return float(num) / float(den)
    return float(r_frame_rate)


def extract_first_frame(video: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video), "-ss", "00:00:00",
        "-vframes", "1", "-q:v", "2", str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video}: {r.stderr}")


def parse_clip_info(name: str) -> tuple[str, int, float] | None:
    m = CLIP_RE.search(name)
    if not m:
        return None
    video_id = m.group(1)
    start_sec = float(m.group(2))
    clip_id = name[:-4] if name.endswith(".mp4") else name
    return clip_id, int(video_id), start_sec


def build_dataset(src_clips: Path, src_manifest: Path | None, out_dir: Path,
                  miner_hotkey: str = "test_miner", netuid: int = 70) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_out = out_dir / "clips"
    frames_out = out_dir / "frames"
    clips_out.mkdir(exist_ok=True)
    frames_out.mkdir(exist_ok=True)

    prompts: dict[str, str] = {}
    if src_manifest and src_manifest.exists():
        with src_manifest.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                clip_id = Path(rec["video"]).stem
                prompts[clip_id] = rec.get("prompt", "")

    rows = []
    manifest_rows = []
    for src in sorted(src_clips.glob("*.mp4")):
        info = parse_clip_info(src.name)
        if not info:
            print(f"skip unrecognized name: {src.name}")
            continue
        clip_id, video_id, start_sec = info

        dst_clip = clips_out / src.name
        if not dst_clip.exists():
            shutil.copy2(src, dst_clip)

        frame_name = f"{clip_id}.jpg"
        dst_frame = frames_out / frame_name
        if not dst_frame.exists():
            try:
                extract_first_frame(dst_clip, dst_frame)
            except Exception as exc:
                print(f"  failed first frame for {src.name}: {exc}")
                continue

        try:
            probe = ffprobe(dst_clip)
        except Exception as exc:
            print(f"  failed ffprobe for {src.name}: {exc}")
            continue

        stream = probe.get("streams", [{}])[0]
        fmt = probe.get("format", {})
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        fps = parse_fps(stream.get("r_frame_rate", "24/1"))
        duration = float(fmt.get("duration", 0))
        nb_frames = stream.get("nb_frames")
        num_frames = int(nb_frames) if nb_frames and nb_frames != "N/A" else int(round(duration * fps))

        rows.append({
            "clip_id": clip_id,
            "clip_uri": f"clips/{src.name}",
            "clip_sha256": sha256_file(dst_clip),
            "first_frame_uri": f"frames/{frame_name}",
            "first_frame_sha256": sha256_file(dst_frame),
            "source_video_id": f"pexels_{video_id}",
            "clip_start_sec": start_sec,
            "duration_sec": duration,
            "width": width,
            "height": height,
            "fps": fps,
            "num_frames": num_frames,
            "source_video_url": f"https://www.pexels.com/video/{video_id}/",
            "caption": prompts.get(clip_id, ""),
        })

        manifest_rows.append({
            "id": clip_id,
            "video": f"clips/{src.name}",
            "prompt": prompts.get(clip_id, ""),
        })

    if not rows:
        raise RuntimeError("No clips processed")

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "dataset.parquet", index=False)

    manifest = {
        "protocol_version": "2.0.0",
        "schema_version": "2.0.0",
        "spec_id": "video_v1",
        "netuid": netuid,
        "miner_hotkey": miner_hotkey,
        "interval_id": 1,
        "created_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "record_count": len(rows),
        "dataset_sha256": sha256_file(out_dir / "dataset.parquet"),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for rec in manifest_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Created dataset at {out_dir}")
    print(f"  clips: {len(rows)}")
    print(f"  duration total: {sum(r['duration_sec'] for r in rows):.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Pexels clips to Nexisgen dataset format")
    ap.add_argument("--src-clips", required=True, type=Path, help="source clips directory")
    ap.add_argument("--src-manifest", type=Path, help="source manifest jsonl with prompts")
    ap.add_argument("--out-dir", required=True, type=Path, help="output dataset directory")
    ap.add_argument("--miner-hotkey", default="test_miner")
    ap.add_argument("--netuid", type=int, default=70)
    args = ap.parse_args()

    build_dataset(args.src_clips, args.src_manifest, args.out_dir,
                  miner_hotkey=args.miner_hotkey, netuid=args.netuid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
