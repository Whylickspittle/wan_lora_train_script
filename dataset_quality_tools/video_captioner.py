"""Standalone multi-frame video captioner for dataset quality tools.

This module captions short video clips by sending evenly spaced keyframes to
an OpenAI-compatible vision API. It is intentionally self-contained so it can
be moved to a separate repository.

Typical usage:

    from video_captioner import Captioner, extract_keyframes, load_env

    env = load_env(".env")
    captioner = Captioner(
        api_key=env["OPENAI_API_KEY"],
        base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=env.get("NEXIS_CAPTION_MODEL", "gpt-4o-mini"),
        num_keyframes=int(env.get("NEXIS_CAPTION_KEYFRAMES", "4")),
    )

    keyframes = extract_keyframes("clip.mp4", "keyframes_dir", num_keyframes=4)
    caption = captioner.caption_frames(keyframes)
"""

from __future__ import annotations

import base64
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_FPS = 24
DEFAULT_NUM_FRAMES = 121

_SINGLE_PROMPT = (
    "Describe this video frame in one short sentence (≤ 20 words) that would "
    "work as a text-to-video generation prompt. Include the subject, setting, "
    "and any motion that is visible or strongly implied. Use present-tense "
    "active verbs. Do not add commentary."
)

_MULTI_PROMPT = (
    "These images are evenly spaced frames from a 5-second video clip, shown "
    "in chronological order. Write one sentence (≤ 25 words) that works as a "
    "text-to-video generation prompt.\n\n"
    "Your caption MUST describe:\n"
    "1. The main subject and setting\n"
    "2. Visible motion: what moves, how it moves, and/or how the camera moves\n"
    "3. Direction when possible (e.g., forward, upward, across the frame)\n\n"
    "Use present-tense active verbs. Avoid words like 'static', 'still', "
    "'photograph', 'image', or 'picture'. Do not describe only the first frame.\n\n"
    "Good example: 'A drone glides forward over a winding river, revealing "
    "layered red rock cliffs under a bright blue sky.'\n"
    "Bad example: 'A scenic view of a canyon under a blue sky.'"
)


def load_env(path: str | Path) -> dict[str, str]:
    """Load a simple KEY=VALUE .env file.

    Values are stripped of surrounding quotes. Lines starting with # and empty
    lines are ignored.
    """
    env: dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _run_ffmpeg(cmd: list[str], timeout_sec: int = 120) -> None:
    """Run an ffmpeg command and raise on failure."""
    logger.debug("ffmpeg command: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=timeout_sec, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "")[-500:]
        raise RuntimeError(f"ffmpeg failed: {stderr}") from exc


def _get_video_duration(path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=True, timeout=30, capture_output=True, text=True)
        data = json.loads(proc.stdout)
        return float(data.get("format", {}).get("duration") or 0.0)
    except Exception as exc:
        logger.warning("duration probe failed for %s: %s", path, exc)
        return 0.0


def extract_first_frame(src: str | Path, dst: str | Path) -> Path:
    """Extract the first frame of a video to a JPEG file."""
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        "select=eq(n\\,0)",
        "-frames:v",
        "1",
        str(dst),
    ]
    _run_ffmpeg(cmd)
    return dst


def extract_keyframes(
    src: str | Path,
    output_dir: str | Path,
    num_keyframes: int,
    *,
    fps: float = DEFAULT_FPS,
    total_frames: int | None = None,
) -> list[Path]:
    """Extract up to ``num_keyframes`` evenly spaced JPEGs from ``src``.

    The returned frames are sorted in chronological order. The output directory
    is cleared before extraction to avoid stale files.

    Args:
        src: Path to the source video.
        output_dir: Directory where keyframes will be written.
        num_keyframes: Number of keyframes to extract.
        fps: Frame rate used when ``total_frames`` is not provided.
        total_frames: Optional total frame count. If not provided, it is derived
            from video duration and ``fps``.
    """
    if num_keyframes < 1:
        return []

    src = Path(src)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear stale files.
    for stale in output_dir.glob("*.jpg"):
        stale.unlink()

    if total_frames is None:
        duration = _get_video_duration(src)
        total_frames = int(round(duration * fps)) if duration > 0 else DEFAULT_NUM_FRAMES
        total_frames = max(total_frames, DEFAULT_NUM_FRAMES)

    if num_keyframes == 1:
        indices = [0]
    else:
        indices = [
            int(round(i * (total_frames - 1) / (num_keyframes - 1)))
            for i in range(num_keyframes)
        ]

    select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
    pattern = output_dir / f"{src.stem}_keyframe_%03d.jpg"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"select={select_expr}",
        "-vsync",
        "vfr",
        str(pattern),
    ]
    _run_ffmpeg(cmd)

    frames = sorted(output_dir.glob("*.jpg"))
    if len(frames) < num_keyframes:
        logger.warning(
            "requested %d keyframes but only got %d from %s",
            num_keyframes,
            len(frames),
            src,
        )
    return frames


def _b64_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


@dataclass
class Captioner:
    """OpenAI-compatible vision captioner supporting single or multi-frame input."""

    api_key: str
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout_sec: int = 120
    num_keyframes: int = 4

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("api_key is required")
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "The 'requests' package is required. Install it with: pip install requests"
            ) from exc
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    def caption_frame(self, frame_path: str | Path) -> str:
        """Caption a single frame."""
        return self.caption_frames([Path(frame_path)])

    def caption_frames(self, frame_paths: list[str | Path]) -> str:
        """Caption using one or more frames."""
        valid_paths = [Path(p) for p in frame_paths if Path(p).exists()]
        if not valid_paths:
            logger.warning("caption_frames: no valid frame paths provided")
            return ""

        prompt = _SINGLE_PROMPT if len(valid_paths) == 1 else _MULTI_PROMPT
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in valid_paths:
            data_url = f"data:image/jpeg;base64,{_b64_image(path)}"
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 80,
            "temperature": 0.1,
        }

        try:
            resp = self._session.post(
                self.endpoint,
                json=payload,
                timeout=float(self.timeout_sec),
            )
            resp.raise_for_status()
            data = resp.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                or ""
            ).strip()
            return text[:300]
        except Exception as exc:
            logger.warning("caption call failed: %s", exc)
            return ""

    def caption_video(
        self,
        video_path: str | Path,
        *,
        workdir: str | Path | None = None,
        num_keyframes: int | None = None,
    ) -> str:
        """Extract keyframes from a video and caption them.

        If ``num_keyframes`` is 1, only the first frame is used. Otherwise,
        evenly spaced keyframes are extracted.
        """
        video_path = Path(video_path)
        if workdir is None:
            workdir = Path(".caption_workdir")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        count = num_keyframes if num_keyframes is not None else self.num_keyframes
        if count <= 1:
            frame_path = workdir / "first_frame.jpg"
            extract_first_frame(video_path, frame_path)
            return self.caption_frame(frame_path)

        keyframes_dir = workdir / "keyframes"
        frames = extract_keyframes(video_path, keyframes_dir, count)
        if not frames:
            logger.warning(
                "multi-keyframe extraction produced no frames, falling back to first frame"
            )
            frame_path = workdir / "first_frame.jpg"
            extract_first_frame(video_path, frame_path)
            return self.caption_frame(frame_path)
        return self.caption_frames(frames)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    env = load_env(".env")
    if "OPENAI_API_KEY" not in env:
        raise SystemExit("OPENAI_API_KEY not found in .env")

    captioner = Captioner(
        api_key=env["OPENAI_API_KEY"],
        base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=env.get("NEXIS_CAPTION_MODEL", "gpt-4o-mini"),
        num_keyframes=int(env.get("NEXIS_CAPTION_KEYFRAMES", "4")),
    )

    import argparse

    parser = argparse.ArgumentParser(description="Caption a single video clip.")
    parser.add_argument("video", type=Path, help="Path to the video file.")
    parser.add_argument(
        "--keyframes",
        type=int,
        default=None,
        help="Override number of keyframes.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(".caption_workdir"),
        help="Working directory for extracted frames.",
    )
    args = parser.parse_args()

    caption = captioner.caption_video(
        args.video,
        workdir=args.workdir,
        num_keyframes=args.keyframes,
    )
    print(caption)
