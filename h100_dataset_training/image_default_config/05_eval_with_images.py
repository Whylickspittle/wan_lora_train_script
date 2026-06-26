#!/usr/bin/env python
"""
Generate videos from (image, prompt) eval pairs using a trained LoRA checkpoint.

Eval manifest format (JSONL, one entry per line):
    {"id": "sample01", "image": "/abs/or/rel/path.jpg", "prompt": "..."}

Usage (inside the docker container):
    python 05_eval_with_images.py \
        --eval_manifest /eval/manifest.jsonl \
        --output_dir   /eval/outputs \
        [--checkpoint  runs/dataset_a/checkpoints/step_002000/lora.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_core"))

from infer_wan22_quality import load_custom_lora, write_video
from train_wan22_ti2v_lora import (
    DEFAULT_NEGATIVE_PROMPT,
    generated_video_to_thwc,
    load_pipeline,
    parse_resolution,
    set_pipeline_eval,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def read_eval_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "image" not in row or "prompt" not in row:
                raise ValueError(f"{path}:{n} must contain 'image' and 'prompt'.")
            row.setdefault("id", Path(row["image"]).stem)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def latest_checkpoint(run_dir: Path) -> Path:
    pointer = run_dir / "latest_checkpoint.txt"
    if pointer.exists():
        p = Path(pointer.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    candidates = sorted((run_dir / "checkpoints").glob("step_*/lora.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints under {run_dir}/checkpoints")
    return candidates[-1]


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    training = cfg.get("training", {})
    inference = cfg.get("inference", {})
    active = cfg["active_dataset"]
    default_run = Path(cfg.get("project_output_root", "runs")) / active

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_manifest", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Path to lora.pt. Default: latest checkpoint of the active dataset.")
    ap.add_argument("--resolution", default=training.get("resolution", "1280x704"))
    ap.add_argument("--num_frames", type=int, default=int(training.get("num_frames", 121)))
    ap.add_argument("--fps", type=int, default=int(training.get("fps", 24)))
    ap.add_argument("--sample_steps", type=int,
                    default=int(inference.get("sample_steps", training.get("sample_steps", 30))))
    ap.add_argument("--guidance_scale", type=float,
                    default=float(inference.get("guidance_scale", 5.0)))
    ap.add_argument("--seed", type=int, default=int(inference.get("seed", 1234)))
    ap.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    args = ap.parse_args()

    width, height = parse_resolution(args.resolution)
    ckpt = args.checkpoint or latest_checkpoint(default_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {ckpt}")
    print(f"Eval manifest: {args.eval_manifest}")
    print(f"Output dir: {args.output_dir}")
    print(f"Resolution: {width}x{height}  frames={args.num_frames}  fps={args.fps}")

    rows = read_eval_manifest(args.eval_manifest)
    device = torch.device(cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    pipe_args = argparse.Namespace(
        model_id=cfg["model_id"],
        device=str(device),
        mixed_precision=training.get("mixed_precision", "bf16"),
        vae_tiling=bool(training.get("vae_tiling", True)),
        task="i2v",
        vae_device=None,
        text_encoder_device=None,
        low_vram=False,
        gradient_checkpointing=False,
    )
    pipe = load_pipeline(pipe_args, device)
    set_pipeline_eval(pipe)
    load_custom_lora(
        pipe,
        str(ckpt),
        int(training.get("lora_rank", 32)),
        float(training.get("lora_alpha", 32.0)),
        float(training.get("lora_dropout", 0.0)),
        training.get("lora_targets", "to_q,to_k,to_v,to_out.0"),
    )

    exec_device = getattr(pipe, "_execution_device", device)
    if not isinstance(exec_device, torch.device):
        exec_device = torch.device(exec_device)
    generator = torch.Generator(device=exec_device).manual_seed(args.seed)

    for row in rows:
        image = Image.open(row["image"]).convert("RGB").resize((width, height), Image.LANCZOS)
        out_path = args.output_dir / f"{row['id']}.mp4"
        print(f"  -> {out_path}")
        with torch.no_grad():
            result = pipe(
                image=image,
                prompt=row["prompt"],
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_frames=args.num_frames,
                num_inference_steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_type="pt",
            ).frames[0]
        write_video(str(out_path), generated_video_to_thwc(result), fps=args.fps)

    print(f"Done. Wrote {len(rows)} videos to {args.output_dir}")


if __name__ == "__main__":
    main()
