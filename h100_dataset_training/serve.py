#!/usr/bin/env python3
"""Local FastAPI server for Wan2.2 TI2V LoRA inference.

POST /generate
  - image: uploaded image file (JPEG/PNG) — the conditioning first frame
  - prompt: text describing the video
  - steps: (optional) number of diffusion steps, default 30
  - guidance_scale: (optional) CFG scale, default 5.0
  - seed: (optional) integer seed, default 1234
  - num_frames: (optional) number of frames, default 121

Returns: mp4 video file download
"""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import threading
import time
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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

# ---------------------------------------------------------------------------
# Config — edit these paths if needed
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path(__file__).resolve().parent / "runs/dataset_a/checkpoints/step_004400/lora.pt"
MODEL_ID        = Path(__file__).resolve().parent / "models/Wan2.2-TI2V-5B-Diffusers"
RESOLUTION      = "1280x704"
NUM_FRAMES      = 121
FPS             = 24
DEVICE          = "cuda:0"
OUTPUT_DIR      = Path(__file__).resolve().parent / "serve_outputs"
# ---------------------------------------------------------------------------

app = FastAPI(title="Wan2.2 TI2V LoRA")

_lock  = threading.Lock()   # one inference at a time on the GPU
_pipe  = None
_dtype = torch.bfloat16
_device: torch.device | None = None
_width  = 0
_height = 0


@app.on_event("startup")
def load_model() -> None:
    global _pipe, _device, _width, _height

    _width, _height = parse_resolution(RESOLUTION)
    _device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if _device.type == "cuda":
        torch.cuda.set_device(_device)

    print(f"Loading pipeline from {MODEL_ID} ...")
    fake_args = argparse.Namespace(
        model_id=str(MODEL_ID),
        task="i2v",
        device=str(_device),
        mixed_precision="bf16",
        width=_width,
        height=_height,
        num_frames=NUM_FRAMES,
        fps=FPS,
        vae_tiling=True,
        gradient_checkpointing=False,
        low_vram=False,
        enable_cpu_offload=False,
        vae_device=None,
        text_encoder_device=None,
        lora_rank=32,
        lora_alpha=32.0,
        lora_dropout=0.0,
        lora_targets="to_q,to_k,to_v,to_out.0",
        seed=1234,
    )
    _pipe = load_pipeline(fake_args, _device)
    set_pipeline_eval(_pipe)

    print(f"Loading LoRA from {CHECKPOINT_PATH} ...")
    load_custom_lora(_pipe, CHECKPOINT_PATH, rank=32, alpha=32.0, dropout=0.0, targets="to_q,to_k,to_v,to_out.0")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Server ready.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "checkpoint": str(CHECKPOINT_PATH)}


@app.post("/generate")
async def generate(
    image: UploadFile = File(..., description="Conditioning first frame (JPEG/PNG)"),
    prompt: str = Form(..., description="Text prompt describing the video"),
    steps: int = Form(30, description="Diffusion steps"),
    guidance_scale: float = Form(5.0, description="CFG guidance scale"),
    seed: int = Form(1234, description="Random seed"),
    num_frames: int = Form(NUM_FRAMES, description="Number of frames to generate"),
) -> FileResponse:
    if not _pipe:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Read and validate uploaded image
    img_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((_width, _height))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    with _lock:
        execution_device = getattr(_pipe, "_execution_device", _device)
        if not isinstance(execution_device, torch.device):
            execution_device = torch.device(execution_device)
        generator = torch.Generator(device=execution_device).manual_seed(seed)

        t0 = time.perf_counter()
        result = _pipe(
            image=pil_image,
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            height=_height,
            width=_width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pt",
        ).frames[0]
        elapsed = time.perf_counter() - t0
        print(f"Generated in {elapsed:.1f}s")

    thwc = generated_video_to_thwc(result)

    # Save to a temp file and return it
    out_path = OUTPUT_DIR / f"gen_{seed}_{int(time.time())}.mp4"
    write_video(str(out_path), thwc, fps=FPS)

    return FileResponse(
        path=str(out_path),
        media_type="video/mp4",
        filename=out_path.name,
    )
