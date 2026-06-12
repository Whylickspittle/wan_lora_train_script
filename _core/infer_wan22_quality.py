#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
import av as _av


def write_video(path, video, fps):
    if isinstance(video, torch.Tensor):
        video = video.numpy()
    T, H, W, C = video.shape
    container = _av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=int(fps))
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    for t in range(T):
        frame = _av.VideoFrame.from_ndarray(video[t], format="rgb24")
        frame = frame.reformat(format="yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

from config_utils import load_json_config
from logging_manager import MetricsLoggingManager, selected_gpu_metrics
from train_wan22_ti2v_lora import (
    DEFAULT_NEGATIVE_PROMPT,
    VBENCH_I2V_DIMENSIONS,
    compute_quality_metrics,
    generated_video_to_thwc,
    inject_lora,
    load_pipeline,
    load_video_for_training,
    parse_resolution,
    read_manifest,
    set_pipeline_eval,
    tensor_to_uint8_video,
)


def get_submodule(root: torch.nn.Module, name: str) -> torch.nn.Module:
    module = root
    for part in name.split("."):
        module = getattr(module, part)
    return module


def load_custom_lora(pipe: Any, checkpoint_path: str | Path, rank: int, alpha: float, dropout: float, targets: str) -> list[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    rank = int(saved_args.get("lora_rank", rank))
    alpha = float(saved_args.get("lora_alpha", alpha))
    dropout = float(saved_args.get("lora_dropout", dropout))
    targets = str(saved_args.get("lora_targets", targets))
    target_tuple = tuple(x.strip() for x in targets.split(",") if x.strip())
    wrapped = []
    if pipe.transformer is not None:
        wrapped.extend([f"transformer.{x}" for x in inject_lora(pipe.transformer, rank, alpha, dropout, target_tuple)])
    if getattr(pipe, "transformer_2", None) is not None:
        wrapped.extend([f"transformer_2.{x}" for x in inject_lora(pipe.transformer_2, rank, alpha, dropout, target_tuple)])
    state: dict[str, torch.Tensor] = checkpoint.get("lora", {})
    missing = []
    for key, tensor in state.items():
        if key.endswith(".alpha") or key.endswith(".rank"):
            continue
        if key.startswith("transformer."):
            root = pipe.transformer
            subkey = key[len("transformer.") :]
        elif key.startswith("transformer_2."):
            root = pipe.transformer_2
            subkey = key[len("transformer_2.") :]
        else:
            missing.append(key)
            continue
        if subkey.endswith(".lora_down.weight"):
            module_name = subkey[: -len(".lora_down.weight")]
            attr = "lora_down"
        elif subkey.endswith(".lora_up.weight"):
            module_name = subkey[: -len(".lora_up.weight")]
            attr = "lora_up"
        else:
            missing.append(key)
            continue
        module = get_submodule(root, module_name)
        getattr(module, attr).weight.data.copy_(tensor.to(getattr(module, attr).weight.device, getattr(module, attr).weight.dtype))
    if missing:
        print(f"Skipped {len(missing)} unrecognized LoRA tensors.")
    applied = sum(1 for k in state if not (k.endswith(".alpha") or k.endswith(".rank")) and k not in missing)
    print(f"LoRA loaded: {checkpoint_path} — wrapped {len(wrapped)} modules, applied {applied} tensors")
    return wrapped


def thwc_to_cthw_minus1_1(video: torch.Tensor) -> torch.Tensor:
    return video.float().permute(3, 0, 1, 2).div(127.5).sub(1.0).contiguous()


def mse_psnr(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a = a.float() / 255.0
    b = b.float() / 255.0
    mse = float(torch.mean((a - b) ** 2))
    mae = float(torch.mean(torch.abs(a - b)))
    psnr = 99.0 if mse <= 1e-12 else float(-10.0 * math.log10(mse))
    return mse, mae, psnr


def extra_video_metrics(thwc: torch.Tensor) -> dict[str, float]:
    v = thwc.float() / 255.0
    if v.shape[0] > 1:
        d1 = (v[1:] - v[:-1]).abs().flatten(1).mean(dim=1)
        d2 = (d1[1:] - d1[:-1]).abs() if d1.numel() > 1 else torch.zeros(1)
    else:
        d1 = torch.zeros(1)
        d2 = torch.zeros(1)
    channel_mean = v.flatten(0, 2).mean(dim=0)
    channel_std = v.flatten(0, 2).std(dim=0)
    luma = 0.2126 * v[..., 0] + 0.7152 * v[..., 1] + 0.0722 * v[..., 2]
    blank_frames = ((luma.flatten(1).std(dim=1) < 0.01) | (luma.flatten(1).mean(dim=1) < 0.02)).float().mean()
    return {
        "gen_rgb_mean_r": float(channel_mean[0]),
        "gen_rgb_mean_g": float(channel_mean[1]),
        "gen_rgb_mean_b": float(channel_mean[2]),
        "gen_rgb_std_r": float(channel_std[0]),
        "gen_rgb_std_g": float(channel_std[1]),
        "gen_rgb_std_b": float(channel_std[2]),
        "gen_temporal_delta_p01": float(torch.quantile(d1, 0.01)),
        "gen_temporal_delta_p50": float(torch.quantile(d1, 0.50)),
        "gen_temporal_delta_p95": float(torch.quantile(d1, 0.95)),
        "gen_temporal_delta_p99": float(torch.quantile(d1, 0.99)),
        "gen_temporal_jerk_mean": float(d2.mean()),
        "gen_temporal_jerk_p95": float(torch.quantile(d2, 0.95)),
        "gen_blank_frame_ratio": float(blank_frames),
        "gen_luma_min": float(luma.min()),
        "gen_luma_max": float(luma.max()),
        "gen_luma_p01": float(np.percentile(luma.flatten().numpy(), 1)),
        "gen_luma_p99": float(np.percentile(luma.flatten().numpy(), 99)),
    }


def write_vbench_command(output_dir: Path, sample_dir: Path, width: int, height: int) -> None:
    dims = " ".join(VBENCH_I2V_DIMENSIONS)
    command = (
        "python /path/to/VBench/evaluate_i2v.py \\\n"
        f"  --videos_path {sample_dir.as_posix()} \\\n"
        f"  --output_path {(output_dir / 'vbench').as_posix()} \\\n"
        f"  --dimension {dims} \\\n"
        "  --mode custom_input \\\n"
        f"  --ratio {width}-{height}\n"
    )
    (output_dir / "vbench_inference_command.txt").write_text(command, encoding="utf-8")


def run_inference_config(config_path: str | Path) -> None:
    config = load_json_config(config_path)
    output_dir = Path(config["output_dir"])
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLoggingManager(output_dir)

    width, height = parse_resolution(config.get("resolution", "1280x704"))
    config.setdefault("width", width)
    config.setdefault("height", height)
    config.setdefault("task", "i2v")
    config.setdefault("model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    config.setdefault("device", "cuda:0")
    config.setdefault("mixed_precision", "bf16")
    config.setdefault("vae_tiling", True)
    config.setdefault("gradient_checkpointing", False)
    config.setdefault("low_vram", False)
    config.setdefault("enable_cpu_offload", bool(config.get("low_vram", False)))
    config.setdefault("vae_device", None)
    config.setdefault("text_encoder_device", None)
    config.setdefault("lora_rank", 32)
    config.setdefault("lora_alpha", 32.0)
    config.setdefault("lora_dropout", 0.0)
    config.setdefault("lora_targets", "to_q,to_k,to_v,to_out.0")
    config.setdefault("num_frames", 121)
    config.setdefault("fps", 24)
    config.setdefault("sample_steps", 30)
    config.setdefault("guidance_scale", 5.0)
    config.setdefault("seed", 1234)
    config.setdefault("max_samples", 4)
    config.setdefault("precompute_prompt_embeds", False)
    args = argparse.Namespace(**config)
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16 if args.mixed_precision == "fp16" else torch.float32

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    rows = read_manifest(Path(args.manifest))
    pipe = load_pipeline(args, device)
    set_pipeline_eval(pipe)
    wrapped = []
    if config.get("checkpoint_path"):
        wrapped = load_custom_lora(
            pipe,
            config["checkpoint_path"],
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            args.lora_targets,
        )
    if bool(config.get("enable_cpu_offload", False)) and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload(device=device)
    logger.event("inference_start", config=config, wrapped_lora_modules=wrapped)

    execution_device = getattr(pipe, "_execution_device", device)
    if not isinstance(execution_device, torch.device):
        execution_device = torch.device(execution_device)
    generator = torch.Generator(device=execution_device).manual_seed(int(args.seed))
    for index, row in enumerate(rows[: int(args.max_samples)]):
        prompt = str(row["prompt"])
        out_path = samples_dir / f"{row['id']}.mp4"
        ref_first_frame = None
        prompt_arg = prompt
        negative_prompt_arg = config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        prompt_embeds = None
        negative_prompt_embeds = None
        if bool(config.get("precompute_prompt_embeds", False)):
            prompt_embeds = pipe.encode_prompt(
                prompt=[prompt],
                negative_prompt=None,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=next(pipe.text_encoder.parameters()).device,
                dtype=next(pipe.text_encoder.parameters()).dtype,
            )[0].to(device=device, dtype=dtype)
            prompt_arg = None
            negative_prompt_arg = None
        start = time.perf_counter()
        if args.task == "i2v":
            video, source_meta = load_video_for_training(row["video"], width, height, int(args.num_frames))
            ref_first_frame = tensor_to_uint8_video(video[:, :1])[0]
            image = Image.fromarray(ref_first_frame.numpy())
            result = pipe(
                image=image,
                prompt=prompt_arg,
                negative_prompt=negative_prompt_arg,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                height=height,
                width=width,
                num_frames=int(args.num_frames),
                num_inference_steps=int(args.sample_steps),
                guidance_scale=float(args.guidance_scale),
                generator=generator,
                output_type="pt",
            ).frames[0]
        else:
            source_meta = {"source_width": width, "source_height": height, "source_frames": args.num_frames, "source_fps": args.fps}
            result = pipe(
                prompt=prompt_arg,
                negative_prompt=negative_prompt_arg,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                height=height,
                width=width,
                num_frames=int(args.num_frames),
                num_inference_steps=int(args.sample_steps),
                guidance_scale=float(args.guidance_scale),
                generator=generator,
                output_type="pt",
            ).frames[0]
        generation_seconds = time.perf_counter() - start
        thwc = generated_video_to_thwc(result)
        write_video(str(out_path), thwc, fps=int(args.fps))
        cthw = thwc_to_cthw_minus1_1(thwc)
        quality = compute_quality_metrics(row, cthw, source_meta, width, height, int(args.num_frames), float(args.fps))
        metrics = {
            **quality.__dict__,
            **extra_video_metrics(thwc),
            "sample_index": index,
            "output_path": str(out_path),
            "output_size_mb": out_path.stat().st_size / 1024**2,
            "generation_seconds": generation_seconds,
            "generation_frames_per_second": float(args.num_frames) / max(generation_seconds, 1e-6),
            "prompt_chars": len(prompt),
            "prompt_words": len(prompt.split()),
            "sample_steps": int(args.sample_steps),
            "guidance_scale": float(args.guidance_scale),
            "checkpoint_path": str(config.get("checkpoint_path", "")),
        }
        if ref_first_frame is not None:
            mse, mae, psnr = mse_psnr(thwc[0], ref_first_frame)
            metrics.update(
                {
                    "first_frame_mse": mse,
                    "first_frame_mae": mae,
                    "first_frame_psnr": psnr,
                }
            )
        metrics.update(selected_gpu_metrics(device))
        logger.log_inference(metrics)

    write_vbench_command(output_dir, samples_dir, width, height)
    logger.event("inference_done", samples_dir=str(samples_dir))
    logger.close()


if __name__ == "__main__":
    run_inference_config(Path("configs/inference_h100_lora.json"))
