#!/usr/bin/env python
"""Wan2.2 TI2V-5B LoRA training harness with dense dataset metrics.

The script is intentionally PyTorch-native: no Accelerate, DeepSpeed, or trainer
framework. Diffusers/Transformers are used only to load the official Wan model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Subset
import av as _av


def read_video(path, pts_unit="sec", output_format="TCHW"):
    container = _av.open(str(path))
    stream = container.streams.video[0]
    fps = float(stream.average_rate)
    frames = [torch.from_numpy(f.to_ndarray(format="rgb24")) for f in container.decode(video=0)]
    container.close()
    video = torch.stack(frames)  # (T, H, W, C)
    if output_format == "TCHW":
        video = video.permute(0, 3, 1, 2)
    return video, torch.zeros(0), {"video_fps": fps}


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
from tqdm import tqdm


DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, static, blurred details, subtitles, watermark, low quality, "
    "jpeg artifacts, malformed hands, malformed face, deformed body, bad anatomy, "
    "fused fingers, extra limbs, messy background, duplicate subjects"
)

VBENCH_I2V_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


@dataclass
class QualityMetrics:
    id: str
    video: str
    source_width: int
    source_height: int
    source_frames: int
    source_fps: float
    target_width: int
    target_height: int
    target_frames: int
    target_fps: float
    width_error_px: int
    height_error_px: int
    frame_error: int
    fps_error: float
    aspect_error: float
    duration_seconds: float
    duplicate_frame_ratio: float
    black_pixel_ratio: float
    white_pixel_ratio: float
    luma_mean: float
    luma_std: float
    contrast_rms: float
    saturation_mean: float
    sharpness_grad_mean: float
    temporal_diff_mean: float
    temporal_diff_std: float
    flicker_luma_std: float
    motion_p95: float
    entropy_mean: float
    conformance_score: float


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self.fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class CsvWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, fieldnames=fieldnames)
        if self.path.stat().st_size == 0:
            self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        clean = {k: row.get(k) for k in self.writer.fieldnames}
        self.writer.writerow(clean)
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.lora_down.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_up.to(device=base.weight.device, dtype=base.weight.dtype)
        self.dropout.to(device=base.weight.device)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scale


def parse_resolution(value: str) -> tuple[int, int]:
    if "x" in value:
        w, h = value.lower().split("x", 1)
    elif "*" in value:
        w, h = value.split("*", 1)
    else:
        raise argparse.ArgumentTypeError("Resolution must look like 1280x704.")
    return int(w), int(h)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "video" not in row or "prompt" not in row:
                raise ValueError(f"{path}:{line_no} must contain 'video' and 'prompt'.")
            row.setdefault("id", Path(row["video"]).stem)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def uniform_frame_indices(total: int, wanted: int) -> torch.Tensor:
    if total <= 0:
        raise ValueError("Video has no frames.")
    if total == wanted:
        return torch.arange(total)
    return torch.linspace(0, total - 1, wanted).round().long().clamp(0, total - 1)


def load_video_for_training(
    video_path: str,
    width: int,
    height: int,
    frames: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    raw, _, info = read_video(video_path, pts_unit="sec", output_format="TCHW")
    if raw.numel() == 0:
        raise ValueError(f"Could not decode frames from {video_path}.")
    source_frames = int(raw.shape[0])
    source_height = int(raw.shape[2])
    source_width = int(raw.shape[3])
    fps = float(info.get("video_fps", 0.0) or 0.0)
    idx = uniform_frame_indices(source_frames, frames)
    sampled = raw[idx].float() / 255.0
    sampled = F.interpolate(sampled, size=(height, width), mode="bilinear", align_corners=False)
    sampled = sampled.mul(2.0).sub(1.0)
    video = sampled.permute(1, 0, 2, 3).contiguous()
    meta = {
        "source_frames": source_frames,
        "source_height": source_height,
        "source_width": source_width,
        "source_fps": fps,
    }
    return video, meta


def tensor_to_uint8_video(video_cthw: torch.Tensor) -> torch.Tensor:
    video = video_cthw.detach().float().clamp(-1, 1).add(1).mul(127.5)
    return video.permute(1, 2, 3, 0).round().byte().cpu()


def generated_video_to_thwc(video: Any) -> torch.Tensor:
    if isinstance(video, list):
        frames = []
        for frame in video:
            if isinstance(frame, Image.Image):
                frames.append(torch.from_numpy(np.asarray(frame.convert("RGB"))))
            else:
                frames.append(torch.as_tensor(frame))
        video = torch.stack(frames, dim=0)
    video = torch.as_tensor(video).detach().float().cpu()
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Generated video must be 4D or 5D, got shape {tuple(video.shape)}.")
    if video.shape[-1] in (1, 3):
        thwc = video
    elif video.shape[0] in (1, 3):
        thwc = video.permute(1, 2, 3, 0)
    elif video.shape[1] in (1, 3):
        thwc = video.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Could not infer generated video layout from shape {tuple(video.shape)}.")
    if thwc.dtype.is_floating_point:
        if float(thwc.max()) <= 1.5 and float(thwc.min()) >= -0.5:
            thwc = thwc.clamp(0, 1).mul(255)
        else:
            thwc = thwc.clamp(0, 255)
    return thwc.round().byte()


def set_pipeline_eval(pipe: Any) -> None:
    for name in ["vae", "text_encoder", "image_encoder", "transformer", "transformer_2"]:
        module = getattr(pipe, name, None)
        if module is not None and hasattr(module, "eval"):
            module.eval()


def rgb_to_luma(video: torch.Tensor) -> torch.Tensor:
    r, g, b = video[:, 0], video[:, 1], video[:, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def compute_entropy(video_0_1: torch.Tensor) -> float:
    gray = rgb_to_luma(video_0_1)
    vals = gray.flatten()
    hist = torch.histc(vals, bins=64, min=0.0, max=1.0)
    prob = hist / hist.sum().clamp_min(1.0)
    entropy = -(prob * (prob + 1e-12).log2()).sum()
    return float(entropy / math.log2(64))


def compute_quality_metrics(
    row: dict[str, Any],
    video_cthw: torch.Tensor,
    source_meta: dict[str, Any],
    target_width: int,
    target_height: int,
    target_frames: int,
    target_fps: float,
) -> QualityMetrics:
    video = video_cthw.permute(1, 0, 2, 3).float().add(1).mul(0.5).clamp(0, 1)
    luma = rgb_to_luma(video)
    frame_delta = (video[1:] - video[:-1]).abs() if video.shape[0] > 1 else torch.zeros_like(video[:1])
    luma_frame_mean = luma.flatten(1).mean(dim=1)
    dx = video[:, :, :, 1:] - video[:, :, :, :-1]
    dy = video[:, :, 1:, :] - video[:, :, :-1, :]
    sharpness = dx.abs().mean() + dy.abs().mean()
    max_delta = frame_delta.flatten(1).mean(dim=1) if frame_delta.numel() else torch.tensor([0.0])
    duplicate_ratio = float((max_delta < 0.002).float().mean())
    mx = video.max(dim=1).values
    mn = video.min(dim=1).values
    saturation = ((mx - mn) / mx.clamp_min(1e-6)).mean()
    source_w = int(source_meta["source_width"])
    source_h = int(source_meta["source_height"])
    source_f = int(source_meta["source_frames"])
    source_fps = float(source_meta["source_fps"])
    aspect_error = abs((source_w / max(source_h, 1)) - (target_width / target_height))
    frame_error = source_f - target_frames
    fps_error = source_fps - target_fps if source_fps > 0 else 0.0
    penalties = [
        min(abs(source_w - target_width) / target_width, 1.0),
        min(abs(source_h - target_height) / target_height, 1.0),
        min(abs(frame_error) / target_frames, 1.0),
        min(abs(fps_error) / target_fps, 1.0) if source_fps > 0 else 0.25,
        min(aspect_error, 1.0),
        duplicate_ratio,
    ]
    conformance = 1.0 - float(sum(penalties) / len(penalties))
    return QualityMetrics(
        id=str(row["id"]),
        video=str(row["video"]),
        source_width=source_w,
        source_height=source_h,
        source_frames=source_f,
        source_fps=source_fps,
        target_width=target_width,
        target_height=target_height,
        target_frames=target_frames,
        target_fps=target_fps,
        width_error_px=source_w - target_width,
        height_error_px=source_h - target_height,
        frame_error=frame_error,
        fps_error=fps_error,
        aspect_error=float(aspect_error),
        duration_seconds=float(source_f / source_fps) if source_fps > 0 else 0.0,
        duplicate_frame_ratio=duplicate_ratio,
        black_pixel_ratio=float((video < 0.01).float().mean()),
        white_pixel_ratio=float((video > 0.99).float().mean()),
        luma_mean=float(luma.mean()),
        luma_std=float(luma.std()),
        contrast_rms=float(luma.std()),
        saturation_mean=float(saturation),
        sharpness_grad_mean=float(sharpness),
        temporal_diff_mean=float(frame_delta.mean()),
        temporal_diff_std=float(frame_delta.std()),
        flicker_luma_std=float(luma_frame_mean.std()),
        motion_p95=float(torch.quantile(max_delta.float(), 0.95)),
        entropy_mean=compute_entropy(video),
        conformance_score=conformance,
    )


class WanVideoDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        width: int,
        height: int,
        frames: int,
        fps: float,
        quality_writer: JsonlWriter | None = None,
        cache_quality: bool = True,
    ):
        self.rows = rows
        self.width = width
        self.height = height
        self.frames = frames
        self.fps = fps
        self.quality_writer = quality_writer
        self.cache_quality = cache_quality
        self._quality_seen: set[str] = set()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        video, meta = load_video_for_training(row["video"], self.width, self.height, self.frames)
        if self.quality_writer is not None and (not self.cache_quality or row["id"] not in self._quality_seen):
            metrics = compute_quality_metrics(row, video, meta, self.width, self.height, self.frames, self.fps)
            self.quality_writer.write(asdict(metrics))
            self._quality_seen.add(str(row["id"]))
        return {
            "id": str(row["id"]),
            "video": video,
            "prompt": str(row["prompt"]),
            "path": str(row["video"]),
            "source_meta": meta,
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": [x["id"] for x in items],
        "videos": torch.stack([x["video"] for x in items], dim=0),
        "prompts": [x["prompt"] for x in items],
        "paths": [x["path"] for x in items],
        "source_meta": [x["source_meta"] for x in items],
    }


def should_lora_wrap(name: str, module: nn.Module, targets: tuple[str, ...]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    return any(token in name for token in targets)


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    targets: tuple[str, ...],
    prefix: str = "",
) -> list[str]:
    wrapped = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if should_lora_wrap(full_name, child, targets):
            setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
            wrapped.append(full_name)
        else:
            wrapped.extend(inject_lora(child, rank, alpha, dropout, targets, full_name))
    return wrapped


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    state = {}
    for name, sub in module.named_modules():
        if isinstance(sub, LoRALinear):
            state[f"{name}.lora_down.weight"] = sub.lora_down.weight.detach().cpu()
            state[f"{name}.lora_up.weight"] = sub.lora_up.weight.detach().cpu()
            state[f"{name}.alpha"] = torch.tensor(float(sub.alpha))
            state[f"{name}.rank"] = torch.tensor(int(sub.rank))
    return state


def trainable_parameters(module: Any) -> list[nn.Parameter]:
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        return [p for p in parameters() if p.requires_grad]

    collected: list[nn.Parameter] = []
    for component_name in ["transformer", "transformer_2", "vae", "text_encoder", "image_encoder"]:
        component = getattr(module, component_name, None)
        component_parameters = getattr(component, "parameters", None)
        if callable(component_parameters):
            collected.extend([p for p in component_parameters() if p.requires_grad])
    return collected


def retrieve_latents(encoder_output: Any, sample_mode: str) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample()
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not retrieve latents from VAE output.")


def latent_stats_tensors(vae: nn.Module, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    std = (1.0 / torch.tensor(vae.config.latents_std)).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    return mean, std


@torch.no_grad()
def encode_videos_to_latents(
    vae: nn.Module,
    videos: torch.Tensor,
    sample_mode: str,
) -> torch.Tensor:
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    videos = videos.to(device=vae_device, dtype=vae_dtype)
    latents = retrieve_latents(vae.encode(videos, return_dict=True), sample_mode)
    mean, std = latent_stats_tensors(vae, latents.device, latents.dtype)
    return (latents - mean) * std


@torch.no_grad()
def encode_first_frame_condition(
    pipe: Any,
    videos: torch.Tensor,
    latents_shape: torch.Size,
    width: int,
    height: int,
    num_frames: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    vae_device = next(pipe.vae.parameters()).device
    vae_dtype = next(pipe.vae.parameters()).dtype
    videos = videos.to(device=vae_device, dtype=vae_dtype)
    first = videos[:, :, 0].unsqueeze(2)
    if bool(pipe.config.expand_timesteps):
        condition_video = first
    else:
        zeros = videos.new_zeros(videos.shape[0], videos.shape[1], num_frames - 1, height, width)
        condition_video = torch.cat([first, zeros], dim=2)
    condition = retrieve_latents(pipe.vae.encode(condition_video.to(device=vae_device, dtype=vae_dtype), return_dict=True), "argmax")
    mean, std = latent_stats_tensors(pipe.vae, condition.device, condition.dtype)
    condition = ((condition - mean) * std).to(device=device, dtype=dtype)
    if bool(pipe.config.expand_timesteps):
        mask = torch.ones(
            latents_shape[0],
            1,
            latents_shape[2],
            latents_shape[3],
            latents_shape[4],
            dtype=dtype,
            device=device,
        )
        mask[:, :, 0] = 0
        return condition, mask
    return condition, None


@torch.no_grad()
def encode_prompts(pipe: Any, prompts: list[str], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    text_device = next(pipe.text_encoder.parameters()).device
    embeds, _ = pipe.encode_prompt(
        prompt=prompts,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=text_device,
        dtype=next(pipe.text_encoder.parameters()).dtype,
    )
    return embeds.to(device=device, dtype=dtype)


def sample_timesteps(
    scheduler: Any,
    batch_size: int,
    device: torch.device,
    weighting: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_index = len(scheduler.sigmas) - 1
    if weighting == "uniform":
        indices = torch.randint(0, max_index, (batch_size,), device=device)
    elif weighting == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, device=device))
        indices = (u * (max_index - 1)).long().clamp(0, max_index - 1)
    else:
        raise ValueError(f"Unknown timestep weighting: {weighting}")
    timesteps = scheduler.timesteps.to(device=device, dtype=torch.float32)[indices]
    sigmas = scheduler.sigmas.to(device=device, dtype=torch.float32)[indices]
    return indices, timesteps, sigmas


def expand_sigma(sigmas: torch.Tensor, ndim: int) -> torch.Tensor:
    while sigmas.ndim < ndim:
        sigmas = sigmas.unsqueeze(-1)
    return sigmas


def build_timestep_input(
    pipe: Any,
    timesteps: torch.Tensor,
    latents: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if bool(pipe.config.expand_timesteps):
        if mask is None:
            mask = torch.ones(latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4], device=latents.device)
        patch_size = pipe.transformer.config.patch_size if pipe.transformer is not None else pipe.transformer_2.config.patch_size
        step_inputs = []
        for i, t in enumerate(timesteps):
            token_ts = (mask[i, 0, :, :: patch_size[1], :: patch_size[2]] * t).flatten()
            step_inputs.append(token_ts)
        return torch.stack(step_inputs, dim=0)
    return timesteps


def temporal_loss_slices(loss_per_element: torch.Tensor) -> dict[str, float]:
    if loss_per_element.ndim < 5:
        return {}
    t = loss_per_element.shape[2]
    if t < 3:
        return {"loss_temporal_all": float(loss_per_element.mean().detach().cpu())}
    a = max(t // 3, 1)
    return {
        "loss_temporal_first": float(loss_per_element[:, :, :a].mean().detach().cpu()),
        "loss_temporal_middle": float(loss_per_element[:, :, a : 2 * a].mean().detach().cpu()),
        "loss_temporal_last": float(loss_per_element[:, :, 2 * a :].mean().detach().cpu()),
    }


def masked_loss_mean(loss_per_element: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return loss_per_element.mean()
    expanded = mask.float().expand_as(loss_per_element)
    return (loss_per_element * expanded).sum() / expanded.sum().clamp_min(1.0)


def grad_norm(parameters: list[nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.detach().float().norm(2).item()
        total += param_norm * param_norm
    return math.sqrt(total)


def current_gpu_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    return {
        "gpu_mem_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "gpu_mem_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "gpu_max_mem_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
    }


def split_rows(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    val_count = int(round(len(rows) * val_fraction))
    val_count = min(max(val_count, 1 if len(rows) > 1 and val_fraction > 0 else 0), len(rows) - 1 if len(rows) > 1 else 0)
    val_ids = set(order[:val_count])
    train = [row for i, row in enumerate(rows) if i not in val_ids]
    val = [row for i, row in enumerate(rows) if i in val_ids]
    return train, val


def save_checkpoint(
    output_dir: Path,
    step: int,
    pipe: Any,
    args: argparse.Namespace,
    wrapped_names: list[str],
) -> Path:
    ckpt_dir = output_dir / "checkpoints" / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {}
    if pipe.transformer is not None:
        state.update({f"transformer.{k}": v for k, v in lora_state_dict(pipe.transformer).items()})
    if getattr(pipe, "transformer_2", None) is not None:
        state.update({f"transformer_2.{k}": v for k, v in lora_state_dict(pipe.transformer_2).items()})
    torch.save(
        {
            "lora": state,
            "args": vars(args),
            "wrapped_names": wrapped_names,
            "step": step,
        },
        ckpt_dir / "lora.pt",
    )
    tensor_state = {k: v for k, v in state.items() if v.ndim > 0}
    if tensor_state:
        save_file(tensor_state, ckpt_dir / "lora.safetensors")
    if getattr(args, "training_mode", "lora") == "full" and getattr(args, "save_full_weights", False):
        if pipe.transformer is not None:
            pipe.transformer.save_pretrained(ckpt_dir / "transformer")
        if getattr(pipe, "transformer_2", None) is not None and getattr(args, "train_transformer_2", False):
            pipe.transformer_2.save_pretrained(ckpt_dir / "transformer_2")
    checkpoint_config = vars(args).copy()
    checkpoint_config.update(
        {
            "checkpoint_step": step,
            "checkpoint_dir": str(ckpt_dir),
            "checkpoint_path": str(ckpt_dir / "lora.pt"),
            "training_mode": getattr(args, "training_mode", "lora"),
            "wrapped_names": wrapped_names,
        }
    )
    (ckpt_dir / "checkpoint_config.json").write_text(json.dumps(checkpoint_config, indent=2), encoding="utf-8")
    (output_dir / "latest_checkpoint.txt").write_text(str(ckpt_dir / "lora.pt"), encoding="utf-8")
    return ckpt_dir


@torch.no_grad()
def generate_samples(
    pipe: Any,
    sample_rows: list[dict[str, Any]],
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_pipeline_eval(pipe)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    for row in sample_rows[: args.num_sample_prompts]:
        out_path = out_dir / f"{row['id']}.mp4"
        if args.task == "i2v":
            video, _ = load_video_for_training(row["video"], args.width, args.height, args.num_frames)
            first = tensor_to_uint8_video(video[:, :1])[0].numpy()
            image = Image.fromarray(first)
            result = pipe(
                image=image,
                prompt=row["prompt"],
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_type="pt",
            ).frames[0]
        else:
            result = pipe(
                prompt=row["prompt"],
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.sample_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_type="pt",
            ).frames[0]
        write_video(str(out_path), generated_video_to_thwc(result), fps=args.fps)


def write_vbench_commands(output_dir: Path, args: argparse.Namespace) -> None:
    dims = " ".join(VBENCH_I2V_DIMENSIONS)
    text = (
        "# Edit /path/to/VBench and the sample step as needed.\n"
        "python /path/to/VBench/evaluate_i2v.py \\\n"
        f"  --videos_path {output_dir.as_posix()}/samples/step_XXXXXX \\\n"
        f"  --output_path {output_dir.as_posix()}/vbench \\\n"
        f"  --dimension {dims} \\\n"
        "  --mode custom_input \\\n"
        f"  --ratio {args.width}-{args.height}\n"
    )
    (output_dir / "vbench_commands.txt").write_text(text, encoding="utf-8")


def summarize_quality(path: Path, output_dir: Path) -> None:
    rows = []
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    numeric: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                numeric[key].append(float(value))
    summary = {"count": len(rows), "metrics": {}}
    for key, values in numeric.items():
        arr = np.asarray(values, dtype=np.float64)
        summary["metrics"][key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p05": float(np.quantile(arr, 0.05)),
            "p50": float(np.quantile(arr, 0.50)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(arr.max()),
        }
    (output_dir / "dataset_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def load_pipeline(args: argparse.Namespace, device: torch.device) -> Any:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16 if args.mixed_precision == "fp16" else torch.float32
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    if args.vae_tiling:
        vae.enable_tiling()
    pipeline_cls = WanImageToVideoPipeline if args.task == "i2v" else WanPipeline
    pipe = pipeline_cls.from_pretrained(args.model_id, vae=vae, torch_dtype=dtype)
    vae_device = torch.device(args.vae_device or str(device))
    text_encoder_device = torch.device(args.text_encoder_device or str(device))
    transformer_device = device
    if args.low_vram:
        vae_device = torch.device(args.vae_device or "cpu")
        text_encoder_device = torch.device(args.text_encoder_device or "cpu")
    pipe.vae.to(device=vae_device, dtype=torch.float32)
    text_dtype = torch.float32 if text_encoder_device.type == "cpu" else dtype
    pipe.text_encoder.to(device=text_encoder_device, dtype=text_dtype)
    pipe.text_encoder.eval()
    pipe.vae.eval()
    if pipe.transformer is not None:
        pipe.transformer.to(device=transformer_device, dtype=dtype)
        pipe.transformer.train()
        if args.gradient_checkpointing and hasattr(pipe.transformer, "enable_gradient_checkpointing"):
            pipe.transformer.enable_gradient_checkpointing()
    if getattr(pipe, "transformer_2", None) is not None:
        pipe.transformer_2.to(device=transformer_device, dtype=dtype)
        pipe.transformer_2.train()
        if args.gradient_checkpointing and hasattr(pipe.transformer_2, "enable_gradient_checkpointing"):
            pipe.transformer_2.enable_gradient_checkpointing()
    return pipe


def freeze_non_lora(pipe: Any) -> None:
    for module_name in ["vae", "text_encoder", "image_encoder", "transformer", "transformer_2"]:
        module = getattr(pipe, module_name, None)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad_(False)


def prepare_trainable_modules(pipe: Any, args: argparse.Namespace) -> list[str]:
    mode = getattr(args, "training_mode", "lora")
    wrapped: list[str] = []
    if mode == "lora":
        freeze_non_lora(pipe)
        targets = tuple(x.strip() for x in args.lora_targets.split(",") if x.strip())
        if pipe.transformer is not None:
            wrapped.extend(
                [
                    f"transformer.{x}"
                    for x in inject_lora(pipe.transformer, args.lora_rank, args.lora_alpha, args.lora_dropout, targets)
                ]
            )
        if getattr(pipe, "transformer_2", None) is not None:
            wrapped.extend(
                [
                    f"transformer_2.{x}"
                    for x in inject_lora(pipe.transformer_2, args.lora_rank, args.lora_alpha, args.lora_dropout, targets)
                ]
            )
        return wrapped
    if mode == "full":
        freeze_non_lora(pipe)
        if pipe.transformer is not None:
            for p in pipe.transformer.parameters():
                p.requires_grad_(True)
            wrapped.append("transformer.full")
        if getattr(pipe, "transformer_2", None) is not None and getattr(args, "train_transformer_2", False):
            for p in pipe.transformer_2.parameters():
                p.requires_grad_(True)
            wrapped.append("transformer_2.full")
        return wrapped
    raise ValueError(f"Unknown training_mode: {mode}")


def run_validation(
    pipe: Any,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
) -> dict[str, float]:
    losses = []
    pipe.transformer.eval()
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        with torch.no_grad():
            videos = batch["videos"].to(device=device, dtype=torch.float32)
            latents = encode_videos_to_latents(pipe.vae, videos, args.vae_sample_mode).to(device=device, dtype=dtype)
            prompt_embeds = encode_prompts(pipe, batch["prompts"], dtype, device)
            _, timesteps, sigmas = sample_timesteps(pipe.scheduler, latents.shape[0], device, args.timestep_weighting)
            noise = torch.randn_like(latents)
            noisy = expand_sigma(sigmas, latents.ndim) * noise + (1 - expand_sigma(sigmas, latents.ndim)) * latents
            target = noise - latents
            condition_mask = None
            model_input = noisy
            if args.task == "i2v":
                condition, condition_mask = encode_first_frame_condition(
                    pipe, videos, latents.shape, args.width, args.height, args.num_frames, dtype, device
                )
                if bool(pipe.config.expand_timesteps):
                    model_input = (1 - condition_mask) * condition + condition_mask * noisy
                else:
                    model_input = torch.cat([noisy, condition], dim=1)
            timestep_input = build_timestep_input(pipe, timesteps, latents, condition_mask)
            pred = pipe.transformer(
                hidden_states=model_input.to(dtype),
                timestep=timestep_input,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            loss_el = F.mse_loss(pred.float(), target.float(), reduction="none")
            losses.append(float(masked_loss_mean(loss_el, condition_mask).cpu()))
    pipe.transformer.train()
    if not losses:
        return {"val_loss": float("nan")}
    return {
        "val_loss": float(np.mean(losses)),
        "val_loss_std": float(np.std(losses)),
        "val_batches": len(losses),
    }


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    args.width, args.height = parse_resolution(args.resolution)
    rows = read_manifest(Path(args.manifest))
    train_rows, val_rows = split_rows(rows, args.validation_fraction, args.seed)
    quality_path = output_dir / "dataset_quality.jsonl"
    quality_writer = JsonlWriter(quality_path)

    if args.preflight_only:
        dataset = WanVideoDataset(rows, args.width, args.height, args.num_frames, args.fps, quality_writer, cache_quality=False)
        for i in tqdm(range(len(dataset)), desc="preflight"):
            _ = dataset[i]
        quality_writer.close()
        summarize_quality(quality_path, output_dir)
        write_vbench_commands(output_dir, args)
        return

    device = torch.device(args.device if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Wan2.2 TI2V-5B training is expected to run on CUDA.")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    pipe = load_pipeline(args, device)
    wrapped = prepare_trainable_modules(pipe, args)
    params = trainable_parameters(pipe)
    if not params:
        raise RuntimeError("No trainable parameters were created. Check training_mode and LoRA targets.")
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.max_train_steps, 1), eta_min=args.learning_rate * 0.1)
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16 if args.mixed_precision == "fp16" else torch.float32

    train_quality_writer = quality_writer if args.num_workers == 0 else None
    train_dataset = WanVideoDataset(train_rows, args.width, args.height, args.num_frames, args.fps, train_quality_writer)
    val_dataset = WanVideoDataset(val_rows, args.width, args.height, args.num_frames, args.fps, None) if val_rows else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        collate_fn=collate_batch,
        drop_last=True,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=bool(args.pin_memory), collate_fn=collate_batch)
        if val_dataset is not None
        else None
    )

    metrics_writer = JsonlWriter(output_dir / "metrics.jsonl")
    validation_writer = JsonlWriter(output_dir / "validation_metrics.jsonl")
    metric_fields = [
        "step",
        "epoch",
        "loss",
        "loss_ema",
        "rmse",
        "mae",
        "lr",
        "grad_norm",
        "sigma_mean",
        "sigma_min",
        "sigma_max",
        "timestep_mean",
        "pred_mean",
        "pred_std",
        "target_mean",
        "target_std",
        "step_seconds",
        "data_seconds",
        "encode_seconds",
        "forward_seconds",
        "backward_seconds",
        "samples_per_second",
        "frames_per_second",
        "gpu_mem_allocated_gb",
        "gpu_mem_reserved_gb",
        "gpu_max_mem_allocated_gb",
        "loss_temporal_first",
        "loss_temporal_middle",
        "loss_temporal_last",
    ]
    csv_writer = CsvWriter(output_dir / "metrics.csv", metric_fields)
    write_vbench_commands(output_dir, args)

    config = vars(args).copy()
    config["train_rows"] = len(train_rows)
    config["val_rows"] = len(val_rows)
    config["wrapped_lora_modules"] = wrapped
    config["trainable_parameters"] = sum(p.numel() for p in params)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    manifest_snapshot = output_dir / "manifest_snapshot.jsonl"
    with manifest_snapshot.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")

    global_step = 0
    loss_ema = None
    start_time = time.perf_counter()
    recent_losses = deque(maxlen=100)
    optimizer.zero_grad(set_to_none=True)
    last_batch_end = time.perf_counter()
    pbar = tqdm(total=args.max_train_steps, desc="train")

    best_val_loss = float("inf")
    best_val_step = 0
    val_no_improve = 0
    stop_training = False

    epoch = 0
    while global_step < args.max_train_steps and not stop_training:
        epoch += 1
        for batch in train_loader:
            if global_step >= args.max_train_steps or stop_training:
                break
            if time.perf_counter() - start_time > args.max_train_seconds:
                global_step = args.max_train_steps
                break
            step_start = time.perf_counter()
            data_seconds = step_start - last_batch_end
            videos = batch["videos"].to(device=device, dtype=torch.float32, non_blocking=True)

            encode_start = time.perf_counter()
            with torch.no_grad():
                latents = encode_videos_to_latents(pipe.vae, videos, args.vae_sample_mode).to(device=device, dtype=dtype)
                videos = videos.to(device=device, dtype=torch.float32)
                prompt_embeds = encode_prompts(pipe, batch["prompts"], dtype, device)
                _, timesteps, sigmas = sample_timesteps(pipe.scheduler, latents.shape[0], device, args.timestep_weighting)
                noise = torch.randn_like(latents)
                sigma_expanded = expand_sigma(sigmas, latents.ndim).to(dtype)
                noisy = sigma_expanded * noise + (1 - sigma_expanded) * latents
                target = noise - latents
                condition_mask = None
                model_input = noisy
                if args.task == "i2v":
                    condition, condition_mask = encode_first_frame_condition(
                        pipe, videos, latents.shape, args.width, args.height, args.num_frames, dtype, device
                    )
                    if bool(pipe.config.expand_timesteps):
                        model_input = (1 - condition_mask) * condition + condition_mask * noisy
                    else:
                        model_input = torch.cat([noisy, condition], dim=1)
                timestep_input = build_timestep_input(pipe, timesteps, latents, condition_mask)
            encode_seconds = time.perf_counter() - encode_start

            forward_start = time.perf_counter()
            pred = pipe.transformer(
                hidden_states=model_input.to(dtype),
                timestep=timestep_input,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]
            loss_el = F.mse_loss(pred.float(), target.float(), reduction="none")
            loss = masked_loss_mean(loss_el, condition_mask)
            scaled_loss = loss / args.gradient_accumulation_steps
            forward_seconds = time.perf_counter() - forward_start

            backward_start = time.perf_counter()
            scaled_loss.backward()
            backward_seconds = time.perf_counter() - backward_start

            do_step = (global_step + 1) % args.gradient_accumulation_steps == 0
            norm = float("nan")
            if do_step:
                norm = grad_norm(params)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            loss_value = float(loss.detach().cpu())
            recent_losses.append(loss_value)
            loss_ema = loss_value if loss_ema is None else args.ema_decay * loss_ema + (1 - args.ema_decay) * loss_value
            step_seconds = time.perf_counter() - step_start
            row = {
                "step": global_step,
                "epoch": epoch,
                "loss": loss_value,
                "loss_ema": loss_ema,
                "loss_rolling_100": float(np.mean(recent_losses)),
                "rmse": float(torch.sqrt(loss.detach()).cpu()),
                "mae": float((pred.float() - target.float()).abs().mean().detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": norm,
                "sigma_mean": float(sigmas.mean().detach().cpu()),
                "sigma_min": float(sigmas.min().detach().cpu()),
                "sigma_max": float(sigmas.max().detach().cpu()),
                "timestep_mean": float(timesteps.mean().detach().cpu()),
                "pred_mean": float(pred.float().mean().detach().cpu()),
                "pred_std": float(pred.float().std().detach().cpu()),
                "target_mean": float(target.float().mean().detach().cpu()),
                "target_std": float(target.float().std().detach().cpu()),
                "step_seconds": step_seconds,
                "data_seconds": data_seconds,
                "encode_seconds": encode_seconds,
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "samples_per_second": float(videos.shape[0] / max(step_seconds, 1e-6)),
                "frames_per_second": float(videos.shape[0] * args.num_frames / max(step_seconds, 1e-6)),
                "batch_ids": batch["ids"],
                "batch_paths": batch["paths"],
            }
            temporal_loss_el = loss_el.detach()
            if condition_mask is not None:
                temporal_loss_el = temporal_loss_el * condition_mask.float()
            row.update(temporal_loss_slices(temporal_loss_el))
            row.update(current_gpu_metrics())
            metrics_writer.write(row)
            csv_writer.write(row)
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss_value:.4f}", ema=f"{loss_ema:.4f}")
            print(f"step={global_step:05d} epoch={epoch} loss={loss_value:.4f} ema={loss_ema:.4f} lr={row['lr']:.2e} grad={norm:.3f}", flush=True)

            if val_loader is not None and args.validation_every > 0 and global_step % args.validation_every == 0:
                val_row = {"step": global_step}
                val_row.update(run_validation(pipe, val_loader, args, device, dtype, args.max_validation_batches))
                validation_writer.write(val_row)
                val_loss = val_row.get("val_loss", float("nan"))
                if args.early_stop_patience > 0 and global_step >= args.early_stop_warmup_steps and not math.isnan(val_loss):
                    if val_loss < best_val_loss - args.early_stop_min_delta:
                        best_val_loss = val_loss
                        best_val_step = global_step
                        val_no_improve = 0
                    else:
                        val_no_improve += 1
                        print(f"[early_stop] no improvement {val_no_improve}/{args.early_stop_patience} (val={val_loss:.4f}, best={best_val_loss:.4f} @ step {best_val_step})", flush=True)
                        if val_no_improve >= args.early_stop_patience:
                            print(f"[early_stop] stopping at step {global_step}: best val_loss={best_val_loss:.4f} at step {best_val_step}", flush=True)
                            stop_training = True

            if args.checkpoint_every > 0 and global_step % args.checkpoint_every == 0:
                save_checkpoint(output_dir, global_step, pipe, args, wrapped)

            if args.sample_every > 0 and global_step % args.sample_every == 0:
                if args.low_vram:
                    raise RuntimeError("--sample_every is not supported with --low_vram; generate samples on the H100 profile.")
                sample_rows = val_rows if val_rows else train_rows
                generate_samples(pipe, sample_rows, output_dir / "samples" / f"step_{global_step:06d}", args, device)

            last_batch_end = time.perf_counter()

    save_checkpoint(output_dir, global_step, pipe, args, wrapped)
    quality_writer.close()
    metrics_writer.close()
    validation_writer.close()
    csv_writer.close()
    summarize_quality(quality_path, output_dir)
    pbar.close()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Wan2.2 TI2V-5B LoRA with dense metrics.")
    parser.add_argument("--manifest", required=True, help="JSONL with video, prompt, and optional id fields.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--training_mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--save_full_weights", action="store_true", help="Only relevant for --training_mode full; writes large transformer checkpoints.")
    parser.add_argument("--train_transformer_2", action="store_true", help="For two-transformer Wan variants, also train the low-noise transformer.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae_device", default=None, help="Defaults to --device, or cpu with --low_vram.")
    parser.add_argument("--text_encoder_device", default=None, help="Defaults to --device, or cpu with --low_vram.")
    parser.add_argument("--low_vram", action="store_true", help="Keep VAE and text encoder on CPU; useful for tiny 3060 smoke runs.")
    parser.add_argument("--task", choices=["i2v", "t2v"], default="i2v")
    parser.add_argument("--resolution", default="1280x704", type=str)
    parser.add_argument("--num_frames", default=121, type=int)
    parser.add_argument("--fps", default=24, type=int)
    parser.add_argument("--train_batch_size", default=1, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--max_train_steps", default=100000, type=int)
    parser.add_argument("--max_train_seconds", default=82800, type=float)
    parser.add_argument("--validation_fraction", default=0.05, type=float)
    parser.add_argument("--validation_every", default=250, type=int)
    parser.add_argument("--max_validation_batches", default=8, type=int)
    parser.add_argument("--checkpoint_every", default=1000, type=int)
    parser.add_argument("--sample_every", default=0, type=int)
    parser.add_argument("--num_sample_prompts", default=4, type=int)
    parser.add_argument("--sample_steps", default=30, type=int)
    parser.add_argument("--guidance_scale", default=5.0, type=float)
    parser.add_argument("--lora_rank", default=32, type=int)
    parser.add_argument("--lora_alpha", default=32.0, type=float)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--lora_targets", default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--mixed_precision", choices=["bf16", "fp16", "no"], default="bf16")
    parser.add_argument("--timestep_weighting", choices=["uniform", "logit_normal"], default="logit_normal")
    parser.add_argument("--vae_sample_mode", choices=["sample", "argmax"], default="argmax")
    parser.add_argument("--vae_tiling", action="store_true", default=True)
    parser.add_argument("--no_vae_tiling", dest="vae_tiling", action="store_false")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--pin_memory", action="store_true", default=False, help="Pin decoded CPU batches before GPU transfer. Leave off for very large video datasets unless RAM is abundant.")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--ema_decay", default=0.98, type=float)
    parser.add_argument("--early_stop_patience", default=0, type=int, help="Stop after this many consecutive validations with no val_loss improvement. 0 disables early stopping.")
    parser.add_argument("--early_stop_min_delta", default=0.005, type=float, help="Absolute val_loss decrease required to count as improvement.")
    parser.add_argument("--early_stop_warmup_steps", default=0, type=int, help="Disable early stop until global_step >= this value.")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
