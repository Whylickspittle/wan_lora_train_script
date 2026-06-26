from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_core"))

from config_utils import namespace_from_config
from dataset_diagnostics import diagnose_dataset
from infer_wan22_quality import run_inference_config
from train_wan22_ti2v_lora import build_argparser, train


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def load_customer_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    active = config["active_dataset"]
    if active not in config["datasets"]:
        raise ValueError(f"active_dataset '{active}' is not listed in datasets.")
    return config


def active_dataset(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    dataset_id = config["active_dataset"]
    return dataset_id, config["datasets"][dataset_id]


def run_dir(config: dict[str, Any], dataset_id: str) -> Path:
    return Path(config["project_output_root"]) / slug(dataset_id)


def base_training_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset_id, dataset = active_dataset(config)
    training = config.get("training", {})
    return {
        "manifest": dataset["manifest"],
        "output_dir": str(run_dir(config, dataset_id)),
        "model_id": config.get("model_id", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
        "training_mode": training.get("mode", "lora"),
        "device": config.get("device", "cuda:0"),
        "task": "i2v",
        "resolution": training.get("resolution", "1280x704"),
        "num_frames": int(training.get("num_frames", 121)),
        "fps": int(training.get("fps", 24)),
        "max_train_seconds": float(training.get("max_train_seconds", 82800)),
        "max_train_steps": int(training.get("max_train_steps", 100000)),
        "train_batch_size": int(training.get("train_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(training.get("learning_rate", 0.0001)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "lora_rank": int(training.get("lora_rank", 32)),
        "lora_alpha": float(training.get("lora_alpha", 32.0)),
        "lora_dropout": float(training.get("lora_dropout", 0.0)),
        "lora_targets": training.get("lora_targets", "to_q,to_k,to_v,to_out.0"),
        "validation_fraction": float(training.get("validation_fraction", 0.05)),
        "validation_every": int(training.get("validation_every", 250)),
        "max_validation_batches": int(training.get("max_validation_batches", 8)),
        "checkpoint_every": int(training.get("checkpoint_every", 1000)),
        "sample_every": int(training.get("sample_every", 1000)),
        "sample_steps": int(training.get("sample_steps", 30)),
        "num_sample_prompts": int(training.get("num_sample_prompts", 4)),
        "guidance_scale": float(training.get("guidance_scale", 5.0)),
        "mixed_precision": training.get("mixed_precision", "bf16"),
        "timestep_weighting": training.get("timestep_weighting", "logit_normal"),
        "vae_sample_mode": training.get("vae_sample_mode", "argmax"),
        "vae_tiling": bool(training.get("vae_tiling", True)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "num_workers": int(training.get("num_workers", 0)),
        "pin_memory": bool(training.get("pin_memory", False)),
        "seed": int(training.get("seed", 1234)),
        "early_stop_patience": int(training.get("early_stop_patience", 0)),
        "early_stop_min_delta": float(training.get("early_stop_min_delta", 0.005)),
        "early_stop_warmup_steps": int(training.get("early_stop_warmup_steps", 0)),
    }


def train_from_customer_config(preflight_only: bool = False) -> None:
    config = load_customer_config()
    train_config = base_training_config(config)
    train_config["preflight_only"] = preflight_only
    defaults = build_argparser().parse_args(["--manifest", "__CONFIG_REQUIRED__", "--output_dir", "__CONFIG_REQUIRED__"])
    args = namespace_from_config(train_config, defaults)
    print(f"Dataset: {config['active_dataset']}")
    print(f"Manifest: {train_config['manifest']}")
    print(f"Output: {train_config['output_dir']}")
    print("Mode: preflight" if preflight_only else f"Mode: {train_config['training_mode']} training")
    diagnostics = config.get("diagnostics", {})
    if preflight_only or diagnostics.get("run_before_training", True):
        diag_dir = Path(train_config["output_dir"]) / "diagnostics"
        diag_summary = diagnose_dataset(
            train_config["manifest"],
            diag_dir,
            train_config["resolution"],
            int(train_config["num_frames"]),
            float(train_config["fps"]),
            allow_resize=bool(diagnostics.get("allow_resize", True)),
            train_batch_size=int(train_config["train_batch_size"]),
            num_workers=int(train_config["num_workers"]),
            pin_memory=bool(train_config["pin_memory"]),
        )
        print(f"Diagnostics report: {diag_dir / 'diagnostics_report.html'}")
        if not diag_summary["ok_to_train"] and not bool(diagnostics.get("allow_training_with_errors", False)):
            raise RuntimeError(
                "Dataset diagnostics found blocking errors. Open diagnostics_report.html, fix the dataset, then rerun."
            )
    train(args)


def latest_checkpoint(output_dir: Path) -> Path:
    latest_file = output_dir / "latest_checkpoint.txt"
    if latest_file.exists():
        path = Path(latest_file.read_text(encoding="utf-8").strip())
        if path.exists():
            return path
    candidates = sorted((output_dir / "checkpoints").glob("step_*/lora.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {output_dir / 'checkpoints'}")
    return candidates[-1]


def inference_from_customer_config() -> None:
    config = load_customer_config()
    dataset_id, dataset = active_dataset(config)
    output = run_dir(config, dataset_id)
    ckpt = latest_checkpoint(output)
    checkpoint_config = ckpt.parent / "checkpoint_config.json"
    if checkpoint_config.exists():
        base = json.loads(checkpoint_config.read_text(encoding="utf-8"))
    else:
        base = base_training_config(config)
    inference = config.get("inference", {})
    infer_config = {
        "manifest": base.get("manifest", dataset["manifest"]),
        "output_dir": str(output / "inference" / ckpt.parent.name),
        "model_id": base.get("model_id", config.get("model_id")),
        "checkpoint_path": str(ckpt),
        "device": config.get("device", base.get("device", "cuda:0")),
        "task": "i2v",
        "resolution": base.get("resolution", "1280x704"),
        "num_frames": int(base.get("num_frames", 121)),
        "fps": int(base.get("fps", 24)),
        "sample_steps": int(inference.get("sample_steps", base.get("sample_steps", 30))),
        "guidance_scale": float(inference.get("guidance_scale", base.get("guidance_scale", 5.0))),
        "max_samples": int(inference.get("max_samples", 8)),
        "mixed_precision": base.get("mixed_precision", "bf16"),
        "vae_tiling": bool(base.get("vae_tiling", True)),
        "seed": int(inference.get("seed", base.get("seed", 1234))),
        "lora_rank": int(base.get("lora_rank", 32)),
        "lora_alpha": float(base.get("lora_alpha", 32.0)),
        "lora_dropout": float(base.get("lora_dropout", 0.0)),
        "lora_targets": base.get("lora_targets", "to_q,to_k,to_v,to_out.0"),
    }
    temp_config = output / "inference" / "latest_inference_config.json"
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    temp_config.write_text(json.dumps(infer_config, indent=2), encoding="utf-8")
    print(f"Using checkpoint: {ckpt}")
    print(f"Inference output: {infer_config['output_dir']}")
    run_inference_config(temp_config)
