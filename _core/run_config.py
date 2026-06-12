from __future__ import annotations

from pathlib import Path

from config_utils import load_namespace
from train_wan22_ti2v_lora import build_argparser, train


def run_training_config(config_path: str | Path) -> None:
    defaults = build_argparser().parse_args(
        [
            "--manifest",
            "__CONFIG_REQUIRED__",
            "--output_dir",
            "__CONFIG_REQUIRED__",
        ]
    )
    args = load_namespace(config_path, defaults)
    if args.manifest == "__CONFIG_REQUIRED__" or args.output_dir == "__CONFIG_REQUIRED__":
        raise ValueError(f"{config_path} must define manifest and output_dir.")
    train(args)
