# Wan2.2 TI2V LoRA 训练代码（提取版）开发文档

本目录是从 `rendixnetwork/train:latest` 镜像中提取出来的独立训练代码，用于在无法使用 Docker-in-Docker 的环境（如 RunPod 标准 PyTorch Pod）里直接跑 LoRA 训练测试。

> 原始镜像里的代码路径：`/workspace/training/Wan2.2DatasetAnalsis/h100_dataset_training/`  
> 核心训练库路径：`/workspace/training/Wan2.2DatasetAnalsis/_core/`

---

## 1. 目录结构

```
/workspace/train_code/
├── README.md                       # 本文件
├── requirements.txt                # Python 依赖
├── _core/                          # 核心训练库（不要改路径）
│   ├── train_wan22_ti2v_lora.py    # 主训练逻辑
│   ├── infer_wan22_quality.py      # 推理 + LoRA 加载
│   ├── dataset_diagnostics.py      # 数据集预飞诊断
│   ├── config_utils.py             # 配置解析工具
│   └── ...
└── h100_dataset_training/          # 入口脚本
    ├── config.json                 # ← 你需要创建/修改这个文件
    ├── 00_download_model.py        # 下载 Wan2.2 模型
    ├── 01_preflight_dataset.py     # 数据集预飞检查
    ├── 02_train_dataset.py         # 开始训练
    ├── 03_infer_latest_checkpoint.py
    ├── 04_make_quality_report.py
    ├── 05_eval_with_images.py      # 用 eval 数据生成视频
    └── customer_workflow.py        # 脚本之间的流程编排
```

**重要**：所有脚本都必须在 `h100_dataset_training/` 目录下执行，因为代码用 `Path(__file__).resolve().parents[1]` 定位 `_core/`。

---

## 2. 环境准备

### 2.1 基础环境要求

- NVIDIA GPU，建议 H100 / A100（训练 1280×704×121 需要大显存）
- CUDA 12.x + cuDNN
- Python 3.10+
- 磁盘空间：模型约 20 GB，数据集和输出另外算

### 2.2 安装依赖

```bash
cd /workspace/train_code/h100_dataset_training
pip install -r ../requirements.txt
```

如果已经装了 PyTorch 但 CUDA 版本不匹配，请先按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 装对应 CUDA 版本的 torch。

---

## 3. 快速开始：最小 LoRA 测试流程

### 3.1 准备数据集

每条训练样本需要一个视频文件和一个文本 prompt。创建一个 `manifest.jsonl`：

```bash
mkdir -p /workspace/my_dataset/clips
cat > /workspace/my_dataset/manifest.jsonl << 'EOF'
{"id": "sample_001", "video": "/workspace/my_dataset/clips/001.mp4", "prompt": "A drone flying over a mountain lake at sunrise"}
{"id": "sample_002", "video": "/workspace/my_dataset/clips/002.mp4", "prompt": "A cat chasing a laser pointer on a wooden floor"}
EOF
```

视频要求：

- 分辨率：**1280×704**（与 config.json 中的 `resolution` 一致）
- 帧数：至少 **121 帧**
- 帧率：**24 fps**
- 格式：mp4（H.264 编码最稳）

> 如果视频尺寸/帧数不完全匹配，训练 loader 会 resize 和采样，但预飞检查会报 WARN。

### 3.2 创建 config.json

在 `h100_dataset_training/` 下创建 `config.json`。最小可运行模板：

```json
{
  "active_dataset": "dataset_a",
  "project_output_root": "runs",
  "model_id": "models/Wan2.2-TI2V-5B-Diffusers",
  "model_download": {
    "source_model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "local_dir": "models/Wan2.2-TI2V-5B-Diffusers",
    "revision": "main",
    "use_local_model_after_download": true
  },
  "device": "cuda:0",
  "diagnostics": {
    "run_before_training": true,
    "allow_resize": true,
    "allow_training_with_errors": false
  },
  "datasets": {
    "dataset_a": {
      "display_name": "my_first_test",
      "manifest": "/workspace/my_dataset/manifest.jsonl",
      "notes": "LoRA smoke test"
    }
  },
  "training": {
    "mode": "lora",
    "resolution": "1280x704",
    "num_frames": 121,
    "fps": 24,
    "max_train_seconds": 82800,
    "max_train_steps": 200,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 0.0001,
    "lora_rank": 32,
    "lora_alpha": 32.0,
    "validation_every": 50,
    "checkpoint_every": 50,
    "sample_every": 0,
    "sample_steps": 30,
    "num_sample_prompts": 4,
    "num_workers": 0,
    "pin_memory": false,
    "seed": 1234
  },
  "inference": {
    "sample_steps": 30,
    "guidance_scale": 5.0,
    "max_samples": 4,
    "seed": 1234
  }
}
```

### 3.3 下载基础模型

```bash
cd /workspace/train_code/h100_dataset_training
python 00_download_model.py
```

这会从 HuggingFace 下载 `Wan-AI/Wan2.2-TI2V-5B-Diffusers` 到 `models/Wan2.2-TI2V-5B-Diffusers/`，并自动把 `config.json` 里的 `model_id` 改为本地路径。

> 如果访问 HuggingFace 慢，可设置镜像或提前把模型文件放到 `models/Wan2.2-TI2V-5B-Diffusers/`。

### 3.4 数据集预飞检查

```bash
python 01_preflight_dataset.py
```

输出：

- `runs/dataset_a/diagnostics/diagnostics_report.html` — 可视化诊断报告
- `runs/dataset_a/diagnostics/diagnostics_summary.json` — 是否 `ok_to_train`

如果 `ok_to_train` 为 `false` 且 `allow_training_with_errors` 为 `false`，训练会拒绝启动。

### 3.5 开始训练

```bash
python 02_train_dataset.py
```

训练输出：

- `runs/dataset_a/checkpoints/step_000050/lora.pt` — LoRA checkpoint
- `runs/dataset_a/latest_checkpoint.txt` — 最新 checkpoint 指针
- `runs/dataset_a/metrics.jsonl` / `metrics.csv` — 训练指标
- `runs/dataset_a/dataset_quality.jsonl` — 数据集质量指标
- `runs/dataset_a/run_config.json` — 本次运行的完整配置

### 3.6 推理测试

```bash
python 03_infer_latest_checkpoint.py
```

会用训练好的 LoRA 对训练 manifest 里的样本做推理，输出到 `runs/dataset_a/inference/`。

### 3.7 生成质量报告

```bash
python 04_make_quality_report.py
```

生成 `runs/dataset_a/report/report.html`，包含 loss 曲线、数据集质量评分等。

---

## 4. config.json 关键字段说明

| 字段 | 说明 |
|------|------|
| `active_dataset` | 当前使用的 dataset key，必须存在于 `datasets` 中 |
| `model_id` | 模型路径，首次运行会被 `00_download_model.py` 覆盖为本地路径 |
| `datasets.<name>.manifest` | 训练 manifest.jsonl 绝对路径 |
| `training.mode` | `lora` 或 `full`（full 会训练整个 transformer，显存爆炸风险） |
| `training.resolution` | 训练分辨率，如 `1280x704` |
| `training.num_frames` | 每段视频采样帧数，默认 121 |
| `training.max_train_steps` | 总训练步数 |
| `training.train_batch_size` | 强烈建议保持 1 |
| `training.gradient_accumulation_steps` | 梯度累积步数，实际 batch = 1 × 该值 |
| `training.lora_rank` / `lora_alpha` | LoRA 参数 |
| `training.lora_targets` | 默认 `to_q,to_k,to_v,to_out.0` |
| `diagnostics.allow_resize` | 是否允许 loader resize 视频 |
| `diagnostics.allow_training_with_errors` | true 时即使预飞报错也继续训练 |

---

## 5. 用独立 eval manifest 跑评估

`05_eval_with_images.py` 通常由 validator 在容器内调用，也可以手动跑：

```bash
# 准备一个 eval manifest，每行包含 image 和 prompt
cat > /workspace/my_dataset/eval_manifest.jsonl << 'EOF'
{"id": "eval_001", "image": "/workspace/my_dataset/frames/001_first_frame.jpg", "prompt": "A drone flying over a mountain lake at sunrise"}
EOF

python 05_eval_with_images.py \
  --eval_manifest /workspace/my_dataset/eval_manifest.jsonl \
  --output_dir /workspace/my_dataset/eval_outputs
```

---

## 6. 常见问题

### 6.1 `ModuleNotFoundError: No module named 'train_wan22_ti2v_lora'`

确认你是在 `h100_dataset_training/` 目录下执行脚本，而不是在 `/workspace/train_code/` 根目录。

### 6.2 `config.json` 找不到

本代码不自带 `config.json`，必须自己创建。可参考第 3.2 节的模板。

### 6.3 显存 OOM

- 保持 `train_batch_size=1`
- 保持 `num_workers=0`
- 降低 `num_frames`（如 81 或 65）做 smoke test
- 开启 `vae_tiling: true`（已在模板中开启）
- 降低 `resolution`（如 `960x544`）

### 6.4 预飞检查报 `WIDTH_MISMATCH` / `FRAME_COUNT_MISMATCH`

如果视频确实没对齐但想先跑通，把 `diagnostics.allow_resize` 设为 `true`；如果是严重错误（如视频不存在），需要修复路径。

### 6.5 下载模型慢 / 失败

```bash
# 设置 HF 镜像（可选）
export HF_ENDPOINT=https://hf-mirror.com
python 00_download_model.py
```

---

## 7. 验证代码是否能导入

不跑训练，只做语法/导入检查：

```bash
cd /workspace/train_code/h100_dataset_training
python -c "import sys; sys.path.insert(0, '../_core'); from train_wan22_ti2v_lora import build_argparser; print('OK')"
```

---

## 8. 与 Validator 的关系

Validator 不会直接跑这些 `.py` 文件，而是启动 Docker 容器，在容器内：

1. 把 `miner_dir` 挂载到 `/workspace/training/<hotkey>/`
2. 把 validator 的 `config.json` 挂载到 `/workspace/training/Wan2.2DatasetAnalsis/h100_dataset_training/config.json`
3. 执行 `python 02_train_dataset.py && python 05_eval_with_images.py ...`

本提取版保留了完全一致的目录结构和导入逻辑，方便你在容器外复现和调试。
