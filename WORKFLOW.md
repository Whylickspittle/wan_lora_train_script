# Wan2.2 TI2V LoRA 训练全流程文档

本文档覆盖从数据获取、质检、训练到推理评估的完整 workflow。

---

## 1. 仓库结构概览

```text
wan_lora_train_script/
├── README.md                              # 仓库说明
├── requirements.txt                       # Python 依赖
├── _core/                                 # 核心训练/推理/诊断代码
│   ├── train_wan22_ti2v_lora.py          # LoRA 训练
│   ├── infer_wan22_quality.py            # 推理
│   ├── dataset_diagnostics.py            # 数据集预检
│   └── config_utils.py                   # 配置解析工具
├── dataset_quality_tools/                 # 数据获取与质检工具
│   ├── README.md
│   ├── download_pexels_quality_pipeline.py   # Pexels 端到端 pipeline
│   ├── clean_dataset.py                  # clip 质量分析
│   ├── prescreen_sources.py              # URL 预筛选
│   ├── generate_quality_review.py        # 生成质检审查清单
│   ├── package_pass_dataset.py           # 打包 PASS clip
│   └── extract_pass_clips.py             # 提取 PASS clip
└── h100_dataset_training/                 # 训练流程入口
    ├── config.example.json               # 配置模板
    ├── customer_workflow.py              # 工作流编排
    ├── quality_report.py                 # 训练后质量报告
    ├── 00_download_model.py              # 下载基础模型
    ├── 01_preflight_dataset.py           # 数据集预检
    ├── 02_train_dataset.py               # 训练
    ├── 03_infer_latest_checkpoint.py     # 推理
    ├── 04_make_quality_report.py         # 生成训练报告
    ├── 05_eval_with_images.py            # 指定图片评估
    └── serve.py                          # 本地推理服务
```

---

## 2. 前置依赖

### 2.1 系统工具

- `ffmpeg`
- `ffprobe`
- `yt-dlp`（`prescreen_sources.py` 样本模式需要）

```bash
# macOS
brew install ffmpeg yt-dlp

# Ubuntu / Debian
sudo apt-get install ffmpeg
pip install yt-dlp
```

### 2.2 Python 依赖

```bash
cd wan_lora_train_script
pip install -r requirements.txt
```

关键包：

- torch>=2.4.0
- diffusers>=0.36.0
- transformers>=4.45.0
- accelerate, huggingface_hub
- datasets
- av (PyAV)
- safetensors
- numpy, pillow, matplotlib
- tqdm, ftfy, regex, sentencepiece, protobuf

---

## 3. 数据准备

### 3.1 方案 A：Pexels 端到端 Pipeline（推荐）

**脚本**：`dataset_quality_tools/download_pexels_quality_pipeline.py`

该脚本完成搜索、下载、切片、`clean_dataset.py` 质检、隔离 FAIL、生成 `manifest.jsonl`。

#### 两阶段 workflow（推荐用于大批量）

```bash
cd dataset_quality_tools

# 阶段 1：下载 200 个 4K 24fps 原视频
python download_pexels_quality_pipeline.py \
    --api-key "YOUR_PEXELS_API_KEY" \
    --query "4k nature scenery drone" \
    --count 200 \
    --min-height 2160 \
    --min-fps 24 \
    --output ./pexels_nature \
    --download-only

# 阶段 2：切片 + 质检 + 隔离 FAIL + 生成 manifest
python download_pexels_quality_pipeline.py \
    --output ./pexels_nature \
    --clips-per-video 2 \
    --quarantine \
    --process-only
```

阶段 2 输出：

```text
pexels_nature/
├── clips/                    # 5.04s 切片（1280x704, 24fps, 121 frames）
├── quality_report/
│   ├── quality_report.csv
│   ├── summary.json
│   └── report.html           # 人工复核 PASS/REVIEW/FAIL
├── quarantine/               # FAIL clip
└── manifest.jsonl            # 训练用 manifest（默认只含 PASS）
```

#### 一次性 workflow

```bash
python download_pexels_quality_pipeline.py \
    --api-key "YOUR_PEXELS_API_KEY" \
    --query "4k nature scenery drone" \
    --count 20 \
    --min-height 2160 \
    --min-fps 24 \
    --clips-per-video 2 \
    --output ./pexels_nature \
    --quarantine \
    --run-diagnostics
```

#### 关键参数

| 参数 | 说明 |
|------|------|
| `--api-key` | Pexels API key（`--process-only` 时可省略） |
| `--query` | 搜索关键词 |
| `--count` | 下载视频数量 |
| `--min-height` | 最小高度，2160=4K |
| `--min-fps` | 最小帧率 |
| `--clips-per-video` | 每个视频切几个 5.04s clip |
| `--download-only` | 只下载原视频 |
| `--process-only` | 只切片/质检已有 `raw/` |
| `--quarantine` | 隔离 FAIL clip |
| `--allow-review` | REVIEW clip 也写入 manifest |
| `--skip-quality-check` | 跳过质检 |
| `--run-diagnostics` | 最后跑 `_core/dataset_diagnostics.py` |
| `--prompt-template` | manifest prompt 模板，支持 `{query}` / `{description}` / `{resolution}` / `{height}` |

#### 质检阈值

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--dup-ratio` | 0.05 | 重复帧比例 FAIL 阈值 |
| `--timelapse-score` | 0.50 | 延时摄影分数上限 |
| `--clipping-ratio` | 0.15 | 纯黑+纯白像素比例上限 |
| `--min-conformance` | 0.50 | 最低合规分数 |
| `--scene-cut-max` | 0 | 每个 clip 最大镜头切换次数 |

---

### 3.2 方案 B：手动数据集 / HuggingFace

**脚本**：`h100_dataset_training/00_prepare_huggingface_dataset.py`

从 HuggingFace 数据集下载视频，写入本地 `manifest.jsonl` 并更新 `config.json`。

---

### 3.3 方案 C：外部来源预筛选

**脚本**：`dataset_quality_tools/prescreen_sources.py`

对 YouTube / Vimeo 等 URL 列表做元数据启发式预筛：

```bash
python prescreen_sources.py urls.txt --output ./prescreen --metadata-only
```

输出：

- `prescreen_report.csv`
- `keep_urls.txt`
- `reject_urls.txt`
- `review_urls.txt`

---

### 3.4 方案 D：已有 clips 质检

**脚本**：`dataset_quality_tools/clean_dataset.py`

如果已经有切好的 clips，直接分析：

```bash
python clean_dataset.py ./clips --output ./report --quarantine
```

核心指标（与训练代码对齐）：

- `conformance_score`：综合合规分数
- `duplicate_frame_ratio`：重复帧比例
- `temporal_diff_mean/std`：帧间差异
- `motion_p95`：运动量 95 分位数
- `flicker_luma_std`：亮度闪烁
- `sharpness_grad_mean`：梯度锐度
- `entropy_mean`：信息熵
- `black_pixel_ratio` / `white_pixel_ratio`：过曝/欠曝
- `timelapse_score`：延时摄影概率
- `scene_cut_count`：镜头切换次数

评分等级：

- **PASS**：conformance ≥ 0.90，重复帧 < 0.03，timelapse < 0.30，clipping < 0.08
- **REVIEW**：minor flags
- **FAIL**：触发任一 reject 条件

---

## 4. 训练配置

### 4.1 创建 config.json

从模板复制：

```bash
cd h100_dataset_training
cp config.example.json config.json
```

关键字段：

```json
{
  "active_dataset": "pexels_nature",
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
    "pexels_nature": {
      "display_name": "Pexels Nature Drone",
      "manifest": "/Users/hongyu/Documents/bt_project/sn70/wan_lora_train_script/dataset_quality_tools/pexels_nature/manifest.jsonl"
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
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lora_rank": 32,
    "lora_alpha": 32.0,
    "lora_dropout": 0.0,
    "lora_targets": "to_q,to_k,to_v,to_out.0",
    "validation_fraction": 0.05,
    "validation_every": 50,
    "max_validation_batches": 8,
    "checkpoint_every": 50,
    "sample_every": 0,
    "sample_steps": 30,
    "num_sample_prompts": 4,
    "guidance_scale": 5.0,
    "mixed_precision": "bf16",
    "timestep_weighting": "logit_normal",
    "vae_sample_mode": "argmax",
    "vae_tiling": true,
    "gradient_checkpointing": true,
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

关键说明：

- `active_dataset` 必须对应 `datasets` 中的 key
- `manifest` 路径可以是绝对路径，也可以是相对 manifest 文件所在目录的相对路径
- `train_batch_size` 建议保持 1
- `num_workers` 建议保持 0
- `vae_tiling` 和 `gradient_checkpointing` 建议开启以节省显存

---

## 5. 核心训练流程

### 5.1 下载基础模型

```bash
cd h100_dataset_training
python 00_download_model.py
```

从 HuggingFace 下载 `Wan-AI/Wan2.2-TI2V-5B-Diffusers` 到 `models/Wan2.2-TI2V-5B-Diffusers/`，并更新 `config.json` 中的 `model_id` 为本地路径。

---

### 5.2 数据集预检

```bash
python 01_preflight_dataset.py
```

调用 `_core/dataset_diagnostics.py::diagnose_dataset()` 检查：

- `manifest.jsonl` 存在且 JSON 格式正确
- 每行包含 `id`、`video`、`prompt`
- 视频文件存在且可解码
- 分辨率是否匹配（`allow_resize=true` 时只警告）
- 帧数是否 ≥ 121
- 帧率是否接近 24fps
- 重复帧比例是否过高
- ID 是否唯一
- 显存占用估算

输出：

- `runs/<dataset>/diagnostics/diagnostics_report.html`
- `runs/<dataset>/diagnostics/diagnostics_summary.json`（含 `ok_to_train`）
- `runs/<dataset>/diagnostics/diagnostics.jsonl`
- `runs/<dataset>/diagnostics/video_inventory.csv`

如果 `ok_to_train=false` 且 `allow_training_with_errors=false`，训练会拒绝启动。

---

### 5.3 训练

```bash
python 02_train_dataset.py
```

训练架构：

- 模型：Wan2.2-TI2V-5B-Diffusers（Diffusers pipeline）
- 任务：I2V（首帧条件生成视频）
- 训练模式：LoRA（默认）或 full fine-tune
- LoRA 目标层：`to_q, to_k, to_v, to_out.0`
- 精度：bf16
- 优化器：AdamW + cosine annealing LR
- 损失：MSE
- 默认梯度累积：8 步（等效 batch size = 8）
- VAE tiling 与 gradient checkpointing 默认开启

训练特性：

- 首次会计算训练集质量指标
- 每 N 步验证（默认 5% 验证集）
- 早停（基于验证 loss）
- 每 N 步保存 checkpoint
- 可选每 N 步生成 sample
- 记录 loss、EMA、RMSE、MAE、LR、grad norm、GPU 显存、throughput 等

输出：

- `runs/<dataset>/checkpoints/step_000050/lora.pt`
- `runs/<dataset>/checkpoints/step_000050/lora.safetensors`
- `runs/<dataset>/checkpoints/step_000050/checkpoint_config.json`
- `runs/<dataset>/latest_checkpoint.txt`
- `runs/<dataset>/metrics.jsonl` / `metrics.csv`
- `runs/<dataset>/validation_metrics.jsonl`
- `runs/<dataset>/dataset_quality.jsonl`
- `runs/<dataset>/run_config.json`
- `runs/<dataset>/manifest_snapshot.jsonl`

---

### 5.4 训练集推理

```bash
python 03_infer_latest_checkpoint.py
```

加载最新 LoRA checkpoint，对训练 manifest 采样做推理，评估首帧 PSNR、时序动态、黑帧比例等。

输出：

- `runs/<dataset>/inference/step_000050/samples/<id>.mp4`
- `runs/<dataset>/inference/step_000050/inference_metrics.jsonl`

---

### 5.5 训练质量报告

```bash
python 04_make_quality_report.py
```

生成 HTML 报告，包含训练 loss 曲线、验证 loss、系统性能、数据集质量概览、推理质量概览和总体 PASS/REVIEW/FAIL 等级。

输出：

- `runs/<dataset>/report/report.html`

---

### 5.6 指定图片评估

```bash
python 05_eval_with_images.py \
    --eval_manifest /path/to/eval_manifest.jsonl \
    --output_dir /path/to/eval_outputs \
    [--checkpoint runs/pexels_nature/checkpoints/step_002000/lora.pt]
```

eval manifest 格式：

```jsonl
{"id": "eval_001", "image": "/path/to/first_frame.jpg", "prompt": "A drone flying over a mountain lake"}
```

---

### 5.7 本地推理服务（可选）

```bash
python serve.py
```

启动 FastAPI 服务，提供 `/generate` 接口：上传首帧图片 + prompt，返回 MP4。

---

## 6. 辅助质检脚本

### 6.1 生成质检审查清单

**脚本**：`dataset_quality_tools/generate_quality_review.py`

根据 `clean_dataset.py` 的 `quality_report.csv` 生成结构化的问题 clip 审查清单，便于人工复核。

### 6.2 打包 PASS 数据集

**脚本**：`dataset_quality_tools/package_pass_dataset.py`

把 PASS / REVIEW clip 复制到输出目录，生成干净的 manifest。

### 6.3 提取 PASS clip

**脚本**：`dataset_quality_tools/extract_pass_clips.py`

从质检报告中提取 PASS clip 列表，用于快速生成训练集。

---

## 7. 常见问题与注意事项

1. **必须在 `h100_dataset_training/` 目录下运行训练脚本**

   脚本通过 `Path(__file__).resolve().parents[1]` 定位 `_core/`。

2. **没有默认 `config.json`**

   必须从 `config.example.json` 复制并修改。

3. **显存安全**

   - 保持 `train_batch_size=1`
   - 保持 `num_workers=0`
   - 保持 `pin_memory=false`
   - 开启 `vae_tiling` 和 `gradient_checkpointing`
   - 测试时可降低 `num_frames` 到 81 或 65

4. **Preflight 阻塞**

   如果诊断发现错误，训练会拒绝启动，除非 `allow_training_with_errors=true`。

5. **Pexels 下载速率**

   - Pexels API 搜索分页每次最多 80 条
   - 200 个视频大约需要 3 次搜索请求
   - 下载耗时主要取决于带宽，建议 `--download-only` 先下完再处理

6. **Prompt 质量**

   Pexels pipeline 默认从搜索关键词生成占位 prompt。高质量训练建议后续用 VLM/LLM 重新打标。

7. **manifest 格式**

   每行一个 JSON 对象，必须包含：

   ```json
   {"id": "sample_001", "video": "clips/clip.mp4", "prompt": "描述主体、运动、镜头、场景、光线的文字"}
   ```

8. **路径约定**

   `video` 路径可以是绝对路径，也可以是相对 `manifest.jsonl` 所在目录的相对路径。

---

## 8. 完整流程图

```text
Pexels API
    │
    ▼
dataset_quality_tools/download_pexels_quality_pipeline.py
    │
    ├── raw/                      # 阶段 1：原视频
    │
    ▼
clips/                            # 阶段 2：切片
    │
    ▼
clean_dataset.py （pipeline 内置）
    │
    ├── quality_report/           # CSV / JSON / HTML 报告
    ├── quarantine/               # FAIL clip
    │
    ▼
manifest.jsonl                    # PASS clip
    │
    ▼
h100_dataset_training/config.json
    │
    ▼
01_preflight_dataset.py           # 数据集预检
    │
    ▼
02_train_dataset.py               # LoRA 训练
    │
    ▼
03_infer_latest_checkpoint.py     # 训练集推理
    │
    ▼
04_make_quality_report.py         # 生成训练报告
    │
    ▼
05_eval_with_images.py            # 指定图片评估
```

---

## 9. 文件路径速查

| 用途 | 路径 |
|------|------|
| 训练配置模板 | `h100_dataset_training/config.example.json` |
| 工作流编排 | `h100_dataset_training/customer_workflow.py` |
| 核心训练 | `_core/train_wan22_ti2v_lora.py` |
| 核心推理 | `_core/infer_wan22_quality.py` |
| 核心诊断 | `_core/dataset_diagnostics.py` |
| Pexels pipeline | `dataset_quality_tools/download_pexels_quality_pipeline.py` |
| clip 质检 | `dataset_quality_tools/clean_dataset.py` |
| URL 预筛选 | `dataset_quality_tools/prescreen_sources.py` |
| 下载模型 | `h100_dataset_training/00_download_model.py` |
| 预检 | `h100_dataset_training/01_preflight_dataset.py` |
| 训练 | `h100_dataset_training/02_train_dataset.py` |
| 推理 | `h100_dataset_training/03_infer_latest_checkpoint.py` |
| 报告 | `h100_dataset_training/04_make_quality_report.py` |
| 评估 | `h100_dataset_training/05_eval_with_images.py` |
| 服务 | `h100_dataset_training/serve.py` |
