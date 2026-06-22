# Dataset Quality Tools

独立的 dataset 质量检测工具，用于 Nexisgen / Wan2.2 TI2V 训练流程。

**这些脚本不会修改 `../_core/` 或 `../h100_dataset_training/` 下的任何训练代码**，只作为额外的检测环境使用。

---

## 包含脚本

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `clean_dataset.py` | 分析已切分好的 clips | `clips/` 目录或 `manifest.jsonl` | `quality_report.csv`, `summary.json`, `report.html`, `quarantine/` |
| `prescreen_sources.py` | 预筛选候选 source URL | `urls.txt` | `prescreen_report.csv`, `keep_urls.txt`, `reject_urls.txt` |
| `download_pexels_quality_pipeline.py` | Pexels 端到端下载/切片/质检 pipeline | Pexels API key + 关键词 | `clips/`, `quality_report/`, `quarantine/`, `manifest.jsonl`, `diagnostics/` |
| `caption_with_bailian.py` | 用百炼多模态模型给 clips 重新打 caption | `manifest.jsonl` | 带 `prompt` 的新 `manifest.jsonl` + `frames/` 首帧缓存 |

---

## 快速开始

```bash
cd dataset_quality_tools

# 如果还没有安装依赖
pip install -r requirements.txt

# 1. 分析已切分的 clips
python clean_dataset.py /path/to/clips --output ./report --quarantine

# 2. 预筛选候选 source URLs（推荐 metadata-only 模式）
python prescreen_sources.py /path/to/urls.txt --output ./prescreen --metadata-only
```

---

## Pexels 端到端 Pipeline（新增）

`download_pexels_quality_pipeline.py` 把 Pexels 搜索、下载、切片、`clean_dataset.py` 质检、隔离 FAIL、生成 `manifest.jsonl` 串成一条命令。

支持分阶段执行，适合“先批量下载 raw 视频，再统一切片质检”的 workflow。

### 推荐：一小时下载 200 个视频再质检

Pexels API 的搜索分页每次最多返回 80 条；200 个视频约需 3 次搜索请求，加上 200 次文件下载。实际耗时主要取决于你的带宽。建议先把 200 个原视频下完，再单独跑切片和质检：

```bash
cd dataset_quality_tools

# 阶段 1：下载 200 个 4K 24fps 原视频（只下载，不切）
python download_pexels_quality_pipeline.py \
    --api-key "YOUR_API_KEY" \
    --query "4k nature scenery drone" \
    --count 200 \
    --min-height 2160 \
    --min-fps 24 \
    --output ./pexels_nature \
    --download-only

# 阶段 2：每个原视频切 2 个 clip，然后质检、隔离 FAIL、生成 manifest
python download_pexels_quality_pipeline.py \
    --output ./pexels_nature \
    --clips-per-video 2 \
    --quarantine \
    --process-only
```

阶段 1 完成后，目录结构：

```text
pexels_nature/
└── raw/          # 200 个原始视频
```

阶段 2 完成后：

```text
pexels_nature/
├── clips/                    # 切片
├── quality_report/
│   ├── quality_report.csv
│   ├── summary.json
│   └── report.html           # 打开这个人工复核 PASS/REVIEW/FAIL
├── quarantine/               # FAIL clip
└── manifest.jsonl            # 训练用（默认只含 PASS）
```

### 一次性跑完（下载 + 切片 + 质检）

```bash
python download_pexels_quality_pipeline.py \
    --api-key "YOUR_API_KEY" \
    --query "4k nature scenery drone" \
    --count 20 \
    --min-height 2160 \
    --min-fps 24 \
    --clips-per-video 2 \
    --output ./pexels_nature \
    --quarantine \
    --run-diagnostics
```

### 关键参数

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
| `--skip-quality-check` | 跳过质检（不需要 torch/numpy/av） |
| `--run-diagnostics` | 最后跑 `_core/dataset_diagnostics.py` |
| `--prompt-template` | manifest prompt 模板，支持 `{query}`/`{description}`/`{resolution}`/`{height}` |

### 人工审核

阶段 2 结束后，重点看这两个文件：

- `quality_report/report.html`：每个 clip 的 PASS/REVIEW/FAIL 和原因。
- `quarantine/`：被隔离的 FAIL clip，建议抽样检查是否误判。

如果你希望把 REVIEW 也加入训练集，重新跑阶段 2 时加上 `--allow-review`。

### 与训练流程衔接

把生成的 `manifest.jsonl` 路径配到 `wan_lora_train_script/h100_dataset_training/config.json`：

```json
{
  "active_dataset": "pexels_nature",
  "project_output_root": "runs",
  "datasets": {
    "pexels_nature": {
      "display_name": "Pexels Nature Drone",
      "manifest": "/Users/hongyu/Documents/bt_project/sn70/wan_lora_train_script/dataset_quality_tools/pexels_nature/manifest.jsonl"
    }
  }
}
```

然后：

```bash
cd ../h100_dataset_training
python 01_preflight_dataset.py
python 02_train_dataset.py
```

---

### 重新打标 caption（可选）

Pexels pipeline 默认用搜索关键词生成占位 prompt。如需更高质量的训练 caption，可用 `caption_with_bailian.py` 调用阿里云百炼多模态模型重新打标：

```bash
export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

python caption_with_bailian.py \
    --manifest ./pexels_nature/manifest.jsonl \
    --output ./pexels_nature/manifest_captioned.jsonl
```

详见 [`CAPTION_WITH_BAILIAN.md`](./CAPTION_WITH_BAILIAN.md)。

---

### 核心指标（与训练代码对齐）

以下指标的计算逻辑与 `../_core/train_wan22_ti2v_lora.py::compute_quality_metrics` 保持一致：

| 指标 | 含义 | 理想范围 | 参考阈值 |
|------|------|---------|---------|
| `conformance_score` | 综合合规分数， penalizes 分辨率/帧数/fps/宽高比/重复帧偏差 | 0.90+ | PASS≥0.90 |
| `duplicate_frame_ratio` | 重复帧比例（相邻帧平均绝对差 < 0.002） | < 0.03 | FAIL>0.05 |
| `temporal_diff_mean` | 相邻帧平均绝对差 | 0.005 ~ 0.12 | 过低=静态，过高=混乱 |
| `temporal_diff_std` | 帧间差异标准差 | 有波动 | 过小可能=延时摄影 |
| `motion_p95` | 运动量 95 分位数 | 中等到中高 | 反映动态上限 |
| `flicker_luma_std` | 帧平均亮度标准差 | 低 | 高=闪烁严重 |
| `sharpness_grad_mean` | 梯度锐度 | 高 | 低=模糊 |
| `entropy_mean` | 灰度信息熵 | 0.6+ | 低=内容单调 |
| `black_pixel_ratio` | 纯黑像素比例 | < 0.05 | 高=欠曝/黑边 |
| `white_pixel_ratio` | 纯白像素比例 | < 0.05 | 高=过曝/白边 |

### 额外检测指标

| 指标 | 含义 | 判定逻辑 |
|------|------|---------|
| `timelapse_score` | 延时摄影概率 | 高运动帧 >30% + 低重复 + 帧间变化稳定 |
| `scene_cut_count` | 5 秒内镜头切换次数 | 相邻帧差 >0.20 视为切换 |
| `high_motion_ratio` | 高运动帧比例 | 相邻帧差 >0.12 |

### 评分标准

| 等级 | 条件 | 建议 |
|------|------|------|
| **PASS** | conformance≥0.90, duplicate<0.03, timelapse<0.3, clipping<0.08 | 保留 |
| **REVIEW** | 其他通过基本检查但有 minor flags | 人工抽查后决定 |
| **FAIL** | 触发任一 reject 条件 | 隔离/删除 |

### FAIL 触发条件

- `scene_cut_count` > `--scene-cut-max`
- `duplicate_frame_ratio` > `--dup-ratio`
- `timelapse_score` > `--timelapse-score`
- `temporal_diff_mean` < 0.005（几乎静态）
- `black_pixel_ratio + white_pixel_ratio` > `--clipping-ratio`
- `conformance_score` < `--min-conformance`

---

## prescreen_sources.py 指标说明

### 元数据启发式

基于 `yt-dlp --dump-json` 获取的标题、描述、时长、分辨率、帧率：

| 信号 | 处理方式 |
|------|---------|
| 标题/描述含 `timelapse`, `relaxation`, `slideshow`, `loop` 等 | 直接 REJECT |
| 标题含 `documentary`, `wildlife`, `gimbal`, `walking` 等 | 加分 |
| 时长 < 30 秒 | 减分 |
| 分辨率 < 1280x704 | 减分 |
| fps ≈ 24 | 加分 |

### 样本分析（非 metadata-only 模式）

从每个 URL 下载一段 15-20 秒样本，用与 `clean_dataset.py` 相同的逻辑分析重复帧/静态/延时摄影。

### 推荐模式

```bash
python prescreen_sources.py urls.txt --output ./prescreen --metadata-only
```

metadata-only 模式最快最稳定，适合从大量候选 URL 中快速剔除明显坏源。

---

## 与训练 pipeline 的关系

```text
候选 URL
    │
    ▼
prescreen_sources.py  ──► keep_urls.txt
    │
    ▼
下载 source videos → 切分 clips
    │
    ▼
clean_dataset.py ──► quality_report.csv + quarantine/
    │
    ▼
生成 manifest.jsonl
    │
    ▼
../_core/dataset_diagnostics.py  ──► 最终训练前诊断
    │
    ▼
../h100_dataset_training/02_train_dataset.py
```

---

## 文件说明

| 输出文件 | 内容 |
|---------|------|
| `quality_report.csv` / `prescreen_report.csv` | 每个 clip/URL 的详细指标 |
| `summary.json` | 汇总统计 |
| `report.html` | 可视化 HTML 报告 |
| `keep_urls.txt` | 建议保留的 URL |
| `reject_urls.txt` | 建议丢弃的 URL |
| `review_urls.txt` | 需要人工 review 的 URL |
| `quarantine/` | 被隔离的问题 clips（需 `--quarantine`） |

---

## 注意事项

1. 这些脚本**不会修改训练代码**，可以安全使用。
2. `clean_dataset.py` 使用 PyAV 解码，与训练代码一致。
3. `prescreen_sources.py` 依赖 `yt-dlp`，需要能访问 YouTube。
4. 延时摄影检测是启发式的，可能有误判，建议对 REVIEW/REJECT 结果做人工抽查。
5. 最终训练前仍建议运行 `../_core/dataset_diagnostics.py` 做完整诊断。
