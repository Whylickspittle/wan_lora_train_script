# 多帧视频 Caption 使用指南

本文档介绍如何使用 `video_captioner.py` 和 `caption_batch.py` 为视频 clip 生成适合 text-to-video / image-to-video 训练的 prompt。

相比传统的单帧 caption，多帧 caption 会抽取 4 张等间距关键帧发给视觉模型，从而获得：

- **相机运动**信息（如 drone sweeps across、glides forward）
- **首帧外的场景变化**（如远处湖泊、城市景观、徒步者）
- 更准确的**运动方向**描述（upward、across、forward）

---

## 目录

- [环境准备](#环境准备)
- [配置文件](#配置文件)
- [单视频测试](#单视频测试)
- [批量处理](#批量处理)
- [输出格式](#输出格式)
- [常见问题](#常见问题)
- [单帧 vs 多帧效果对比](#单帧-vs-多帧效果对比)

---

## 环境准备

```bash
cd dataset_quality_tools

# 安装依赖（如果还没有安装）
pip install -r requirements.txt
```

确保系统已安装 `ffmpeg`：

```bash
ffmpeg -version
```

---

## 配置文件

复制示例配置：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
# 必填：API key
OPENAI_API_KEY=sk-your-key-here

# 可选：自定义 OpenAI 兼容接口地址，用于 DMX API 等中转平台
OPENAI_BASE_URL=https://www.dmxapi.cn/v1

# 可选：模型名称
NEXIS_CAPTION_MODEL=gpt-4o-mini

# 可选：关键帧数量，1=单帧，4=多帧（默认）
NEXIS_CAPTION_KEYFRAMES=4
```

---

## 单视频测试

### 多帧 caption（默认 4 帧）

```bash
python video_captioner.py /path/to/clip.mp4
```

### 单帧 caption（用于对比）

```bash
python video_captioner.py /path/to/clip.mp4 --keyframes 1
```

### 指定工作目录

```bash
python video_captioner.py /path/to/clip.mp4 --workdir ./my_workdir
```

---

## 批量处理

对整个 `clips/` 目录生成单帧/多帧对比：

```bash
python caption_batch.py \
    --input_dir /path/to/clips \
    --output captions.jsonl \
    --num_keyframes 4 \
    --workdir ./caption_workdir
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--input_dir` | 输入 clips 目录 |
| `--output` | 输出 JSONL 文件路径 |
| `--num_keyframes` | 多帧模式的关键帧数量 |
| `--workdir` | 临时抽帧目录 |

---

## 输出格式

`caption_batch.py` 输出的 JSONL 每行格式如下：

```json
{
  "video": "/path/to/clip_28543569_node0.11.mp4",
  "single_frame_caption": "Aerial view captures a river flowing beside steep, red rock cliffs under bright sunlight.",
  "multi_frame_caption": "A drone soars upward over a rugged canyon, revealing hikers exploring the cliff's edge against a backdrop of vibrant red rock formations.",
  "num_keyframes": 4
}
```

你可以用 `jq` 查看：

```bash
jq -r '.multi_frame_caption' captions.jsonl | head
```

---

## 常见问题

### Q1: 为什么多帧 caption 比单帧长？

多帧模型能看到时间变化，所以会补充相机运动和场景展开的描述。这是正常的，也是多帧的价值所在。

### Q2: 可以只生成多帧 caption 吗？

可以。`video_captioner.py` 默认就是多帧。`caption_batch.py` 输出里虽然有单帧对比，但你完全可以只用 `multi_frame_caption` 字段。

### Q3: 成本如何？

多帧 caption 约是单帧的 3-4 倍 token 成本（因为发了 4 张图）。以 GPT-4o-mini 为例，一个 400 clip 的 interval 大约多花费 $0.20-$0.50。

### Q4: 可以用 Gemini / Claude 吗？

可以。只要 API 是 OpenAI 兼容格式，修改 `.env` 里的模型名称和 base_url 即可。

推荐配置：

```bash
# Gemini via OpenAI-compatible endpoint
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
NEXIS_CAPTION_MODEL=gemini-1.5-flash
```

### Q5: 抽帧目录可以删除吗？

可以。`--workdir` 下的文件只是临时关键帧，处理完后可以删除。

---

## 单帧 vs 多帧效果对比

### 示例 1：峡谷航拍

| 模式 | Caption |
|------|---------|
| 单帧 | Aerial view captures a river flowing beside steep, red rock cliffs under bright sunlight. |
| 多帧 | A drone soars upward over a rugged canyon, revealing hikers exploring the cliff's edge against a backdrop of vibrant red rock formations. |

**差异**：多帧捕捉到无人机上升运动和徒步者。

### 示例 2：熔岩原航拍

| 模式 | Caption |
|------|---------|
| 单帧 | Aerial view captures lush greenery surrounding rugged volcanic rock formations under a cloudy sky. |
| 多帧 | A drone sweeps across a lush, green lava field, revealing rugged rock formations and a distant shimmering lake under a cloudy sky. |

**差异**：多帧写出无人机横扫运动，并发现远处湖泊。

### 示例 3：瀑布

| 模式 | Caption |
|------|---------|
| 单帧 | Water cascades down cliffs, creating mist and a vibrant rainbow over the rushing river below. |
| 多帧 | A drone sweeps across a vibrant waterfall, capturing cascading water and a rainbow arching over the rushing river below. |

**差异**：多帧补充了相机横扫运动和彩虹横跨河流的画面感。

---

## 接入训练流程

生成的 caption 可以直接写入 trainer manifest 的 `prompt` 字段：

```json
{"video": "clip.mp4", "image": "frame.jpg", "prompt": "A drone sweeps across a vibrant waterfall..."}
```

对于 Nexisgen 训练流程，你可以用 `caption_batch.py` 的输出替换 `manifest.jsonl` 里的 prompt 字段。

---

## 参考

- `video_captioner.py`：核心 caption 模块
- `caption_batch.py`：批量处理脚本
- `.env.example`：配置模板
