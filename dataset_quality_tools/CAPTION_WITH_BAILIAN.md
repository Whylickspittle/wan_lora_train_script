# Bailian/DashScope Caption Annotator

`caption_with_bailian.py` 是一个独立的 caption 标注脚本，为已有的视频 clip manifest 生成训练用的 `prompt`。

它复用了 `nexisgen` 中 miner captioner 的实现思路：

* 抽取每个 clip 的**首帧**；
* 通过阿里云**百炼（DashScope）**的 OpenAI-compatible API 调用多模态模型；
* 将模型返回的一句话描述写入 manifest 的 `prompt` 字段。

> 目前只传首帧，与 `nexisgen/nexis/miner/captioner.py` 的行为保持一致。

---

## 前置依赖

* Python 3.10+
* `ffmpeg`（用于抽取首帧）
* `openai>=1.40.0`

安装 Python 依赖：

```bash
cd /workspace/wan_lora_train_script
pip install -r requirements.txt
```

或者单独安装：

```bash
pip install openai>=1.40.0
```

---

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `DASHSCOPE_API_KEY` | 是（或 `--api-key`） | 阿里云 DashScope API key |
| `NEXIS_CAPTION_MODEL` | 否 | 默认 `qwen3.5-omni-plus` |
| `NEXIS_CAPTION_TIMEOUT_SEC` | 否 | 默认 `30` |

脚本会自动读取当前目录或脚本所在目录下的 `.env` 文件。你也可以在执行前手动 `export`：

```bash
export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx
```

每行一个 JSON 对象，至少包含 `video` 字段：

```jsonl
{"id": "clip_001", "video": "clips/clip_001.mp4"}
{"id": "clip_002", "video": "clips/clip_002.mp4"}
```

`video` 可以是：

* 相对路径（相对于 manifest 所在目录，或 `--clips-dir` 指定的目录）
* 绝对路径

---

## 输出

脚本会生成一个新的 `manifest.jsonl`，每行在原有字段基础上加入 `prompt`：

```jsonl
{"id": "clip_001", "video": "clips/clip_001.mp4", "prompt": "A drone flying over a calm mountain lake at sunrise."}
```

首帧图片会缓存在 `frames/` 目录下，方便重跑或检查。

---

## 使用示例

### 1. 最基本的用法

```bash
cd /workspace/wan_lora_train_script/dataset_quality_tools

export DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxx

python caption_with_bailian.py \
    --manifest ./pexels_nature/manifest.jsonl \
    --output ./pexels_nature/manifest_captioned.jsonl
```

### 2. 指定 clips 目录

如果 `video` 字段是相对路径且 clips 不在 manifest 同级目录：

```bash
python caption_with_bailian.py \
    --manifest ./pexels_nature/manifest.jsonl \
    --clips-dir ./pexels_nature/clips \
    --output ./pexels_nature/manifest_captioned.jsonl
```

### 3. 直接覆盖原 manifest

```bash
python caption_with_bailian.py \
    --manifest ./pexels_nature/manifest.jsonl \
    --update-in-place
```

### 4. 调整并发和模型

```bash
python caption_with_bailian.py \
    --manifest ./pexels_nature/manifest.jsonl \
    --model qwen-vl-max \
    --max-workers 2 \
    --sleep 0.5 \
    --output ./pexels_nature/manifest_captioned.jsonl
```

* `--max-workers`：并发请求数，默认为 4。
* `--sleep`：每个 worker 在两次请求之间休息的秒数，用于控制 QPS。

---

## 参数说明

| 参数 | 说明 |
|------|------|
| `--manifest` | 输入 manifest.jsonl 路径 |
| `--output` | 输出 manifest.jsonl 路径（默认：`<manifest>.captioned.jsonl`） |
| `--clips-dir` | clips 所在目录（默认：manifest 所在目录） |
| `--frames-dir` | 首帧缓存目录（默认：`<manifest_dir>/frames`） |
| `--api-key` | DashScope API key |
| `--model` | 模型名，默认 `qwen3.5-omni-plus` |
| `--base-url` | OpenAI-compatible endpoint，默认 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `--timeout` | 单次 API 调用超时，默认 30 秒 |
| `--max-workers` | 并发数，默认 4 |
| `--sleep` | 请求间隔（秒），默认 0.2 |
| `--update-in-place` | 直接覆盖输入 manifest |

---

## 模型名说明

脚本默认使用 `qwen3.5-omni-plus`。如果你在百炼控制台看到的实际模型 ID 不同，请通过 `--model` 或 `NEXIS_CAPTION_MODEL` 环境变量指定正确的名称。

常见可选模型（以控制台实际名称为准）：

* `qwen-vl-max`
* `qwen-vl-plus`
* `qwen2.5-vl-72b-instruct`
* `qwen-omni-turbo`

---

## 与 Pexels Pipeline 配合使用

```bash
# 1. 下载并切片（生成占位 prompt）
python download_pexels_quality_pipeline.py \
    --output ./pexels_nature \
    --process-only \
    --clips-per-video 2

# 2. 用 Bailian 重新打标
python caption_with_bailian.py \
    --manifest ./pexels_nature/manifest.jsonl \
    --output ./pexels_nature/manifest_captioned.jsonl \
    --update-in-place

# 3. 将 manifest_captioned.jsonl 配到训练 config.json 中
```

---

## 故障排查

### `caption call failed ...`

* 检查 `DASHSCOPE_API_KEY` 是否有效；
* 检查 `--model` 是否为百炼控制台中实际存在的模型 ID；
* 检查网络是否能访问 `dashscope.aliyuncs.com`。

### `ffmpeg is required but not found`

安装 ffmpeg：

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### 生成的 caption 是空的

* 百炼 compatible-mode 对图片 base64 格式有要求，脚本已使用 `data:image/jpeg;base64,...`；
* 如果问题持续，尝试换 `--model qwen-vl-max` 测试。
