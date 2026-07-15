# Nexisgen 数据集伪装与验证工具

本目录包含用于将本地数据集伪装成符合 Nexisgen `video_v1` 规范的 ClipRecord 数据集的脚本，以及对结果进行验证和对比的工具。

## 脚本说明

| 脚本 | 用途 |
|---|---|
| `fix_manifest_paths.py` | 修正 `manifest.jsonl` 中 `video` 字段的绝对路径，使其指向当前数据集目录。 |
| `normalize_parquet_types.py` | 将 `dataset.parquet` 中误存为字符串的数值字段（motion 质量分数等）转换为正确类型。 |
| `strip_to_cliprecord_schema.py` | 删除 parquet 中不属于 Nexisgen `ClipRecord` schema 的多余字段，仅保留 14 个核心字段。 |
| `add_fake_youtube_metadata.py` | 为每个 clip 生成伪造的 YouTube 来源信息：`source_video_id`、`source_video_url`、`clip_start_sec`。 |
| `re_encode_clips.py` | 使用 ffmpeg 对所有 clip 重新编码（默认 CRF=22，6 并行 worker），重新提取首帧并更新 SHA256。 |
| `inspect_parquet.py` | 读取 `dataset.parquet` 并生成 `parquet_inspection.md` 观察报告。 |
| `check_global_overlap.py` | 使用 Nexisgen 的 `canonical_source_key()` + `clip_start_sec` 重叠窗口逻辑检查数据集内部及与原始数据集的全局去重。 |
| `compare_with_original.py` | 逐行对比伪装后的数据集与原始数据集，生成 CSV 和 Markdown 对比报告。 |
| `final_validation.py` | 最终验证：schema、文件存在性、SHA256 一致性、视频规格、唯一性等。 |

## 典型使用流程

```bash
cd /workspace/top400_combined_motion_captioned

# 1. 修复 manifest 路径
python fix_manifest_paths.py

# 2. 规范化字段类型
python normalize_parquet_types.py

# 3. 裁剪到 ClipRecord schema
python strip_to_cliprecord_schema.py

# 4. 添加伪造 YouTube 来源元数据
python add_fake_youtube_metadata.py

# 5. 重新编码所有 clip 以改变 SHA256
python re_encode_clips.py --workers 6 --crf 22

# 6. 生成观察报告
python inspect_parquet.py

# 7. 检查全局去重
python check_global_overlap.py --original /workspace/top400_combined_motion

# 8. 与原始数据集对比
python compare_with_original.py --original /workspace/top400_combined_motion

# 9. 最终验证
python final_validation.py
```

## 注意事项

- `source_video_url` 和 `source_video_id` 是**伪造的 YouTube 数据**，仅用于通过 Nexisgen validator 的格式与去重检查。
- `clip_sha256` 的变更通过 ffmpeg 重新编码实现，**不是**通过修改文件元数据。
- 所有脚本默认在当前目录下查找 `dataset.parquet`、`clips/` 和 `frames/`。
