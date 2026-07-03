# 运动风光素材批量获取工作流（Motion-Scenery Acquisition）

针对 Wan2.2 TI2V LoRA 的核心短板 **dynamic_degree 塌缩**，本工作流批量下载"带真实运动的自然风光"，并用 **两段漏斗** 保证留下来的片既动态达标、又不混乱、画面干净。

## 文件

| 文件 | 作用 |
|------|------|
| `motion_scenery_keywords.py` | 运动风光关键词矩阵（19 query / 8 类，含奇观 ~17%）|
| `run_vbench_dataset.py` | 采集引擎：超量下载→Tier-A 质检→Tier-B CLIP 代理→Top-K（新增 `--keywords-module`）|
| `motion_aesthetic_gate.py` | **最后的 RAFT + 美学精筛**（第二段漏斗）|
| `pexels_download_history.json` | 全局去重（已含 3041 ID，不会重下）|

## 两段漏斗

```
Pexels ──(超采 2.5×)──▶ Tier-A 质检 ──▶ Tier-B CLIP代理 Top-K ──▶ 【RAFT+美学 gate】──▶ 最终训练片
            mean_delta 粗筛            composite 排序          motion_mean∈[33,80]
         [0.010, 0.20] 挡静态/混乱                              且 aesthetic≥0.58
```

第一段（引擎内）用 **mean_delta** 做廉价粗筛；mean_delta 与 RAFT 仅弱相关（r≈0.18），所以**必须**有第二段用 **VBench 精确 RAFT 光流** 复筛——这是 dynamic_degree 不再塌的关键。

## 运行

```bash
cd dataset_quality_tools
source /venv/main/bin/activate

# 0)（可选）先小批量验证出片率：只跑前 3 个关键词、每个 target 4
python run_vbench_dataset.py \
    --keywords-module motion_scenery_keywords \
    --output ./motion_scenery_raw \
    --log-dir ./motion_scenery_logs \
    --limit-keywords 3 --target-override 4

# 1) 正式采集（19 关键词，预筛前目标 ~198 选中片）
python run_vbench_dataset.py \
    --keywords-module motion_scenery_keywords \
    --output ./motion_scenery_raw \
    --log-dir ./motion_scenery_logs

# 2) 最后一步：RAFT + 美学精筛 → 最终训练片 + manifest
python motion_aesthetic_gate.py \
    --dataset-root ./motion_scenery_raw \
    --output ./motion_scenery_final \
    --require-dynamic --motion-hi 80 --aesthetic-floor 0.55
```

> **gate 动态门槛用 VBench 口径**（`--require-dynamic`：motion>16.5 即 dynamic=1），不是 strong 桶的 33。
> 原因见下"阈值校准"。`--motion-hi 80` 仍挡过载混乱，`--aesthetic-floor 0.55` 挡画面差。

产物：
- `motion_scenery_final/clips/`        —— 过了精筛的最终训练片
- `motion_scenery_final/final_manifest.jsonl` —— `{id, video, prompt}`（prompt 沿用各关键词的语义）
- `motion_scenery_final/gate_scores.csv`  —— 每个选中片的 motion_mean / aesthetic / 是否保留 / 淘汰原因

## 数量与调参

- 目标：净增 **~100 条** 过精筛的优质片
- 预筛前 Top-K 目标 198（运动类 `_M=10`/query，奇观 `_W=12`/query）
- **精筛留存率（实测 smoke test）：~33%**（dynamic=1 且 aes≥0.55 口径）→ 预筛 ~198 选中 → 最终 ~65 条；要净增 ~100 需把 `_M`/`_W` 提到约 1.5×，或多跑几个关键词
- **要更多/更少**：改 `motion_scenery_keywords.py` 顶部的 `_M` / `_W`
- **出片太少**：把 gate 的 `--aesthetic-floor` 降到 0.52 重跑（gate 用缓存分数秒出，不必重下）

## 阈值校准（smoke test 的关键发现）

`motion_mean≥33` 是你现有数据集 **strong 桶（top 19% 精华）** 的门槛，**不适合做原始下载的准入标准**——随机下载的平缓河流/海浪航拍 motion 普遍 11–33。改用 **VBench 自己的二值 dynamic 判定**（`--require-dynamic`，motion>16.5）+ 美学下限，留存才健康（实测 0/12 → 4/12）。
道理：VBench 的 dynamic_degree 本就是二值聚合，训练片只要 dynamic=1 就对该维度有正贡献，不需要个个都是 strong。

## 关于奇观/地貌（geo_wonders 类）

- 4 个 query，全部"带运动"写法（云移/浪打/风扫/雾飘 + drone orbit），且照样过 RAFT gate，静态明信片会被自动毙
- 角色是**美学 + 内容多样性**，不是动态来源；出片率天然偏低，属正常
- 想扩：在 `geo_wonders` 段继续加（棉花堡梯田流水、巨人之路海浪、火山口蒸汽、张掖丹霞云影…），保持"运动动词"

## 合并进现有数据集

精筛产物是独立目录，**不动**现有 `merged_dataset_nexisgen`。确认质量后，用现成的 `merge_nexisgen_datasets.py` / `build_nexisgen_merged_dataset.py` 合并，并重算 manifest 行数 + parquet sha256。
