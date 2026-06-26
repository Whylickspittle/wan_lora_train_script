# TODO — VBench 数据集提升（2026-06-26）

> 配套需求文档：[`VBENCH_DATASET_REQUIREMENTS.md`](./VBENCH_DATASET_REQUIREMENTS.md)
> 核心矛盾：模型生成视频塌缩成静态（dynamic_degree 0.075），因训练集运动太弱。
> 主线：**按运动强度（motion_mean≥33）重筛 + 补采强运动素材**，美学保持 ≥0.55。

---

## P0 — 立即可做（不下载，只在现有素材上）

- [ ] **导出强运动子集清单**：从 `merged_dataset_nexisgen/motion_strength.csv` 选 `motion_mean≥33`（76 个）
      和 `≥24.8`（132 个）两版，各生成一个 manifest，供"重筛后训练集"使用。
- [ ] **join 美学+运动总表**：把 `vbench_aes_dyn.csv` 与 `motion_strength.csv` 按 clip 合并，
      输出一张含 `aesthetic_quality, motion_mean, dynamic` 的总表，按"强运动+高美学"排序。
- [ ] **量化现有可用量**：同时满足 `motion_mean≥33` 且 `aesthetic_quality≥0.55` 的有多少个？
      （预判：远不足一个训练集，需补采。）

## P1 — 升级采集门槛

- [ ] **改 `collect_vbench_nature.py` 的过滤标准**：从"美学0.55 + dynamic=1"升级为
      **"美学≥0.55 + motion_mean≥33"**（接入 `score_motion_strength.py`）。
- [ ] 采集脚本统一加 **一源一片去重**（复用 `dedup_review80.py` 逻辑）。
- [ ] **强运动题材关键词**：在 `vbench_keywords.py` 加/换为偏强运动的词
      （crashing waves / rushing river / waterfall / FPV drone / running animals / fast tracking shot），
      少用 calm/serene/slow/aerial drift。

## P2 — 补采强运动素材

- [ ] 用升级后的脚本补采，目标量待定（建议先试水每类 ~15 强运动 unique，再放量）。
- [ ] 监控产出率：强运动门槛高，预计超采倍数要大、RAFT 打分慢（~26s/clip），后台跑 + 断点续跑。
- [ ] 采集后人工抽查：确认 motion_mean 高的片确实是"好动"而非镜头抖动/快切（快切应已被 Tier-A scene_cut 挡掉）。

## P3 — 组装与回训验证

- [ ] 用"强运动+高美学+去重"的最终子集，经 `build_nexisgen_merged_dataset.py` 打包成 nexisgen 数据集。
- [ ] caption：用 `caption_with_bailian.py`（已含 VBench 对齐模板）按真实首帧重打标，**显式描述运动/动作**。
- [ ] **回训一轮**，再对生成视频跑 VBench，**重点看 dynamic_degree 是否从 0.075 显著回升**。
- [ ] 若仍偏静：考虑训练侧（更多 step、运动权重、降低 I2V 首帧条件强度），但数据是主杠杆。

## 待决策（需用户拍板）

- [ ] 运动地板定 `motion_mean≥33`（强，量少）还是 `≥24.8`（中等，量多一倍）？
- [ ] 现有 400 个里的弱/擦边片（179+88）是**删除**、**移走存档**、还是**保留作低权重样本**？
- [ ] 补采的类目范围：维持 scenery/plant，还是纳入强运动题材（water/wildlife/vehicles）？
- [ ] 目标训练集规模？（决定补采量和时间预算。）

## 已完成（备查）

- [x] 确认 nexisgen 用官方 VBench-I2V 8 维、无额外归一化。
- [x] 建 `vbench_exact_scorer.py`（精确复刻 aesthetic+dynamic，与官方同口径）。
- [x] 建 `score_motion_strength.py`（连续运动强度），二值与官方一致性已验证（246/400）。
- [x] 诊断 0.62→0.075 崩塌根因 = 训练集运动太弱/擦边（中位 motion_mean≈18，强运动仅 19%）。
- [x] 对 `merged_dataset_nexisgen` 400 个完成 aesthetic+dynamic+motion_strength 全量打分。
- [x] 环境就绪（vbench --no-deps + RAFT/aesthetic 权重，torch 栈未动）。
