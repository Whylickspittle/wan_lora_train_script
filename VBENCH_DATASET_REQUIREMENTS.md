# VBench 对齐的训练数据需求（v2 · 2026-06-26 更新）

> 本文件记录当前 Wan2.2 TI2V LoRA 训练数据集的**最新筛选需求**，取代之前"二值 dynamic 即可"的旧标准。
> 配套工具与脚本都在 `dataset_quality_tools/`。

---

## 1. 背景与目标

- **目标**：训练 Wan2.2 TI2V LoRA，使其在 **VBench-I2V** 上拿高分。
- **评测方**：nexisgen 验证器（`/workspace/nexisgen`），用官方 VBench，镜像 `rendixnetwork/vbench:latest`，
  8 个维度简单平均：
  `i2v_subject, i2v_background, subject_consistency, background_consistency,
   motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality`。
  **VBench 各维度输出本身就是 0–1，nexisgen 不做额外归一化。**

## 2. 这次发现的核心问题（必须解决）

对 `step_001750/lora.pt` 的 40 个生成视频跑 VBench（`results_2026-06-25-..._eval_results.json`）：

| 维度 | 训练素材(我方打分) | 生成视频(VBench) | 结论 |
|---|---|---|---|
| aesthetic_quality | 0.593 | 0.571 | ✅ 一致，美学成功迁移 |
| **dynamic_degree** | 0.615 | **0.075** | ❌ **崩塌**：37/40 生成视频是静态 |

- 其他一致性维度虚高（subject 0.984 / background 0.976 / motion_smoothness 0.995）——
  **这是"模型塌缩成冻结首帧"的典型特征**（静态视频天然在这些维度拿满分）。
- **根因**：训练集运动太弱/擦边。对 400 个训练素材跑 `score_motion_strength.py`：
  - 全体 `motion_mean` 中位数 ≈ 18（紧贴 VBench 阈值 16.5）
  - 强运动(`motion_mean≥33`)只占 **19%**；弱(<16.5) 45% + 擦边(16.5–24.8) 22%。
  - 模型学到偏弱的中心分布，一回归到均值就整体掉到阈值以下 → 生成静态。
  - **结论：只用"二值 dynamic=1"筛选不够，擦边片扛不住训练回归。**

## 3. VBench dynamic_degree 判定机制（务必理解）

源码：`vbench/dynamic_degree.py`。对一个 clip：
1. 抽帧降到 ~8fps：`interval = round(fps/8)`（24fps→每3帧取1，121帧→约41帧）。
2. 相邻抽样帧用 **RAFT(raft-things)** 算光流，每像素幅值 `rad=√(u²+v²)`（单位=像素）。
3. 取每帧对**最快 5% 像素的平均幅值** = `max_rad`。
4. 阈值 `thres = 6.0 × min(H,W) / 256`。**704p → 16.5，1080p → 25.3，4K → 50.6**（越高清要求越高）。
5. 计数门 `count_num = round(4 × 帧数/16)` ≈ 10。
6. `max_rad > thres` 的帧对 ≥ count_num → **判"动"(1)**，否则 **"静"(0)**。
7. 数据集 `dynamic_degree` = 判"动"的比例。

## 4. 新的数据筛选标准（硬性）

| 指标 | 工具 | 标准 |
|---|---|---|
| **运动强度** | `score_motion_strength.py` | **`motion_mean ≥ 33`（≈2× 阈值，留足余量对抗回归）**；至少 `≥24.8`(1.5×) |
| dynamic 二值 | `vbench_exact_scorer.py` | 必须 `dynamic=1`（被上面的强度门自然覆盖） |
| 美学 | `vbench_exact_scorer.py` | `aesthetic_quality ≥ 0.55`（VBench 尺度；0.6 可达但仅约 1/5 素材） |
| 去重 | `dedup_review80.py` 逻辑 | **一源一片**：同一源视频只留 node 最小（开头）的切片 |
| 题材/来源 | — | 优先 **激流/海浪/瀑布、跟拍/FPV、奔跑动物、行驶车辆**；**Pexels 运动强于 YouTube**（中位 23.3 vs 12.1）；避开慢航拍、锁定空镜、YouTube 慢镜头 |

> 注意：标准提高后产出率骤降——现有 400 个里 `motion_mean≥33` 的只有 76 个。**必须补采强运动新素材**才能凑够训练量。

## 5. 已建工具与环境（dataset_quality_tools/）

| 脚本 | 作用 |
|---|---|
| `vbench_exact_scorer.py` | **精确复刻** VBench `aesthetic_quality`(LAION sa_0_4 linear÷10) + `dynamic_degree`(RAFT 二值)。`--resume` 增量。与 nexisgen 同口径。 |
| `score_motion_strength.py` | 连续**运动强度**：`motion_mean/median/max, moving_ratio, thres, dynamic`。用于按余量排序选片。 |
| `vbench_keywords.py` | VBench 内容类目关键词矩阵。 |
| `collect_vbench_nature.py` | 采集→Tier-A→VBench 打分→阈值过滤→Top-K（当前阈值是美学0.55+dynamic=1，**待升级为 motion_mean≥33**）。 |
| `collect_vbench_review80.py` / `run_vbench_dataset.py` | 多类目批量采集驱动。 |
| `dedup_review80.py` | 一源一片去重。 |
| `score_vbench_proxy.py` / `vbench_compose_select.py` | 早期 CLIP 代理打分（已被 vbench_exact 取代，保留备用）。 |

**环境（已就绪，勿动 torch）**：`vbench`(--no-deps) + `decord/opencv-headless/easydict/scipy`；
权重在 `~/.cache/vbench/`：`raft_model/models/raft-things.pth`、`aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth`；
CLIP ViT-L/14 在 `~/.cache/clip/`。torch 栈保持 `2.12.0+cu130` 未受影响。

## 6. 已产出的评分数据

- `merged_dataset_nexisgen/vbench_aes_dyn.csv` — 400 个训练素材的 aesthetic + dynamic(二值)。
- `merged_dataset_nexisgen/motion_strength.csv` — 400 个的运动强度连续值。
- `dataset_quality_tools/vbench_nature/` — 试水采集的 9 个 unique（美学均值 0.612，dynamic 全 1）。
- `dataset_quality_tools/vbench_review80/` — 早期 review80 的 scenery/plant 保留物（已删 animals 等非风光类）。

## 7. 关键阈值速查

- VBench dynamic 阈值（704p）：**16.5 像素**；count_num ≈ **10**；抽样 **8fps**；top-**5%** 光流。
- 训练运动地板：**motion_mean ≥ 33**（强），最低 ≥24.8（中等）。
- 美学地板：**aesthetic_quality ≥ 0.55**（VBench 尺度 = LAION÷10）。
