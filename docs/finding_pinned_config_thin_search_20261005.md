# 重大发现：钉死基线配置里的搜索**几乎是空转**（2026-10-05）

> 状态：**读数级证据**（代码 + 历史日志 + 注释），**未做新对局**。强度：**中**
> （每条都有 文件:行号 或 run 读数可复核；但"到底强多少"要等 §4 的实验）。
> 触发点：核对"我们 vs rule_v4 的 54.3% 是在什么配置下测的"时发现的，
> 起因是 `docs/next_step_survey_20261005.md` 里 H2 的先例不可比。

## 1. 结论（一句话）

`code/test_match/az_bridge_ai.py` 里 `AZAI_K` 默认 **1**、`AZAI_SAMPLE_MULT` 默认 **1**
（`:167-168`），而任务章程 §2 的钉死配置**没有设这两个** ⇒ 本轮基线跑的 MCTS
**每次扩展只采样 1 个 bundle**（`bundle_mcts.py:295` `n_samples = k * sample_mult = 1`，
`:336-357` 聚合成 1 个候选、`[:k]` 取前 1 个）⇒ **根节点永远只有 1 个子节点**
⇒ `chosen_bundle` 与"跑 256 次迭代"无关，**搜索实际只做了一件事：从策略里随机采一个动作**；
那 256 次迭代（每次一次价值网前向，`bundle_mcts.py:400-409` 注释：占单次搜索墙钟 ~85%）
在单候选树上**不改变任何被读取的东西**。

## 2. 逐条证据

| # | 读数 | 出处 |
|---|---|---|
| 1 | 桥壳默认 `k=1`、`sample_mult=1`、`skip_single_candidate=0`、`mode=joint` | `code/test_match/az_bridge_ai.py:167-175` |
| 2 | 本轮基线的实际环境**只有**章程列的那些变量（无 `AZAI_K`/`AZAI_SAMPLE_MULT`） | `training_history/runs/20261005_012937_B3_baseline_rulev4_128/env.txt` |
| 3 | 引擎构造日志同样只报 iters/depth/mode/pospin/tclass/tpos/temp，**不报 k** ⇒ 默认值静默生效 | `match_results/ai0_az_bridge_ai_seed7.stderr.log` |
| 4 | 每次扩展采样次数 `n_samples = k * sample_mult`；只保留按出现次数排序的前 `k` 个 | `bundle_mcts.py:295, 336-359` |
| 5 | 单候选时 256 迭代是纯浪费（作者已实测：pos-only 下 25.8×，748ms→5ms/回合） | `bundle_mcts.py:400-409`（强制走子跳过那个 OPT-IN 开关的注释） |
| 6 | **历史强度的"标准配置"完全不同**：`--k 24 --t-pos 1.0 --search-mode pos-only --skip-single-candidate`（`eval.py` **不传** `sample_mult` ⇒ 默认 **15**，`bundle_mcts.py:199`）⇒ n_samples=**360** | `docs/experiment_log.md:3698`；`code/my_ai/az_intent/eval.py:243-247`；`bundle_mcts.py:195-202` |
| 7 | `posnet_A_k5_m32` 的 **54.3%**（256 局 vs rule_v4）就是在上一条那个配置下测的 | `docs/experiment_log.md:3720-3727` |
| 8 | 那个 run（3 臂 × 256 局 + 3 臂 × 64 局 = 960 局）总用时 **6536 s**（16 workers）⇒ **≈1.8 min/局** | `training_history/runs/20260920_224913_vp_net_criteria/meta.txt` |
| 9 | 对照：本轮基线（k=1/sm=1）实测 **12–19 min/局**（前 8 局），速率 ~0.42 局/min | `training_history/runs/20261005_012937_B3_baseline_rulev4_128/output.log` |
| 10 | 厚搜索下"搜索选择 == raw 选择"仍高达 **97.7%**，有得选（≥2 候选）的回合只有 **2.8%**（pos-only） | `.../20260920_224913_vp_net_criteria/output.log` 末尾 `[diag]` 行 |

## 3. 这说明什么（解读，可能被推翻）

1. **本轮的 128 局基线不是"我们 + 搜索"，而是"我们 + 每回合从策略里采一个动作"**
   （`t_class=0.5` 的锐化采样，位置按 `t_pos=1.0` 采）⇒ 它的读数**不能**与历史上
   任何"带搜索"的读数（54.3% / 46.9% / …）相减（协议不同：搜索强度不同）。
   ⚠️ 这也意味着**章程 §2 的钉死配置与"我们已知的最好配置"不是同一个东西**，
   这是**章程本身的疏漏**（只列了部分 `AZAI_*`，把两个决定性默认值留在了代码里）。
2. **它还慢**：单候选树上照样付 256 次价值前向/回合 ⇒ 12–19 min/局，而厚配置是 ~1.8 min/局。
   ⇒ 只要把配置改对，整条循环的**迭代速度可以提高近一个数量级**
   （快读数 64 局 ≈ 15–30 min；验收 128 局 ≈ 0.5–1 h，而不是 6–7 h）。
3. **因此"配置修复"应当排在所有方法层实验之前**：否则我们会拿一个"几乎没搜索"的起点去比
   方法改动，读出来的差异很可能只是"搜索有没有生效"。
   ⚠️ 但按纪律，这**必须**如实标注：它是**工程/协议层的修复**，**不是**"我们的方法学会了优化"。

## 4. 下一步（已按纪律写成预注册）

见 `docs/prereg_20261005_search_config_ladder.md`：用**钉死判据**（vs rule_v4 的配对分）
比较 ① 现行钉死配置 ② 历史标准配置（k24/sm15/pos-only/skip1）③ 厚 joint 配置，
先在 seed 前缀上快读数筛，赢家再跑 128 局验收。
