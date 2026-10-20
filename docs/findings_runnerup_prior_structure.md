# 主线判决：亚军的"先验 + 价值函数"我们能不能复刻？

> 2026-10-20。工具链：`dump_plan_prior.exe`（状态 JSON → 亚军 `generate_root_plans` / rollout）
> + `dump_prior_labels.py` / `analyze_prior_decomposition.py` / `check_prior_rule.py`
> / `measure_rollout_noise.py` / `compare_one_step_vs_search.py`
> 计划：`docs/plan_prior_distill.md`。实验记录：`docs/experiment_log.md`（`prior_rollout_noise`、`prior_1step_vs_search`）。

## 0. 一句话结论

**亚军的强度不来自"聪明的先验"或"聪明的价值函数"，而来自 8 回合 × 125 次的 rollout 前瞻。**
它的先验是 **2 条 if**（312,771 条 plan 上 100% 可复现），它的价值函数是**叶子评估**
（我们已蒸馏到 R² 0.9969），但**把叶子评估提前 1 回合用，决策一致率就等于"永远不动"的基线**——
说明信息全在"多步前瞻"里，不在函数表达力里。

⇒ 对我们架构的含义：**表达力够用**（叶子价值能拟合），**缺的是前瞻深度与精度**。

---

## 1. 先验的真实结构：2 条谓词，100% 可复现

在 4 条亚军日志（seed 732/733 × 双方，共 **312,771 条 plan**）上逐项核对：

| 检验 | 结果 |
|---|---|
| A `base_heuristic == 80·[key=="hold"]` | **312,771 / 312,771 = 100%** |
| B `lure_heuristic == lightning_heuristic == followup_penalty == 0` | **100%** |
| C `operation_penalty ∈ {0, 400}` | **100%** |
| D 恒等式 `heuristic = base + lure + lightning − op_pen − followup_pen` | **100%** |
| E `heuristic == 80·[hold] − operation_penalty` | **100%** |

```
heuristic = 80 · [计划是「不动」]  −  400 · [计划会降级一座狙击塔]
```

`operation_penalty` 取值分布：`0.0` 出现 281,020 次，`400.0` 出现 31,751 次。
带降级操作的计划里 **115,215 个没有罚分**（降的不是狙击塔）、31,751 个有罚分。

源码印证（`lure_strategy_v4_params.hpp` 的手调常数表）：
`hold_bonus=80`、`sniper_downgrade_penalty=400`，而 `c1_build_bonus=0`、
`followup_plan_penalty=0`、`hold_followup_enabled=false`、`enable_producer_medic_branch=false`、
`c1_quick_trans_bonus/c1_sniper_trans_bonus` 在默认配置下**根本触发不到**。
`c1_build_heuristic(code)` 的实现就一句 `return code == C1 ? c1_build_bonus : 0.0;`。

**⇒ 先验不是函数，是查表。主线里"先验可否复刻"这一问，答案是"可以精确复刻，且不需要学习"。**

## 2. 先验的作用：一个 80 分的"该不该动"阈值

1024 个决策（`ai0_runnerup_seed732` + `ai1_ai3_linear_seed732`）统计：

| 指标 | 数值 |
|---|---|
| 决策数 / 候选计划总数 | 1024 / 161,365（均值 157.6 个候选/决策） |
| 先验取值种类 | **3**（`0` 91.2%、`−400` 8.2%、`+80` 0.63%） |
| **最终选择 = 不动（hold）** | **846 / 1024 = 82.6%** |
| `argmax(heuristic) == 最终选择` | 82.6%（因为 80 是唯一正数，argmax(heuristic) **恒为 hold**） |
| `argmax(mean_terminal_score) == 最终选择` | 44.7%（**先验翻盘 52.6%**） |
| top1 − top2 的 `score` 分差 | 均值 62.2 / **中位 71.1** / p90 100.0 |
| `rank==1` 与 `max(score)` 一致 | 1024 / 1024 |

`score = mean_terminal_score + heuristic`，而 top1−top2 的分差中位 **71** ≈ hold 的加成 **80**
⇒ 先验的 80 分就是**设计成卡在这个分差量级上的阈值**：动作的期望收益要超过 ~80 才值得动手。

## 3. 价值函数：叶子能拟合，但"提前一步用"就失效

先看搜索本身的噪声（4 seeds × 22 rounds × 双方，1408 条 计划×状态，64 rollouts）：

| 指标 | 数值 |
|---|---|
| 单次 rollout 分数 σ | 均值 123.6 / 中位 110.9 / p90 245.9 |
| **SE（N=125，亚军每臂实际量级）** | **均值 11.1 / 中位 9.9** |
| 同状态内 top1−top2 的**真实均值差**（信号） | 均值 47.0 / 中位 33.4 / p10 7.3 |
| 信号 / SE(125) | 中位 **3.37** |
| 信号 < SE 的决策占比 | **12.5%**（这些决策本身由噪声决定） |

⇒ 亚军自己重复同一决策的一致率**上限约 87.5%**；要"复现它的选择"，价值估计误差必须 ≲ **10 分**
（满分尺度约 10,000，即 0.1%）。而我们蒸馏的 `evaluate_terminal` MAE ≈ **80**（0.8%），**差 8 倍**。
这就是之前 top-1 全错的定量原因，也是"表达力够、精度不够"的分界。

再看**形态**问题（4 seeds × 10 rounds × 双方 = 80 个状态，32 rollouts，全部候选）：

| 指标 | 数值 |
|---|---|
| Spearman(1 步评估, 8 步 rollout) | 均值 +0.556 / **中位 +0.737** / p10 +0.018 |
| 决策一致率（1 步评估 + 80 先验） | **51 / 80 = 63.8%** |
| 「永远 hold」基线一致率 | **63.8%**（完全相同） |
| 纯评估 argmax 一致率（不含先验） | **12.5%** |
| top5 排名重叠率 | 0.13 |
| 计划间离散度（full / 1step） | 894 / 717 |
| 单次采样 σ（full / 1step） | 104 / 5 |

分解那 80 个状态：

```
都选 hold 51 | 都动手 3 | 该动没动 26 | 不该动却动 0
```

**1 步终止评估从不误动，但也几乎从不正确动手（26 个漏掉 23 个）**——
它等价于"永远 hold"，对决策**零增量信息**。

## 4. 对主线的回答

| 问题 | 答案 | 证据 |
|---|---|---|
| 能复刻**先验**吗？ | **能，而且平凡**（2 条谓词，精确 100%） | 312,771 条 plan 逐条核对 |
| 能复刻**价值函数**（叶子评估）吗？ | **能**（R² 0.9969 / 单边仿真 0.9987） | 前一轮蒸馏实验 |
| 用这个价值函数能复刻**决策**吗？ | **不能**（= 永远 hold 的基线，63.8%） | 80 状态实验 |
| 差在哪？ | **前瞻深度**（1 步 vs 8 步）+ **精度**（MAE 80 vs 需要 ≲10） | 噪声/信号分析 |
| 是架构表达力不够吗？ | **不是**。叶子价值拟合得动，缺的是多步前瞻与精度 | 以上全部 |

**⇒ 如果目标是"复刻亚军"，要抄的东西按重要性排序是：
① 8 回合前瞻 + 125 次采样（决定性的）；② hold +80 阈值规则（免费的 1 bit）；
③ 叶子价值函数（我们已有，但精度还差 8 倍）。**

## 5. 顺带关掉的一条死路

原计划 `我们的状态 → replay.json → sdk_lure_inspector`：**不可行**。
`load_replay_round_start` 只能从 seed 逐回合推演，而我们只有中局状态快照、没有 op 历史；
且它要的 `round_state` 是对象 schema，我们的 dump 是数组 schema（报
`cannot use value() with array`）。改成直接把状态灌进 `PublicState` 后就通了。

**工具可信度验证**：`seed732 / round100 / player0`，我的工具 vs 亚军自己的 stderr 日志
→ 候选 key **209/209 完全一致**，`heuristic` **max diff = 0.0**。

## 6. 补测：把目标上移到"8 步期望值"后，能学吗？（下界基线）

既然叶子评估没用，就把蒸馏目标换成亚军真正在用的量 `mean_terminal_score`（8 回合期望值）。
数据：`training_history/distill_top2/q8/q8_dataset_v2.jsonl`（**50,296 行 / 304 状态**，
每行 = (状态, 候选计划) → 64 次 rollout 的均值，+92 个状态特征 + 计划特征）。

先用一个**下界模型**（HistGradientBoosting，按**状态**随机切分）探路：

| 指标 | 数值 |
|---|---|
| 绝对 R²（参照；被状态间差异灌水） | 0.296 |
| **状态内残差 R²** | **−5.54** |
| 状态内 Spearman 中位 | **+0.616** |
| **top-1 一致率** | **36.8%（28/76）** |
| **「永远 hold」基线** | **76%**（58/76 个状态真值就是 hold） |
| 前 12 名内部成对排序正确率 | 45.6% |

两条结论：

1. **目标里确实有状态内结构**（ρ=0.62），不是纯噪声 ⇒ 值得用实体网络再试；
2. 但手工聚合特征 + GBDT **连"永远 hold"都打不过**，**不能替代搜索**。这不构成对实体网络的
   判决（它能看到整块棋盘），但钉死了评估规范：**任何"复刻亚军决策"的指标都必须与
   「永远 hold」基线并列报**，否则 63.8%、76% 这类数字会被误读成"模型有用"。

> 另注：第一版基线（`q8_dataset.jsonl`，88 状态）**按 seed 切分**得到 test R² = −0.24，
> 那是"跨对局迁移"而非"状态内判别"，是**实验设计错误**，已用按状态切分的 v2 取代。

### 6.1 放大到 28.6 万行 + 上实体网络（2026-10-20 结论）

数据 `training_history/distill_top2/q8/q8_dataset_big.jsonl`：286,434 行 / 1,750 状态
（40 局 × 22 回合 × 双方，64 次 rollout 的均值）。按**整局**切分（30 局训练 / 10 局测试）：

| 模型 | 状态内 Spearman | top-1 | **「永远 hold」基线** | 该动时动了 | 前 12 名成对排序 |
|---|---|---|---|---|---|
| GBDT（手工聚合特征） | +0.656 | 42.7% | **76.9%** | 29.7% | 43.7% |
| **实体 Transformer**（带 plan token，0.52M 参数） | **+0.822** | 77.2% | **76.9%** | 5.9% | 49.5%（≈随机） |

**两个模型都打不过「永远 hold」。** 实体网络把状态内整体排序相关从 0.656 提到 **0.822**，
但**前 12 名之间的细排序等于随机**、该动时动了只有 5.9% ⇒ **学出来的 8 步期望值替代不了搜索**。

原因：决策需要 **~71 分的分辨率**（top1−top2 分差中位），而模型在这批数据上的残差远大于它；
先验的 +80 又正好压在这个量级，所以预测几乎永远被 hold 压住。

⇒ **主线最终判决（三层）**：
1. **先验**：2 条 if，可精确复刻（312,771 条 plan 100%）；
2. **叶子价值** `evaluate_terminal`：可表达（R² 0.9969）；
3. **决策真正依赖的 8 步期望值**：我们的架构能解释大部分**粗**排序（ρ 0.82），
   但**细粒度排序与"该不该动手"都做不到**（top-1 仅比「永远 hold」高 0.3pp）。
   **所以亚军的强度确实来自搜索，而不是任何一个可蒸的函数。**

## 7. 复现命令

```powershell
# 编译（ASCII junction + mingw64 PATH，坑见 docs/runnerup_build_note.md）
& code\my_ai\distill_top2\cpp_test\build_dump_plan_prior.ps1

# 对齐验证（PASS 表示工具与亚军逐位一致）
python code\my_ai\distill_top2\dump_prior_labels.py `
  --globs "training_history/distill_top2/abl_linear2_raw/ents_seed*.jsonl" `
  --align-log-dir match_results --align-seed 732 --align-round 100

# 先验 = 2 条谓词（100% 检验）
python code\my_ai\distill_top2\check_prior_rule.py `
  match_results\ai0_runnerup_seed732.stderr.log match_results\ai1_ai3_linear_seed732.stderr.log

# 先验 vs 搜索的决策归因
python code\my_ai\distill_top2\analyze_prior_decomposition.py `
  match_results\ai0_runnerup_seed732.stderr.log match_results\ai1_ai3_linear_seed732.stderr.log

# rollout 噪声下限
python code\my_ai\distill_top2\measure_rollout_noise.py --seeds 732 733 734 735 --rollouts 64 --max-plans 8

# 1 步评估 vs 8 步搜索
python code\my_ai\distill_top2\compare_one_step_vs_search.py --seeds 732 733 734 735 --rollouts 32
```
