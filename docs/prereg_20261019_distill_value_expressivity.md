# 预注册：亚军 value 函数蒸馏（架构表达力验证）

> 2026-10 新方向。计划见 `docs/distill_top2_expressivity_plan.md`。
> **不是刷分/里程碑**，只回答"现有 value 架构能不能表示一个非 ML 的手写价值函数"。

## 假设

**H1**：现有 `AntWarNetwork` 的 value 通路（board 28ch → Conv+ResBlock → **GAP** → +stats42 → value head）
**能**从 `(board, stats)` 学会亚军的 `evaluate_terminal` 标签。
- 能学会 ⇒ 架构表达力够，RL 的瓶颈在"发现"而非"表示"。
- 学不会 ⇒ 先排除输入表示（board 缺塔 HP/类型），再判架构/池化不足。

## 数据与切分

- 数据：`training_history/distill_top2/value_ds_12g.npz`（5942 样本 / 12 局，C++ 引擎状态 + 亚军标签）。
- **按 seed 切**（防同局相邻回合泄漏）：**train = seed 100..107（8 局）**，**val = seed 108..111（4 局）**。

## 模型与训练

- 模型：`create_model(value_pool=<gap|grid|...>)`，从 `checkpoints/grafted_sdk_A.pt` 的 `value_state` **warm-start**。
- 目标：`y = (value - mean_train) / std_train`（归一化，避免 10000 量级主导梯度）。
- 损失：MSE；Adam lr=1e-3；batch=128；early stop（patience，看 val）。
- **只动 `value_state`**（三网独立 ⇒ 天然不影响 class/pos）。
- 变体：
  1. `head`（冻结 encoder，只训 value_head）——"现有特征能否表达"；
  2. `full`（encoder+head 一起训）——"架构能否学出来"；
  3. `value_pool=grid`（4×4 粗网格池化）——测"GAP 池化是不是瓶颈"。

## 判据（预注册，先写死后跑）

主判据 = **val R²（原始 value 尺度）**，并和线性基线比：

| 结果 | 判读 |
|---|---|
| net R² ≥ 0.95 且 ≥ 线性基线 + 0.05 | **表达力够**：能学会 money+threat 的空间项 |
| net R² ≈ 线性基线 | 只学会 base_hp/money（stats 里的线性项），**没学到空间 threat** ⇒ 架构/池化不足 |
| net R² < 线性基线 | 训练/容量问题，需查 |

辅助：val MAE；`value_pool=grid` 相对 `gap` 的提升（判断池化是否瓶颈）。

## 基线

- 常数（train mean）⇒ R²=0。
- **ridge 线性回归**（只用 42 维 stats）⇒ "不看棋盘"的上限。
- ridge 线性回归（只用前 22 维 summarize 特征）。

## 不做

- 不改 class_state / pos_state；不碰判据 seed 的对局数据；结果只当**表达力诊断**，不当里程碑证据。
