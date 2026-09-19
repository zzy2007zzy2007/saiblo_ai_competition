# PPO v1 — SelfPlay 与 Baseline Battle 对战逻辑详解

> **版本**：v3.1（基于 `ppo/src/ppo_antwar` 源码分析 + `docs/design/antwar_rules.md` + `docs/design/antwar_rules_review.md` + Ant-Game SDK）
>
> 本文档为导航索引，实际内容按章节拆分为 11 个独立文件。

---

## 文档结构

| 章节 | 内容 | 文件 | 行数 |
|------|------|------|------|
| **1** | 总体架构概览 | [sec1](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec1.md) | ~230 |
| **2** | 游戏基础：环境与回合机制 | [sec2](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec2.md) | ~410 |
| **3** | SelfPlay 训练流程 | [sec3](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec3.md) | ~900 |
| **4** | Baseline Battle 流程 | [sec4](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec4.md) | ~1,100 |
| **5** | 动作空间与 Action Mask | [sec5](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec5.md) | ~480 |
| **6** | Observation 编码（观察空间） | [sec6](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec6.md) | ~400 |
| **7** | 奖励函数 | [sec7](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec7.md) | ~350 |
| **8** | PPO Agent 决策流程 | [sec8](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec8.md) | ~500 |
| **9** | 对手管理与 League 机制 | [sec9](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec9.md) | ~640 |
| **10** | 日志与报告系统 | [sec10](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec10.md) | ~750 |
| **11** | 关键配置参数速查 | [sec11](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec11.md) | ~280 |

> **共 11 章，约 6,100 行。**

---

## 快速导航

### 想了解整体架构？
→ [第 1 章：总体架构概览](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec1.md)

### 想了解游戏回合机制？
→ [第 2 章：游戏基础：环境与回合机制](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec2.md)

### 想了解 SelfPlay 训练全流程？
→ [第 3 章：SelfPlay 训练流程](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec3.md)（最核心章节）

### 想了解 Baseline Battle 并行评估？
→ [第 4 章：Baseline Battle 流程](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec4.md)

### 想了解动作空间 / 动作掩码？
→ [第 5 章：动作空间与 Action Mask](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec5.md)

### 想了解观察空间 / 特征编码？
→ [第 6 章：Observation 编码](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec6.md)

### 想了解奖励设计？
→ [第 7 章：奖励函数](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec7.md)

### 想了解 PPO 算法决策细节？
→ [第 8 章：PPO Agent 决策流程](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec8.md)

### 想了解对手管理 / League / TrueSkill？
→ [第 9 章：对手管理与 League 机制](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec9.md)

### 想了解日志 / 报告系统？
→ [第 10 章：日志与报告系统](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec10.md)

### 想快速查阅配置参数？
→ [第 11 章：关键配置参数速查](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_sec11.md)

---

## 完整大纲

详见 [文档大纲](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/docs/design/ppo_v1_selfplay_battle_logic_outline.md)

---

## 源文件索引

| 模块 | 文件路径 | 章节 |
|------|---------|------|
| PPO 训练 | `ppo/src/ppo_antwar/trainer/selfplay.py` | §3 |
| PPO 算法 | `ppo/src/ppo_antwar/trainer/ppo_trainer.py` | §3, §8 |
| PPO 网络 | `ppo/src/ppo_antwar/network/antwar_net.py` | §5, §6, §8 |
| Baseline 协调器 | `ppo/src/ppo_antwar/battle/battle_coordinator.py` | §4 |
| Baseline 模拟器 | `ppo/src/ppo_antwar/battle/battle_simulator.py` | §2, §4 |
| Baseline 配置 | `ppo/src/ppo_antwar/battle/battle_config.py` | §4, §11 |
| 兼容适配 | `ppo/src/ppo_antwar/compat/adapters.py` | §2 |
| 对手选择器 | `ppo/src/ppo_antwar/league/opponent_selector.py` | §3, §9 |
| TrueSkill/Payoff | `ppo/src/ppo_antwar/league/payoff.py` | §3, §9 |
| 动作常量 | `ppo/src/ppo_antwar/utils/action_constants.py` | §5 |
| 日志系统 | `ppo/src/ppo_antwar/battle/battle_logger.py` | §10 |
| 日志系统 | `ppo/src/ppo_antwar/trainer/selfplay_logger.py` | §10 |
| 报告生成 | `ppo/src/ppo_antwar/battle/report_generator.py` | §10 |
| 结果聚合 | `ppo/src/ppo_antwar/battle/result_aggregator.py` | §10 |
| 配置文件 | `ppo/src/ppo_antwar/configs/ppo_antwar.yaml` | §11 |
| 配置解析 | `ppo/src/ppo_antwar/config/config_parser.py` | §11 |
| SDK 特征 | `Ant-Game/SDK/utils/features.py` | §6 |
| SDK 动作 | `Ant-Game/SDK/utils/actions.py` | §5 |
| SDK 环境 | `Ant-Game/SDK/training/env.py` | §7 |
| 游戏规则 | `docs/design/antwar_rules.md` | §2 |
| 规则解读 | `docs/design/antwar_rules_review.md` | §2 |
