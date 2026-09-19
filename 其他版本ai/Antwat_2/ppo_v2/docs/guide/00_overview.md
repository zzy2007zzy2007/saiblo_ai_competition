# 总体概述

## 1. 项目背景

AntWar 是一款双人回合制塔防对战游戏。两名玩家各自拥有防御塔和自动进攻的蚂蚁单位，通过建造、升级、降级防御塔，升级科技，使用超级武器等操作来消灭对手的全部血量。游戏在一个 19×19 的六边形棋盘上进行，双方的目标是在回合限制内击败对手。

本项目（ppo_v2）实现了一套基于 **PPO（Proximal Policy Optimization）强化学习算法** 的 **自对弈（Self-Play）训练系统**，用于训练一个能够自主游玩 AntWar 游戏的 AI 智能体。训练过程中，智能体通过与自身历史版本对战，持续提升策略强度，最终达到甚至超越人类水平。

### 核心设计理念

- **自对弈训练**：当前策略与历史上的策略版本对战，避免策略过拟合到固定对手
- **TrueSkill 评分**：基于贝叶斯推断的评分系统，用于衡量各策略版本的相对强度，指导对手选择
- **PPO 算法**：使用 PPO-Clip 进行策略优化，配合 GAE（广义优势估计）降低方差
- **辅助任务学习**：预测未来塔伤害和金币收入作为辅助监督信号，加速训练
- **结构化动作空间**：分层采样（先选择动作类型，再选择目标），处理 96 维离散动作空间

---

## 2. 架构总览

系统划分为 **8 个逻辑模块**，每个模块服务于特定的业务目标，遵循高内聚、低耦合的设计原则。

### 模块依赖关系图

```
┌────────────────────┐
│  模块8: 基础设施    │ ← 被所有模块依赖
│  (config + compat) │
└────────┬───────────┘
         │
   ┌─────┴──────────────────────────────────────────┐
   │                                                  │
┌──▼──────────────┐    ┌──────────────────┐    ┌─────▼──────────┐
│ 模块1: 游戏环境  │◄───│ 模块3: PPO训练引擎│───►│ 模块7: 监控    │
│ 与动作空间       │    │                  │    │ 与可观测性      │
└─────────────────┘    └────────┬─────────┘    └────────────────┘
         │                      │
         │              ┌───────▼──────────┐
         └──────────────► 模块2: 神经网络  │
                        │ 模型              │
                        └──────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
┌────────▼──────┐   ┌───────────▼───────┐   ┌─────────▼──────┐
│ 模块6: 训练   │   │ 模块4: 自对弈     │   │ 模块5: 对战    │
│ 回调系统       │   │ 训练编排          │   │ 评估系统       │
└───────────────┘   └───────────────────┘   └────────────────┘
```

### 模块职责速览

| 模块 | 名称 | 核心职责 |
|------|------|---------|
| 模块1 | 游戏环境与动作空间 | 封装游戏交互接口，编码观测/动作/奖励 |
| 模块2 | 神经网络模型 | 策略-价值网络架构，动作采样与评估 |
| 模块3 | PPO 训练引擎 | PPO 算法核心，GAE，梯度更新，数值稳定性 |
| 模块4 | 自对弈训练编排 | 训练主循环，对手池管理，TrueSkill 评分 |
| 模块5 | 对战评估系统 | 多进程并行对战，基线 AI 评估，报告生成 |
| 模块6 | 训练回调系统 | 可扩展钩子机制，检查点/TensorBoard |
| 模块7 | 监控与可观测性 | 日志、系统资源、指标持久化 |
| 模块8 | 基础设施 | 配置管理，路径管理，SDK 兼容适配 |

---

## 3. 训练流程概览

### 3.1 完整训练循环

一个完整的训练周期由 [SelfPlayTrainer](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/selfplay.py) 驱动，流程如下：

```
启动
  │
  ├── 1. 初始化：加载配置，创建网络，初始化对手池（3个随机对手）
  │
  └── 2. 主循环（每个 episode）:
        │
        ├── 2.1 选择对手（探索/利用策略，基于 TrueSkill 评分）
        ├── 2.2 收集对战数据（EpisodeCollector）
        │       └── env.reset() → 循环 [agent.act() → env.step()] → 记录步数据
        ├── 2.3 累积到 batch_pool
        ├── 2.4 当 batch 满足条件时：
        │       └── merge batches → 质量检查 → GAE → PPO update → 记录指标
        │
        ├── 2.5 周期性任务：
        │       ├── 每 N 局: 保存当前策略到对手池（添加新对手）
        │       ├── 每 M 局: 基线评估对战（PPO vs 规则 AI）
        │       ├── 每 K 局: 保存检查点
        │       ├── 每 J 局: 保存联赛状态（pool.json + payoff.json）
        │       └── 每 I 局: 保存训练状态（training_status.json）
        │
        └── 3. 训练结束：保存最终检查点，写入训练完成日志
```

### 3.2 单次 PPO 更新的数据流

```
┌──────────────────────────────────────────────────────────────┐
│ EpisodeBatch (多局对战数据)                                   │
│  observations, actions, rewards, values, log_probs, dones   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
              ┌───────────────────────┐
              │ 1. GAE 优势估计       │  ← compute_gae()
              │    advantages, returns│
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ 2. 多 Epoch 训练      │  ← _train_ppo_epochs()
              │   ┌─────────────────┐ │
              │   │ Minibatch 循环   │ │
              │   │   evaluate_actions() → new_log_probs, entropy
              │   │   compute PPO-Clip loss
              │   │   compute auxiliary loss (塔伤害/金币)
              │   │   backward + clip grad + optimizer step
              │   └─────────────────┘ │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ 3. 指标聚合与记录      │
              │   loss/entropy/kl/... │
              └───────────────────────┘
```

---

## 4. 技术栈

| 技术 | 用途 |
|------|------|
| **Python 3.x** | 主要编程语言 |
| **PyTorch** | 深度学习框架，神经网络定义与训练 |
| **Gymnasium** | RL 环境接口标准（reset / step） |
| **TrueSkill** | 贝叶斯评分系统，用于对手强度评估 |
| **TensorBoard** | 训练指标可视化 |
| **loguru** | 结构化日志记录 |
| **NumPy** | 数值计算 |
| **cloudpickle** | Agent 序列化（多进程对战） |
| **concurrent.futures** | 多进程并行对战 |
| **psutil + pynvml** | 系统资源监控（CPU/内存/GPU） |

---

## 5. 目录结构

```
ppo_v2/
├── train.py                          # 训练入口脚本
├── requirements.txt                  # Python 依赖
├── configs/
│   └── custom_antwar.yaml            # 自定义训练配置
├── docs/
│   ├── code/
│   │   └── logical_modules.md        # 模块划分文档
│   ├── core_define/                  # 需求与工具定义
│   ├── issue/                        # 问题记录
│   ├── refactor/                     # 重构文档
│   └── guide/                        # 【本文档目录】模块详细指南
└── src/ppo_antwar/
    ├── __init__.py                   # 包入口，公共接口导出
    ├── config/                       # ⚙ 模块8：基础设施
    │   ├── config_parser.py          #   配置加载、合并、验证
    │   └── path_config.py            #   路径管理
    ├── compat/                       # ⚙ 模块8：基础设施
    │   └── adapters.py               #   SDK 后端兼容适配
    ├── env/                          # 🎮 模块1：游戏环境与动作空间
    │   ├── antwar_env.py             #   Gymnasium 环境封装
    │   ├── observation.py            #   观测编码器
    │   └── action_mask.py            #   动作掩码处理
    ├── network/                      # 🧠 模块2：神经网络模型
    │   └── antwar_net.py             #   策略-价值网络架构
    ├── trainer/                      # 🔧 模块3：PPO 训练引擎 + 模块4：自对弈编排
    │   ├── ppo_trainer.py            #   PPO 算法核心
    │   ├── agent.py                  #   推理代理
    │   ├── batch.py                  #   数据批次容器
    │   ├── episode_collector.py      #   单局数据收集器
    │   ├── safety.py                 #   NaN 检测与恢复
    │   ├── lr_scheduler.py           #   学习率调度
    │   ├── checkpoint_manager.py     #   检查点管理
    │   ├── metrics_schema.py         #   训练指标定义
    │   ├── selfplay.py               #   自对弈训练主循环
    │   ├── selfplay_logger.py        #   自对弈日志
    │   └── training_status.py        #   训练状态汇总
    ├── league/                       # 🏆 模块4：自对弈训练编排
    │   ├── payoff.py                 #   TrueSkill 评分与对手池
    │   └── opponent_selector.py      #   对手选择策略
    ├── battle/                       # ⚔ 模块5：对战评估系统
    │   ├── battle_coordinator.py     #   对战协调器
    │   ├── battle_simulator.py       #   多进程对战执行
    │   ├── battle_config.py          #   评估配置
    │   ├── agent_loader.py           #   Agent 加载器
    │   ├── opponent_agent.py         #   对手 Agent
    │   ├── report_generator.py       #   报告生成
    │   ├── result_aggregator.py      #   结果聚合
    │   ├── battle_logger.py          #   对战日志
    │   └── utils/                    #   对战工具
    │       ├── exception_logging.py
    │       ├── process_isolation.py
    │       └── timing.py
    ├── callbacks/                    # 🔗 模块6：训练回调系统
    │   ├── base_ppo_callback.py      #   回调基类
    │   ├── checkpoint_callback.py    #   检查点回调
    │   ├── metrics_callback.py       #   指标日志回调
    │   ├── tensorboard_callback.py   #   TensorBoard 回调
    │   └── callback_factory.py       #   回调工厂
    ├── monitor/                      # 📊 模块7：监控与可观测性
    │   ├── logger.py                 #   训练日志
    │   ├── constants.py              #   日志常量
    │   ├── time_tracker.py           #   时间追踪
    │   ├── system_metrics_sampler.py #   系统资源采样
    │   ├── batch_metrics_writer.py   #   批量指标写入
    │   ├── episode_batch_writer.py   #   Episode 聚合写入
    │   ├── selfplay_battle_writer.py #   自对弈记录写入
    │   ├── eval_battle_writer.py     #   评估记录写入
    │   └── metrics_cache.py          #   [已废弃]
    └── utils/                        # 🛠 通用工具（被多个模块使用）
        ├── action_constants.py       #   游戏常量（属模块1）
        ├── gae.py                    #   GAE 计算（属模块3）
        ├── aux_labels.py             #   辅助标签计算（属模块3）
        └── obs_utils.py              #   观测张量转换（属模块3）
```

---

## 6. 数据闭环

下图展示了从游戏交互到参数更新的完整数据闭环：

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练数据闭环                             │
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ AntWarEnv│───►│Observation│───►│ Network  │───►│ Action   │ │
│  │ (游戏状态)│    │ Encoder  │    │ forward  │    │ Sampling │ │
│  └──────────┘    └──────────┘    └──────────┘    └─────┬────┘ │
│       ▲                                                │      │
│       │                    ┌───────────────────────────┘      │
│       │                    ▼                                   │
│  ┌────┴─────┐    ┌──────────────┐    ┌──────────────┐        │
│  │ env.step │◄───│ ActionMask   │◄───│ action_id    │        │
│  │ (执行动作)│    │ .action_id   │    │ (int 0-95)   │        │
│  └────┬─────┘    │ _to_op()     │    └──────────────┘        │
│       │          └──────────────┘                              │
│       ▼                                                        │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│  │ Reward + │───►│Episode   │───►│Episode   │                │
│  │ Next Obs │    │Collector │    │Batch     │                │
│  └──────────┘    └──────────┘    └────┬─────┘                │
│                                       │                        │
│                          ┌────────────┘                        │
│                          ▼                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                │
│  │ Optimizer│◄───│ PPO Loss │◄───│  GAE +   │                │
│  │ .step()  │    │ Backward │    │Eval Acts │                │
│  └──────────┘    └──────────┘    └──────────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

关键数据转换节点：

| 节点 | 输入 | 输出 | 负责模块 |
|------|------|------|---------|
| 观测编码 | 游戏 state | `{board, global_vec, action_mask}` | 模块1 |
| 前向推理 | 观测张量 | `action_logits, value` | 模块2 |
| 动作采样 | logits + mask | `action_id, log_prob` | 模块2 |
| 动作解码 | action_id | SDK Operation | 模块1 |
| 环境步进 | Operation | `next_obs, reward, done` | 模块1 |
| 数据收集 | 步数据流 | `EpisodeBatch` | 模块3 |
| GAE 估计 | rewards + values | `advantages, returns` | 模块3 |
| PPO 更新 | batch + advantages | 梯度更新 | 模块3 |

---

## 7. 阅读指南

建议按照以下顺序阅读各模块文档，以获得最佳理解效果：

1. **[模块8：基础设施](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/docs/guide/08_infrastructure.md)** — 了解配置管理和路径管理，这是所有模块的基础
2. **[模块1：游戏环境与动作空间](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/docs/guide/01_game_environment.md)** — 理解游戏交互接口、观测编码和奖励设计
3. **[模块2：神经网络模型](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/docs/guide/02_neural_network.md)** — 理解策略-价值网络的架构设计
4. **[模块3：PPO 训练引擎](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/docs/guide/03_ppo_training_engine.md)** — 理解 PPO 算法实现和数据流
5. **[模块4：自对弈训练编排](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/docs/guide/04_selfplay_orchestration.md)** — 理解完整的训练循环和对手管理
6. **[模块5：对战评估系统](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/docs/guide/05_battle_evaluation.md)** — 理解如何评估模型强度
7. **[模块6：训练回调系统](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/docs/guide/06_training_callbacks.md)** — 理解插件化的训练钩子
8. **[模块7：监控与可观测性](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/docs/guide/07_monitoring_observability.md)** — 理解日志、指标和系统监控
