# PPO AntWar 逻辑业务模块划分

> 本文档基于 `ppo_v2/src/ppo_antwar/` 源代码分析，按**逻辑功能**而非目录结构进行模块划分。
> 划分原则：高内聚低耦合、辅助代码归入所服务的业务模块、通用功能归入基础设施。

---

## 模块总览

| 编号 | 模块名称 | 核心业务目标 | 源文件数 |
|------|----------|-------------|---------|
| 1 | 游戏环境 (Game Environment) | 封装 AntWar 游戏为 Gymnasium 风格强化学习环境 | 4 |
| 2 | 神经网络模型 (Neural Network Model) | 定义 Actor-Critic 网络架构，支持结构化动作输出和辅助任务预测 | 3 |
| 3 | 算法训练 (Algorithm Training) | 实现 PPO-Clip 算法及 GAE 优势估计、梯度安全、检查点管理 | 6 |
| 4 | 自对弈训练 (Self-Play Training) | 驱动完整的自对弈训练循环，协调对手选择、数据收集与 PPO 更新 | 7 |
| 5 | 联赛系统 (League System) | 管理对手池、TrueSkill 评分及自适应探索/利用对手选择 | 2 |
| 6 | 基线对战评估 (Baseline Battle Evaluation) | 评估 PPO 智能体对基线 AI 的对抗性能 | 13 |
| 7 | 基础设施 (Infrastructure) | 提供配置管理、路径管理、日志框架、时间追踪、系统监控等跨切面服务 | 12 |
| — | **无法明确划分的代码** | 与 GitHub 标准 workflow 集成，不属于任何业务模块 | 1 |

---

## 模块 1：游戏环境 (Game Environment)

**业务目标**：将 AntWar 游戏封装为标准化的强化学习环境，提供观测编码、动作掩码生成和奖励计算功能。

| 源文件 | 归属目录 | 核心函数/类 | 功能描述 |
|--------|---------|-------------|---------|
| `env/antwar_env.py` | `env/` | `AntWarEnv` | 核心环境类，实现 `reset()` / `step()` 接口，支持批量与顺序两种交互模式；负责回合推进、多维度奖励计算（HP 攻击、塔伤害、金币、科技、终局奖励等） |
| `env/action_mask.py` | `env/` | `ActionMaskHandler` | 生成动作合法性掩码，将扁平 action_id 转换为 SDK Operation 对象，确保策略网络仅输出合法动作 |
| `env/observation.py` | `env/` | `ObservationEncoder` | 将游戏状态编码为神经网络输入格式（棋盘特征图 `28×19×19` + 全局特征向量 `33维` + 动作掩码 `96维`） |
| `compat/adapters.py` | `compat/` | `BackendAdapter` | 封装 SDK 版本差异，提供统一的后端运行时创建接口；包装 `MatchRuntime` 以兼容旧版 `resolve_turn` API |

---

## 模块 2：神经网络模型 (Neural Network Model)

**业务目标**：定义 PPO 策略-价值网络架构，支持六边形卷积、结构化动作空间分层输出、辅助任务预测头。

| 源文件 | 归属目录 | 核心函数/类 | 功能描述 |
|--------|---------|-------------|---------|
| `network/antwar_net.py` | `network/` | `AntWarPolicyValueNetwork` | 主网络（独立 Policy/Value 编码器）；内含 `HexCNNEncoder`（六边形卷积）、`StructuredActionHead`（分层动作头）、`TowerDamageHead` / `GoldIncomeHead`（辅助任务预测头） |
| `utils/action_constants.py` | `utils/` | (常量定义) | 定义动作空间全部常量：`ACTION_DIM=96`、`TYPE_CONFIG`（9种动作类型的起止位置）、`TOWER_POSITIONS`、`REWARD_CONFIG`、`OBS_NORMALIZATION` 等；提供 `action_id_to_type_and_target()` 工具函数。**该文件与网络结构紧密耦合**，因为结构化动作头的 `TYPE_CONFIG` 决定了网络分层头的输出维度。 |
| `utils/obs_utils.py` | `utils/` | `observation_to_tensors()` | 将 numpy 观测字典转换为 PyTorch 张量，添加 batch 维度。**仅被 PPOAgent 和 PPOTrainer._select_action 调用**，是模型推理的前置步骤。 |

---

## 模块 3：算法训练 (Algorithm Training)

**业务目标**：实现 PPO-Clip 核心算法，包括多 epoch 训练、GAE 优势估计、双优化器（策略/价值）、NaN 安全保护、检查点管理。

| 源文件 | 归属目录 | 核心函数/类 | 功能描述 |
|--------|---------|-------------|---------|
| `trainer/ppo_trainer.py` | `trainer/` | `PPOTrainer` | PPO 训练核心：动作选择、GAE 计算、多 epoch minibatch 迭代、PPO-Clip 损失函数（策略+价值+熵+辅助任务）、双优化器梯度裁剪、学习率调度（余弦退火+warmup）、batch 质量校验 |
| `trainer/batch.py` | `trainer/` | `EpisodeBatch` | 单局轨迹数据容器，管理观测/动作/奖励/价值/log_prob/done 序列；提供 `to_tensors()` 和 `merge()` 方法 |
| `trainer/agent.py` | `trainer/` | `PPOAgent` | 智能体封装，对外提供 `act()` / `get_action_values()` / `get_value()` 接口。**被自对弈系统和基线对战评估共同使用** |
| `trainer/checkpoint_manager.py` | `trainer/` | `CheckpointManager` | 模型检查点的 save/load，保存策略权重、双优化器状态、训练配置 |
| `trainer/safety.py` | `trainer/` | `NanRecoveryHandler` | NaN/Inf 检测与恢复机制：跟踪 NaN 次数、维护健康检查点、超过阈值时回滚权重 |
| `utils/gae.py` | `utils/` | `compute_gae()` | 完整 GAE(λ) 算法实现，计算优势函数和折扣回报。**仅被 PPOTrainer 调用** |

---

## 模块 4：自对弈训练 (Self-Play Training)

**业务目标**：驱动完整的自对弈训练主循环，协调对手选择、数据收集、PPO 更新、对战评估、日志记录和状态持久化。

| 源文件 | 归属目录 | 核心函数/类 | 功能描述 |
|--------|---------|-------------|---------|
| `trainer/selfplay.py` | `trainer/` | `SelfPlayTrainer` | 自对弈主控制器：训练主循环、episode 数据收集（含先手/后手交替）、batch 累积与 PPO 更新触发、对手管理（创建/更新/选择）、基线对战评估触发、联赛状态持久化、CUDA 缓存管理 |
| `trainer/selfplay_logger.py` | `trainer/` | `SelfPlayLogger` | 自对弈专属分级别日志（sp_debug/info/warning/error/all.log），记录对战生命周期和对手变更事件 |
| `trainer/training_status.py` | `trainer/` | `compute_and_save_training_status()` | 收集并持久化训练状态快照：平均奖励/损失/胜率/训练速度等指标 |
| `monitor/selfplay_battle_writer.py` | `monitor/` | `SelfPlayBattleWriter` + `EpisodeRecord` | 每局自对弈对战数据（奖励、回合数、HP/Coins、动作统计等）写入 `selfplay_battle_log.jsonl`。**仅被 SelfPlayTrainer 调用** |
| `monitor/batch_metrics_writer.py` | `monitor/` | `BatchMetricsWriter` | 每次 PPO update 后将训练指标写入 `batch_metrics.jsonl`。**仅被 SelfPlayTrainer 调用** |
| `monitor/episode_batch_writer.py` | `monitor/` | `EpisodeBatchWriter` | 窗口满时从原始 JSONL 读取并聚合对战和训练指标，生成 episode batch 级别统计。**仅被 SelfPlayTrainer 调用** |
| `utils/aux_labels.py` | `utils/` | `compute_aux_labels_from_trajectory()` | 从对战轨迹快照计算辅助任务标签（多视界塔伤害、金币收入）。**仅被 SelfPlayTrainer 调用** |

---

## 模块 5：联赛系统 (League System)

**业务目标**：管理对手池的生命周期，基于 TrueSkill 评分实现自适应探索/利用对手选择策略。

| 源文件 | 归属目录 | 核心函数/类 | 功能描述 |
|--------|---------|-------------|---------|
| `league/opponent_selector.py` | `league/` | `SelfPlayManager` + `OpponentSelector` | 自对弈管理器：整合对手池、评分矩阵和选择策略；提供对手添加、选择、评分更新、状态持久化接口。`OpponentSelector` 实现探索/利用平衡的对手选择策略（sigmoid/exp/linear 调度）。 |
| `league/payoff.py` | `league/` | `BattleSharedPayoff` + `OpponentPool` | 对战评分核心：TrueSkill 评分管理与更新、贝叶斯胜率计算（含信度加权收缩）、对局衰减、可利用性评分。`OpponentPool` 负责对手的添加/淘汰（按 eviction score 淘汰最差对手）。 |

---

## 模块 6：基线对战评估 (Baseline Battle Evaluation)

**业务目标**：将已训练的 PPO 智能体与内置基线 AI（及外部实现）进行多场对战评估，聚合结果并生成报告。

| 源文件 | 归属目录 | 核心函数/类 | 功能描述 |
|--------|---------|-------------|---------|
| `battle/battle_coordinator.py` | `battle/` | `BattleCoordinator` | 对战协调入口，整合配置、智能体加载、模拟器、结果聚合和报告生成 |
| `battle/battle_simulator.py` | `battle/` | `BattleSimulator` | 多进程并行对战模拟（ProcessPoolExecutor），失败时降级为顺序执行 |
| `battle/agent_loader.py` | `battle/` | `AgentLoader` + `PPOWarAgent` + 内置 AI 类 | Agent 加载框架：PPO 智能体包装、内置基线 AI（`BasicRandomAI` / `BasicTowerAI` / `MediumRuleAI`）、外部 Agent 动态加载（importlib）、序列化/反序列化（cloudpickle） |
| `battle/battle_config.py` | `battle/` | `BaselineBattleConfig` | 对战参数配置 |
| `battle/battle_logger.py` | `battle/` | `BattleLogger` + `SubProcessLogger` | 对战会话日志：按级别分文件输出、对战统计计数、计时和超时日志 |
| `battle/opponent_agent.py` | `battle/` | `OpponentAgent` | 从 checkpoint 加载对手模型，提供统一的 `act()` 接口。**被自对弈系统（selfplay.py）和 battle_simulator 共同使用** |
| `battle/result_aggregator.py` | `battle/` | `aggregate_results()` | 聚合多场对战结果：胜负统计、先手/后手胜率、平均回合数 |
| `battle/report_generator.py` | `battle/` | `BattleReporter` | 生成可读的对战评估报告 |
| `monitor/eval_battle_writer.py` | `monitor/` | `EvalBattleWriter` | 评估对战数据写入 `eval_battle_log.jsonl`、回合级日志、baseline 胜率聚合。**仅被 BattleCoordinator 和 SelfPlayTrainer 调用** |
| `battle/utils/exception_logging.py` | `battle/utils/` | `log_exception()` | 对战异常记录与重新抛出 |
| `battle/utils/process_isolation.py` | `battle/utils/` | `process_isolation()` | 子进程模块隔离上下文管理器，确保子进程有干净的环境 |
| `battle/utils/timing.py` | `battle/utils/` | `BattleTimer` | 对战计时与超时警告 |
| *(internal)* `_execute_battle()` | `battle/battle_simulator.py` | 对战中实际执行单场对战的核心函数，调用 SDK 推进回合 | 内联于 battle_simulator.py |

---

## 模块 7：基础设施 (Infrastructure)

**业务目标**：提供与具体业务逻辑无关的跨切面服务，包括配置管理、路径管理、日志框架、时间追踪、系统监控、回调框架等。

> 本模块内的辅助代码虽然原目录分布在 `config/`、`monitor/`、`callbacks/` 等目录中，但它们本质上不服务于某个特定业务，而是为所有业务模块提供通用服务。

| 源文件 | 归属目录 | 核心函数/类 | 功能描述 |
|--------|---------|-------------|---------|
| `config/config_parser.py` | `config/` | `PPORLConfigParser` + `create_ppo_config()` | YAML 配置加载与 CLI 覆盖合并、类型校验和转换 |
| `config/path_config.py` | `config/` | `PathConfig` | 统一路径管理：项目根目录解析、输出目录（training/checkpoint/battle/selfplay/system/evaluation）自动创建与访问 |
| `monitor/logger.py` | `monitor/` | `Logger` | 训练主日志管理器：训练启动/完成日志、对战统计持久化、JSON 状态变化日志 |
| `monitor/constants.py` | `monitor/` | (常量定义) | 日志格式常量、采样间隔、缓存大小等全局配置常量 |
| `monitor/time_tracker.py` | `monitor/` | `TimeTracker` | 训练/对战时间追踪：Episode 耗时统计、每分钟训练速度、时间统计持久化 |
| `monitor/system_metrics_sampler.py` | `monitor/` | `SystemMetricsSampler` | 后台线程周期采样：CPU 使用率、内存用量、GPU util/显存/温度 |
| `monitor/metrics_cache.py` | `monitor/` | `MetricsCache` | 训练指标环形缓存，提供窗口统计查询 |
| `callbacks/base_ppo_callback.py` | `callbacks/` | `BasePPOCallback` | PPO 训练回调基类，定义训练生命周期钩子接口 |
| `callbacks/callback_factory.py` | `callbacks/` | `create_ppo_callbacks()` | 回调工厂函数，统一创建 MetricsLogging + Checkpoint + TensorBoard 回调列表 |
| `callbacks/metrics_callback.py` | `callbacks/` | `MetricsLoggingCallback` | 定期输出训练指标摘要到 loguru |
| `callbacks/checkpoint_callback.py` | `callbacks/` | `CheckpointCallback` | 定期保存模型检查点并清理旧检查点 |
| `callbacks/tensorboard_callback.py` | `callbacks/` | `TensorBoardCallback` | 将训练指标写入 TensorBoard 便于可视化监控 |

---

## 无法明确划分的代码

以下文件无法划分到任何业务模块，**单独标注**：

| 源文件 | 功能描述 | 备注 |
|--------|---------|------|
| `configs/ppo_antwar.yaml` | 默认 PPO 训练 YAML 配置文件 | 属于配置数据而非代码逻辑，是各模块的配置来源，不归属某个特定模块 |

---

## 模块依赖关系图（简要）

```
自对弈训练 (Self-Play Training)
   ├── 联赛系统 (League System)         ← 对手选择和评分
   ├── 算法训练 (Algorithm Training)     ← PPO 更新和 GAE
   ├── 游戏环境 (Game Environment)       ← 环境交互
   ├── 神经网络模型 (Neural Network)     ← 策略网络推理
   ├── 基线对战评估 (Battle Evaluation)  ← 周期性基线评估
   └── 基础设施 (Infrastructure)         ← 日志、时间追踪等

基线对战评估 (Battle Evaluation)
   ├── 游戏环境 (Game Environment)       ← 环境交互
   ├── 神经网络模型 (Neural Network)     ← PPOAgent 推理
   └── 基础设施 (Infrastructure)         ← 日志

算法训练 (Algorithm Training)
   ├── 神经网络模型 (Neural Network)     ← 前向/反向传播
   └── 基础设施 (Infrastructure)         ← 检查点管理

基础设施 (Infrastructure)
   └── (无其他模块依赖)
```

依赖方向：**自对弈训练** → 所有其他业务模块，**基础设施** 在最底层，不依赖任何业务模块。
