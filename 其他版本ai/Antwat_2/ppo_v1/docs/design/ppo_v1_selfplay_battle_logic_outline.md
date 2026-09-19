# PPO v1 — SelfPlay 与 Baseline Battle 对战逻辑详解（大纲）

> 基于 `ppo/src/ppo_antwar` 源码分析，结合 `docs/design/antwar_rules.md`、`docs/design/antwar_rules_review.md` 和 Ant-Game SDK（`Ant-Game/SDK/`）。

---

## 1. 总体架构概览

### 1.1 两大对战通路
- SelfPlay 通路：目的、频率、核心类
- Baseline Battle 通路：目的、频率、核心类
- 对比表格（对战类型 | 目的 | 频率 | 核心类 | 环境方式）

### 1.2 关键代码位置
- `ppo/src/ppo_antwar/` 完整目录树
- 核心源文件说明

### 1.3 核心类关系图
- PPOTrainer → SelfPlayTrainer → SelfPlayManager
- BattleCoordinator → BattleSimulator → run_single_battle_process

---

## 2. 游戏基础：环境与回合机制

### 2.1 先手/后手机制（重要）
- 两个 step 构成一个完整游戏回合
- step 序号与 round_index 的变化关系
- MAX_ROUND = 512：实际最多 256 个完整回合
- 参考：`docs/design/antwar_rules_review.md` §1.3

### 2.2 SelfPlay 使用的环境方式
- env_factory 创建环境（外部注入）
- 双 step 模式：`env.step(player_0_action)` → `env.step(player_1_action)`
- 奖励只在后手 step 后发放
- 参考源码：`trainer/selfplay.py` `_collect_episode_with_swap()`

### 2.3 Baseline Battle 使用的环境方式
- 直接操作 `PythonBackendState`
- 单刀模式：`state.resolve_turn(player0_ops, player1_ops)`
- 使用 `load_backend(prefer_native=False)` 加载后端
- 参考源码：`battle/battle_simulator.py` `_execute_battle()`

### 2.4 BackendAdapter — 兼容适配层
- `compat/adapters.py` 中的 `BackendAdapter` 和 `MatchRuntimeWrapper`
- 提供旧版 `resolve_turn` API 兼容

---

## 3. SelfPlay 训练流程

### 3.1 整体训练循环
- `SelfPlayTrainer.train()` 主循环（`trainer/selfplay.py`）
- 8 个步骤的流程图

### 3.2 对手选择机制（OpponentSelector）
- exploit vs explore 概率
- 自适应 exploit_prob 线性增长（0.3 → 0.9）
- confident 判断（sigma < 5.0）
- 参考源码：`league/opponent_selector.py`

### 3.3 先后手交替（Swap Positions）
- `swap_positions = (battle_count % 2 == 1)`
- 偶数局 Agent 先手、奇数局 Agent 后手

### 3.4 单局对战数据收集（_collect_episode_with_swap）
- 观察获取与 swap 的关系
- Agent 动作选择（`trainer._select_action(player_obs)`）
- 对手动作选择（agent.act / 随机合法动作）
- 双 step 执行顺序
- 奖励获取与 swap 的关系
- 结束条件（terminated / truncated）
- 结果判定（基于 final HP 比较）
- 参考源码：`trainer/selfplay.py`

### 3.5 PPO 更新
- GAE 计算（`_compute_gae`）
- 多 epoch 小批量更新
- Policy Loss + Value Loss + Entropy Bonus
- 梯度裁剪（policy 和 value 分别裁剪）
- NaN 鲁棒性机制
- 参考源码：`trainer/ppo_trainer.py`

### 3.6 学习率与熵退火
- 预热阶段（前 1000 个 episode 线性增长）
- 可选余弦退火
- 熵退火公式

### 3.7 对手池管理（OpponentPool）
- 最大池大小、最小对战场次阈值
- 添加对手（每 opponent_update_interval 个 episode）
- 淘汰策略（mu 最低 + 场次保护）
- 初始对手创建
- 持久化（保存/加载 league state）
- 参考源码：`league/opponent_selector.py`、`trainer/selfplay.py`

### 3.8 Payoff 更新
- `SelfPlayManager.update_payoff()` 
- TrueSkill `rate_1vs1()` 更新
- 正向/反向战绩记录
- 参考源码：`league/payoff.py`

---

## 4. Baseline Battle 流程

### 4.1 触发时机
- SelfPlay 训练循环中触发
- BattleEvalCallback 中触发
- 频率：`battle_interval`（默认 1000 个 episode）

### 4.2 对战次数计算
- `num_episodes = n_battles // 2`
- 每 episode 包含先手 + 后手两局
- 默认：2 episodes × 2 先后手 = 4 场实际对局

### 4.3 BattleCoordinator 协调流程
- `run_ppo_evaluation_with_agent()` 完整流程
- 遍历 baseline_agents
- 对每个 baseline，创建 logger → 加载 agent → 创建 simulator → 并行对战
- 参考源码：`battle/battle_coordinator.py`

### 4.4 BattleSimulator — 并行对战引擎
- `run_battles_parallel()`：多进程并行架构
- cloudpickle 序列化 agent
- ProcessPoolExecutor 并行执行
- 对局编号规则（episode vs episode+1000 区分先后手）
- 参考源码：`battle/battle_simulator.py`

### 4.5 子进程对战逻辑（run_single_battle_process）
- 进程隔离：清理 sys.modules、恢复 sys.path
- SDK 加载：`load_backend()` → `PythonBackendState`
- cloudpickle 反序列化 agent
- 调用 `_execute_battle()`
- 异常处理和日志记录

### 4.6 _execute_battle() — 单局对战核心
- 状态初始化：`backend.initial_state(seed)` → `python_state_class(game_state)`
- 回合循环：获取双方操作 → `state.resolve_turn(player0_ops, player1_ops)`
- 操作获取：`_get_operations(agent, state, player)`
- 每回合记录 HP、coins、operations
- 结果判定：`_determine_result(state, first_player)`

### 4.7 结果判定（_determine_result）
- 优先使用 `state.winner`
- 降级方案：比较 HP
- HP 相等 → draw
- 结果类型：`agent1_win` / `agent2_win` / `draw` / `error`
- 参考源码：`battle/battle_simulator.py`

### 4.8 降级策略
- 多进程失败 → 自动降级为顺序执行 `_run_battles_sequential()`

### 4.9 结果聚合（ResultAggregator）
- 聚合字段：total_battles、wins、draws、先后手胜率、avg_rounds 等
- 参考源码：`battle/result_aggregator.py`

### 4.10 报告生成（ReportGenerator）
- 总体统计 + 对局记录 + 先后手胜率
- 参考源码：`battle/report_generator.py`

---

## 5. 动作空间与 Action Mask

### 5.1 动作空间结构（96 维）
- action_id 范围：0 ~ 95
- NO_OP（0）
- BUILD_TOWER（1-10）
- UPGRADE_TOWER（11-15，5 tower × 3 direction）
- DOWNGRADE_TOWER（16-20）
- 4 种超级武器（各 5 个位置，21-40）
- 科技升级（41-42）
- 保留/未使用（43-95）
- 参考源码：`utils/action_constants.py`

### 5.2 Action ID 与 Operation 映射
- `ACTION_OFFSETS` 字典
- `ACTION_SPACE_CONFIG` 字典
- 位置映射：`TOWER_POSITIONS`、`SUPER_WEAPON_POSITIONS`

### 5.3 Action Mask 的使用方式
- observation 中的 `action_mask` 字段
- 网络前向时用于屏蔽非法动作（mask 值 0 → 对应 logit 设为 -inf）
- NO_OP 始终合法（mask[0] = 1.0）

---

## 6. Observation 编码（观察空间）

### 6.1 Board 特征（28 × 19 × 19）
- 参考 `ppo_antwar/__init__.py` 中引用 `ObservationEncoder`
- 28 个 channel 的详细列表
- 参考：`Ant-Game/SDK/utils/features.py`（实际特征编码逻辑在此）

### 6.2 Global 特征（33 维）
- 33 个标量特征的索引表
- 参考：`Ant-Game/SDK/utils/features.py`

### 6.3 综合观察结构
- `{'board': np.ndarray(28, 19, 19), 'global': np.ndarray(33,), 'action_mask': np.ndarray(96,)}`

---

## 7. 奖励函数

### 7.1 奖励来源
- 奖励来自 env（由 Ant-Game SDK 提供）
- `env.step()` 返回的 rewards 字典

### 7.2 终端奖励
- 胜利：+10.0
- 失败：-10.0

### 7.3 增量奖励
- HP 变化、金币变化、建塔、蚂蚁数量、蚂蚁推进进度
- 权重参数

### 7.4 NO_OP 惩罚
- NO_OP_PENALTY = -0.3

### 7.5 奖励裁剪
- 范围限制

### 7.6 参考
- `Ant-Game/SDK/training/env.py`（奖励计算逻辑）

---

## 8. PPO Agent 决策流程

### 8.1 PPOAgent 类
- `act(observation, deterministic=False)` — 训练中使用
- `select_action(obs)` — 评估中使用（deterministic=True）
- `evaluate(observation)` — 返回概率分布和 value
- 参考源码：`trainer/ppo_trainer.py` L899-952

### 8.2 _select_action 内部流程
- observation → tensor → device
- `policy.get_action(board, global_obs, action_mask, deterministic)`
- NaN 检测和清理
- 返回 (action, log_prob, value)
- 参考源码：`trainer/ppo_trainer.py` L628-660

### 8.3 网络前向流程
- Board: CNN 编码
- Global: MLP 编码
- 合并 → Policy Head（logits） + Value Head（scalar）
- Softmax + action mask → 概率分布
- 参考源码：`network/antwar_net.py`

### 8.4 Deterministic vs Sampling
- deterministic=True: `argmax(probs)`，评估/Baseline Battle 中使用
- deterministic=False: `Categorical(probs).sample()`，训练中使用

### 8.5 PPO 更新流程（补充）
- GAE 计算 advantages 和 returns
- Policy Loss: Clipped PPO surrogate objective
- Value Loss: Value Clipping 版 MSE
- Entropy Bonus: `current_ent_coef * entropy_loss`
- 梯度分别裁剪（policy max_grad_norm=0.5, vf max_grad_norm_vf=0.2）
- 参考源码：`trainer/ppo_trainer.py` L662-823

---

## 9. 对手管理与 League 机制

### 9.1 OpponentSelector — 对手选择
- exploit 模式（默认 0.7）：选 TrueSkill mu 最高的 confident 对手
- explore 模式（默认 0.3）：随机选 sigma 不太高的对手
- 自适应调度：exploit_prob 随训练进度从 0.3 线性增长到 0.9
- 参考源码：`league/opponent_selector.py`

### 9.2 OpponentPool — 对手池
- 最大池大小：10
- 最小对战场次阈值：8
- 添加对手流程
- 淘汰策略（mu 最低 + 低场次保护加成）
- 持久化：save_state / load_state
- 参考源码：`league/opponent_selector.py`

### 9.3 TrueSkill 评分系统
- `TrueSkillRating`: mu, sigma, games_played
- `rate_1vs1()` 更新
- confident 判断：sigma < 5.0
- exploitability 计算
- 参考源码：`league/payoff.py`

### 9.4 BattleSharedPayoff — 战绩记录
- 双向战绩记录（home-away 和 away-home）
- 胜率计算（带 confidence weight）
- 历史衰减（decay=0.99）
- 持久化
- 参考源码：`league/payoff.py`

### 9.5 SelfPlayManager — 整合管理
- 组合 OpponentPool + Payoff + OpponentSelector
- 完整的对手生命周期管理
- 统计信息聚合
- 参考源码：`trainer/selfplay.py`

---

## 10. 日志与报告系统

### 10.1 BattleLogger
- 对战日志记录器（主进程）
- 多进程安全的实现方式
- 日志级别枚举（LogLevel）
- 参考源码：`battle/battle_logger.py`

### 10.2 SubProcessLogger
- 子进程日志记录器
- 与 BattleLogger 的交互方式
- 参考源码：`battle/battle_logger.py`

### 10.3 SelfPlayLogger
- SelfPlay 专用日志
- 记录每回合状态快照
- 记录对战结果和统计
- 参考源码：`trainer/selfplay_logger.py`

### 10.4 ReportGenerator
- 报告格式：总体统计、对局记录、先后手胜率
- 参考源码：`battle/report_generator.py`

### 10.5 ResultAggregator
- 聚合逻辑：total_battles、wins、draws、先后手分离
- 参考源码：`battle/result_aggregator.py`

### 10.6 日志系统工具模块
- `battle/utils/exception_logging.py`：异常日志记录
- `battle/utils/process_isolation.py`：进程隔离装饰器
- `battle/utils/timing.py`：计时工具

---

## 11. 关键配置参数速查

### 11.1 PPO 参数
- lr, gamma, gae_lambda, clip_eps, ent_coef, vf_coef 等
- 参考：`configs/ppo_antwar.yaml` `ppo:` 段

### 11.2 训练参数
- total_episodes, battle_interval, n_battles, opponent_update_interval 等
- 参考：`configs/ppo_antwar.yaml` `training:` 段

### 11.3 SelfPlay 参数
- opponent_pool_size, exploit_prob, explore_prob 等
- 参考：`configs/ppo_antwar.yaml` `selfplay:` 段

### 11.4 网络参数
- hidden_dim, board_shape, global_dim, action_dim
- 参考：`configs/ppo_antwar.yaml` `network:` 段

### 11.5 BaselineBattleConfig 参数
- max_rounds, n_battles, log_level, timing_warning_threshold 等
- 参考：`battle/battle_config.py`

### 11.6 数值安全边界
- logits_min/max, values_min/max, returns_min/max 等
- 参考：`configs/ppo_antwar.yaml` `clip_values:` 段
