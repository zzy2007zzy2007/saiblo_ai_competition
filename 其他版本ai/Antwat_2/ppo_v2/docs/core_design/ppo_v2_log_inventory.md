# PPO v2 日志文件体系盘点

> 自动生成于 2026-06-11，基于 `ppo_v2/src/ppo_antwar/` 代码库全量扫描。

---

## 目录结构概览

所有日志文件基于 `PathConfig` 管理，根目录为 `outputs/{run_id}/`，按子目录分类：

```
outputs/{run_id}/
├── training/          # 训练主日志
├── checkpoint/        # 模型检查点
├── selfplay/          # 自对弈日志 + 联赛状态
│   └── league/        # 对手池 + 评分矩阵
├── evaluation/        # 基线评估日志
└── system/            # 系统指标 + 时间统计
```

---

## 一、训练主日志（training/）

### 1.1 `training_{time}.log`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/training_{time}.log` |
| **格式** | 纯文本，loguru 格式：`{time} \| {level} \| {name}:{function}:{line} - {message}` |
| **输出频率** | 持续（loguru handler 实时写入） |
| **触发条件** | 所有 loguru INFO 及以上级别消息 |
| **写入代码位置** | `trainer/selfplay.py:159-165` |
| **Rotation** | 100 MB |
| **Retention** | 30 天 |

**内容说明**：训练过程的全局日志，包含 PPO 更新摘要、学习率变化、检查点保存、异常警告等所有 INFO+ 级别消息。EpisodeCollector 的 `[ROUND]` 逐回合日志也写入此文件。

---

### 1.2 `training_error_{time}.log`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/training_error_{time}.log` |
| **格式** | 纯文本，同 training log 格式 |
| **输出频率** | 持续（loguru handler 实时写入） |
| **触发条件** | 所有 loguru ERROR 及以上级别消息 |
| **写入代码位置** | `trainer/selfplay.py:166-172` |
| **Rotation** | 100 MB |
| **Retention** | 90 天 |

**内容说明**：仅记录错误级别消息，用于快速定位训练异常（如梯度 NaN、对手加载失败、对战评估异常等）。

---

### 1.3 `training_start.log`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/training_start.log` |
| **格式** | 纯文本，key-value 逐行 |
| **输出频率** | 一次（训练启动时） |
| **触发条件** | `SelfPlayTrainer.train()` 开始时调用 `Logger.log_training_start()` |
| **写入代码位置** | `monitor/logger.py:27-57` |

**内容字段**：

| Key | 说明 |
|-----|------|
| Training started at | 训练启动时间 |
| Run ID | 运行标识 |
| Episodes | 总 episode 数 |
| Batch size | PPO 批次大小 |
| Learning rate | 初始学习率 |
| Entropy coef | 熵系数 |
| Gamma | 折扣因子 |
| GAE lambda | GAE 参数 |
| Clip epsilon | PPO clip 范围 |
| n_envs | 并行环境数 |
| Save interval | 检查点保存间隔 |
| Opponent update interval | 对手更新间隔 |
| Network hidden_size | 网络隐藏层维度 |
| Network num_layers | 网络层数 |
| Device | 计算设备 |
| PyTorch version | PyTorch 版本 |
| System | 系统平台信息 |
| CPU cores | CPU 核心数 |
| PID | 进程 ID |

---

### 1.4 `training_complete.log`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/training_complete.log` |
| **格式** | 纯文本，key-value 逐行 |
| **输出频率** | 一次（训练结束时） |
| **触发条件** | `SelfPlayTrainer._finalize_training()` 调用 `Logger.log_training_complete()` |
| **写入代码位置** | `monitor/logger.py:59-78` |

**内容字段**：

| Key | 说明 |
|-----|------|
| Total episodes | 实际训练 episode 数 |
| Total steps | 总步数 |
| Total time | 总训练时间 |
| Best avg reward | 最佳平均奖励 |
| Best avg reward episode | 最佳奖励对应 episode |
| Final win rate | 最终胜率 |
| Final avg reward | 最终平均奖励 |
| Final avg loss | 最终平均损失 |
| Win rate vs {opponent} | 对各对手的胜率（动态） |
| Completed at | 完成时间 |

---

### 1.5 `battle_stats.json`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/battle_stats.json` |
| **格式** | JSON（覆盖写入） |
| **输出频率** | 周期性（默认每 50 episode） |
| **触发条件** | `SelfPlayTrainer._run_periodic_tasks()` 中 `episode_count % status_save_interval == 0` |
| **写入代码位置** | `monitor/logger.py:85-87`，调用点 `trainer/selfplay.py:648` |

**内容字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| wins | int | 累计胜场 |
| losses | int | 累计负场 |
| draws | int | 累计平局 |

---

### 1.6 `*.json.log`（通用 JSON 状态日志）

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/{filename}.json.log` |
| **格式** | JSONL（每行一个 JSON 对象） |
| **输出频率** | 按需 |
| **触发条件** | 调用 `Logger.log_json_state(filename, data, context)` |
| **写入代码位置** | `monitor/logger.py:91-104` |

**每条记录字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| timestamp | str | 写入时间 |
| data | dict | 状态数据 |
| context | dict/null | 附加上下文 |

> **注意**：当前代码库中此接口已定义但未被主动调用，为预留的通用状态变更日志接口。

---

### 1.7 `selfplay_battle_log.jsonl`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/selfplay_battle_log.jsonl` |
| **格式** | JSONL（每局一行） |
| **输出频率** | 每局对战结束 |
| **触发条件** | `SelfPlayTrainer._collect_and_update_payoff()` 中每局结束后调用 `SelfPlayBattleWriter.write_episode()` |
| **写入代码位置** | `monitor/selfplay_battle_writer.py:20-23`，调用点 `trainer/selfplay.py:368-392` |

**每条记录字段**（来自 `EpisodeRecord.to_json()`，`monitor/episode_record.py:33-63`）：

| Key | 类型 | 说明 |
|-----|------|------|
| episode | int | Episode 编号 |
| swap | bool | 是否后手 |
| opponent_id | str | 对手 ID |
| result | str | 对战结果（win/loss/draw） |
| rounds | int | 回合数 |
| reward | float | 总奖励（4 位小数） |
| length | int | 步数 |
| start_time | float | 开始时间戳 |
| end_time | float | 结束时间戳 |
| duration | float | 持续时间秒（3 位小数） |
| end_own_hp | float | 终局己方塔 HP 总和 |
| end_own_coins | float | 终局己方金币 |
| max_own_coins | float | 己方最高金币 |
| total_own_coin_income | float | 己方总金币收入 |
| end_enemy_hp | float | 终局敌方塔 HP 总和 |
| end_enemy_coins | float | 终局敌方金币 |
| max_enemy_coins | float | 敌方最高金币 |
| total_enemy_coin_income | float | 敌方总金币收入 |
| action_counts | dict | 动作类型计数 `{type_name: count}` |
| action_rewards | dict | 动作类型奖励 `{type_name: reward}` |
| reward_sources | dict | 奖励来源分解 `{source_key: value}` |

**reward_sources 子字段**：

| Key | 说明 |
|-----|------|
| rw_hp_attack_base | 基地伤害奖励 |
| rw_hp_attack_tower | 塔伤害奖励 |
| rw_coin_gain | 金币获取奖励 |
| rw_tower_survival | 塔存活奖励 |
| rw_balance | 平衡奖励 |
| rw_tech_bonus | 科技奖励 |
| rw_die_penalty | 死亡惩罚 |
| rw_end_reward | 终局奖励 |

---

### 1.8 `batch_metrics.jsonl`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/batch_metrics.jsonl` |
| **格式** | JSONL（每批次一行） |
| **输出频率** | 每次 PPO update 后 |
| **触发条件** | `SelfPlayTrainer._flush_batch_to_update()` 中调用 `BatchMetricsWriter.write_batch()` |
| **写入代码位置** | `monitor/batch_metrics_writer.py:34-56`，调用点 `trainer/selfplay.py:476` |

**每条记录字段**（由 `MetricsSchema` 定义，`trainer/metrics_schema.py`，仅列出 `log_jsonl=True` 的字段）：

#### 训练损失类

| Key | 类型 | 精度 | 说明 |
|-----|------|------|------|
| policy_loss | float | 6 | PPO 策略 clip loss |
| value_loss | float | 6 | PPO 价值 loss |
| entropy | float | 6 | 策略熵（type + target） |
| entropy_type | float | 6 | 类型选择熵 |
| entropy_target | float | 6 | 目标选择熵 |
| clip_fraction | float | 6 | PPO clip 比例 |
| approx_kl | float | 6 | 近似 KL 散度 |

#### 辅助任务损失

| Key | 类型 | 精度 | 说明 |
|-----|------|------|------|
| aux_tower_loss | float | 6 | 己方塔伤害辅助损失 |
| aux_gold_loss | float | 6 | 己方金币辅助损失 |
| aux_enemy_tower_loss | float | 6 | 敌方塔伤害辅助损失 |
| aux_enemy_gold_loss | float | 6 | 敌方金币辅助损失 |
| aux_base_loss | float | 6 | 己方基地伤害辅助损失 |
| aux_enemy_base_loss | float | 6 | 敌方基地伤害辅助损失 |

#### 梯度类

| Key | 类型 | 精度 | 说明 |
|-----|------|------|------|
| grad_norm | float | 6 | 总梯度范数 |
| grad_norm_policy | float | 6 | 策略网络梯度范数 |
| grad_norm_value | float | 6 | 价值网络梯度范数 |
| grad_norm_max_layer | float | 6 | 最大单层梯度范数 |

#### 奖励/价值统计

| Key | 类型 | 精度 | 说明 |
|-----|------|------|------|
| reward_mean | float | 4 | 批次平均奖励 |
| reward_std | float | 4 | 批次奖励标准差 |
| reward_min | float | 4 | 批次最小奖励 |
| reward_max | float | 4 | 批次最大奖励 |
| return_mean | float | 4 | 批次平均回报 |
| return_std | float | 4 | 批次回报标准差 |
| advantages_std | float | 4 | 优势标准差 |

#### Value 统计

| Key | 类型 | 精度 | 说明 |
|-----|------|------|------|
| value_input_mean | float | 6 | 输入价值均值 |
| value_input_std | float | 6 | 输入价值标准差 |
| value_pred_mean | float | 6 | 预测价值均值 |
| value_pred_std | float | 6 | 预测价值标准差 |
| ratio_mean | float | 6 | 重要性采样比率均值 |
| ratio_std | float | 6 | 重要性采样比率标准差 |
| td_error_mean | float | 6 | TD 误差均值 |
| td_error_std | float | 6 | TD 误差标准差 |

#### Action 统计

| Key | 类型 | 精度 | 说明 |
|-----|------|------|------|
| valid_actions_mean | float | 6 | 合法动作数均值 |
| valid_actions_min | float | 6 | 合法动作数最小值 |
| batch_size | int | - | 批次大小 |
| num_collected | int | - | 收集的 episode 数 |
| nan_skip_count | int | - | 因 NaN 跳过的 minibatch 数 |

#### 超参类

| Key | 类型 | 精度 | 说明 |
|-----|------|------|------|
| learning_rate | float | 原始值 | 当前策略学习率 |
| entropy_coef | float | 原始值 | 当前熵系数 |

#### 修正后的 loss 指标

| Key | 类型 | 精度 | 说明 |
|-----|------|------|------|
| training_loss | float | 6 | 总训练 loss = policy_loss + vf_coef*value_loss - ent_coef*entropy |
| loss | float | 6 | 旧版 loss（policy_loss + value_loss） |

#### 动态附加字段

| Key | 类型 | 说明 |
|-----|------|------|
| type_ratios | dict | 各动作类型占比 `{type_name_ratio: value}` |
| type_probs | dict | 各动作类型概率 `{type_name_prob: value}` |

---

### 1.9 `episode_batch_battle_stats.jsonl`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/episode_batch_battle_stats.jsonl` |
| **格式** | JSONL（每窗口一行） |
| **输出频率** | 每 `window_size`（默认 8）局后 |
| **触发条件** | `EpisodeBatchWriter.on_batch_complete()` 中 `episode_count_in_window >= window_size` |
| **写入代码位置** | `monitor/episode_batch_writer.py:76-87` |

**每条记录字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| episode_start | int | 窗口起始 episode |
| episode_end | int | 窗口结束 episode |
| num_episodes | int | 窗口内 episode 数 |
| win_rate | float | 窗口内胜率 |
| draw_rate | float | 窗口内平局率 |
| avg_reward | float | 窗口内平均奖励 |
| avg_rounds | float | 窗口内平均回合数 |
| avg_end_own_hp | float | 平均终局己方 HP |
| avg_end_enemy_hp | float | 平均终局敌方 HP |
| avg_end_own_coins | float | 平均终局己方金币 |
| avg_max_own_coins | float | 平均最高己方金币 |
| avg_total_own_coin_income | float | 平均己方金币收入 |
| avg_end_enemy_coins | float | 平均终局敌方金币 |
| avg_max_enemy_coins | float | 平均最高敌方金币 |
| avg_total_enemy_coin_income | float | 平均敌方金币收入 |
| avg_action_counts | dict | 平均动作计数 |
| avg_action_rewards | dict | 平均动作奖励 |
| avg_rw_hp_attack_base | float | 平均基地伤害奖励 |
| avg_rw_hp_attack_tower | float | 平均塔伤害奖励 |
| avg_rw_coin_gain | float | 平均金币获取奖励 |
| avg_rw_tower_survival | float | 平均塔存活奖励 |
| avg_rw_balance | float | 平均平衡奖励 |
| avg_rw_tech_bonus | float | 平均科技奖励 |
| avg_rw_die_penalty | float | 平均死亡惩罚 |
| avg_rw_end_reward | float | 平均终局奖励 |

---

### 1.10 `episode_batch_train_stats.jsonl`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/training/episode_batch_train_stats.jsonl` |
| **格式** | JSONL（每窗口一行） |
| **输出频率** | 每 `window_size`（默认 8）局后 |
| **触发条件** | 与 `episode_batch_battle_stats.jsonl` 同步写入 |
| **写入代码位置** | `monitor/episode_batch_writer.py:89-100` |

**每条记录字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| episode_start | int | 窗口起始 episode |
| episode_end | int | 窗口结束 episode |
| num_batches | int | 窗口内 PPO update 次数 |
| avg_policy_loss | float | 平均策略损失 |
| avg_value_loss | float | 平均价值损失 |
| avg_aux_tower_loss | float | 平均塔伤害辅助损失 |
| avg_aux_gold_loss | float | 平均金币辅助损失 |
| avg_aux_enemy_tower_loss | float | 平均敌方塔伤害辅助损失 |
| avg_aux_enemy_gold_loss | float | 平均敌方金币辅助损失 |
| avg_aux_base_loss | float | 平均基地伤害辅助损失 |
| avg_aux_enemy_base_loss | float | 平均敌方基地伤害辅助损失 |
| avg_entropy | float | 平均策略熵 |
| avg_clip_fraction | float | 平均 clip 比例 |
| avg_grad_norm | float | 平均梯度范数 |
| avg_learning_rate | float | 平均学习率 |
| avg_reward_mean | float | 平均批次奖励均值 |
| avg_return_mean | float | 平均批次回报均值 |

---

## 二、自对弈日志（selfplay/）

### 2.1 `sp_all.log`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/selfplay/sp_all.log` |
| **格式** | 纯文本，loguru 格式：`{time} \| {level} \| [selfplay] \| {message}` |
| **输出频率** | 事件驱动 |
| **触发条件** | 对战开始/结束、对手添加/淘汰 |
| **写入代码位置** | `trainer/selfplay_logger.py:25-30`（handler 注册），`selfplay_logger.py:38-61`（事件写入） |

**事件类型**：

| 事件 | 格式 | 触发位置 |
|------|------|----------|
| 对战开始 | `START episode={ep} vs={opp_id} swap={swap}` | `trainer/selfplay.py:574-576` |
| 对战结束 | `END episode={ep} vs={opp_id} result={result} reward={reward} rounds={rounds}` | `trainer/selfplay.py:594-600` |
| 对手添加 | `OPPONENT_ADDED episode={ep} id={opp_id} pool_size={size}` | `trainer/selfplay.py:780-782` |
| 对手淘汰 | `OPPONENT_EVICTED id={id} mu={mu} reason={reason}` | 当前代码中未调用（接口已预留） |

---

### 2.2 `league/pool.json`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/selfplay/league/pool.json` |
| **格式** | JSON（覆盖写入） |
| **输出频率** | 周期性（每 `save_interval * 5` episode）+ 训练结束 |
| **触发条件** | `SelfPlayTrainer._save_league_state()` → `SelfPlayManager.save_state()` → `OpponentPool.save_state()` |
| **写入代码位置** | `league/opponent_pool.py:99-108`，调用链 `trainer/selfplay.py:870-881` |

**内容字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| opponents | list[str] | 对手 ID 列表（按添加顺序） |
| checkpoint_paths | dict | 对手 ID → 检查点路径映射 |
| games_played | dict | 对手 ID → 对战场次映射 |
| added_episodes | dict | 对手 ID → 添加时 episode 映射 |

---

### 2.3 `league/payoff.json`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/selfplay/league/payoff.json` |
| **格式** | JSON（覆盖写入） |
| **输出频率** | 同 `pool.json` |
| **触发条件** | 同 `pool.json` |
| **写入代码位置** | `league/battle_shared_payoff.py:188-201` |

**内容字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| players | list[str] | 所有玩家 ID 列表 |
| players_ids | dict | 玩家 ID → 索引映射 |
| data | dict | 对战记录 `{home-away: {wins, draws, losses, games}}` |
| trueskill_ratings | dict | TrueSkill 评分 `{pid: {mu, sigma, games_played}}` |

---

### 2.4 `opponent_checkpoints/opponent_ep{episode}.pt`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/selfplay/opponent_checkpoints/opponent_ep{episode}.pt` |
| **格式** | PyTorch 二进制检查点 |
| **输出频率** | 周期性（每 `opponent_update_interval` episode） |
| **触发条件** | `SelfPlayTrainer._update_opponent()` → `PPOTrainer.save_checkpoint()` → `CheckpointManager.save()` |
| **写入代码位置** | `trainer/checkpoint_manager.py:13-31`，调用点 `trainer/selfplay.py:774` |

**内容字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| episode | int | 保存时的 episode 编号 |
| policy_state_dict | dict | 策略网络权重 |
| optimizer_state_dict | dict | 策略优化器状态 |
| value_optimizer_state_dict | dict | 价值优化器状态 |
| metrics | dict | 保存时的指标 |
| config | dict | 训练配置 |

---

### 2.5 `opponent_checkpoints/initial_opponent_{uuid}.pt`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/selfplay/opponent_checkpoints/initial_opponent_{uuid}.pt` |
| **格式** | PyTorch 二进制检查点 |
| **输出频率** | 一次（训练启动时，仅当对手池为空） |
| **触发条件** | `SelfPlayTrainer._create_initial_opponents()` |
| **写入代码位置** | `trainer/selfplay.py:739-757` |

**内容字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| policy_state_dict | dict | 随机初始化策略网络权重 |
| config | dict | 训练配置 |

> 默认创建 3 个初始对手（`INITIAL_OPPONENT_COUNT = 3`）。

---

## 三、基线评估日志（evaluation/）

### 3.1 `eval_battle_log.jsonl`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/evaluation/eval_battle_log.jsonl` |
| **格式** | JSONL（每局一行） |
| **输出频率** | 每局基线评估对战 |
| **触发条件** | `SelfPlayTrainer._log_battle_evaluation_results()` → `EvalBattleWriter.write_battle()` |
| **写入代码位置** | `monitor/eval_battle_writer.py:56-91`，调用点 `trainer/selfplay.py:846-856` |

**每条记录字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| episode | int | 评估时的 episode 编号 |
| baseline_agent_name | str | 基线智能体名称 |
| battle_mode | str | 对战模式（默认 "selfplay_eval"） |
| swap | bool | 是否后手 |
| opponent_id | str | 对手 ID |
| result | str | 对战结果（win/loss/draw） |
| rounds | int | 回合数 |
| reward | float | 奖励（4 位小数） |
| length | int | 步数 |
| start_time | float | 开始时间戳 |
| end_time | float | 结束时间戳 |
| duration | float | 持续时间（3 位小数） |
| **extra | any | 附加字段（通过 **kwargs 传入） |

---

### 3.2 `eval_battle_round_log_{timestamp}.log`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/evaluation/eval_battle_round_log_{YYYYMMDD_HHMMSS}.log` |
| **格式** | 纯文本，loguru 格式：`{time} \| {level} \| [EVAL] \| {message}` |
| **输出频率** | 每回合 |
| **触发条件** | 基线评估对战中每回合调用 `EvalBattleWriter.write_round()` |
| **写入代码位置** | `monitor/eval_battle_writer.py:40-54`（handler 注册），`eval_battle_writer.py:96-109`（写入），调用点 `battle/battle_coordinator.py:115-147` |

**每条消息字段**（以 `key=value` 格式嵌入消息文本）：

| Key | 说明 |
|-----|------|
| episode | Episode 编号 |
| round_num | 回合编号 |
| baseline | 基线智能体名称 |
| battle_idx | 对战索引 |
| time | 时间戳 |
| agent1_name | PPO agent 名称 |
| agent2_name | 基线 agent 名称 |
| agent1_ops | PPO 操作列表（紧凑格式） |
| agent2_ops | 基线操作列表（紧凑格式） |
| agent1_hp / agent2_hp | 操作后基地 HP |
| agent1_hp_before / agent2_hp_before | 操作前基地 HP |
| agent1_hp_dmg / agent2_hp_dmg | 基地伤害 |
| agent1_twr_hp_before/after/dmg | 己方塔 HP 变化 |
| agent2_twr_hp_before/after/dmg | 敌方塔 HP 变化 |
| agent1_coins / agent2_coins | 操作后金币 |
| agent1_coins_before / agent2_coins_before | 操作前金币 |
| agent1_ops_cost / agent2_ops_cost | 操作花费 |
| agent1_coins_earned / agent2_coins_earned | 金币收入 |
| agent1_towers / agent2_towers | 塔数量 |

---

### 3.3 `episode_batch_baseline_winrates.jsonl`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/evaluation/episode_batch_baseline_winrates.jsonl` |
| **格式** | JSONL（每窗口一行） |
| **输出频率** | 每 EpisodeBatch 窗口边界 |
| **触发条件** | `SelfPlayTrainer._flush_batch_to_update()` 中 `episode_count % batch_window == 0` 时调用 `EvalBattleWriter.on_episode_batch_complete()` |
| **写入代码位置** | `monitor/eval_battle_writer.py:111-175`，调用点 `trainer/selfplay.py:487-491` |

**每条记录字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| episode_start | int | 窗口起始 episode |
| episode_end | int | 窗口结束 episode |
| num_episodes | int | 窗口内评估局数 |
| baseline_winrates | dict | 各基线智能体的胜率统计 |

**baseline_winrates 子字段**（每个基线智能体）：

| Key | 类型 | 说明 |
|-----|------|------|
| battles | int | 对战场次 |
| wins | int | 胜场 |
| losses | int | 负场 |
| draws | int | 平局 |
| win_rate | float | 胜率（4 位小数） |
| avg_rounds | float | 平均回合数（1 位小数） |
| avg_reward | float | 平均奖励（4 位小数） |

---

## 四、系统指标日志（system/）

### 4.1 `system_metrics.json`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/system/system_metrics.json` |
| **格式** | JSON（覆盖写入） |
| **输出频率** | 每 `SAMPLES_PER_WRITE`（默认 6）次采样后 |
| **触发条件** | `SystemMetricsSampler._maybe_flush()` 中采样数达到阈值 |
| **写入代码位置** | `monitor/system_metrics_sampler.py:129-149` |

**内容字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| latest | dict | 最新一次采样数据 |
| latest_samples | list[dict] | 最近 10 次采样数据 |
| stats | dict | 全局统计 |
| sample_count | int | 总采样数 |

**latest / latest_samples 中每条采样字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| timestamp | str | 采样时间 |
| phase | str | 当前阶段（training/battle） |
| cpu_percent | float | CPU 使用率 |
| ram_percent | float | 内存使用率 |
| ram_used_gb | float | 内存使用量 GB |
| ram_total_gb | float | 内存总量 GB |
| gpu_util | float | GPU 使用率（-1 表示不可用） |
| gpu_mem_used_mb | int | GPU 显存使用量 MB（-1 表示不可用） |
| gpu_mem_total_mb | int | GPU 显存总量 MB（-1 表示不可用） |
| gpu_temp | int | GPU 温度（-1 表示不可用） |

**stats 字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| cpu_avg | float | CPU 平均使用率 |
| cpu_max | float | CPU 最大使用率 |
| ram_avg | float | 内存平均使用率 |
| ram_max | float | 内存最大使用率 |
| gpu_avg | float? | GPU 平均使用率（有 GPU 时） |
| gpu_max | float? | GPU 最大使用率（有 GPU 时） |
| training_cpu_avg | float? | 训练阶段 CPU 平均使用率 |
| battle_cpu_avg | float? | 对战阶段 CPU 平均使用率 |

---

### 4.2 `system_metrics.json.log`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/system/system_metrics.json.log` |
| **格式** | JSONL（追加写入，每次 flush 写入最近 `SAMPLES_PER_WRITE` 条） |
| **输出频率** | 同 `system_metrics.json` |
| **触发条件** | 同 `system_metrics.json` |
| **写入代码位置** | `monitor/system_metrics_sampler.py:151-153` |

**每条记录字段**：同 `system_metrics.json` 中 `latest` 的采样字段。

---

### 4.3 `time_statistics.json`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/system/time_statistics.json` |
| **格式** | JSON（覆盖写入） |
| **输出频率** | 周期性（默认每 50 episode）+ 训练结束 |
| **触发条件** | `SelfPlayTrainer._run_periodic_tasks()` 和 `_finalize_training()` 中调用 `TimeTracker.save_time_statistics()` |
| **写入代码位置** | `monitor/time_tracker.py:81-95`，调用点 `trainer/selfplay.py:646-647,694-695` |

**内容字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| training_episodes_count | int | 训练 episode 总数 |
| training_time_total | float | 训练总时间（秒） |
| battle_evaluations_count | int | 基线评估次数 |
| battle_time_total | float | 基线评估总时间（秒） |
| last_training_start_time | str/null | 最近训练开始时间 |
| last_training_end_time | str/null | 最近训练结束时间 |
| last_battle_start_time | str/null | 最近评估开始时间 |
| last_battle_end_time | str/null | 最近评估结束时间 |
| last_battle_duration | float | 最近评估持续时间 |
| timestamp | str | 保存时间 |

---

## 五、检查点目录（checkpoint/）

### 5.1 `checkpoint_ep{episode}.pt`

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `outputs/{run_id}/checkpoint/checkpoint_ep{episode}.pt` |
| **格式** | PyTorch 二进制检查点 |
| **输出频率** | 周期性（每 `save_interval` episode） |
| **触发条件** | `CheckpointCallback._on_step()` → `PPOTrainer.save_checkpoint()` → `CheckpointManager.save()` |
| **写入代码位置** | `callbacks/checkpoint_callback.py:31-43` |
| **清理策略** | 仅保留最近 `keep_last_n`（默认 5）个检查点 |

**内容字段**：

| Key | 类型 | 说明 |
|-----|------|------|
| episode | int | 保存时的 episode 编号 |
| policy_state_dict | dict | 策略网络权重 |
| optimizer_state_dict | dict | 策略优化器状态 |
| value_optimizer_state_dict | dict | 价值优化器状态 |
| metrics | dict | 保存时的指标 |
| config | dict | 训练配置 |

---

## 六、TensorBoard 日志

### 6.1 TensorBoard 事件文件

| 属性 | 说明 |
|------|------|
| **文件路径模式** | `{log_dir}/events.out.tfevents.*`（由 `SummaryWriter` 自动生成） |
| **格式** | TensorBoard 二进制事件文件 |
| **输出频率** | 每 `log_interval`（默认 10）步 |
| **触发条件** | `TensorBoardCallback._on_step()` → `SummaryWriter.add_scalar()` |
| **写入代码位置** | `callbacks/tensorboard_callback.py:29-46` |

**记录字段**（由 `MetricsSchema.get_tensorboard_keys()` 过滤，仅 `log_tensorboard=True` 的字段）：

排除的字段（`log_tensorboard=False`）：entropy_type, entropy_target, approx_kl, grad_norm_max_layer, reward_min, reward_max, value_input_mean/std, value_pred_mean/std, ratio_mean/std, td_error_mean/std, valid_actions_mean/min, batch_size, num_collected, nan_skip_count, loss (legacy)。

包含的主要字段：policy_loss, value_loss, entropy, clip_fraction, aux_tower_loss, aux_gold_loss, aux_enemy_tower_loss, aux_enemy_gold_loss, aux_base_loss, aux_enemy_base_loss, grad_norm, grad_norm_policy, grad_norm_value, reward_mean, reward_std, return_mean, return_std, advantages_std, training_loss, learning_rate, entropy_coef。

---

## 七、日志关联关系

### 7.1 数据流向图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        训练主循环 (SelfPlayTrainer)                  │
│                                                                     │
│  每局结束 ──┬── SelfPlayBattleWriter ──→ selfplay_battle_log.jsonl  │
│             ├── EpisodeBatchWriter.on_episode_complete()            │
│             │   (battle_buffer 累积)                                 │
│             └── SelfPlayLogger ──→ sp_all.log                       │
│                                                                     │
│  每次PPO更新 ──┬── BatchMetricsWriter ──→ batch_metrics.jsonl        │
│               └── EpisodeBatchWriter.on_batch_complete()            │
│                   (batch_buffer 累积)                                │
│                                                                     │
│  窗口满时 ──── EpisodeBatchWriter._flush_*()                        │
│               ├──→ episode_batch_battle_stats.jsonl                 │
│               └──→ episode_batch_train_stats.jsonl                  │
│                                                                     │
│  基线评估 ────┬── EvalBattleWriter.write_battle()                   │
│              │   ──→ eval_battle_log.jsonl                          │
│              ├── EvalBattleWriter.write_round()                     │
│              │   ──→ eval_battle_round_log_{ts}.log                 │
│              └── EvalBattleWriter.on_episode_batch_complete()       │
│                  ──→ episode_batch_baseline_winrates.jsonl          │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 核心聚合关系

| 源日志 | 聚合日志 | 聚合方式 | 聚合器 |
|--------|----------|----------|--------|
| `selfplay_battle_log.jsonl` 的同源数据 → `battle_buffer` | `episode_batch_battle_stats.jsonl` | 窗口内均值聚合 | `EpisodeBatchWriter._aggregate_battle()` |
| `batch_metrics.jsonl` 的同源数据 → `batch_buffer` | `episode_batch_train_stats.jsonl` | 窗口内均值聚合 | `EpisodeBatchWriter._aggregate_train()` |
| `eval_battle_log.jsonl` 的同源数据 → `eval_buffer` | `episode_batch_baseline_winrates.jsonl` | 按基线智能体分组统计 | `EvalBattleWriter.on_episode_batch_complete()` |

> **关键设计**：EpisodeBatchWriter 使用内存环形缓冲区（`_battle_buffer` / `_batch_buffer`）替代从 JSONL 文件回读，避免性能随训练时间退化。数据在写入 JSONL 的同时存入内存缓冲，窗口满时从缓冲聚合。

### 7.3 日志并行关系

| 日志 A | 日志 B | 关系说明 |
|--------|--------|----------|
| `selfplay_battle_log.jsonl` | `sp_all.log` | 同一局对战的结构化数据 vs 人类可读文本，数据源相同但格式和详细程度不同 |
| `training_{time}.log` | `training_error_{time}.log` | 全量 INFO+ vs 仅 ERROR+，后者是前者的子集 |
| `system_metrics.json` | `system_metrics.json.log` | 快照+统计 vs 历史采样流，同时写入 |
| `pool.json` | `payoff.json` | 联赛状态的两部分，必须成对使用 |
| `battle_stats.json` | `selfplay_battle_log.jsonl` | battle_stats.json 是当前累计统计（覆盖写入），selfplay_battle_log.jsonl 是逐局明细（追加） |

### 7.4 触发时序

```
训练启动
  ├──→ training_start.log (一次)
  ├──→ initial_opponent_*.pt (一次，仅池为空时)
  └──→ system_metrics 采样开始

每局结束 (episode++)
  ├──→ selfplay_battle_log.jsonl (追加)
  ├──→ sp_all.log (END 事件)
  ├──→ EpisodeBatchWriter.on_episode_complete() (缓冲)
  └──→ _flush_batch_to_update() (满足条件时)
       ├──→ batch_metrics.jsonl (追加)
       ├──→ EpisodeBatchWriter.on_batch_complete() (缓冲)
       └──→ [窗口满时]
            ├──→ episode_batch_battle_stats.jsonl (追加)
            ├──→ episode_batch_train_stats.jsonl (追加)
            └──→ episode_batch_baseline_winrates.jsonl (追加)

周期性任务
  ├── 每 opponent_update_interval 局
  │   └──→ opponent_ep{ep}.pt
  ├── 每 battle_interval 局
  │   ├──→ eval_battle_log.jsonl (追加)
  │   └──→ eval_battle_round_log_{ts}.log (追加)
  ├── 每 status_save_interval 局 (默认 50)
  │   ├──→ battle_stats.json (覆盖)
  │   └──→ time_statistics.json (覆盖)
  ├── 每 save_interval*5 局
  │   ├──→ pool.json (覆盖)
  │   └──→ payoff.json (覆盖)
  └── 每 save_interval 局 (via CheckpointCallback)
      └──→ checkpoint_ep{ep}.pt

训练结束
  ├──→ episode_batch_battle_stats.jsonl (flush 残留窗口)
  ├──→ episode_batch_train_stats.jsonl (flush 残留窗口)
  ├──→ pool.json + payoff.json (最终保存)
  ├──→ time_statistics.json (最终保存)
  └──→ training_complete.log (一次)
```

---

## 八、配置常量汇总

| 常量 | 默认值 | 定义位置 | 说明 |
|------|--------|----------|------|
| SAMPLE_INTERVAL | 10 秒 | `monitor/constants.py:20` | 系统指标采样间隔 |
| SAMPLES_PER_WRITE | 6 | `monitor/constants.py:21` | 每次写入的采样数 |
| TRAINING_DETAIL_LOG_INTERVAL | 1 | `monitor/constants.py:24` | 训练详情日志间隔 |
| TRAINING_STATUS_SAVE_INTERVAL | 50 | `monitor/constants.py:25` | 训练状态保存间隔 |
| RECENT_METRICS_WINDOW | 10 | `monitor/constants.py:27` | 近期指标窗口 |
| HISTORICAL_METRICS_WINDOW | 100 | `monitor/constants.py:28` | 历史指标窗口 |
| METRICS_CACHE_SIZE | 100 | `monitor/constants.py:18` | 指标缓存大小 |
| episode_batch_window | n_envs (默认 8) | `trainer/selfplay.py:186-189` | EpisodeBatch 聚合窗口大小 |
| keep_last_n | 5 | `callbacks/checkpoint_callback.py:23` | 保留的检查点数量 |
| LEAGUE_SAVE_INTERVAL_MULTIPLIER | 5 | `trainer/selfplay.py:91` | 联赛保存间隔倍数 |
| INITIAL_OPPONENT_COUNT | 3 | `trainer/selfplay.py:92` | 初始对手数量 |
| CUDA_CACHE_CLEAN_INTERVAL | 3 | `trainer/selfplay.py:90` | CUDA 缓存清理间隔 |
| LR_LOG_INTERVAL | 500 | `trainer/ppo_trainer.py:48` | 学习率日志间隔 |

---

## 九、日志子系统架构

所有日志组件通过 `LoggingSubsystem`（`trainer/logging_subsystem.py`）统一管理生命周期：

```
LoggingSubsystem
├── loguru_handler_id: int          # training_{time}.log 的 handler
├── time_tracker: TimeTracker       # 时间追踪
├── logger: Logger                  # training_start/complete, battle_stats, json.log
├── selfplay_logger: SelfPlayLogger # sp_all.log
├── battle_logger: SelfPlayBattleWriter   # selfplay_battle_log.jsonl
├── batch_logger: BatchMetricsWriter      # batch_metrics.jsonl
├── ep_batch_logger: EpisodeBatchWriter   # episode_batch_*_stats.jsonl
├── eval_logger: EvalBattleWriter         # eval_battle_log/round_log/baseline_winrates
└── system_sampler: SystemMetricsSampler  # system_metrics.*
```

关闭顺序（`close_all()`）：battle_logger → batch_logger → ep_batch_logger → eval_logger → selfplay_logger → system_sampler → loguru handler。
