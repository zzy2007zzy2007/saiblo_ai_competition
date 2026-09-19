# SelfPlay 训练阶段日志文档

## 目录

1. [目录结构总览](#1-目录结构总览)
2. [训练文本日志 (training.log)](#2-训练文本日志-traininglog)
3. [训练指标数据 (training_metrics.json / training_metrics_history.jsonl)](#3-训练指标数据)
4. [训练状态快照 (training_status.json)](#4-训练状态快照-training_statusjson)
5. [逐局统计 (per_episode_stats.jsonl)](#5-逐局统计-per_episode_statsjsonl)
6. [权重统计 (weight_stats.jsonl)](#6-权重统计-weight_statsjsonl)
7. [异常与警告日志 (error.log / warning.log)](#7-异常与警告日志-errorlog--warninglog)
8. [系统资源采样 (system_metrics.json)](#8-系统资源采样-system_metricsjson)
9. [时间统计 (time_statistics.json)](#9-时间统计-time_statisticsjson)
10. [SelfPlay 对战详情 (selfplay_battle_stats.json + detailed_battles/)](#10-selfplay-对战详情)
11. [Baseline 对战日志](#11-baseline-对战日志)
12. [检查点 (checkpoint/)](#12-检查点-checkpoint)
13. [联赛状态 (league_state/)](#13-联赛状态-league_state)
14. [日志输出对应关系速查表](#14-日志输出对应关系速查表)
15. [日志工具使用](#15-日志工具使用)

---

## 1. 目录结构总览

训练输出目录为 `outputs/<timestamp>/`，子目录由 [path_config.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/config/path_config.py) 定义：

```
outputs/<timestamp>/
├── training/                          # 训练核心日志与指标
│   ├── training.log                   # 全量文本日志（loguru, DEBUG 级别）
│   ├── training.log.1, ...            # 日志轮转备份（100MB/个，30天保留）
│   ├── training_metrics.json          # 最新指标 + 统计摘要（覆盖写）
│   ├── training_metrics_history.jsonl # 全量指标历史（追加写，每轮 PPO update 一行）
│   ├── training_status.json           # 训练状态快照（覆盖写）
│   ├── per_episode_stats.jsonl        # 逐局统计（追加写）
│   ├── weight_stats.jsonl             # 权重统计（追加写）
│   ├── error.log                      # 异常记录（JSONL）
│   ├── warning.log                    # 警告记录（JSONL）
│   ├── training_start.log             # 训练启动信息
│   ├── training_details.log           # 训练详情
│   ├── training_complete.log          # 训练完成摘要
│   ├── model_saves.log                # 模型保存记录
│   ├── battle_results.log             # 对战结果
│   └── opponent_selection.log         # 对手选择记录
│
├── checkpoint/                        # 模型检查点
│   ├── checkpoint_*.pt                # 定期快照（每 100 次 PPO update）
│   ├── model_*.pt                     # 对手池 checkpoint（每 opponent_update_interval）
│   ├── initial_opponent_*.pt          # 随机初始对手
│   ├── final_model.pt                 # 训练结束后的最终模型
│   └── league_state/                  # 联赛状态
│       ├── opponent_pool.json         # 对手池
│       ├── payoff.json                # TrueSkill 评分
│       └── eviction_history.jsonl     # 对手淘汰历史
│
├── battle/                            # Baseline 对战输出
│   ├── battle_stats.json              # 对战统计汇总
│   └── <agent1>_vs_<agent2>_<ts>/    # 逐场对战目录
│       ├── battle_debug.log
│       ├── battle_info.log
│       ├── battle_warning.log
│       ├── battle_error.log
│       ├── battle_all.log
│       └── battle_results.json
│
├── selfplay/                          # SelfPlay 对战详情
│   └── selfplay_battles/
│       ├── selfplay_battle_stats.json # 按对手聚合的统计
│       └── detailed_battles/          # 逐场详情
│           └── <battle_id>.json
│
└── system/                            # 系统资源
    ├── system_metrics.json            # 最新采样 + 统计
    ├── system_metrics.json.log        # 全量采样历史（JSONL）
    └── time_statistics.json           # 时间统计
```

---

## 2. 训练文本日志 (training.log)

**文件路径**: `<training_dir>/training.log`

**写入方式**: loguru file handler，配置于 [logger.py:L107-L114](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/monitor/logger.py#L107-L114)

```
格式: {time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}
级别: DEBUG
轮转: 100 MB
保留: 30 天
```

### 2.1 包含的日志内容

此文件是**所有日志输出的集合**，包含：

| 来源 | 内容 |
|---|---|
| `selfplay.py` 的 logger.info/warning/error/debug | 训练进度、对手更新、baseline 评估等 |
| `ppo_trainer.py` 的 logger.info/warning/error | 网络初始化、PPO 更新、奖励统计、NaN/Inf 检测 |
| `logger.py` 的 log_exception/log_warning | 异常和警告的文本副本（带 `[ERROR]` / `[WARNING]` 前缀） |
| `path_config.py` | SDK/Baseline 路径验证 |

### 2.2 关键日志内容清单

#### selfplay.py (29 条)

| 级别 | 时机 | 内容 |
|---|---|---|
| `INFO` | 训练开始 | `"Starting self-play training for {num_episodes} episodes"` |
| `DEBUG` | 每个 episode 开始 | `"[Phase:SELFPLAY] Episode {episode}: ε=0"` |
| `WARNING` | 对手加载失败 | `"Checkpoint not found for opponent {opponent_id}: {checkpoint_path}"` |
| `WARNING` | 保存详情失败 | `"保存SelfPlay对战详情失败: {e}"` |
| `DEBUG` | 缓存刷盘 | `"SelfPlay对战缓存已保存并清空"` |
| `INFO` | 保存训练状态 | `"[_save_training_status] 开始/完成"` |
| `ERROR` | 训练状态异常 | `"⚠ 保存训练状态异常: {e}"` |
| `WARNING` | 批次无效 | `"[警告] 批次数据无效: {', '.join(issues)}"` |
| `INFO` | 新对手加入 | `"Added new opponent checkpoint: {checkpoint_path}"` |
| `INFO` | Baseline 评估 | `"Episode {episode}: 开始/完成 Baseline对战评估"` |
| `INFO` | Baseline 结果 | `"Episode {episode}: Baseline - Win:{}, Loss:{}, Draw:{}"` |
| `ERROR` | Baseline 失败 | `"Episode {episode}: 对战评估失败 - {e}"` |
| `DEBUG` | Payoff 衰减 | `"Applied history decay at episode {episode}"` |
| `INFO` | 初始对手创建 | `"Creating {n} initial opponents..."` / `"Created initial opponent: {id}"` |
| `ERROR` | 随机对手失败 | `"Failed to create random agent checkpoint: {e}"` |
| `INFO` | 联赛加载 | `"Loaded payoff state / opponent pool state"` |

#### ppo_trainer.py (重点 30+ 条)

| 级别 | 时机 | 内容 |
|---|---|---|
| `INFO` | 初始化 | `"Initializing PPOTrainer..."` / `"Device: {device}"` / `"Log directory: {log_dir}"` |
| `INFO` | 初始化 | `"Creating policy network..."` / `"Creating optimizer..."` / `"Initializing logging system..."` |
| `INFO` | 初始化完成 | `"PPOTrainer initialized successfully"` |
| `ERROR` | 初始化失败 | `"Failed to initialize PPOTrainer: {e}"` |
| `INFO` | PPO 更新 | `"[奖励统计] mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}"` (每 10 轮一次) |
| `WARNING` | 奖励异常 | `"[警告] 奖励统计异常: mean={}, std={}"` |
| `WARNING` | NaN/Inf | `"[警告] {name} 包含异常值: NaN={}, Inf={}"` |
| `WARNING` | NaN 次数 | `"[警告] 检测到第 {n} 次 NaN"` → 达到阈值后 `"[错误] 连续 {n} 次检测到 NaN，训练异常"` |
| `WARNING` | loss 异常 | `"[警告] loss 异常: {}，跳过此批次"` |
| `WARNING` | 梯度 NaN | `"[权重监控] 梯度 NaN/INF 首次出现: {name}, grad_norm={}"` |
| `WARNING` | 梯度过大 | `"[警告] 价值/策略网络参数 {name} 梯度范数过大: {}"` |
| `WARNING` | 总梯度报警 | `"[警告] 梯度范数 {} 远超过阈值 {}"` |
| `WARNING` | 权重 NaN | `"[权重监控] 权重 NaN 已写入: {name}, episode={}"` |
| `WARNING` | 观察 NaN | `"[警告] 观察输入包含 NaN: board_nan={}, global_nan={}"` |
| `WARNING` | 输出 NaN | `"[警告] 网络输出包含 NaN: action={}, log_prob={}, value={}"` |
| `INFO` | 探索 | `"[EXPLORE-DEBUG] explore#{}: mask_valid={}, sel_action={}, all_valid={}"` |
| `INFO` | 训练配置 | `"训练配置摘要"` |

---

## 3. 训练指标数据

### 3.1 training_metrics.json (最新 + 统计)

**文件路径**: `<training_dir>/training_metrics.json`

**格式**: JSON（覆盖写）

**结构**: `{"latest": {...}, "stats": {...}}`

**写入时机**: 每次 PPO update 后，[logger.py:L303-L310](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/monitor/logger.py#L303-L310)

#### latest 字段

| 字段名 | 类型 | 来源 | 说明 |
|---|---|---|---|
| `episode` | int | 训练循环 | PPO update 对应的 episode 编号 |
| `avg_reward` | float | episode_reward / N | 本批平均奖励 |
| `avg_loss` | float | loss / update_count | 本批平均 PPO loss |
| `avg_rounds` | float | episode_length_total / N | 本批平均回合数 |
| `steps` | int | trainer.total_steps | 累计总步数 |
| `timestamp` | str | time.strftime() | 记录时间 |
| `policy_loss` | float | ppo_metrics | PPO policy loss |
| `value_loss` | float | ppo_metrics | PPO value loss |
| `entropy` | float | ppo_metrics | 策略熵 |
| `entropy_type` | float | ppo_metrics | 类型熵 |
| `entropy_target` | float | ppo_metrics | 目标熵 |
| `entropy_coef` | float | trainer.current_ent_coef | 当前熵系数 |
| `learning_rate` | float | trainer.current_lr | 当前学习率 |
| `reward_mean` | float | ppo_metrics | 本批奖励均值 |
| `reward_std` | float | ppo_metrics | 本批奖励标准差 |
| `reward_min` | float | ppo_metrics | 本批奖励最小值 |
| `reward_max` | float | ppo_metrics | 本批奖励最大值 |
| `advantages_mean` | float | ppo_metrics | 优势函数均值 |
| `advantages_std` | float | ppo_metrics | 优势函数标准差 |
| `return_mean` | float | ppo_metrics | 回报均值 |
| `return_std` | float | ppo_metrics | 回报标准差 |
| `clip_fraction` | float | ppo_metrics | PPO clip 触发比例 |
| `gradient_norm` | float | ppo_metrics | 总梯度范数 |
| `hard_entropy_penalty` | float | ppo_metrics | 硬熵惩罚 |
| `value_input_mean` | float | ppo_metrics | 价值输入均值 |
| `value_input_std` | float | ppo_metrics | 价值输入标准差 |
| `value_pred_mean` | float | ppo_metrics | 价值预测均值 |
| `value_pred_std` | float | ppo_metrics | 价值预测标准差 |
| `ratio_mean` | float | ppo_metrics | 重要性采样 ratio 均值 |
| `ratio_std` | float | ppo_metrics | 重要性采样 ratio 标准差 |
| `grad_norm_policy` | float | ppo_metrics | 策略网络梯度范数 |
| `grad_norm_value` | float | ppo_metrics | 价值网络梯度范数 |
| `grad_norm_max_layer` | float | ppo_metrics | 最大单层梯度范数 |
| `nan_skip_count` | int | ppo_metrics | 本批跳过的 NaN 次数 |
| `valid_actions_mean` | float | ppo_metrics | 平均有效动作数 |
| `valid_actions_min` | float | ppo_metrics | 最小有效动作数 |
| `td_error_mean` | float | ppo_metrics | TD 误差均值 |
| `td_error_std` | float | ppo_metrics | TD 误差标准差 |
| `batch_size` | int | ppo_metrics | 本批总 timestep 数 |
| `num_collected` | int | batch_pool 长度 | 收集的 episode 数 |
| `avg_our_coins` | float | details_pool 汇总 | 我方平均金币 |
| `avg_enemy_coins` | float | details_pool 汇总 | 敌方平均金币 |
| `avg_our_cumulative_coins` | float | details_pool 汇总 | 我方累积金币 |
| `avg_enemy_cumulative_coins` | float | details_pool 汇总 | 敌方累积金币 |
| `rw_hp_attack` | float | details_pool 汇总 | HP 攻击奖励 |
| `rw_coin_gain` | float | details_pool 汇总 | 金币获取奖励 |
| `rw_tower_survival` | float | details_pool 汇总 | 塔生存奖励 |
| `rw_tech_bonus` | float | details_pool 汇总 | 科技奖励 |
| `rw_die_penalty` | float | details_pool 汇总 | 死亡惩罚 |
| `rw_end_reward` | float | details_pool 汇总 | 终局奖励/惩罚 |
| `rw_speed_bonus` | float | details_pool 汇总 | 效率奖励 |
| `coins_*` | float | details_pool 汇总 | 金币分桶比例（our/enemy × 4 桶） |
| `type_*` | float | ppo_metrics | 各动作类型的选择频率 |
| `logit_*` | float | ppo_metrics | 各动作类型的 logits 均值 |
| `prob_*` | float | ppo_metrics | 各动作类型的概率均值 |
| `actrw_*` | float | details_pool 汇总 | 各动作类型的 reward |
| `actcnt_*` | float | details_pool 汇总 | 各动作类型的执行次数 |
| `opponent_*` | float | details_pool 汇总 | 对手维度指标（胜率、奖励等） |

#### stats 字段 (history 的计算统计)

| 字段 | 说明 |
|---|---|
| `reward_mean` | 历史奖励均值 |
| `reward_std` | 历史奖励标准差 |
| `reward_max` | 历史奖励最大值 |
| `reward_min` | 历史奖励最小值 |
| `loss_mean` | 历史 loss 均值 |
| `avg_rounds_mean` | 历史平均回合数均值 |
| `entropy_mean` | 历史熵均值 |

### 3.2 training_metrics_history.jsonl (全量历史)

**文件路径**: `<training_dir>/training_metrics_history.jsonl`

**格式**: JSONL（追加写），每行一个完整的 metrics 字典

**写入时机**: 每次 PPO update 后，[logger.py:L315-L319](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/monitor/logger.py#L315-L319)

每行结构与 `training_metrics.json` 的 `latest` 字段相同（不含 stats）。

### 3.3 training_metrics.json.log (变更历史)

**文件路径**: `<training_dir>/training_metrics.json.log`

**格式**: JSONL

**写入时机**: 每次 `save_training_metrics` 时，通过 `log_json_state()` 写入

---

## 4. 训练状态快照 (training_status.json)

**文件路径**: `<training_dir>/training_status.json`

**格式**: JSON（覆盖写）

**写入时机**: 每 100 个 episode（由 `_save_training_status` 触发），[selfplay.py:L326-L419](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L326-L419)

```json
{
    "timestamp": "...",
    "episode": 320,
    "total_steps": 78400,
    "best_avg_reward": 2841.06,
    "avg_reward_last_100": 2500.0,
    "avg_reward_last_10": 2450.0,
    "avg_loss_last_100": 0.15,
    "avg_rounds": 245.0,
    "training_speed": 1.5,
    "total_wins": 200,
    "total_losses": 180,
    "total_draws": 20,
    "win_rate": 0.5,
    "ppo_metrics": {
        "policy_loss": 0.15,
        "value_loss": 0.85,
        "entropy": 0.28
    },
    "training_stability": {
        "reward_std": 240.0,
        "loss_std": 0.03
    }
}
```

---

## 5. 逐局统计 (per_episode_stats.jsonl)

**文件路径**: `<training_dir>/per_episode_stats.jsonl`

**格式**: JSONL（追加写），每局一行

**写入时机**: 每 N 局 PPO update 后，[selfplay.py:L1083-L1088](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L1083-L1088)

**构建位置**: [selfplay.py:L815-L823](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L815-L823)

```json
{
    "episode": 320,
    "reward": 2418.31,
    "length": 245,
    "opponent_id": "opponent_003",
    "result": "win",
    "rounds": 245,
    "swap": false
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `episode` | int | 全局 episode 编号 |
| `reward` | float | 本局总奖励 |
| `length` | int | 本局步数 |
| `opponent_id` | str | 对手 ID（或 "none"） |
| `result` | str | "win" / "loss" / "draw" / "unknown" |
| `rounds` | int | 本局回合数 |
| `swap` | bool | 是否交换先手 |

---

## 6. 权重统计 (weight_stats.jsonl)

**文件路径**: `<training_dir>/weight_stats.jsonl`

**格式**: JSONL（追加写）

**写入时机**: 每次 PPO update 后，[selfplay.py:L1064-L1081](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L1064-L1081)

统计 policy 网络中每层的参数 norm：

```json
{
    "episode": 100,
    "timestamp": "2026-05-28 18:20:00",
    "weights": {
        "encoder.conv1.weight": {"mean": 0.01, "std": 0.05, "norm": 1.23},
        "encoder.conv2.weight": {"mean": -0.01, "std": 0.04, "norm": 1.15},
        ...
    }
}
```

---

## 7. 异常与警告日志

### 7.1 error.log (JSONL)

**文件路径**: `<training_dir>/error.log`

**写入时机**: `loguru_logger.exception()` 调用时，[logger.py:L155-L186](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/monitor/logger.py#L155-L186)

```json
{"timestamp": "2026-05-28 18:20:00", "exception_type": "RuntimeError", "message": "CUDA out of memory", "context": {"episode": 100}}
```

每行同时以 `[ERROR] ` 前缀追加到 `training.log`。

### 7.2 warning.log (JSONL)

**文件路径**: `<training_dir>/warning.log`

**写入时机**: `loguru_logger.warning()` 调用时，[logger.py:L189-L217](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/monitor/logger.py#L189-L217)

```json
{"timestamp": "2026-05-28 18:20:00", "message": "梯度 NaN/INF", "context": {"episode": 100}}
```

每行同时以 `[WARNING] ` 前缀追加到 `training.log`。

---

## 8. 系统资源采样

### 8.1 system_metrics.json

**文件路径**: `<system_dir>/system_metrics.json`

**写入时机**: 每 6 个采样点（约 60 秒），由 `SystemMetricsSampler` 后台线程执行

**采样间隔**: 10 秒

```json
{
    "latest": {
        "timestamp": "2026-05-28 18:20:00",
        "phase": "training",
        "cpu_percent": 45.2,
        "ram_percent": 32.1,
        "ram_used_gb": 8.2,
        "ram_total_gb": 25.5,
        "gpu_util": 85.0,
        "gpu_mem_used_mb": 8192,
        "gpu_mem_total_mb": 24576,
        "gpu_temp": 65
    },
    "latest_samples": [...],  // 最近 10 个采样点
    "stats": {
        "cpu_avg": 45.0,
        "cpu_max": 78.3,
        "ram_avg": 32.0,
        "ram_max": 35.1,
        "gpu_avg": 82.5,
        "gpu_max": 100.0,
        "training_cpu_avg": 44.8,
        "training_gpu_avg": 83.1,
        "battle_cpu_avg": 50.2,
        "battle_gpu_avg": 80.0
    },
    "sample_count": 120
}
```

### 8.2 system_metrics.json.log

**文件路径**: `<system_dir>/system_metrics.json.log`

**格式**: JSONL，每个采样点一行，全量历史。

---

## 9. 时间统计 (time_statistics.json)

**文件路径**: `<system_dir>/time_statistics.json`

**写入时机**: 每 100 个 episode，[selfplay.py:L1062](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L1062)

```json
{
    "timestamp": "2026-05-28 18:20:00",
    "training_start": "2026-05-28 18:18:00",
    "training_time_total": 120.5,
    "battle_time_total": 0.0,
    "training_episodes_count": 100,
    "battle_evaluations_count": 5,
    "last_training_start_time": "2026-05-28 18:19:45",
    "last_training_end_time": "2026-05-28 18:20:00",
    "last_battle_start_time": "",
    "last_battle_end_time": "",
    "last_battle_duration": 0.0
}
```

---

## 10. SelfPlay 对战详情

### 10.1 selfplay_battle_stats.json

**文件路径**: `<selfplay_dir>/selfplay_battles/selfplay_battle_stats.json`

**写入时机**: 每个 episode 后，[selfplay.py:L288-L324](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L288-L324)

**格式**: JSON，按 opponent_id 聚合的累计统计：

```json
{
    "opponent_003": {
        "wins": 50,
        "losses": 30,
        "draws": 5,
        "avg_reward": 2450.0,
        "avg_rounds": 245.0,
        "recent_rewards": [2418.3, 2300.6, ...],
        "recent_results": ["win", "loss", ...],
        "recent_opponents": ["opponent_003", ...]
    }
}
```

### 10.2 detailed_battles/<battle_id>.json

**文件路径**: `<selfplay_dir>/selfplay_battles/detailed_battles/<battle_id>.json`

**写入时机**: 同上

**格式**: JSON，逐场完整对战记录：

```json
{
    "battle_id": "opponent_003_20260528_182000_001",
    "episode": 320,
    "opponent_id": "opponent_003",
    "player_position": 0,
    "start_time": "2026-05-28 18:20:00",
    "end_time": "2026-05-28 18:20:15",
    "duration": 15.2,
    "result": "win",
    "total_reward": 2418.31,
    "rounds": 245,
    "our_cumulative_coins": 896.0,
    "enemy_cumulative_coins": 956.9,
    "first_move_wins": 1,
    "first_move_losses": 0,
    "second_move_wins": 0,
    "second_move_losses": 0,
    "round_details": [
        {
            "step": 0,
            "round": 1,
            "our_hp": 100,
            "enemy_hp": 100,
            "our_coins": 50,
            "enemy_coins": 50,
            "our_cumulative_coins": 50,
            "enemy_cumulative_coins": 50,
            "our_towers": [...],
            "enemy_towers": [...],
            "our_ants": 0,
            "enemy_ants": 0,
            "our_ant_kinds": {},
            "enemy_ant_kinds": {},
            "our_tech": {},
            "enemy_tech": {}
        }
    ],
    "actions": [
        {
            "step": 0,
            "action": 15,
            "opponent_action": 0,
            "log_prob": -1.23,
            "value": 0.12,
            "reward": 0.0,
            "reward_detail": {},
            "mask_valid": 30,
            "type_probs": [0.1, 0.2, ...]
        }
    ],
    "action_counts": {0: 210, 15: 32, 16: 2},
    "final_hp": {
        "our_hp": 100,
        "enemy_hp": 0
    }
}
```

---

## 11. Baseline 对战日志

### 11.1 battle_stats.json

**文件路径**: `<battle_dir>/battle_stats.json`

```json
{
    "BasicRandomAI": {
        "wins": 123,
        "losses": 5,
        "draws": 0,
        "total_battles": 128,
        "win_rate": 0.96
    },
    "BasicTowerAI": { ... },
    "MediumRuleAI": { ... },
    "sample": { ... }
}
```

### 11.2 逐场对战目录

**路径**: `<battle_dir>/<agent1>_vs_<agent2>_<timestamp>/`

每个 baseline 对战创建一个独立目录，包含：

| 文件 | 说明 |
|---|---|
| `battle_debug.log` | DEBUG 级别日志 |
| `battle_info.log` | INFO 级别日志 |
| `battle_warning.log` | WARNING 级别日志 |
| `battle_error.log` | ERROR 级别日志 |
| `battle_all.log` | 全量日志 |
| `battle_results.json` | 逐场对战结果 (JSONL) |

---

## 12. 检查点 (checkpoint/)

### 12.1 文件类型

| 命名模式 | 触发条件 | 内容 |
|---|---|---|
| `checkpoint_{episode_count}.pt` | `episode_count % 100 == 0` 且 `update_count == 1`（PPO update 内部） | policy_state_dict + optimizer_state_dict + value_optimizer_state_dict + episode_count + total_steps + config |
| `model_{episode}.pt` | `episode % opponent_update_interval == 0`（默认 40） | 同上，用于加入对手池 |
| `initial_opponent_{uuid}.pt` | 训练开始时创建初始对手 | 同上，随机权重 |
| `final_model.pt` | 训练结束后 | 同上，最终模型 |

### 12.2 Checkpoint 管理

- CheckpointCallback 保留最近 5 个 checkpoint，自动清理旧文件
- 每次 PPO update 保存的快照在 [ppo_trainer.py:L912-L914](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L912-L914)

---

## 13. 联赛状态 (league_state/)

### 13.1 opponent_pool.json

**文件路径**: `<checkpoint_dir>/league_state/opponent_pool.json`

对手池状态，包含所有对手的 ID、checkpoint 路径、加入时间和对战次数。

### 13.2 payoff.json

**文件路径**: `<checkpoint_dir>/league_state/payoff.json`

TrueSkill 评分矩阵，包含所有对手的 mu、sigma 和 pairwise 胜率。

### 13.3 eviction_history.jsonl

**文件路径**: `<checkpoint_dir>/league_state/eviction_history.jsonl`

对手淘汰历史，每行记录被淘汰对手的信息和淘汰原因。

---

## 14. 日志输出对应关系速查表

### 按文件分类

| 文件 | 写入源 | 格式 | 写入模式 | 写入频率 |
|---|---|---|---|---|
| `training.log` | loguru file handler | 文本 | 追加 | 每条日志一行 |
| `training_metrics.json` | Logger.save_training_metrics | JSON | 覆盖 | 每次 PPO update |
| `training_metrics_history.jsonl` | Logger.save_training_metrics | JSONL | 追加 | 每次 PPO update |
| `training_status.json` | Logger.save_training_status | JSON | 覆盖 | 每 100 episode |
| `per_episode_stats.jsonl` | SelfPlayTrainer._flush_per_episode_stats | JSONL | 追加 | 每次 PPO update |
| `weight_stats.jsonl` | SelfPlayTrainer._save_weight_stats | JSONL | 追加 | 每次 PPO update |
| `error.log` | Logger.log_exception | JSONL | 追加 | 每次异常 |
| `warning.log` | Logger.log_warning | JSONL | 追加 | 每次警告 |
| `system_metrics.json` | SystemMetricsSampler | JSON | 覆盖 | 每 60 秒 |
| `system_metrics.json.log` | SystemMetricsSampler | JSONL | 追加 | 每 10 秒 |
| `time_statistics.json` | TimeTracker | JSON | 覆盖 | 每 100 episode |
| `selfplay_battle_stats.json` | SelfPlayTrainer._flush_selfplay_battle_cache | JSON | 覆盖 | 每个 episode |
| `detailed_battles/*.json` | SelfPlayTrainer._flush_selfplay_battle_cache | JSON | 新建 | 每个 episode |
| `battle_stats.json` | BattleLogger | JSON | 覆盖 | 每次 baseline 评估 |
| `checkpoint_*.pt` | PPOTrainer.save_checkpoint | PyTorch | 新建 | 每 100 PPO update |
| `model_*.pt` | PPOTrainer.save_checkpoint | PyTorch | 新建 | 每 opponent_update_interval |
| `opponent_pool.json` | OpponentPool.save_state | JSON | 覆盖 | intermittent |
| `payoff.json` | BattleSharedPayoff.save_state | JSON | 覆盖 | intermittent |
| `eviction_history.jsonl` | OpponentPool._evict | JSONL | 追加 | 对手淘汰时 |

### 按功能分类

| 功能 | 涉及文件 |
|---|---|
| **训练进度监控** | `training.log`, `training_metrics.json`, `training_metrics_history.jsonl`, `training_status.json` |
| **逐局详情** | `per_episode_stats.jsonl` |
| **权重分析** | `weight_stats.jsonl`, `checkpoint_*.pt` |
| **异常诊断** | `error.log`, `warning.log`, `training.log` |
| **系统资源** | `system_metrics.json`, `system_metrics.json.log` |
| **时间分析** | `time_statistics.json` |
| **SelfPlay 对战** | `selfplay_battle_stats.json`, `detailed_battles/*.json` |
| **Baseline 评估** | `battle_stats.json`, `battle/*/battle_*.log` |
| **联赛管理** | `opponent_pool.json`, `payoff.json`, `eviction_history.jsonl` |
| **模型保存** | `checkpoint_*.pt`, `model_*.pt`, `final_model.pt` |

---

## 15. 日志工具使用

### 15.1 tools/manager.sh 命令

| 命令 | 读取的文件 | 说明 |
|---|---|---|
| `train_metrics_table` | `training_metrics.json` | SelfPlay 技术参数表 |
| `train_action_table` | `training_metrics_history.jsonl` | 每局各动作执行次数统计 |
| `train_action_reward_table` | `training_metrics_history.jsonl` | 各类 Action 的 Reward 统计 |
| `train_battle_table` | `battle/` 目录 | Baseline 对战横表 |
| `train_opponent_table` | `checkpoint/league_state/` | 对手池统计 |
| `scan_anomalies` | `training_metrics_history.jsonl` + `training.log` | NaN/ratio/梯度异常扫描 |
| `train_report` | `training_metrics.json` + `battle/` | 完整训练报告 |
| `train_progress_check` | 多个文件（聚合脚本） | 综合训练进度 |
| `train_status` | 远程 screen + 文件检查 | 训练运行状态 |

### 15.2 使用示例

```bash
# 查看 SelfPlay 技术参数表
bash tools/manager.sh --server server9 --model ppo_v1 train_metrics_table

# 查看动作统计表
bash tools/manager.sh --server server9 --model ppo_v1 train_action_table

# 查看 Action Reward 统计
bash tools/manager.sh --server server9 --model ppo_v1 train_action_reward_table

# 查看 Baseline 对战横表
bash tools/manager.sh --server server9 --model ppo_v1 train_battle_table

# 扫描训练异常
bash tools/manager.sh --server server9 --model ppo_v1 scan_anomalies

# 查看完整训练报告
bash tools/manager.sh --server server9 --model ppo_v1 train_report
```
