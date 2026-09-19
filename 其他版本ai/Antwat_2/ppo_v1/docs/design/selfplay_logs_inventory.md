# SelfPlay 日志清单

> 梳理所有 SelfPlay 训练输出的日志文件、格式、路径、写入方式和数据颗粒度。

---

## 目录结构总览

```
{output_dir}/
├── training/                          ← 训练主日志目录
│   ├── training_metrics.json          ← 最近一条训练指标
│   ├── training_metrics_history.jsonl ← 历史训练指标（核心指标序列）
│   ├── training_status.json           ← 训练状态快照
│   ├── training.log                   ← loguru 训练日志
│   ├── error.log                      ← 异常 JSONL
│   ├── warning.log                    ← 警告 JSONL
│   ├── training_start.log             ← 启动信息
│   ├── training_details.log           ← 每 episode 详情
│   ├── training_complete.log          ← 完成信息
│   ├── model_saves.log                ← 模型保存记录
│   ├── battle_results.log             ← 每场对战简记
│   ├── opponent_selection.log         ← 对手选择日志
│   ├── time_statistics.json           ← 时间统计
│   ├── weight_stats.jsonl             ← 网络权重统计
│   └── per_episode_stats.jsonl        ← 每 episode 简记
│
├── checkpoint/
│   ├── model_{episode}.pt             ← PyTorch 模型
│   ├── initial_opponent_{uuid}.pt     ← 初始对手模型
│   └── league_state/
│       ├── opponent_pool.json         ← 对手池状态
│       └── payoff.json                ← Payoff 矩阵 + TrueSkill
│
├── battle/
│   ├── battle_stats.json              ← 对战统计
│   └── {agent1}_vs_{agent2}_{ts}/
│       ├── battle_results.json        ← 对战结果 JSONL
│       ├── battle_debug.log
│       ├── battle_info.log
│       ├── battle_warning.log
│       ├── battle_error.log
│       └── battle_all.log
│
├── selfplay/
│   ├── sp_debug.log
│   ├── sp_info.log
│   ├── sp_warning.log
│   ├── sp_error.log
│   ├── sp_all.log
│   ├── sp_results.json                ← SelfPlay 对战结果 JSONL
│   └── selfplay_battles/
│       ├── selfplay_battle_stats.json  ← 对手维度聚合统计
│       └── detailed_battles/
│           └── selfplay_{ep}_{opp}_{ts}.json  ← 单场完整详情
│
└── system/
    ├── system_metrics.json             ← 系统资源采样统计
    └── system_metrics.json.log         ← 系统资源采样流水
```

---

## 一、训练主指标

### 1.1 training_metrics.json

| 属性 | 值 |
|------|-----|
| **路径** | `{training_dir}/training_metrics.json` |
| **格式** | JSON |
| **写入方式** | 覆盖写入（每次 PPO update 后） |
| **频率** | 每个 PPO update 一次（目前=每 episode 一次） |
| **数据颗粒度** | 最新一条 + 历史统计 |

**字段结构：**

```
{
  "latest": {                           ← 最近一条完整指标
    "episode": int,
    "avg_reward": float,
    "avg_loss": float,
    "avg_rounds": float,
    "steps": int,
    "policy_loss": float,
    "value_loss": float,
    "entropy": float,
    "entropy_coef": float,
    "learning_rate": float,
    "timestamp": str,
    // PPO 训练指标
    "reward_mean/ std/ min/ max": float,
    "advantages_mean/ std": float,
    "return_mean/ std": float,
    "clip_fraction": float,
    "gradient_norm": float,
    "grad_norm_policy/ value/ max_layer": float,
    "nan_skip_count": int,
    "valid_actions_mean/ min": float,
    "td_error_mean/ std": float,
    "batch_size": int,
    "num_collected": int,
    // 金币
    "avg_our_coins": float,             ← 每步余额均值
    "avg_enemy_coins": float,
    "avg_our_cumulative_coins": float,  ← 每局累计金币均值（含初始50）
    "avg_enemy_cumulative_coins": float,
    // 奖励分解
    "rw_hp_attack": float,
    "rw_coin_gain": float,
    "rw_tower_survival": float,
    "rw_tech_bonus": float,
    "rw_die_penalty": float,
    "rw_end_reward": float,
    "rw_speed_bonus": float,
    // 动态字段 - 动作分布
    "type_noop": float,                 ← NO-OP 比例
    "type_build_tower": float,
    "type_upgrade_tower": float,
    "type_downgrade_tower": float,
    "type_lightning_storm": float,
    "type_emp_blaster": float,
    "type_deflector": float,
    "type_evasion": float,
    "type_tech_upgrade": float,
    // 动态字段 - 对手
    "opponent_win_rate_opp_xxx": float,
    "opponent_reward_opp_xxx": float,
  },
  "stats": {
    "reward_mean": float,               ← 历史 reward 均值
    "reward_std": float,
    "reward_max": float,
    "reward_min": float
  }
}
```

### 1.2 training_metrics_history.jsonl

| 属性 | 值 |
|------|-----|
| **路径** | `{training_dir}/training_metrics_history.jsonl` |
| **格式** | JSONL |
| **写入方式** | append（每次 PPO update 追加一行） |
| **频率** | 每个 PPO update 一次 |
| **数据颗粒度** | 每行一条完整的指标记录（同 `latest` 字段） |

### 1.3 training_status.json

| 属性 | 值 |
|------|-----|
| **路径** | `{training_dir}/training_status.json` |
| **格式** | JSON |
| **写入方式** | 覆盖写入 |
| **频率** | 每次 `save_training_status()` 调用（定期） |
| **数据颗粒度** | 训练状态快照 |

**字段结构：**

```json
{
  "episode": int,
  "total_steps": int,
  "best_avg_reward": float,
  "avg_reward_last_100": float,
  "avg_reward_last_10": float,
  "avg_loss_last_100": float,
  "avg_rounds": float,
  "total_wins/ losses/ draws": int,
  "win_rate": float,
  "training_speed": float,
  "ppo_metrics": { ... },
  "training_stability": { "reward_std": float, "loss_std": float },
  "start_time": str,
  "current_time": str
}
```

---

## 二、文本日志（loguru）

| 文件 | 路径 | 等级 | 内容示例 |
|------|------|------|---------|
| training.log | `training/training.log` | 所有级别 | 训练全过程日志 |
| error.log | `training/error.log` | ERROR | `{timestamp, exception_type, message, context}` |
| warning.log | `training/warning.log` | WARNING | `{timestamp, message, context}` |
| sp_debug/ info/ warning/ error/ all.log | `selfplay/` | 对应级别 | SelfPlay 子进程日志 |
| battle_debug/ info/ warning/ error/ all.log | `battle/{agent1}_vs_{agent2}_{ts}/` | 对应级别 | 对战单场日志 |

---

## 三、文本简记日志

| 文件 | 路径 | 写入方式 | 频率 | 每行内容 |
|------|------|---------|------|---------|
| training_start.log | `training/` | 一次性 | 训练启动时 | episodes, batch_size, lr, device, 系统信息 |
| training_details.log | `training/` | append | 每 episode | `{ts} \| Episode: {n} \| Avg Reward: {x} \| Avg Loss: {x} \| Avg Rounds: {x} \| Steps: {x}` |
| training_complete.log | `training/` | 一次性 | 训练完成时 | total_episodes, total_steps, best_avg_reward, total_time, win_rate |
| model_saves.log | `training/` | append | 每次保存模型 | `{ts} \| Model saved: {path}` |
| battle_results.log | `training/` | append | 每场对战 | `{ts} \| Episode: {n} \| Opponent: {x} \| Result: {x} \| Reward: {x}` |
| opponent_selection.log | `training/` | append | 每次选择对手 | `{ts} \| Episode: {n} \| Opponent: {x} \| Random prob: {x}` |

---

## 四、逐 episode 统计

### 4.1 per_episode_stats.jsonl

| 属性 | 值 |
|------|-----|
| **路径** | `{training_dir}/per_episode_stats.jsonl` |
| **格式** | JSONL，append |
| **数据颗粒度** | 每 episode 一行 |

**字段：**

```json
{
  "episode": int,
  "reward": float,
  "length": int,
  "opponent_id": str,
  "result": "win" | "loss" | "draw",
  "rounds": int,
  "swap": bool     ← 是否交换先后手
}
```

---

## 五、权重统计

### 5.1 weight_stats.jsonl

| 属性 | 值 |
|------|-----|
| **路径** | `{training_dir}/weight_stats.jsonl` |
| **格式** | JSONL，append |
| **频率** | 每次 `save_checkpoint()` 调用 |
| **数据颗粒度** | 每 episode 一行，含所有网络层参数统计 |

**字段结构：**

```json
{
  "episode": int,
  "weights": {
    "policy_net.fc1.weight": {
      "mean": float,
      "std": float,
      "min": float,
      "max": float,
      "grad_mean": float,
      "grad_std": float,
      "grad_norm": float
    },
    ...   // 每个网络参数一条
  }
}
```

---

## 六、对战详情

### 6.1 selfplay_battle_stats.json（对手维度聚合）

| 属性 | 值 |
|------|-----|
| **路径** | `{base_dir}/selfplay/selfplay_battles/selfplay_battle_stats.json` |
| **格式** | JSON，覆盖写入 |
| **数据颗粒度** | 每个 opponent_id 一条聚合记录 |

```json
{
  "opp_xxx": {
    "wins": int,
    "losses": int,
    "draws": int,
    "total_battles": int,
    "total_reward": float,
    "avg_reward": float,
    "avg_rounds": float
  }
}
```

### 6.2 回合级对战详情（最详细）

| 属性 | 值 |
|------|-----|
| **路径** | `{base_dir}/selfplay/selfplay_battles/detailed_battles/selfplay_{episode}_{opponent}_{timestamp_ms}.json` |
| **格式** | JSON，独立文件 |
| **数量** | 每 episode × 每个对手 × 2（swap先后手）= N 个文件 |
| **数据颗粒度** | 单场对战，含每步详情 |

**顶层字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `episode` | int | episode 编号 |
| `opponent_id` | str | 对手 ID |
| `player_position` | int | 0=先手, 1=后手 |
| `start_time` | str | 开始时间 |
| `end_time` | str | 结束时间 |
| `duration` | float | 耗时（秒） |
| `result` | str | win / loss / draw |
| `total_reward` | float | 总奖励 |
| `rounds` | int | 总回合数 |
| `final_hp` | dict | {our_hp, enemy_hp} |
| `action_counts` | dict | {action_id: count} |
| `first_move_wins` | int | 先手胜（该场） |
| `first_move_losses` | int | 先手负 |
| `second_move_wins` | int | 后手胜 |
| `second_move_losses` | int | 后手负 |
| `our_cumulative_coins` | float | 我方累计赚取金币 |
| `enemy_cumulative_coins` | float | 敌方累计赚取金币 |

**`round_details[]` 每步详情（回合级）：**

| 字段 | 说明 |
|------|------|
| `step` | 步数 |
| `round` | 回合数 |
| `our_hp` | 我方血量 |
| `enemy_hp` | 敌方血量 |
| `our_coins` | 我方当前金币余额 |
| `enemy_coins` | 敌方当前金币余额 |
| `our_cumulative_coins` | 我方累计赚取金币（含初始50） |
| `enemy_cumulative_coins` | 敌方累计赚取金币 |
| `timestamp` | 时间戳 |
| `our_towers` | 我方塔信息 `[{id, type, level, position}]` |
| `enemy_towers` | 敌方塔信息 |
| `our_ants` | 我方蚂蚁数量 |
| `enemy_ants` | 敌方蚂蚁数量 |
| `our_ant_kinds` | 我方蚂蚁种类分布 |
| `enemy_ant_kinds` | 敌方蚂蚁种类分布 |
| `our_tech` | 我方科技 `{gen_speed, ant_strength}` |
| `enemy_tech` | 敌方科技 |

**`actions[]` 每步动作：**

| 字段 | 说明 |
|------|------|
| `step` | 步数 |
| `action` | 我方 action ID |
| `opponent_action` | 对手 action ID |
| `log_prob` | 动作 log 概率 |
| `value` | 价值函数输出 |
| `reward` | 该步奖励 |
| `reward_detail` | dict，奖励分解各项 |
| `mask_valid` | 有效动作数量 |

### 6.3 sp_results.json（SelfPlay 对战结果简记）

| 属性 | 值 |
|------|-----|
| **路径** | `{selfplay_dir}/sp_results.json` |
| **格式** | JSONL，append |
| **数据颗粒度** | 每场对战一行 |

### 6.4 battle_results.log（训练日志中的对战简记）

| 属性 | 值 |
|------|-----|
| **路径** | `{training_dir}/battle_results.log` |
| **格式** | 纯文本，append |
| **内容** | `{timestamp} | Episode: {n} | Opponent: {x} | Result: {x} | Reward: {x}` |

---

## 七、对手池 & Payoff

### 7.1 opponent_pool.json

| 属性 | 值 |
|------|-----|
| **路径** | `{checkpoint_dir}/league_state/opponent_pool.json` |
| **格式** | JSON，覆盖写入 |
| **数据颗粒度** | 文件级（整个对手池的完整状态） |

```json
{
  "max_size": int,
  "min_games_threshold": int,
  "opponent_ids": ["opp_xxx", ...],
  "checkpoint_paths": {"opp_xxx": "path/model.pt"},
  "added_timestamps": {"opp_xxx": float},
  "added_episodes": {"opp_xxx": int},
  "games_played": {"opp_xxx": int}
}
```

### 7.2 payoff.json

| 属性 | 值 |
|------|-----|
| **路径** | `{checkpoint_dir}/league_state/payoff.json` |
| **格式** | JSON，覆盖写入 |
| **数据颗粒度** | 文件级（整个 payoff 矩阵） |

```json
{
  "players": ["current_agent", "opp_xxx", ...],
  "players_ids": ["current_agent", "opp_xxx", ...],
  "trueskill_ratings": {
    "current_agent": {"mu": float, "sigma": float, "games_played": int},
    "opp_xxx": {"mu": float, "sigma": float, "games_played": int}
  },
  "battle_records": {
    "current_agent-opp_xxx": {"wins": int, "draws": int, "losses": int, "games": int}
  },
  "decay": float,
  "min_win_rate_games": int
}
```

---

## 八、系统资源

### 8.1 system_metrics.json

| 属性 | 值 |
|------|-----|
| **路径** | `{system_dir}/system_metrics.json` |
| **格式** | JSON，每6次采样覆盖写入 |
| **数据颗粒度** | 最近10个采样 + 统计 |

```json
{
  "latest": { "timestamp", "phase", "cpu_percent", "ram_percent", "ram_used_gb", "gpu_util", ... },
  "latest_samples": [最近10个采样],
  "stats": { "cpu_avg/max", "ram_avg/max", "gpu_avg/max", ... }
}
```

### 8.2 system_metrics.json.log

| 属性 | 值 |
|------|-----|
| **路径** | `{system_dir}/system_metrics.json.log` |
| **格式** | JSONL，每次写入最近6个采样 |
| **数据颗粒度** | 每个采样一行 |

---

## 九、时间统计

### time_statistics.json

| 属性 | 值 |
|------|-----|
| **路径** | `{training_dir}/time_statistics.json` |
| **格式** | JSON，覆盖写入 |
| **数据颗粒度** | 训练/对战的时间汇总 |

```json
{
  "training_episodes_count": int,
  "training_time_total": float,
  "battle_evaluations_count": int,
  "battle_time_total": float,
  "last_training_start_time": str,
  "last_training_end_time": str,
  "last_battle_start_time": str,
  "last_battle_duration": float,
  "timestamp": str
}
```

---

## 十、Model Checkpoint

### model_{episode}.pt

| 属性 | 值 |
|------|-----|
| **路径** | `{checkpoint_dir}/model_{episode}.pt` |
| **格式** | PyTorch 序列化（torch.save） |
| **数据颗粒度** | 文件级，单个 checkpoint |

```python
{
  "policy_state_dict": ...,             # 策略网络参数
  "optimizer_state_dict": ...,          # 策略优化器状态
  "value_optimizer_state_dict": ...,    # 价值网络优化器状态
  "episode_count": int,
  "total_steps": int,
  "config": { ... }                     # 训练时的完整配置
}
```

---

## 颗粒度分级汇总

| 层级 | 颗粒度 | 代表文件 | 用途 |
|------|--------|---------|------|
| **L1 - 聚合指标** | 多 episode 聚合 | `training_metrics.json` / `.jsonl`, `training_status.json` | 训练监控、趋势分析 |
| **L2 - 每 episode 简记** | 每 episode | `per_episode_stats.jsonl` | 快速检索单 episode 结果 |
| **L3 - 权重统计** | 每 episode | `weight_stats.jsonl` | 网络收敛分析 |
| **L4 - 对手维度聚合** | 对手 | `selfplay_battle_stats.json` | 对手表现概览 |
| **L5 - 单场对战** | 单场 | `selfplay_{ep}_{opp}_{ts}.json` | 完整回放 + 逐步数据 |
| **L6 - 对手池/Payoff** | 全局 | `opponent_pool.json`, `payoff.json` | 对手管理、TrueSkill |
| **L7 - 系统资源** | 每 6 次采样 | `system_metrics.json` | 资源监控 |
| **L8 - 模型** | 每 save_interval | `model_{ep}.pt` | 模型恢复、推理部署 |

```
                   颗粒度由粗到细
                   ────────────>
L1 指标      L2 简记  L3 权重   L4 对手聚合  L5 单场详情  L6 对手池   L7 系统    L8 模型
Training    PerEp     Weight   Battle       Detailed   OppPool    System    Checkpoint
Metrics     Stats     Stats    Stats        Battles    +Payoff    Metrics
```

---

## 文件数量估算（2000 episode 训练，save_interval=20）

| 文件类型 | 数量 |
|---------|:----:|
| training_metrics_history.jsonl | 250 条记录 |
| per_episode_stats.jsonl | 2000 条记录 |
| weight_stats.jsonl | 100 条记录 |
| selfplay_battle_stats.json | 1 文件 |
| detailed_battles/*.json | ~2000 文件（每 ep × baseline数量 × 2） |
| model_{ep}.pt | 100 文件 |
| opponent_pool.json / payoff.json | 2 文件 |
| system_metrics.json / .log | 2 文件 |
