# 模块7：监控与可观测性

## 1. 概述

### 1.1 模块定位

监控与可观测性模块提供训练过程的 **全面可观测性支持**，包括结构化日志记录、系统资源监控、时间追踪统计、对战数据持久化和批量聚合输出，服务于训练监控、问题调试和事后分析。

### 1.2 在整体架构中的位置

```
模块4: SelfPlayTrainer
    │
    │  直接使用 LoggingSubsystem
    ▼
┌───────────────────────────────────────────────────┐
│            模块7: 监控与可观测性                      │
│                                                     │
│  ┌─────────────────────────────────────────┐       │
│  │  Logger           训练主日志              │       │
│  │  TimeTracker      时间追踪               │       │
│  │  SystemMetricsSampler  系统资源监控       │       │
│  ├─────────────────────────────────────────┤       │
│  │  数据写入器                               │       │
│  │  ├── BatchMetricsWriter    训练指标      │       │
│  │  ├── EpisodeBatchWriter    episode聚合   │       │
│  │  ├── SelfPlayBattleWriter  自对弈记录     │       │
│  │  └── EvalBattleWriter      评估记录      │       │
│  └─────────────────────────────────────────┘       │
└───────────────────────────────────────────────────┘
    │
    │  输出文件
    ▼
outputs/{run_id}/
  ├── training/
  │   ├── training_start.log
  │   ├── training_complete.log
  │   └── battle_stats.json
  ├── selfplay/
  │   ├── selfplay_battle_log.jsonl
  │   ├── batch_metrics.jsonl
  │   ├── episode_batch_battle_stats.jsonl
  │   └── episode_batch_train_stats.jsonl
  ├── system/
  │   ├── time_statistics.json
  │   ├── system_metrics.json
  │   └── system_metrics.json.log
  └── evaluation/
      ├── eval_battle_log.jsonl
      └── episode_batch_baseline_winrates.jsonl
```

### 1.3 业务目标

- 记录训练全生命周期的关键日志（启动快照、完成汇总）
- 监控系统资源使用情况（CPU/内存/GPU）通过后台线程周期性采样
- 追踪训练时间统计（速度、效率）
- 持久化训练指标（每次 PPO update 后的详细指标）
- 记录每局自对弈的完整数据
- 持久化基线评估结果

---

## 2. 背景与概念

### 2.1 可观测性（Observability）

在长时间的训练过程中（可能持续数小时到数天），需要持续监测：
- **训练是否正常**：loss 是否下降，entropy 是否合理
- **资源是否充足**：GPU 显存是否溢出，CPU 是否瓶颈
- **速度是否正常**：episode 处理速度是否稳定
- **效果是否提升**：胜率是否在增长

### 2.2 JSONL 格式

模块中大部分数据写入器使用 **JSONL（JSON Lines）** 格式：每行一个独立的 JSON 对象。优势：
- 追加写入高效（无需重写整个文件）
- 逐行读取方便（适合流式处理）
- 支持事后用 `jq`、`pandas` 等工具分析

### 2.3 窗口聚合

EpisodeBatchWriter 和 EvalBattleWriter 使用 **滑动窗口聚合**：每收集满 `window_size` 个 episode 的数据后，计算窗口内的平均值并写入。这降低了数据量，同时保留了趋势信息。

---

## 3. 架构设计

### 3.1 模块内部结构

```
┌─────────────────────────────────────────────────────┐
│                 监控与可观测性模块                      │
│                                                       │
│  ┌──────────────┐  ┌───────────────┐                │
│  │   Logger     │  │  TimeTracker  │                │
│  │  (训练日志)   │  │  (时间追踪)    │                │
│  └──────────────┘  └───────────────┘                │
│                                                       │
│  ┌──────────────────────────────────┐                │
│  │  SystemMetricsSampler            │                │
│  │  (后台线程: CPU/内存/GPU 采样)     │                │
│  └──────────────────────────────────┘                │
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │              数据写入器层                       │    │
│  │                                                │    │
│  │  ┌──────────────────┐  ┌───────────────────┐  │    │
│  │  │BatchMetricsWriter│  │EpisodeBatchWriter │  │    │
│  │  │ (训练指标 JSONL)  │  │ (episode 聚合)     │  │    │
│  │  └──────────────────┘  └───────────────────┘  │    │
│  │                                                │    │
│  │  ┌────────────────────┐  ┌──────────────────┐ │    │
│  │  │SelfPlayBattleWriter│  │EvalBattleWriter  │ │    │
│  │  │ (自对弈记录)        │  │ (评估记录)        │ │    │
│  │  └────────────────────┘  └──────────────────┘ │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  ┌────────────┐                                      │
│  │ constants  │  日志格式常量、采样配置                 │
│  └────────────┘                                      │
└─────────────────────────────────────────────────────┘
```

### 3.2 组件职责

| 组件 | 文件 | 职责 |
|------|------|------|
| `Logger` | [monitor/logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/logger.py) | 训练主日志：启动快照、完成汇总、对战统计 |
| `TimeTracker` | [monitor/time_tracker.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/time_tracker.py) | 训练/对战耗时统计、速率计算 |
| `SystemMetricsSampler` | [monitor/system_metrics_sampler.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/system_metrics_sampler.py) | 后台线程周期性采样系统资源 |
| `BatchMetricsWriter` | [monitor/batch_metrics_writer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/batch_metrics_writer.py) | 训练指标写入 `batch_metrics.jsonl` |
| `EpisodeBatchWriter` | [monitor/episode_batch_writer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/episode_batch_writer.py) | Episode 聚合写入（对战+训练统计） |
| `SelfPlayBattleWriter` | [monitor/selfplay_battle_writer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/selfplay_battle_writer.py) | 自对弈单局记录写入 |
| `EvalBattleWriter` | [monitor/eval_battle_writer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/eval_battle_writer.py) | 评估记录 + baseline 胜率 |
| `constants` | [monitor/constants.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/constants.py) | 日志格式常量和采样配置 |

---

## 4. 核心组件详解

### 4.1 Logger — 训练主日志

位置：[monitor/logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/logger.py)

```python
Logger(training_dir: str, run_id: str)
```

| 方法 | 说明 |
|------|------|
| `log_training_start(config, device)` | 写入 `training_start.log`（配置摘要、系统信息） |
| `log_training_complete(summary)` | 写入 `training_complete.log`（最终统计） |
| `save_battle_stats(stats_data)` | 写入 `battle_stats.json`（对手胜率矩阵） |
| `log_json_state(filename, data, context)` | 追加写入 `{filename}.json.log` |

---

### 4.2 TimeTracker — 时间追踪器

位置：[monitor/time_tracker.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/time_tracker.py)

```python
TimeTracker()
```

| 方法 | 说明 |
|------|------|
| `start()` | 重置并开始计时 |
| `record_episode(duration)` | 记录单个 episode 的耗时 |
| `start_training()` / `end_training()` | 训练计时段 |
| `start_battle()` / `end_battle()` | 对战计时段 |
| `elapsed_seconds` | 自 `start()` 起的累计秒数 |
| `episodes_per_minute` | 每分钟处理的 episode 数 |
| `avg_episode_time` | 平均每个 episode 耗时（秒） |
| `get_summary()` | 获取完整时间统计摘要 |
| `save_time_statistics(path)` | 写入 `time_statistics.json` |

---

### 4.3 SystemMetricsSampler — 系统资源采样器

位置：[monitor/system_metrics_sampler.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/system_metrics_sampler.py)

```python
SystemMetricsSampler(output_dir: str, sample_interval: int)
```

使用 daemon 后台线程周期性采样：

| 采样指标 | 来源 | 说明 |
|---------|------|------|
| `cpu_percent` | `psutil` | CPU 使用率（%） |
| `ram_percent` | `psutil` | 内存使用率（%） |
| `ram_used_gb` | `psutil` | 已用内存（GB） |
| `ram_total_gb` | `psutil` | 总内存（GB） |
| `gpu_util` | `pynvml` | GPU 利用率（%，如有） |
| `gpu_mem_used_mb` | `pynvml` | GPU 显存已用（MB，如有） |
| `gpu_mem_total_mb` | `pynvml` | GPU 显存总量（MB，如有） |
| `gpu_temp` | `pynvml` | GPU 温度（℃，如有） |

| 方法 | 说明 |
|------|------|
| `start()` | 启动后台 daemon 线程 |
| `stop()` | 停止线程并 flush 数据 |
| `set_phase(phase)` | 设置采样阶段（`"training"` / `"battle"`） |

Flush 输出：
- `system_metrics.json`：最新值 + 统计量（均值/最大/最小）
- `system_metrics.json.log`：每个采样点的追加记录

---

### 4.4 数据写入器

四个写入器分别负责不同维度的数据持久化：

#### BatchMetricsWriter

```python
BatchMetricsWriter(filepath: str)
    write_batch(metrics: Dict)     # 写入每次 PPO update 的纯技术指标
    close()
```

输出：`batch_metrics.jsonl`，每行包含 loss/entropy/grad_norm/clip_fraction 等训练指标。

#### EpisodeBatchWriter

```python
EpisodeBatchWriter(battle_log_path, batch_metrics_path,
                   battle_stats_path, train_stats_path,
                   window_size=8)
    on_episode_complete(episode, battle_record)   # 缓冲对战数据
    on_batch_complete(episode, batch_metrics)      # 触发窗口聚合和 flush
    flush_now()                                    # 强制刷新
```

窗口满（`window_size=8`）时聚合并写入两个 JSONL 文件：
- **对战统计**：win_rate, draw_rate, avg_reward, avg_rounds, avg_action_counts 等
- **训练统计**：avg_policy_loss, avg_value_loss, avg_entropy 等

#### SelfPlayBattleWriter

```python
SelfPlayBattleWriter()
    write_episode(record: EpisodeRecord)
```

输出：`selfplay_battle_log.jsonl`，每行记录单局对战的完整数据：

```
{episode, swap, opponent_id, result, rounds, reward, length,
 hp/coin 终局数据, action_counts, action_rewards, reward_sources}
```

#### EvalBattleWriter

```python
EvalBattleWriter(eval_dir: str, window_size: int = 8)
    write_battle(episode, baseline_agent_name, battle_mode, result, ...)
    write_round(episode, round_num, **fields)
    on_episode_batch_complete(episode_start, episode_end)
```

输出：
- `eval_battle_log.jsonl`：每场评估对战的详细数据
- `episode_batch_baseline_winrates.jsonl`：按窗口聚合的 baseline 胜率

---

## 5. 数据流与时序

### 5.1 训练全生命周期的日志记录

```
训练启动
  │
  ├── Logger.log_training_start()          # training_start.log
  ├── TimeTracker.start()
  └── SystemMetricsSampler.start()         # 后台线程启动
  │
训练循环中
  │
  ├── 每局后:
  │   ├── TimeTracker.record_episode()
  │   ├── SelfPlayBattleWriter.write_episode()   # selfplay_battle_log.jsonl
  │   └── EpisodeBatchWriter.on_episode_complete()
  │
  ├── 每次 PPO update 后:
  │   ├── BatchMetricsWriter.write_batch()       # batch_metrics.jsonl
  │   └── EpisodeBatchWriter.on_batch_complete()  # 聚合 flush
  │
  └── 每次基线评估后:
      └── EvalBattleWriter.write_battle()        # eval_battle_log.jsonl
  │
训练结束
  │
  ├── Logger.log_training_complete()       # training_complete.log
  ├── TimeTracker.save_time_statistics()   # time_statistics.json
  └── SystemMetricsSampler.stop()          # flush system_metrics.json
```

### 5.2 输出文件目录结构

```
outputs/{run_id}/
├── training/
│   ├── training_start.log           # 训练启动快照
│   ├── training_complete.log        # 训练完成汇总
│   └── battle_stats.json            # 对手胜率统计
├── selfplay/
│   ├── selfplay_battle_log.jsonl    # 每局对战数据
│   ├── batch_metrics.jsonl          # 每次 PPO update 指标
│   ├── episode_batch_battle_stats.jsonl  # 窗口聚合对战统计
│   ├── episode_batch_train_stats.jsonl   # 窗口聚合训练统计
│   └── sp_all.log                   # 自对弈事件日志
├── system/
│   ├── time_statistics.json         # 时间统计
│   ├── system_metrics.json          # 最新资源指标 + 统计量
│   └── system_metrics.json.log      # 原始采样点
└── evaluation/
    ├── eval_battle_log.jsonl        # 评估对战详记录
    └── episode_batch_baseline_winrates.jsonl  # 聚合胜率
```

---

## 6. 对外交互

### 6.1 被依赖关系

| 调用方 | 场景 | 使用方法 |
|--------|------|---------|
| **SelfPlayTrainer** (模块4) | 训练生命周期中 | `LoggingSubsystem` 中的各组件 |
| **BattleCoordinator** (模块5) | 评估对战中 | `EvalBattleWriter` |

### 6.2 对下游模块的依赖

| 依赖模块 | 使用方式 |
|---------|---------|
| **模块8（基础设施）** | `PathConfig` 确定输出目录结构 |

---

## 7. 配置参数

### 7.1 系统监控参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `sample_interval` | 系统指标采样间隔（秒） | 配置在 `constants.py` |

### 7.2 窗口聚合参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `window_size` | 聚合窗口大小（episode 数） | 8 |

---

## 8. 常见问题与注意事项

### 8.1 为什么需要多个独立 Writer 而非统一日志文件？

不同类型的日志有不同的消费场景：
- **batch_metrics.jsonl**：用于绘制训练曲线（loss/seeking），需要高频写入
- **selfplay_battle_log.jsonl**：用于分析对局模式（对手选择、奖励结构），每局一条
- **episode_batch 聚合文件**：用于宏观趋势分析，低频写入
- **eval_battle_log.jsonl**：用于跟踪 baseline 评估，周期性写入

独立文件避免了混合数据带来的解析困难。

### 8.2 系统监控的后台线程

`SystemMetricsSampler` 使用 Python `threading.Thread(daemon=True)`。daemon 线程在主线程退出时会自动终止，不需要显式 join。但最佳实践是在 `on_training_end()` 中调用 `stop()` 确保数据被 flush。

### 8.3 JSONL 的事后分析

可以用以下方式分析 JSONL 文件：

```bash
# 查看最近的训练指标
tail -5 batch_metrics.jsonl | jq .

# 用 pandas 加载
import pandas as pd
df = pd.read_json('selfplay_battle_log.jsonl', lines=True)

# 计算平均胜率
df['win_rate'] = (df['result'] == 'win').rolling(100).mean()
```

### 8.4 已废弃的 MetricsCache

[metrics_cache.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/metrics_cache.py) 已被标记为 DEPRECATED，不应再使用。指标获取通过 `BasePPOCallback.get_last_metrics()` 完成。

### 8.5 GPU 监控的条件可用性

`pynvml` 是 NVIDIA 的 Python 绑定，仅在安装了 NVIDIA 驱动和 CUDA 的环境中可用。在 CPU-only 环境中，`SystemMetricsSampler` 会优雅降级（跳过 GPU 指标，不报错）。
