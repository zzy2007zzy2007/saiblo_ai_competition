# 10. 日志与报告系统

> 本章详细描述 PPO v1 中 SelfPlay 训练与 Baseline Battle 对战流程的日志和报告系统。涵盖主进程日志记录器（`BattleLogger`）、子进程日志记录器（`SubProcessLogger`）、SelfPlay 训练专用日志（`SelfPlayLogger`）、报告生成器（`ReportGenerator`）、结果聚合器（`ResultAggregator`）以及三个底层工具模块（异常日志、进程隔离、计时工具）。

---

## 10.1 LogLevel — 日志级别枚举

[LogLevel](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_logger.py#L13-L41) 定义了四个标准日志级别，用于控制 BattleLogger 和 SubProcessLogger 的日志输出粒度：

```python
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
```

**关键方法：**

| 方法 | 说明 |
|------|------|
| `from_string(level_str)` | 从字符串（不区分大小写）创建枚举值，若无法匹配则默认返回 `INFO` |
| `to_loguru_level()` | 转换为 loguru 兼容的日志级别（小写字符串，如 `"info"`） |
| `get_level_number()` | 获取数值表示，用于比较日志级别高低：`DEBUG=0, INFO=1, WARNING=2, ERROR=3` |

**级别过滤逻辑：** 在 `debug()`、`info()`、`warning()` 方法中通过 `get_level_number()` 比较实现按级别过滤；`error()` 始终输出（无过滤）。

---

## 10.2 BattleLogger — 主进程日志记录器

### 10.2.1 概述

[BattleLogger](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_logger.py#L43-L367) 是 Baseline Battle 流程中主进程使用的日志记录器，负责：

- 配置和管理 loguru 日志处理器（handlers）
- 记录每次对战的开始、回合进度、结束
- 记录对战结果并持久化为 JSON 文件
- 累计统计信息（胜/负/平、耗时、平均回合数）
- 提供计时功能用于性能分析
- 最终输出汇总摘要和性能报告

**底层架构：** 基于 [loguru](https://github.com/Delgan/loguru) 库，日志格式中使用 `extra` 字段绑定 `category` 参数，实现在统一日志流中对不同模块（`battle`、`round`、`timing`、`load`、`summary` 等）进行分类标识。

### 10.2.2 构造函数

```python
def __init__(self, agent1_name: str, agent2_name: str,
             log_level: LogLevel = LogLevel.INFO, log_to_console: bool = True,
             enable_timing: bool = True, path_config: PathConfig = None,
             warning_threshold: float = 2.0, log_dir: str = None):
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `agent1_name` | `str` | — | 玩家1（当前策略 Agent）的名称标识 |
| `agent2_name` | `str` | — | 玩家2（baseline 对手）的名称标识 |
| `log_level` | `LogLevel` | `INFO` | 控制台和文件日志的过滤级别 |
| `log_to_console` | `bool` | `True` | 是否启用控制台 stderr 输出 |
| `enable_timing` | `bool` | `True` | 是否启用计时功能 |
| `path_config` | `PathConfig` | — | 路径配置对象（推荐方式），用于确定日志根目录 |
| `warning_threshold` | `float` | `2.0` | 单次操作超时警告阈值（秒） |
| `log_dir` | `str` | — | 日志目录字符串路径（向后兼容方式） |

`path_config` 和 `log_dir` **必须提供其一**，推荐使用 `path_config`。构造时自动执行：

1. 通过 `_create_log_dir()` 创建带时间戳的日志子目录，命名格式为 `{agent1_name}_vs_{agent2_name}_{YYYYMMDD_HHMMSS}`，位于 `path_config.battle_dir` 之下。
2. 调用 `_configure_logger()` 配置 loguru handlers。
3. 注册 `atexit` 清理回调 `_cleanup()`。

**内部统计计数器：**

| 字段 | 类型 | 初始值 | 说明 |
|------|------|--------|------|
| `_total_battles` | `int` | 0 | 总对战场次（含异常） |
| `_completed_battles` | `int` | 0 | 正常完成的对战场次 |
| `_error_battles` | `int` | 0 | 异常终止的对战场次 |
| `_agent1_wins` | `int` | 0 | Agent1 获胜场次 |
| `_agent2_wins` | `int` | 0 | Agent2 获胜场次 |
| `_draws` | `int` | 0 | 平局场次 |
| `_total_duration` | `float` | 0.0 | 总耗时（秒） |
| `_total_rounds` | `int` | 0 | 总回合数 |
| `_overall_start_time` | `datetime` | 构造时间 | 整体开始时间 |
| `agent1_total_time` | `float` | 0.0 | Agent1 总耗时（计时聚合） |
| `agent2_total_time` | `float` | 0.0 | Agent2 总耗时（计时聚合） |
| `resolve_total_time` | `float` | 0.0 | 回合结算总耗时（计时聚合） |

### 10.2.3 日志处理器配置（`_configure_logger`）

[`_configure_logger()`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_logger.py#L115-L177) 配置 **6 个 loguru handler**：

| Handler | 目标 | 级别 | enqueue | 说明 |
|---------|------|------|---------|------|
| Console | `sys.stderr` | 按 `log_level` | `False` | 控制台实时输出，带颜色高亮 |
| `battle_debug.log` | 文件 | `DEBUG` | `True` | 所有 DEBUG 及以上级别日志 |
| `battle_info.log` | 文件 | `INFO` | `True` | INFO 及以上级别日志 |
| `battle_warning.log` | 文件 | `WARNING` | `True` | WARNING 及以上级别日志 |
| `battle_error.log` | 文件 | `ERROR` | `True` | 仅 ERROR 级别日志 |
| `battle_all.log` | 文件 | `DEBUG` | `True` | 全量日志（与 `battle_debug.log` 内容相同） |

**日志格式：**

```
<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>[{extra[category]}]</cyan> | <level>{message}</level>
```

每个 handler ID 被保存在 `_handlers` 列表中，便于后续清理时逐一移除。

**关键设计：** 文件 handlers 使用 `enqueue=True` 确保多进程安全——日志消息通过内部队列汇入后台 writer 线程，实现均匀输出，避免多进程写同一文件的竞态问题。控制台 handler 使用 `enqueue=False` 以降低延迟。

### 10.2.4 日志输出方法

四种分类日志方法，均通过 `logger.bind(category=category)` 动态绑定分类标签：

```python
def debug(self, category: str, message: str):
def info(self, category: str, message: str):
def warning(self, category: str, message: str):
def error(self, category: str, message: str):
```

`debug`、`info`、`warning` 在输出前会检查当前 `log_level` 是否允许该级别；`error` 始终输出。

**常用的 category 值：** `battle`、`round`、`timing`、`load`、`summary`、`perf_report`、`timeout`、`gpu`

### 10.2.5 对战生命周期日志方法

#### `log_battle_start(episode: int)`

记录对战开始。输出格式：

```
[INFO] [battle] START #{episode}: {agent1_name} vs {agent2_name}
```

#### `log_round(episode, round_num, hp1, hp2, coins1, coins2, player0_ops, player1_ops, first_player)`

记录每一回合的详细状态，输出于 DEBUG 级别。根据 `first_player` 参数确定先手/后手顺序，在日志中标注 `(先手)` / `(后手)` 后缀：

```python
agent1_is_player0 = (first_player == 0)
player0_name = self.agent1_name if agent1_is_player0 else self.agent2_name
```

#### `log_battle_end(result: Dict)`

记录对战结束。输出两行：

```
[INFO] [battle] END #{episode}: {agent1} vs {agent2}
[INFO] [battle]   Result: {result_str}, Rounds: {rounds}, Time: {duration:.2f}s
```

#### `log_result(result: Dict)`

将单局对战结果写入两个位置：

1. **JSON 文件持久化：** 以追加模式写入 `{log_dir}/battle_results.json`，每行一条 JSON 记录（JSONL 格式）
2. **内存计数器累加：** 更新 `_total_battles`、`_completed_battles`、`_error_battles`、`_agent1_wins`、`_agent2_wins`、`_draws`、`_total_duration`、`_total_rounds`

```python
def log_result(self, result: Dict):
    results_json_path = os.path.join(self._log_dir, 'battle_results.json')
    with open(results_json_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()

    self._total_battles += 1
    self._total_duration += result['duration']
    self._total_rounds += result.get('total_rounds', 0)

    if result['result'] == 'error':
        self._error_battles += 1
    else:
        self._completed_battles += 1
        if result['result'] == 'agent1_win':
            self._agent1_wins += 1
        elif result['result'] == 'agent2_win':
            self._agent2_wins += 1
        else:
            self._draws += 1
```

#### `log_error(episode: int, error: Exception)`

记录对战异常，输出为 ERROR 级别，格式如：

```
[ERROR] [battle] Battle #{episode} failed: {ExceptionType}: {message}
```

#### `log_agent_load(agent_name: str, success: bool, error_msg: str = None)`

记录 Agent 加载结果。成功时输出 `INFO` 级别带 `✓` 标记，失败时输出 `ERROR` 级别带 `✗` 标记。

### 10.2.6 计时方法

BattleLogger 内置了一套独立的计时系统（独立于 `timing.py` 中的 `Timer` 类），通过 `enable_timing` 标志控制。

| 方法 | 说明 |
|------|------|
| `log_timing(operation, duration)` | 输出单次操作的耗时日志（DEBUG 级别） |
| `timing_start(operation)` | 记录操作开始时间，存入 `_timing_starts` 字典 |
| `timing_stop(operation, duration=None)` | 计算耗时并累加到 `_timings[operation]`，自动聚合到 `agent1_total_time` / `agent2_total_time` / `resolve_total_time`；超阈值时触发 WARNING |
| `log_timing_summary(battle_total_time)` | 输出本轮对战的各 Agent 耗时和结算耗时汇总 |
| `reset_timing_summary()` | 重置计时统计（每局对战结束后调用） |
| `get_timing_total(operation)` | 查询某操作的累计耗时 |
| `print_performance_report()` | 输出所有操作的累计耗时排名（从高到低排序） |

**超时警告机制：** 当 `timing_stop()` 检测到单次操作耗时超过 `warning_threshold`（默认 2.0 秒），将输出 WARNING 级别日志：

```
[WARNING] [timeout] {operation} took {duration:.2f}s (exceeds {warning_threshold}s)
```

### 10.2.7 `write_summary()` — 汇总摘要

[`write_summary()`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_logger.py#L317-L359) 在所有对战完成后生成格式化摘要，输出到 INFO 级别的 `summary` 分类。

**摘要内容：**

- **总体统计：** `Total Battles`、`Completed Battles`、`Error Battles`
- **对战结果：** 双方各自胜场数 + 胜率百分比、平局数 + 平局率
- **时间统计：** `Overall Start Time`、`Overall End Time`、`Total Duration`、`Average Rounds`、`Average Duration`

**输出格式示例：**
```
============================================================
Summary:
============================================================
Total Battles: 4
Completed Battles: 4
Error Battles: 0

【对战结果】
agent1 Wins: 3 (75.0%)
agent2 Wins: 1 (25.0%)
Draws: 0 (0.0%)

【时间统计】
Overall Start Time: 2025-05-15 10:30:00
Overall End Time: 2025-05-15 10:32:15
Total Duration: 45.230s
Average Rounds: 128.5
Average Duration: 11.308s
============================================================
```

**胜率计算逻辑：** 分母为 `_agent1_wins + _agent2_wins + _draws`（即 `total_wins`），而非 `_total_battles`，这意味着异常对战不影响胜率计算。

### 10.2.8 `close()` — 资源清理

调用 `write_summary()` 输出最终摘要，然后调用 `_cleanup()` 移除所有 loguru handler 并等待后台消息队列排空。

### 10.2.9 `log_gpu_downgrade(message)` — GPU 降级日志

当 GPU 环境出现问题时（如 CUDA OOM 自动降级到 CPU），记录 WARNING 级别的 `gpu` 分类日志。

---

## 10.3 SubProcessLogger — 子进程日志记录器

### 10.3.1 设计定位

[SubProcessLogger](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_logger.py#L370-L420) 是 BattleLogger 的**轻量级子集**，专用于多进程 `fork` 出的子进程环境。

**与 BattleLogger 的核心差异：**

| 维度 | BattleLogger | SubProcessLogger |
|------|-------------|-------------------|
| 实例化位置 | 主进程 | 每个子进程内部 |
| 日志 handler 注册 | 负责配置 loguru handlers | **不注册** handlers（继承主进程配置） |
| 计时系统 | 内置完整的 timing_start/stop/summary | **不包含**计时功能 |
| 统计计数器 | 维护完整的胜/负/平统计 | **不包含**统计聚合 |
| 结果持久化 | `log_result()` 写入 JSON 文件 | **不包含**结果持久化 |
| 构造函数参数 | `path_config`/`log_dir` 二选一必传 | 仅需 agent 名称和 log_level |

### 10.3.2 构造与实现

```python
class SubProcessLogger:
    def __init__(self, agent1_name: str, agent2_name: str,
                 log_level: LogLevel = LogLevel.DEBUG):
```

构造函数极简，仅保留 Agent 名称和日志级别。子进程通过 `fork` 继承主进程中 BattleLogger 配置的所有 loguru handlers（包括 enqueue 队列），因此无需重复配置。

### 10.3.3 支持的日志方法

SubProcessLogger 提供与 BattleLogger 完全相同的分类日志接口——`debug()`、`info()`、`warning()`、`error()`——以及以下对战生命周期方法：

| 方法 | 说明 |
|------|------|
| `log_battle_start(episode)` | 记录子进程内的对战开始 |
| `log_round(episode, round_num, hp1, hp2, ...)` | 记录回合状态，包含先手/后手标注 |
| `log_battle_end(result)` | 记录对战结束及胜负结果 |
| `log_gpu_downgrade(message)` | GPU 降级警告（与 BattleLogger 相同） |

这些方法的实现逻辑与 BattleLogger 对应方法完全一致。

### 10.3.4 多进程安全原理

**关键设计：** 子进程的 SubProcessLogger 不持有独立的 handler 引用，不进行 handler 注册/清理。日志消息直接通过继承的 loguru 全局 `logger` 对象输出，消息通过 `enqueue=True` 的队列汇入后台 writer 线程，天然避免了多进程同时写文件的冲突。

这种设计意味着：

1. 子进程内的 SubProcessLogger 可以与主进程 BattleLogger 共享同一套日志文件，日志按时间顺序交叉写入
2. 子进程崩溃不会影响主进程日志队列，日志不会丢失
3. 无需进程间通信传递日志数据

---

## 10.4 SelfPlayLogger — SelfPlay 训练日志记录器

### 10.4.1 设计定位

[SelfPlayLogger](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay_logger.py#L25-L175) 是 SelfPlay 训练专用的日志记录器，独立于 Baseline Battle 的 `BattleLogger` / `SubProcessLogger`。该记录器跟踪当前训练中的 Agent 与对手池中各个对手的对战表现，而非两个特定 Agent 之间的对战。

### 10.4.2 SelfPlayLogLevel 枚举

SelfPlayLogger 使用独立的 [SelfPlayLogLevel](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay_logger.py#L11-L22) 枚举：

```python
class SelfPlayLogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
```

与 `LogLevel` 不同，`SelfPlayLogLevel` 使用整数值（0~3），通过 `value` 属性比较级别高低。

### 10.4.3 构造函数

```python
def __init__(self, agent_name: str = "current_agent",
             log_level: SelfPlayLogLevel = SelfPlayLogLevel.INFO,
             log_dir: Optional[str] = None,
             enable_timing: bool = True):
```

**内部统计计数器：**

| 字段 | 类型 | 初始值 | 说明 |
|------|------|--------|------|
| `_total_battles` | `int` | 0 | 总对战数 |
| `_wins` | `int` | 0 | 获胜场次 |
| `_losses` | `int` | 0 | 失败场次 |
| `_draws` | `int` | 0 | 平局场次 |
| `_total_duration` | `float` | 0.0 | 总耗时 |
| `_total_rounds` | `int` | 0 | 总回合数 |
| `_overall_start_time` | `datetime` | 构造时间 | 开始时间 |
| `_timings` | `dict` | `{}` | 计时记录（operation → 累计秒数） |

### 10.4.4 日志输出方法

与 BattleLogger 相同的四种分类方法（`debug`、`info`、`warning`、`error`），级别判断基于 `SelfPlayLogLevel.value`。

### 10.4.5 对战生命周期方法

#### `log_battle_start(episode, opponent_id, player_position)`

记录对战开始，区别于 BattleLogger：这里对手是动态变化的 `opponent_id`，且标注当前 Agent 是先手（`player_position == 0`）还是后手。

#### `log_battle_end(episode, result, total_rounds, duration, final_hp)`

记录对战结束。额外记录双方最终 HP：

```python
self.info('selfplay', f"  Final HP - Our: {our_hp}, Enemy: {enemy_hp}")
```

#### `log_round(episode, round_num, our_hp, enemy_hp, our_coins, enemy_coins)`

记录回合快照，视角固定为"我方（当前 Agent）vs 敌方（对手）"，不再区分先手/后手标注。

### 10.4.6 结果记录与统计

#### `record_battle_result(result, duration, total_rounds)`

累加统计计数器。`result` 的取值与 BattleLogger 不同，使用 `'win'` / `'loss'` / `'draw'` 三元值（而非 `'agent1_win'` / `'agent2_win'`），因为 SelfPlayLogger 始终以当前训练的 Agent 为视角。

#### `print_summary(episode=None)`

输出汇总摘要，格式与 BattleLogger 的 `write_summary()` 风格一致，但标题标注为 `[SelfPlay]`。

#### `get_statistics() -> Dict`

返回统计信息的字典，包含 `total_battles`、`wins`、`losses`、`draws`、`win_rate`、`avg_rounds`、`avg_duration`。

#### `reset_statistics()`

重置所有统计计数器（在训练过程中可周期性地重置）。

### 10.4.7 计时方法

| 方法 | 说明 |
|------|------|
| `log_timing(operation, duration)` | 输出单次操作耗时（DEBUG 级别） |
| `record_timing(operation, duration)` | 累加操作耗时到 `_timings` 并调用 `log_timing` |
| `print_performance_report()` | 输出所有操作的耗时排名报告 |

与 BattleLogger 不同，SelfPlayLogger 的计时系统更简单：没有 `timing_start`/`timing_stop` 配对机制，仅在外部计算好耗时后通过 `record_timing()` 直接传入。

---

## 10.5 ReportGenerator — 报告生成器

### 10.5.1 概述

[ReportGenerator](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/report_generator.py#L5-L65) 将 `ResultAggregator` 聚合后的结果字典格式化为可读的文本报告。

### 10.5.2 `generate(aggregated, agent1_name, agent2_name) -> str`

接收聚合结果字典和双方 Agent 名称，返回格式化后的多行文本字符串。

### 10.5.3 报告结构

报告分为以下几个部分：

#### （1）报告头部

```
============================================================
        Baseline 对战评测报告
============================================================
执行时间: {start_time}
结束时间: {end_time}
对战双方: {agent1_name} vs {agent2_name}
============================================================
```

#### （2）总体统计

```python
report.append("【总体统计】")
report.append(f"总对战数: {aggregated['total_battles']}")
report.append(f"完成对战: {aggregated['completed_battles']}")
report.append(f"异常对战: {aggregated['error_battles']}")
report.append(f"总耗时: {aggregated['total_duration']:.2f}秒")
report.append(f"平均回合数: {aggregated.get('avg_rounds', 0.0):.1f}")
```

#### （3）对战结果表格

使用 Unicode 框线字符绘制表格：

```
【对战结果】
┌────────────┬────────────┬───────┬───────┬──────┬────────┐
│   Agent1   │   Agent2   │  胜   │  负   │  平   │  胜率   │
├────────────┼────────────┼───────┼───────┼──────┼────────┤
│  agent_1   │  agent_2   │   3   │   1   │   0   │  75.0%  │
└────────────┴────────────┴───────┴───────┴──────┴────────┘
```

胜率计算：`agent1_wins / (agent1_wins + agent2_wins + draws) * 100`，即以完成对战（排除异常）为分母，展示的是 agent1 的胜率。

#### （4）详细对战记录

文本格式的胜负统计：
```
agent1 vs agent2: 3胜 1负 0平 (胜率 75.0%)
```

#### （5）先后手胜率

按先手/后手分别统计 agent1 的表现：

```python
# 先手统计
agent1_first_move_battles  # agent1 为先手的对局数
agent1_first_move_wins     # 其中 agent1 获胜的对局数
first_win_rate = first_wins / first_battles * 100 (当 first_battles > 0)
first_losses = first_battles - first_wins

# 后手统计
agent1_second_move_battles  # agent1 为后手的对局数
agent1_second_move_wins     # 其中 agent1 获胜的对局数
second_win_rate = second_wins / second_battles * 100 (当 second_battles > 0)
second_losses = second_battles - second_wins
```

输出示例：
```
【先后手胜率】
agent1 - 先手: 2胜 0负 (100.0%)
agent1 - 后手: 1胜 1负 (50.0%)
```

#### （6）报告尾部

```
============================================================
报告生成时间: 2025-05-15 10:32:15
============================================================
```

### 10.5.4 便捷函数

[`generate_report()`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/report_generator.py#L70-L72) 是一个模块级便捷函数，等同于创建 `ReportGenerator` 实例并调用 `generate()`。

---

## 10.6 ResultAggregator — 结果聚合器

### 10.6.1 概述

[ResultAggregator](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/result_aggregator.py#L4-L62) 接收所有单局对战结果列表，聚合生成统一的统计摘要。

### 10.6.2 `aggregate(results: List[Dict]) -> Dict`

输入为对战结果的列表（每个元素是单局对战的字典），输出为聚合后的统计字典。

### 10.6.3 聚合字段

**输入字段（每条 result 记录需包含）：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `start_time` | datetime-like | 对战开始时间 |
| `end_time` | datetime-like | 对战结束时间 |
| `duration` | float | 对战耗时（秒） |
| `result` | str | 结果类型：`'agent1_win'` / `'agent2_win'` / `'draw'` / `'error'` |
| `total_rounds` | int | 总回合数 |
| `first_player` | int | 先手方：0=agent1 先手，否则 agent2 先手 |

**输出聚合字典字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `total_battles` | `int` | 总对战数（= `len(results)`） |
| `completed_battles` | `int` | 正常完成的对战数（result != 'error'） |
| `error_battles` | `int` | 异常对战数 |
| `agent1_wins` | `int` | Agent1 胜场数 |
| `agent2_wins` | `int` | Agent2 胜场数 |
| `draws` | `int` | 平局数 |
| `agent1_first_move_wins` | `int` | Agent1 先手时获胜场次 |
| `agent1_first_move_battles` | `int` | Agent1 先手的对局总数 |
| `agent1_second_move_wins` | `int` | Agent1 后手时获胜场次 |
| `agent1_second_move_battles` | `int` | Agent1 后手的对局总数 |
| `total_duration` | `float` | 总耗时（秒） |
| `avg_rounds` | `float` | 平均回合数（仅统计完成对战） |
| `start_time` | datetime | 最早的对战开始时间 |
| `end_time` | datetime | 最晚的对战结束时间 |

### 10.6.4 先后手分离逻辑

聚合时，仅对 `result != 'error'` 的正常对战进行先后手统计：

```python
if result['first_player'] == 0:
    aggregated['agent1_first_move_battles'] += 1
    if result['result'] == 'agent1_win':
        aggregated['agent1_first_move_wins'] += 1
else:
    aggregated['agent1_second_move_battles'] += 1
    if result['result'] == 'agent1_win':
        aggregated['agent1_second_move_wins'] += 1
```

注意：先后手统计只记录 **agent1 视角**——不单独统计 agent2 的先后手表现，agent2 的先后手表现是 agent1 视角的互补（agent1 先手 = agent2 后手，反之亦然）。

### 10.6.5 边界处理

- 如果 `results` 为空列表，返回的聚合字典中所有计数字段为 0，`start_time` 和 `end_time` 为 `None`
- `avg_rounds` 仅在 `completed_battles > 0` 时计算，否则保持默认值 0.0

### 10.6.6 便捷函数

[`aggregate_results()`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/result_aggregator.py#L65-L67) 是一个模块级便捷函数。

---

## 10.7 日志系统工具模块

### 10.7.1 `battle/utils/timing.py` — 计时工具

#### Timer 类

[Timer](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/utils/timing.py#L6-L106) 是一个独立的计时器类，提供 `timing_start()` / `timing_stop()` 配对机制、阈值警告功能和时间聚合。

```python
class Timer:
    def __init__(self, warning_threshold: float = 2.0):
```

**核心方法：**

| 方法 | 说明 |
|------|------|
| `timing_start(operation)` | 记录操作开始时间 |
| `timing_stop(operation, duration=None)` | 停止计时，返回耗时。支持直接传入 `duration` 跳过内部计算 |
| `timing_warning(operation) -> bool` | 检查是否有超时警告 |
| `get_timing_total(operation) -> float` | 获取某操作的累计总时间 |
| `get_all_timings() -> Dict[str, float]` | 获取所有操作的计时记录 |
| `get_warning_count(operation) -> int` | 获取某操作的警告次数 |
| `reset()` | 重置所有记录 |

**内部数据结构：**

- `_timing_starts: Dict[str, datetime]` — 进行中的计时起点
- `_timings: Dict[str, float]` — 各操作的累计耗时
- `_timing_warnings: Dict[str, int]` — 各操作的超时警告计数

**超时警告机制：** 当 `timing_stop()` 中计算出的 `duration > warning_threshold` 时，对应操作的警告计数 +1。

#### timer_decorator 装饰器

[`timer_decorator()`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/utils/timing.py#L109-L140) 提供了函数级别的计时能力：

```python
def timer_decorator(timer=None, operation=None, warning_threshold=2.0):
```

- 若不传入 `timer`，装饰器内部自动创建一个新的 `Timer` 实例
- 若不传入 `operation`，默认使用被装饰函数的 `__name__`
- 使用 `functools.wraps` 保留原函数的元数据
- 在 `finally` 块中执行 `timing_stop()`，确保异常情况下也能记录耗时

### 10.7.2 `battle/utils/exception_logging.py` — 异常日志

#### `log_exception(logger_obj, category, context, exception)`

[`log_exception()`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/utils/exception_logging.py#L17-L43) 是统一的异常日志记录函数：

- 拼接异常类型、异常消息、上下文信息和完整堆栈跟踪
- 如果 `logger_obj` 不为 `None`，调用 `logger_obj.error(category, error_msg)` 记录
- 如果 `logger_obj` 为 `None`，降级使用全局 loguru logger

#### `get_error_context(**kwargs) -> Dict`

[`get_error_context()`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/utils/exception_logging.py#L46-L60) 返回包含环境信息的标准错误上下文：

- `pid`: 当前进程 ID
- `cwd`: 当前工作目录
- `python_version`: Python 版本号
- `sys_path_length`: `sys.path` 的长度
- 支持通过 `**kwargs` 追加自定义字段

#### `safe_execute(func, default_return, log_level, context, logger_obj, reraise)`

[`safe_execute()`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/utils/exception_logging.py#L63-L92) 是一个安全执行包装函数（也可作装饰器使用）：

- 在 `try/except` 中执行 `func()`
- 异常时调用 `log_exception()` 记录
- `reraise=True`（默认）时重新抛出异常；`reraise=False` 时返回 `default_return`

### 10.7.3 `battle/utils/process_isolation.py` — 进程隔离

#### `process_isolation(extra_isolated_modules, preserve_modules)`

[`process_isolation()`](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/utils/process_isolation.py#L11-L62) 是一个 `@contextmanager` 上下文管理器，用于在多进程 `fork` 出的子进程中隔离特定模块的加载。

**隔离逻辑：**
1. 保存原始的 `sys.path`、`sys.modules`（部分）和 `os.getcwd()`
2. 遍历 `sys.modules`，移除所有以 `ISOLATED_MODULE_PREFIXES`（`'ai'`、`'common'`、`'SDK'`、`'AI'`）或 `extra_isolated_modules` 为前缀的模块
3. 同时支持通过精确模块名（`key in ISOLATED_MODULE_PREFIXES`）匹配
4. 跳过 `preserve_modules` 集合中指定的模块（不会被删除）

**恢复逻辑（`finally` 块）：**
1. 恢复 `os.getcwd()`
2. 恢复 `sys.path`
3. 将被删除的模块重新放回 `sys.modules`

**使用示例：**

```python
with process_isolation(extra_isolated_modules=['my_module']):
    # 在此上下文中 'my_module' 及 ISOLATED_MODULE_PREFIXES 下的模块不可见
    ...
```

**设计目的：** 在子进程中进行 Baseline Battle 时，不同 Agent 可能依赖不同版本的 SDK 或模型库，通过进程隔离确保每个子进程加载的是独立、干净的模块环境，避免模块缓存导致的版本冲突。

---

## 10.8 系统交互流程

### 10.8.1 Baseline Battle 日志流程

```
主进程:
  BattleLogger.__init__()
    ├── _create_log_dir()           # 创建 {agent1}_vs_{agent2}_{timestamp}/ 目录
    ├── _configure_logger()         # 配置 6 个 loguru handler
    └── atexit.register(_cleanup)   # 退出时清理

  for 并行子进程:
    ├── cloudpickle 序列化 agent
    └── fork 子进程

子进程:
  SubProcessLogger.__init__()
    ├── log_battle_start(episode)
    ├── (for 每个回合) log_round(...)
    ├── log_battle_end(result)
    └── 返回 result dict → 主进程

主进程:
  └── BattleLogger.log_result(result)    # 追加 JSONL + 累加计数器

所有对战完成后:
  ├── ResultAggregator.aggregate(all_results)  # 聚合
  ├── ReportGenerator.generate(aggregated, ...) # 生成文本报告
  └── BattleLogger.close()
        ├── write_summary()            # 输出摘要
        └── _cleanup()                 # 移除 handler + 排空队列
```

### 10.8.2 SelfPlay 日志流程

```
SelfPlayLogger.__init__()
  # 无额外 handler 配置，复用 loguru 全局 logger

for 每个 episode:
  ├── log_battle_start(episode, opponent_id, player_position)
  ├── (for 每个回合) log_round(episode, round_num, ...)
  ├── log_battle_end(episode, result, total_rounds, duration, final_hp)
  └── record_battle_result(result, duration, total_rounds)   # 累加统计

周期性:
  ├── print_summary()                # 输出当前进度汇总
  ├── get_statistics()               # 获取统计 dict（用于上报）
  ├── print_performance_report()     # 性能报告
  └── reset_statistics()            # 可选：重置计数器
```

---

## 10.9 源码文件索引

| 文件 | 说明 |
|------|------|
| [battle/battle_logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/battle_logger.py) | LogLevel 枚举、BattleLogger（主进程）、SubProcessLogger（子进程） |
| [trainer/selfplay_logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay_logger.py) | SelfPlayLogLevel 枚举、SelfPlayLogger（训练日志） |
| [battle/report_generator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/report_generator.py) | ReportGenerator（报告生成） |
| [battle/result_aggregator.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/result_aggregator.py) | ResultAggregator（结果聚合） |
| [battle/utils/exception_logging.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/utils/exception_logging.py) | log_exception、get_error_context、safe_execute |
| [battle/utils/process_isolation.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/utils/process_isolation.py) | process_isolation 上下文管理器 |
| [battle/utils/timing.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/battle/utils/timing.py) | Timer 类、timer_decorator 装饰器 |
