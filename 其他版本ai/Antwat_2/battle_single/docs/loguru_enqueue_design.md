# Loguru enqueue=True 均匀输出设计方案

## 1. 现状分析

### 1.1 当前实现架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           主进程 (Parent)                            │
│  ┌─────────────┐                                                     │
│  │ BattleLogger │── logger.remove() 移除默认 handler                  │
│  └─────────────┘                                                     │
│          │                                                           │
│          ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  logger.add(sys.stderr, enqueue=False)  # 控制台             │   │
│  │  logger.add(debug.log, enqueue=True)     # 后台线程队列      │   │
│  │  logger.add(info.log, enqueue=True)      # 后台线程队列      │   │
│  │  logger.add(warning.log, enqueue=True)   # 后台线程队列      │   │
│  │  logger.add(error.log, enqueue=True)     # 后台线程队列      │   │
│  │  logger.add(all.log, enqueue=True)       # 后台线程队列      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              │ fork()                                │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                         子进程 (Child)                         │   │
│  │  setup_subprocess_logger() 调用 logger.remove() ──✗ 问题！    │   │
│  │  然后重新添加 enqueue=True 的 handlers                         │   │
│  │  这会导致创建新的队列，而不是使用父进程的队列                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 当前实现问题

| 问题 | 描述 | 影响 |
|------|------|------|
| **Redundant Queue Creation** | 子进程调用 `logger.remove()` 后重新添加 handlers，创建新的队列 | 内存浪费，队列管理混乱 |
| **Queue Inheritance Issue** | fork 后子进程继承父进程的 SimpleQueue，但 remove() 会关闭它 | 日志可能丢失或写入失败 |
| **Duplicate report.txt** | CLI 和子进程都写入 report.txt | 文件内容重复或损坏 |
| **time.log Redundancy** | timing 信息同时写入 debug.log 和 time.log | 冗余存储 |
| **Console Handler in Child** | 子进程继承了父进程的控制台 handler | 子进程日志输出到控制台 |

### 1.3 日志文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| debug.log | DEBUG 级别日志 | 保留 |
| info.log | INFO 级别日志 | 保留 |
| warning.log | WARNING 级别日志 | 保留 |
| error.log | ERROR 级别日志 | 保留 |
| all.log | 所有级别日志 | 保留 |
| time.log | 性能计时日志 | **合并到 debug.log** |
| report.txt | 对战报告（CLI 生成） | **删除**（改为仅通过 logger 输出） |
| results.json | 对战结果 JSON | 保留 |

---

## 2. 目标架构

### 2.1 正确使用 enqueue=True 的架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           主进程 (Parent)                            │
│  ┌─────────────┐                                                     │
│  │ BattleLogger │                                                   │
│  └─────────────┘                                                     │
│          │                                                           │
│          ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  logger.add(sys.stderr, enqueue=False)  # 控制台，非阻塞      │   │
│  │  logger.add(debug.log, enqueue=True, level="DEBUG")         │   │
│  │  logger.add(info.log, enqueue=True, level="INFO")           │   │
│  │  logger.add(warning.log, enqueue=True, level="WARNING")     │   │
│  │  logger.add(error.log, enqueue=True, level="ERROR")         │   │
│  │  logger.add(all.log, enqueue=True, level="DEBUG")            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│         ┌────────────────────┼────────────────────┐                 │
│         │ fork()             │ fork()             │ fork()         │
│         ▼                    ▼                    ▼                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐         │
│  │   子进程 1   │     │   子进程 2   │     │   子进程 N   │         │
│  │              │     │              │     │              │         │
│  │  直接使用    │     │  直接使用    │     │  直接使用    │         │
│  │  继承的      │     │  继承的      │     │  继承的      │         │
│  │  logger     │     │  logger     │     │  logger     │         │
│  │              │     │              │     │              │         │
│  │  无需重新    │     │  无需重新    │     │  无需重新    │         │
│  │  配置        │     │  配置        │     │  配置        │         │
│  └─────────────┘     └─────────────┘     └─────────────┘         │
│         │                    │                    │               │
│         └────────────────────┼────────────────────┘               │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │         multiprocessing.SimpleQueue (继承自父进程)            │   │
│  │                    所有进程的日志消息                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              loguru-writer 后台线程 (daemon)                  │   │
│  │                  统一处理所有队列消息                          │   │
│  │                  实现均匀输出                                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计原则

1. **单一队列来源**：所有进程共享同一个 SimpleQueue，由一个后台 writer 线程统一写出
2. **fork 继承**：子进程通过 fork 继承父进程的 logger 状态，包括队列
3. **子进程不重新配置**：子进程直接使用继承的 logger，不调用 `logger.remove()`
4. **主进程控制清理**：主进程调用 `logger.complete()` 等待所有日志处理完成
5. **子进程主动完成**：子进程在退出前调用 `logger.complete()` 确保本地日志入队

---

## 3. 详细设计

### 3.1 battle_single_logger.py 修改

**需要修改的内容：**

```python
# 修改 _configure_logger 方法
def _configure_logger(self):
    """配置 loguru handlers"""
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>[{extra[category]}]</cyan> | "
        "<level>{message}</level>"
    )

    # 控制台 handler（enqueue=False，即时输出）
    if self.log_to_console:
        logger.add(
            sys.stderr,
            format=log_format,
            level="DEBUG",
            enqueue=False,
            colorize=True
        )

    # 文件 handlers（enqueue=True，异步队列）
    log_files = {
        'debug.log': 'DEBUG',
        'info.log': 'INFO',
        'warning.log': 'WARNING',
        'error.log': 'ERROR',
        'all.log': 'DEBUG'
    }

    for filename, level in log_files.items():
        logger.add(
            os.path.join(self._log_dir, filename),
            format=log_format,
            level=level,
            enqueue=True,
            serialize=False
        )
```

**需要删除的内容：**
- 无需删除内容，但 `_cleanup` 方法需要增强

**需要添加的内容：**

```python
def _cleanup(self):
    """清理资源，确保所有日志写入完成"""
    logger.info('battle', "Logger cleanup: waiting for queue to flush")
    logger.complete()
    import time
    time.sleep(0.1)  # 等待 daemon 线程写入完成
```

### 3.2 battle_single_simulator.py 修改

**需要删除的内容：**

```python
# 删除整个 setup_subprocess_logger 函数（约 50 行）
def setup_subprocess_logger(log_dir: str, log_level: str = "DEBUG"):
    """在子进程中配置 loguru logger（fork 后调用）"""
    # ... 整个函数删除
```

**需要修改的内容：**

```python
def run_single_battle_process(task):
    """
    在独立进程中执行单局对战

    Args:
        task: 任务参数 (episode, seed, first_player, agent1_name, agent2_name, max_rounds, log_dir)

    Returns:
        对战结果字典
    """
    episode, seed, first_player, agent1_name, agent2_name, max_rounds, log_dir = task

    # 删除：setup_subprocess_logger(log_dir)  # 不再需要！
    # 子进程直接使用 fork 继承的 logger

    try:
        # ... 后续代码不变 ...

        # 在返回前确保日志写入
        logger.complete()
        return result

    except Exception as e:
        # ...
        logger.complete()
        return result
```

### 3.3 battle_single_cli.py 修改

**需要删除的内容：**

```python
# 删除 save_results 函数中对 report.txt 的写入
# 因为 report.txt 是冗余的，报告内容已经通过 logger.info() 输出

# 原代码:
def save_results(results, aggregated, output_report_file, output_result_json):
    output_dir = os.path.dirname(output_report_file) or '/tmp/battle_single'
    os.makedirs(output_dir, exist_ok=True)

    # 删除这一行：写入 report.txt 是冗余的
    # with open(output_report_file, 'w', encoding='utf-8') as f:
    #     f.write(generate_report(aggregated, ...))

    # 保留 results.json 的写入即可
    with open(output_result_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
```

或者更简单地，**直接删除 `output_report_file` 参数**：

```python
def save_results(results, aggregated, output_result_json):
    output_dir = os.path.dirname(output_result_json) or '/tmp/battle_single'
    os.makedirs(output_dir, exist_ok=True)

    with open(output_result_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
```

对应修改调用处：
```python
# 在 main() 中
save_results(results, aggregated, config.output_result_json)  # 删除 output_report_file
```

### 3.4 battle_single_config.py 修改

**需要删除的内容：**

```python
# 删除 output_report_file 配置项
self.output_report_file: str = "/tmp/battle_single/report.txt"  # 删除此行
```

**需要修改的内容：**

```python
# 在 load_from_file 中删除 report_file 的加载
if 'output' in config:
    output = config['output']
    # 删除这段：
    # if 'report_file' in output:
    #     self.output_report_file = output['report_file']
    if 'result_json' in output:
        self.output_result_json = output['result_json']

# 在 load_from_args 中删除 --output 参数的处理
# if hasattr(args, 'output') and args.output is not None:
#     self.output_report_file = args.output
```

---

## 4. 子进程控制台 Handler 风险分析

### 4.1 风险点

当子进程通过 fork 继承父进程的 logger 时，如果父进程配置了 `enqueue=False` 的控制台 handler（写入 `sys.stderr`），子进程也会继承这个 handler。这意味着：

- 子进程的日志会直接写入**父进程终端**（通过继承的文件描述符）
- 这可能导致输出混乱（多进程同时写同一终端）

### 4.2 解决方案

**方案一：子进程不输出到控制台（推荐）**

由于 `enqueue=False` 的控制台 handler 是继承的，子进程的日志会直接写到终端。但这是**可接受的行为**，因为：
- 每个子进程的输出会混在一起（这正是我们想要的"均匀输出"效果）
- 如果用户不想看到子进程输出，可以设置 `log_to_console=False`

```python
# 在 CLI 中
logger = BattleLogger(
    agent1_name=config.agent1_name,
    agent2_name=config.agent2_name,
    log_level=log_level,
    log_to_console=False,  # 子进程继承后也不会输出到控制台
    enable_timing=config.logging_enable_timing
)
```

**方案二：子进程显式禁用控制台**

如果需要更精确的控制，可以在子进程中显式移除控制台 handler：

```python
def run_single_battle_process(task):
    # 移除控制台 handler（继承自父进程）
    logger.remove(0)  # 移除第一个 handler（控制台）
    # 保留文件 handlers（enqueue=True 的那些）
    # ... 后续代码 ...
```

**推荐方案一**：让用户通过 `--no-console-log` 参数控制，这样架构最简单。

---

## 5. 进程启动方式与 enqueue=True 兼容性

### 5.1 fork vs spawn

| 启动方式 | enqueue=True 行为 | 适用场景 |
|----------|-------------------|----------|
| **fork** | 子进程继承父进程的 SimpleQueue 和文件描述符 | Linux 默认，推荐 |
| spawn | 子进程不继承父进程状态，需要重新配置 logger | Windows 或明确需要隔离 |

**当前实现使用 fork（默认在 Linux 上）**，这是正确的选择。

### 5.2 fork 时的队列继承

当使用 `fork` 时：
1. 父进程创建 `SimpleQueue` 并启动后台 writer 线程
2. 子进程通过 fork 继承队列对象和文件描述符
3. 子进程的日志消息进入同一个队列
4. 后台 writer 线程（在父进程中）统一处理所有消息

```
Parent Process:
├── SimpleQueue (共享)
├── loguru-writer thread (处理队列)
└── File handlers (写入文件)

Child Process (fork):
├── SimpleQueue (继承自父进程)
└── File handlers (继承自父进程)
```

### 5.3 日志完整性的保证

为确保所有日志都被写入：

1. **主进程结束时调用 `logger.complete()`**
   ```python
   logger.close()  # 内部调用 logger.complete()
   ```

2. **子进程退出前调用 `logger.complete()`**
   ```python
   def run_single_battle_process(task):
       try:
           # ... 执行任务 ...
       finally:
           logger.complete()  # 确保本地日志入队
           return result
   ```

3. **使用 atexit 确保清理**
   ```python
   atexit.register(self._cleanup)
   ```

---

## 6. 日志初始化和销毁时序

### 6.1 初始化时序

```
1. main() 创建 BattleLogger 实例
   └── BattleLogger.__init__()
       ├── _create_log_dir()     # 创建日志目录
       ├── _configure_logger()   # 配置 handlers (enqueue=True)
       └── atexit.register(_cleanup)  # 注册退出清理

2. 创建 BattleSingleSimulator 实例

3. 启动多进程对战
   └── ProcessPoolExecutor.submit(run_single_battle_process, task)
       └── 每个子进程继承父进程的 logger
```

### 6.2 销毁时序

```
1. 所有对战任务完成（子进程返回）

2. 主进程收集结果
   └── executor.shutdown(wait=True)

3. 主进程调用 logger.close()
   └── BattleLogger.close()
       ├── write_summary()  # 写入汇总信息
       └── _cleanup()
           ├── logger.complete()  # 等待队列清空
           └── time.sleep(0.1)    # 确保写入完成

4. 子进程退出前
   └── run_single_battle_process() 返回前
       └── logger.complete()  # 确保本地日志入队
```

### 6.3 异常情况处理

| 场景 | 处理方式 |
|------|----------|
| 子进程异常退出 | 父进程 `future.result()` 会捕获异常 |
| 主进程异常终止 | daemon writer 线程会被强制终止，可能丢失未处理的日志 |
| 队列积压过多 | 无界队列可能导致内存增长，需监控 |

---

## 7. 代码修改清单

### 7.1 需要删除的代码

| 文件 | 删除内容 | 原因 |
|------|----------|------|
| battle_single_simulator.py | `setup_subprocess_logger` 函数（约 50 行） | 子进程不需要重新配置 logger |
| battle_single_cli.py | `save_results` 中写入 report.txt 的代码 | report.txt 是冗余的 |
| battle_single_cli.py | `generate_report` 函数（如不需要保留） | 报告已通过 logger 输出 |
| battle_single_config.py | `output_report_file` 配置项 | 不再需要 |

### 7.2 需要修改的代码

| 文件 | 修改内容 |
|------|----------|
| battle_single_logger.py | `_configure_logger` 方法（保持不变，可优化） |
| battle_single_logger.py | `_cleanup` 方法（增强，确保日志写入） |
| battle_single_simulator.py | `run_single_battle_process` 函数（删除 setup_subprocess_logger 调用） |
| battle_single_cli.py | `save_results` 函数调用（删除 output_report_file 参数） |

### 7.3 需要添加的代码

| 文件 | 添加内容 |
|------|----------|
| battle_single_simulator.py | 在 `run_single_battle_process` 返回前添加 `logger.complete()` 调用（如果还没有） |

---

## 8. 验证检查清单

### 8.1 功能验证

- [ ] 多进程对战正常执行
- [ ] 日志文件正常生成（debug.log, info.log, warning.log, error.log, all.log）
- [ ] results.json 正常生成
- [ ] 子进程的日志正确汇入主进程的队列
- [ ] 日志输出均匀，无明显交错或丢失

### 8.2 文件验证

- [ ] report.txt 不再生成
- [ ] time.log 不再生成（或内容已合并到 debug.log）
- [ ] debug.log 包含 timing 信息

### 8.3 清理验证

- [ ] 主进程退出时所有日志都已写入
- [ ] 子进程退出时日志都已入队
- [ ] 无僵尸 writer 线程

---

## 9. 最小化改动原则

本次改造遵循**最小化改动原则**：

1. **不改变日志目录命名规则**：保持 `{agent1}_vs_{agent2}_{timestamp}` 格式
2. **不改变日志内容**：日志消息格式保持不变
3. **只删除冗余**：删除 report.txt 和 time.log 的生成逻辑
4. **利用现有机制**：利用 fork 的继承特性和 enqueue=True 的队列共享

### 9.1 改动统计

| 类型 | 数量 | 说明 |
|------|------|------|
| 删除函数 | 1 个 | `setup_subprocess_logger` |
| 修改函数 | 2 个 | `run_single_battle_process`, `save_results` |
| 删除配置项 | 1 个 | `output_report_file` |
| 代码行数减少 | ~70 行 | 删除冗余代码 |

---

## 10. 注意事项汇总

### 10.1 子进程控制台 Handler

- 如果 `log_to_console=True`，子进程会继承控制台 handler，日志直接输出到终端
- 这是预期行为，实现了"均匀输出"
- 如不需要，可设置 `log_to_console=False`

### 10.2 进程启动方式

- 当前使用 `fork`（Linux 默认）
- 子进程继承父进程的 SimpleQueue 和文件描述符
- 适合 Linux 运行环境

### 10.3 日志初始化

- `BattleLogger.__init__` 中完成所有配置
- 子进程直接使用继承的 logger，无需重新配置
- 使用 `atexit` 注册清理函数

### 10.4 日志销毁

- 子进程退出前调用 `logger.complete()` 确保日志入队
- 主进程退出前调用 `logger.complete()` 等待队列处理完成
- 使用 `time.sleep(0.1)` 确保 daemon 线程写入完成

### 10.5 内存考虑

- enqueue=True 使用无界队列
- 如果日志产生速度 > 处理速度，可能导致内存增长
- 当前场景（日志量可控）无风险
