# Battle Single 日志系统详细设计文档

## 1. 需求分析

### 1.1 核心需求

| 序号 | 需求 | 描述 | 优先级 |
|------|------|------|--------|
| 1 | 性能调试 | 能够识别 battle 运行缓慢的原因，定位主要耗时步骤 | 高 |
| 2 | 分级日志 | 不同内容输出到不同日志级别（DEBUG/INFO/WARNING/ERROR） | 高 |
| 3 | 日志命名 | 使用 `agent1_vs_agent2_yyyymmdd_hhmiss` 命名规则 | 高 |
| 4 | 结果输出 | 对战结果单独输出到文件，每局结束立即输出 | 高 |

### 1.2 日志级别定义

| 级别 | 用途 | 输出内容示例 |
|------|------|--------------|
| **DEBUG** | 详细调试信息 | 每回合操作、状态变化、函数调用耗时 |
| **INFO** | 一般信息 | 对战开始/结束、进度更新、配置信息 |
| **WARNING** | 警告信息 | 异常情况但不影响运行、性能警告 |
| **ERROR** | 错误信息 | 严重错误、导致对战失败的异常 |

---

## 2. 日志系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Battle Single 日志系统                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐    │
│   │   CLI入口    │───▶│   日志管理器  │───▶│   日志输出器     │    │
│   └──────────────┘    └──────────────┘    └──────────────────┘    │
│                              │                      │               │
│                              ▼                      ▼               │
│                     ┌─────────────┐       ┌────────────────┐       │
│                     │ 性能计时器  │       │ 结果汇总器     │       │
│                     └─────────────┘       └────────────────┘       │
│                              │                      │               │
│                              ▼                      ▼               │
│                     ┌───────────────────────────────────────┐       │
│                     │          日志存储目录                   │       │
│                     │  /tmp/battle_single/logs/             │       │
│                     │  ├─ agent1_vs_agent2_20260508_103000/ │       │
│                     │  │   ├─ debug.log                    │       │
│                     │  │   ├─ info.log                     │       │
│                     │  │   ├─ warning.log                   │       │
│                     │  │   ├─ error.log                     │       │
│                     │  │   ├─ results.json                  │       │
│                     │  │   └─ timing.log                     │       │
│                     │  └─ ...                              │       │
│                     └───────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 核心组件说明

| 组件 | 职责 | 说明 |
|------|------|------|
| **日志管理器** | 统一管理日志输出，按级别分流 | 负责日志级别控制、文件创建、输出路由 |
| **性能计时器** | 记录各步骤耗时 | 用于定位性能瓶颈 |
| **结果汇总器** | 收集和输出对战结果 | 每局结束立即输出结果 |
| **日志输出器** | 实际写入日志文件 | 支持按级别分离输出 |

---

## 3. 日志文件结构

### 3.1 目录结构

```
/tmp/battle_single/
├── logs/
│   └── sample_vs_BasicRandomAI_20260508_103000/
│       ├── debug.log        # DEBUG级别日志
│       ├── info.log         # INFO级别日志
│       ├── warning.log      # WARNING级别日志
│       ├── error.log        # ERROR级别日志
│       ├── results.json     # 对战结果JSON
│       ├── results.txt      # 对战结果文本报告
│       └── timing.log       # 性能计时日志
```

### 3.2 文件命名规则

```
{agent1}_vs_{agent2}_{yyyyMMdd}_{HHmmss}/
├── {level}.log               # 分级日志文件
├── results.json              # 结构化结果
├── results.txt               # 可读文本结果
└── timing.log                # 性能计时
```

---

## 4. 日志内容设计

### 4.1 DEBUG 级别日志

**内容类型**：详细调试信息，用于问题定位

| 类别 | 格式 | 示例 |
|------|------|------|
| 函数调用 | `[DEBUG] [func] {function_name}() - {args}` | `[DEBUG] [func] run_battle(agent1=sample, agent2=BasicRandomAI)` |
| 回合操作 | `[DEBUG] [round] #{episode}-{round} player={player} ops={operations}` | `[DEBUG] [round] #0-1 player=0 ops=[BUILD_TOWER(4,6)]` |
| 状态变化 | `[DEBUG] [state] #{episode} hp1={hp1} hp2={hp2} coins1={c1} coins2={c2}` | `[DEBUG] [state] #0 hp1=50 hp2=45 coins1=30 coins2=25` |
| 耗时记录 | `[DEBUG] [time] {operation} took {duration}ms` | `[DEBUG] [time] agent1.choose_operations took 156ms` |

### 4.2 INFO 级别日志

**内容类型**：一般信息，用于追踪流程

| 类别 | 格式 | 示例 |
|------|------|------|
| 对战开始 | `[INFO] [battle] START #{episode}: {agent1} vs {agent2}` | `[INFO] [battle] START #0: sample vs BasicRandomAI` |
| 对战结束 | `[INFO] [battle] END #{episode}: {result} (rounds={rounds}, time={time}s)` | `[INFO] [battle] END #0: agent1_win (rounds=15, time=2.3s)` |
| 进度更新 | `[INFO] [progress] {completed}/{total} battles done` | `[INFO] [progress] 10/20 battles done` |
| 配置信息 | `[INFO] [config] episodes={episodes}, workers={workers}, max_rounds={max_rounds}` | `[INFO] [config] episodes=10, workers=4, max_rounds=512` |

### 4.3 WARNING 级别日志

**内容类型**：警告信息，不影响运行但需关注

| 类别 | 格式 | 示例 |
|------|------|------|
| 超时警告 | `[WARNING] [timeout] {operation} exceeded {threshold}ms` | `[WARNING] [timeout] agent.choose_operations exceeded 500ms` |
| 异常恢复 | `[WARNING] [recover] {error_type} occurred, continuing` | `[WARNING] [recover] ValueError occurred, continuing` |
| 性能警告 | `[WARNING] [perf] High memory usage detected: {usage}MB` | `[WARNING] [perf] High memory usage detected: 2048MB` |

### 4.4 ERROR 级别日志

**内容类型**：严重错误，影响运行

| 类别 | 格式 | 示例 |
|------|------|------|
| Agent加载失败 | `[ERROR] [load] Failed to load agent {agent_name}: {error}` | `[ERROR] [load] Failed to load agent gen99: ModuleNotFoundError` |
| 对战失败 | `[ERROR] [battle] Battle #{episode} failed: {error}` | `[ERROR] [battle] Battle #5 failed: RuntimeError` |
| 致命错误 | `[ERROR] [fatal] {error} - shutting down` | `[ERROR] [fatal] Out of memory - shutting down` |

### 4.5 结果文件 (results.json)

**每局输出一条记录**：

```json
{
    "episode": 0,
    "agent1_name": "sample",
    "agent2_name": "BasicRandomAI",
    "first_player": "sample",
    "result": "agent1_win",
    "total_rounds": 15,
    "duration": 2.345,
    "final_hp": {"agent1": 50, "agent2": 0},
    "final_coins": {"agent1": 120, "agent2": 45},
    "start_time": "2026-05-08 10:30:01",
    "end_time": "2026-05-08 10:30:03",
    "error": null
}
```

### 4.6 计时日志 (timing.log)

**记录各步骤耗时**：

```
[2026-05-08 10:30:01.000] [TIMING] init_battle: 0.005s
[2026-05-08 10:30:01.005] [TIMING] agent1_choose_operations: 0.156s
[2026-05-08 10:30:01.161] [TIMING] agent2_choose_operations: 0.089s
[2026-05-08 10:30:01.250] [TIMING] resolve_turn: 0.023s
[2026-05-08 10:30:01.273] [TIMING] round_1_total: 0.268s
...
[2026-05-08 10:30:03.345] [TIMING] battle_total: 2.345s
[2026-05-08 10:30:03.345] [TIMING] agent1_total_time: 1.820s
[2026-05-08 10:30:03.345] [TIMING] agent2_total_time: 0.450s
[2026-05-08 10:30:03.345] [TIMING] resolve_total_time: 0.075s
```

---

## 5. 性能调试方案

### 5.1 耗时分析维度

| 维度 | 测量内容 | 定位目的 |
|------|----------|----------|
| **Agent加载** | 加载agent所需时间 | 识别加载慢的agent |
| **操作决策** | `choose_operations`调用耗时 | 识别思考慢的agent |
| **回合执行** | `resolve_turn`执行耗时 | 识别SDK性能问题 |
| **线程调度** | 等待线程池时间 | 识别线程竞争问题 |
| **整体耗时** | 单场对战总耗时 | 评估整体性能 |

### 5.2 性能指标阈值

| 指标 | 正常范围 | 警告阈值 | 错误阈值 |
|------|----------|----------|----------|
| 单回合耗时 | < 100ms | 500ms | 2000ms |
| Agent决策耗时 | < 50ms | 200ms | 1000ms |
| Agent加载耗时 | < 5s | 30s | 120s |
| 单场对战耗时 | < 30s | 120s | 600s |

### 5.3 性能瓶颈定位流程

```
1. 启动对战时记录开始时间
2. 每回合记录：
   - agent1决策耗时
   - agent2决策耗时
   - 回合执行耗时
3. 对战结束时汇总：
   - 总回合数
   - 各agent总决策时间
   - 总执行时间
4. 输出性能报告：
   - 耗时占比分析
   - 瓶颈定位建议
```

---

## 6. 代码实现设计

### 6.1 日志管理器类设计

```python
class BattleLogger:
    def __init__(self, agent1_name: str, agent2_name: str, log_level: LogLevel = LogLevel.INFO):
        self.agent1_name = agent1_name
        self.agent2_name = agent2_name
        self.log_level = log_level
        self.log_dir = self._create_log_dir()
        self._open_files()
        
    def _create_log_dir(self) -> str:
        """创建日志目录，命名格式：agent1_vs_agent2_yyyymmdd_hhmiss"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dir_name = f"{self.agent1_name}_vs_{self.agent2_name}_{timestamp}"
        dir_path = os.path.join('/tmp/battle_single/logs', dir_name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
        
    def _open_files(self):
        """打开各级别日志文件"""
        self.debug_file = open(os.path.join(self.log_dir, 'debug.log'), 'w', encoding='utf-8')
        self.info_file = open(os.path.join(self.log_dir, 'info.log'), 'w', encoding='utf-8')
        self.warning_file = open(os.path.join(self.log_dir, 'warning.log'), 'w', encoding='utf-8')
        self.error_file = open(os.path.join(self.log_dir, 'error.log'), 'w', encoding='utf-8')
        self.results_file = open(os.path.join(self.log_dir, 'results.json'), 'w', encoding='utf-8')
        self.timing_file = open(os.path.join(self.log_dir, 'timing.log'), 'w', encoding='utf-8')
        
    def debug(self, category: str, message: str):
        if self.log_level.value <= LogLevel.DEBUG.value:
            self._write_log(self.debug_file, 'DEBUG', category, message)
            
    def info(self, category: str, message: str):
        if self.log_level.value <= LogLevel.INFO.value:
            self._write_log(self.info_file, 'INFO', category, message)
            
    def warning(self, category: str, message: str):
        if self.log_level.value <= LogLevel.WARNING.value:
            self._write_log(self.warning_file, 'WARNING', category, message)
            
    def error(self, category: str, message: str):
        self._write_log(self.error_file, 'ERROR', category, message)
        
    def log_result(self, result: dict):
        """输出单局对战结果"""
        self.results_file.write(json.dumps(result, ensure_ascii=False) + '\n')
        self.results_file.flush()
        
    def log_timing(self, operation: str, duration: float):
        """记录操作耗时"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        self.timing_file.write(f"[{timestamp}] [TIMING] {operation}: {duration:.3f}s\n")
        self.timing_file.flush()
```

### 6.2 性能计时器类设计

```python
class PerformanceTimer:
    def __init__(self, logger: BattleLogger):
        self.logger = logger
        self.timings = {}
        self.start_times = {}
        
    def start(self, operation: str):
        """开始计时"""
        self.start_times[operation] = time.time()
        
    def stop(self, operation: str) -> float:
        """停止计时并返回耗时（秒）"""
        if operation not in self.start_times:
            return 0.0
        duration = time.time() - self.start_times[operation]
        self.timings[operation] = self.timings.get(operation, 0) + duration
        self.logger.log_timing(operation, duration)
        
        if duration > 0.5:
            self.logger.warning('timeout', f"{operation} took {duration:.2f}s (exceeds 0.5s)")
            
        return duration
    
    def get_total(self, operation: str) -> float:
        """获取操作总耗时"""
        return self.timings.get(operation, 0.0)
    
    def report(self):
        """输出性能报告"""
        self.logger.info('perf_report', "=== Performance Report ===")
        for operation, total_time in sorted(self.timings.items(), key=lambda x: -x[1]):
            self.logger.info('perf_report', f"{operation}: {total_time:.3f}s")
```

---

## 7. 输出文件格式

### 7.1 results.txt 格式

```
========================================
Battle Results: sample vs BasicRandomAI
========================================

[Episode 0]
Result: agent1_win
First Player: sample
Total Rounds: 15
Duration: 2.345s
Final HP: sample=50, BasicRandomAI=0
Final Coins: sample=120, BasicRandomAI=45
Start: 2026-05-08 10:30:01
End: 2026-05-08 10:30:03

[Episode 1]
Result: agent2_win
First Player: BasicRandomAI
Total Rounds: 12
Duration: 1.890s
Final HP: sample=0, BasicRandomAI=35
Final Coins: sample=80, BasicRandomAI=95
Start: 2026-05-08 10:30:03
End: 2026-05-08 10:30:05

...

========================================
Summary:
Total Battles: 20
Agent1 Wins: 12 (60.0%)
Agent2 Wins: 6 (30.0%)
Draws: 2 (10.0%)
Total Duration: 45.234s
Average Duration: 2.262s
========================================
```

---

## 8. 部署与配置

### 8.1 配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `log_level` | INFO | 日志输出级别 |
| `log_dir` | /tmp/battle_single/logs | 日志存储目录 |
| `log_to_console` | true | 是否输出到控制台 |
| `enable_timing` | true | 是否启用性能计时 |
| `results_format` | json | 结果格式（json/txt/both） |

### 8.2 命令行参数

```bash
python -m battle_single.src.battle_single_cli \
    --agent1 sample \
    --agent2 BasicRandomAI \
    --episodes 10 \
    --log-level DEBUG \
    --enable-timing
```

---

## 9. 安全考虑

| 风险点 | 描述 | 缓解措施 |
|--------|------|----------|
| 日志文件过大 | 长时间运行可能产生大量日志 | 设置日志大小限制，自动轮转 |
| 敏感信息泄露 | agent名称可能包含敏感信息 | 对输出进行脱敏处理 |
| 路径遍历攻击 | 恶意agent名称可能导致路径遍历 | 对文件名进行严格验证 |
| 资源耗尽 | 并发对战可能创建大量日志文件 | 限制同时进行的对战数量 |

---

## 10. 设计原则

### 10.1 可观测性原则

1. **完整性**：记录所有关键步骤和决策点
2. **可读性**：日志格式清晰，易于理解和分析
3. **可追溯性**：每局对战有独立的日志文件，便于回溯
4. **实时性**：结果即时输出，便于监控

### 10.2 性能影响最小化

1. 日志写入采用异步方式
2. 可配置日志级别，生产环境可关闭DEBUG
3. 避免在热点路径中记录过多日志