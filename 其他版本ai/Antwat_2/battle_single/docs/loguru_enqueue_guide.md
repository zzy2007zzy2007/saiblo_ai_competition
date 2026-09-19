# Loguru enqueue=True 特性技术指南

## 目录
1. [概述](#概述)
2. [工作原理](#工作原理)
3. [主要功能](#主要功能)
4. [应用场景](#应用场景)
5. [使用示例](#使用示例)
6. [主要注意事项](#主要注意事项)
7. [最佳实践](#最佳实践)

---

## 概述

`enqueue=True` 是 Loguru 日志库提供的一个强大特性，它通过将日志消息先放入多进程安全的队列，再由后台线程处理，实现了非阻塞的日志记录和多进程安全的日志写入。

### 核心价值
- **非阻塞日志**：日志调用不会阻塞主线程执行
- **多进程安全**：支持多进程环境下的安全日志写入
- **异步处理**：日志格式化和写入由后台线程异步完成

---

## 工作原理

### 架构设计

```
┌─────────────┐
│  主线程     │─── logger.info()
└─────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  multiprocessing.SimpleQueue    │  (线程/进程安全队列)
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  后台线程 (loguru-writer)       │─── 处理队列中的消息
└─────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  Sink       │─── 写入文件/控制台等
└─────────────┘
```

### 详细流程

1. **消息入队**：当调用 `logger.info()` 等方法时，如果启用了 `enqueue=True`，日志消息会被放入 `multiprocessing.SimpleQueue` 队列中
2. **立即返回**：主线程立即返回，不会等待日志写入完成
3. **后台处理**：Loguru 启动一个名为 `loguru-writer` 的 daemon 线程，持续从队列中取出消息
4. **日志输出**：后台线程负责格式化消息并写入到对应的 sink（文件、控制台等）

### 内部实现

根据 Loguru 源代码分析，关键实现包括：

- 使用 `multiprocessing.SimpleQueue` 作为队列（支持跨进程通信）
- 使用 `multiprocessing.Event` 和 `multiprocessing.Lock` 进行同步
- 后台线程以 daemon 模式运行，随主进程退出而退出
- 每个启用 `enqueue` 的 sink 都有独立的队列和线程

---

## 主要功能

### 1. 非阻塞日志调用

**特点**：日志操作不会阻塞主线程执行

**优势**：
- 提高应用响应速度
- 避免 I/O 密集型日志影响性能
- 适合高性能场景

### 2. 多进程安全

**特点**：支持多进程环境下的安全日志写入

**技术实现**：
- 使用 `multiprocessing` 模块的原语
- 队列本身是进程安全的
- 自动处理进程间同步

### 3. 多 Sink 独立队列

**特点**：每个启用 `enqueue` 的 sink 都有独立的队列和线程

**优势**：
- 不同 sink 的处理速度互不影响
- 一个 sink 阻塞不会影响其他 sink
- 更精细的性能控制

### 4. 配置灵活

**参数**：
- `enqueue=True/False`：启用/禁用队列
- `context`：指定 multiprocessing 上下文（可选）

---

## 应用场景

### 场景 1：异步 Web 应用

**适用**：FastAPI、Sanic、aiohttp 等异步框架

**原因**：
- 避免日志写入阻塞事件循环
- 保持高并发性能
- 防止日志 I/O 影响请求响应时间

**示例**：
```python
from fastapi import FastAPI
from loguru import logger

app = FastAPI()

# 配置异步日志
logger.add("app.log", enqueue=True, rotation="100 MB")

@app.get("/")
async def root():
    logger.info("请求已接收")  # 不会阻塞事件循环
    return {"message": "Hello"}
```

### 场景 2：多进程应用

**适用**：使用 `multiprocessing` 的应用

**原因**：
- 避免多个进程同时写入文件导致的日志混乱
- 确保日志完整性和顺序性

**示例**：
```python
from multiprocessing import Process
from loguru import logger

logger.add("multi_process.log", enqueue=True)

def worker(process_id):
    logger.info(f"进程 {process_id} 开始工作")

if __name__ == "__main__":
    processes = [Process(target=worker, args=(i,)) for i in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
```

### 场景 3：高性能计算

**适用**：需要极致性能的场景

**原因**：
- 日志操作不占用主线程时间
- 批量处理日志提高效率

### 场景 4：慢速 Sink

**适用**：写入网络、数据库等慢速 sink

**原因**：
- 将慢速 I/O 操作移到后台
- 主线程不受慢速 sink 影响

---

## 使用示例

### 基础用法

```python
from loguru import logger
import sys

# 移除默认 sink
logger.remove()

# 添加控制台 sink，启用队列
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="DEBUG",
    enqueue=True,
    colorize=True
)

# 添加文件 sink，启用队列
logger.add(
    "app.log",
    rotation="100 MB",
    retention="30 days",
    compression="zip",
    encoding="utf-8",
    enqueue=True
)

# 使用日志
logger.info("这是一条异步日志")
logger.error("这是一条错误日志")
```

### 结合完整功能

```python
from loguru import logger

# 完整配置示例
logger.add(
    "app_{time:YYYY-MM-DD}.log",
    rotation="00:00",        # 每天午夜轮转
    retention="7 days",      # 保留7天
    compression="gz",        # 压缩旧日志
    enqueue=True,            # 启用队列
    backtrace=True,          # 完整堆栈跟踪
    diagnose=True,           # 变量诊断（生产环境建议关闭）
    level="INFO"
)

try:
    1 / 0
except Exception:
    logger.exception("发生异常")
```

### 多 Sink 配置

```python
from loguru import logger
import sys

# 控制台（实时，不使用队列）
logger.add(
    sys.stderr,
    enqueue=False,  # 控制台可以不使用队列，获得即时反馈
    level="DEBUG"
)

# 文件（使用队列，不阻塞）
logger.add(
    "app.log",
    enqueue=True,
    level="INFO",
    rotation="50 MB"
)

# 错误日志（使用队列）
logger.add(
    "error.log",
    enqueue=True,
    level="ERROR"
)
```

---

## 主要注意事项

### 1. 日志丢失风险

**问题**：如果主进程异常终止，队列中未处理的消息可能丢失

**缓解措施**：
- 使用 `logger.complete()` 等待队列清空（仅适用于主线程）
- 在子进程退出前调用 `logger.complete()`
- 对于关键日志，考虑同时使用同步和异步 sink

**示例**：
```python
import time
from loguru import logger

logger.add("app.log", enqueue=True)

# 主程序结束前
logger.info("程序即将退出")
logger.complete()  # 等待队列处理完成
time.sleep(0.1)    # 额外等待确保写入完成
```

### 2. 内存增长（无界队列）

**问题**：当前 Loguru 使用无界队列，如果 sink 处理速度跟不上日志产生速度，会导致内存无限增长

**影响**：
- 内存占用持续增加
- 可能导致 OOM（Out of Memory）

**缓解措施**：
- 监控队列积压情况
- 避免过度日志
- 对于慢速 sink，考虑限流或降级
- 关注 Loguru 未来版本的有界队列支持（Issue #1419）

### 3. 输出交错

**问题**：在多进程环境下，`enqueue=True` 可以避免日志交错，但如果同时使用 `print()` 或其他方式写入 `sys.stderr`，仍可能出现交错

**缓解措施**：
- 统一使用 logger，避免直接使用 print
- 如果必须混用，注意同步问题

### 4. 性能开销

**问题**：使用 `enqueue=True` 会有一定的性能开销

**开销来源**：
- 队列操作
- 线程上下文切换
- 进程间通信（多进程环境）

**建议**：
- 对于性能极度敏感的场景，进行性能测试
- 对比 `enqueue=True` 和 `enqueue=False` 的性能差异
- 根据实际需求选择

### 5. 子进程处理

**问题**：在使用 `multiprocessing` 时，子进程的 logger 需要特殊处理

**注意事项**：
- 使用 `spawn` 方式启动进程时，logger 需要重新配置
- 子进程退出前建议调用 `logger.complete()`
- 可能出现信号量泄漏警告（macOS 常见）

### 6. Daemon 线程

**特点**：后台线程是 daemon 线程

**影响**：
- 主进程退出时，daemon 线程会被立即终止
- 队列中未处理的消息可能丢失
- 无法保证所有日志都被写入

---

## 最佳实践

### 1. 根据场景选择是否启用 enqueue

| 场景 | 推荐设置 | 原因 |
|------|----------|------|
| 控制台输出 | `enqueue=False` | 获得即时反馈，性能影响小 |
| 文件日志 | `enqueue=True` | 避免阻塞主线程 |
| 异步框架 | `enqueue=True` | 不阻塞事件循环 |
| 多进程 | `enqueue=True` | 保证进程安全 |
| 关键日志 | 同时使用同步和异步 | 兼顾性能和可靠性 |

### 2. 分层配置

```python
from loguru import logger
import sys

# 1. 控制台（同步，即时反馈）
logger.add(
    sys.stderr,
    enqueue=False,
    level="DEBUG",
    colorize=True
)

# 2. 普通文件（异步，高性能）
logger.add(
    "app.log",
    enqueue=True,
    level="INFO",
    rotation="100 MB"
)

# 3. 错误文件（同步+异步，双重保险）
logger.add(
    "error_sync.log",
    enqueue=False,
    level="ERROR"
)
logger.add(
    "error_async.log",
    enqueue=True,
    level="ERROR"
)
```

### 3. 优雅关闭

```python
import atexit
from loguru import logger

logger.add("app.log", enqueue=True)

def cleanup():
    """程序退出前的清理工作"""
    logger.info("程序正在关闭...")
    logger.complete()  # 等待队列处理完成

atexit.register(cleanup)
```

### 4. 监控和告警

```python
from loguru import logger
import time

class MonitoredSink:
    def __init__(self):
        self.queue_size = 0
    
    def write(self, message):
        # 实际写入逻辑
        print(message, end="")
        self.queue_size -= 1
    
    def update_queue_size(self, size):
        self.queue_size = size
        if size > 1000:
            logger.warning(f"日志队列积压: {size} 条消息")

# 使用自定义 sink（需要结合 Loguru 内部机制）
```

### 5. 多进程最佳实践

```python
from multiprocessing import Process, get_context
from loguru import logger
import sys

def setup_logger():
    """配置 logger，在每个进程中调用"""
    logger.remove()
    logger.add(
        "multi_process.log",
        enqueue=True,
        context=get_context("spawn")  # 显式指定 context
    )

def worker(process_id):
    setup_logger()
    logger.info(f"进程 {process_id} 启动")
    # ... 工作逻辑 ...
    logger.info(f"进程 {process_id} 结束")
    logger.complete()  # 确保队列处理完成

if __name__ == "__main__":
    setup_logger()
    processes = [Process(target=worker, args=(i,)) for i in range(4)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    logger.complete()
```

---

## 总结

`enqueue=True` 是 Loguru 的一个强大特性，它通过队列化日志处理，实现了非阻塞、多进程安全的日志记录。在使用时需要注意：

✅ **适用场景**：异步应用、多进程、高性能场景、慢速 sink  
⚠️ **注意事项**：日志丢失风险、内存增长、输出交错、性能开销  
🏆 **最佳实践**：分层配置、优雅关闭、监控告警、合理选择同步/异步

正确使用 `enqueue=True` 可以显著提升应用性能和日志系统的可靠性，但需要根据具体场景权衡利弊，制定合适的日志策略。

---

## 参考资料

- [Loguru 官方文档](https://loguru.readthedocs.io/)
- [Loguru GitHub 仓库](https://github.com/Delgan/loguru)
- [Issue #836: Document tradeoffs of using enqueue=True](https://github.com/Delgan/loguru/issues/836)
- [Issue #1419: enqueue=True causes unbounded memory growth](https://github.com/Delgan/loguru/issues/1419)
