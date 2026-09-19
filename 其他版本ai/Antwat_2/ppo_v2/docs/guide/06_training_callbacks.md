# 模块6：训练回调系统

## 1. 概述

### 1.1 模块定位

训练回调系统提供 **可扩展的插件化钩子（Hooks）机制**，将检查点保存、指标日志输出和 TensorBoard 可视化等附加功能与核心训练逻辑解耦。

### 1.2 在整体架构中的位置

```
模块4: SelfPlayTrainer.train()
    │
    │  训练循环中周期性调用
    ├── callbacks.on_training_start()
    ├── callbacks.on_step()
    └── callbacks.on_training_end()
    │
    ▼
┌─────────────────────────────────────┐
│        模块6: 训练回调系统            │
│                                       │
│  BasePPOCallback (基类)               │
│      │                                │
│      ├── CheckpointCallback           │
│      ├── MetricsLoggingCallback       │
│      └── TensorBoardCallback          │
│                                       │
│  CallbackList (容器)                  │
│  callback_factory (工厂)             │
└─────────────────────────────────────┘
```

### 1.3 业务目标

- 提供标准的回调生命周期接口（init / on_training_start / on_step / on_training_end）
- 支持按间隔（episode 数）自动保存检查点
- 支持按间隔输出训练指标摘要到日志
- 支持将训练指标写入 TensorBoard 事件文件
- 通过工厂模式统一创建和组装回调链

---

## 2. 背景与概念

### 2.1 回调模式（Callback Pattern）

回调模式允许在不修改核心训练循环的情况下，注入额外的行为。每个回调实现一组标准接口（钩子方法），训练循环在特定时机调用这些方法。

典型场景：
- **检查点保存**：每隔 N 局保存一次模型
- **指标日志**：每隔 M 局打印一次当前训练指标
- **TensorBoard**：每隔 K 局写入可视化数据

### 2.2 回调生命周期

```
init_callback(trainer)          # 注入训练器引用
    │
on_training_start()             # 训练开始时调用一次
    │
    │   (训练循环中)
    ├── on_step() → True/False  # 每个 episode 后调用
    │
on_training_end()               # 训练结束时调用一次
```

`on_step()` 返回 `True` 表示回调执行成功，`False` 表示跳过（未到触发间隔）。

---

## 3. 架构设计

### 3.1 组件职责

| 组件 | 文件 | 职责 |
|------|------|------|
| `BaseCallback` | [callbacks/base_ppo_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/base_ppo_callback.py) | 回调基础接口（空类） |
| `BasePPOCallback` | [callbacks/base_ppo_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/base_ppo_callback.py) | PPO 专用回调基类：生命周期 + 指标访问 |
| `CheckpointCallback` | [callbacks/checkpoint_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/checkpoint_callback.py) | 定期保存检查点 + 过期清理 |
| `MetricsLoggingCallback` | [callbacks/metrics_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/metrics_callback.py) | 定期输出指标摘要 |
| `TensorBoardCallback` | [callbacks/tensorboard_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/tensorboard_callback.py) | TensorBoard 可视化写入 |
| `CallbackList` | [callbacks/callback_factory.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/callback_factory.py) | 回调列表容器 |
| `create_ppo_callbacks()` | [callbacks/callback_factory.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/callback_factory.py) | 回调工厂函数 |

---

## 4. 核心组件详解

### 4.1 BasePPOCallback — 回调基类

位置：[callbacks/base_ppo_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/base_ppo_callback.py)

```python
class BasePPOCallback(BaseCallback):
    def init_callback(self, trainer) -> None:
        """注入训练器引用和指标提供者"""
    
    def get_last_metrics(self) -> Optional[Dict]:
        """获取最近一次 PPO update 的指标"""
    
    def on_training_start(self) -> None:
        """训练开始钩子（可重写）"""
    
    def on_training_end(self) -> None:
        """训练结束钩子（可重写）"""
    
    def on_step(self) -> bool:
        """每步钩子，子类重写 _on_step()"""
```

`init_callback()` 注入 `_trainer_ref` 和 `_last_metrics_provider`，使回调可以访问训练器和最近的指标。

---

### 4.2 CheckpointCallback — 检查点回调

位置：[callbacks/checkpoint_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/checkpoint_callback.py)

```python
CheckpointCallback(
    save_interval: int,
    checkpoint_dir: Optional[str],
    path_config,
    keep_last_n: int = 5
)
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `save_interval` | 每隔多少个 episode 保存一次 | 必填 |
| `checkpoint_dir` | 检查点目录（可选，默认用 path_config） | `None` |
| `path_config` | 路径管理器 | 必填 |
| `keep_last_n` | 保留最近 N 个检查点 | 5 |

**行为**：
- 每个 `save_interval` 步保存 `checkpoint_ep{N}.pt`
- 自动清理：只保留最近 N 个检查点文件

---

### 4.3 MetricsLoggingCallback — 指标日志回调

位置：[callbacks/metrics_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/metrics_callback.py)

```python
MetricsLoggingCallback(
    log_interval: int = 10,
    window_size: int = 100
)
```

每隔 `log_interval` 步输出核心指标摘要到日志：

```
Episode XXX | loss: 0.1234 | policy_loss: 0.0567 | value_loss: 0.0890
           | entropy: 1.2345 | reward_mean: 12.34
```

---

### 4.4 TensorBoardCallback — TensorBoard 可视化回调

位置：[callbacks/tensorboard_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/tensorboard_callback.py)

```python
TensorBoardCallback(
    log_dir: str,
    log_interval: int = 10
)
```

- 每隔 `log_interval` 步将 `MetricsSchema` 中定义的指标写入 TensorBoard 事件文件
- 按 `MetricsSchema` 做指标筛选（只写入已注册的指标）
- `on_training_end()` 时关闭 `SummaryWriter`

---

### 4.5 CallbackList — 回调容器

位置：[callbacks/callback_factory.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/callback_factory.py)

统一管理回调列表的容器，对外暴露相同接口：

```python
callback_list.append(callback)
callback_list.init_callback(trainer)
callback_list.on_training_start()
callback_list.on_step()        # 依次调用所有回调，短路到第一个返回 False 的
callback_list.on_training_end()
```

---

### 4.6 callback_factory — 回调工厂

位置：[callbacks/callback_factory.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/callback_factory.py)

```python
create_ppo_callbacks(
    save_interval: int,
    log_interval: int = 10,
    checkpoint_dir: Optional[str] = None,
    path_config = None,
    enable_checkpoint: bool = True,
    enable_metrics_logging: bool = True,
    enable_tensorboard: bool = False,
    tensorboard_dir: Optional[str] = None,
    tensorboard_interval: int = 10,
) -> Optional[CallbackList]
```

通过开关参数控制启用哪些回调，统一创建和组装。

---

## 5. 数据流与时序

### 5.1 回调在训练循环中的调用时机

```
SelfPlayTrainer.train()
  │
  ├── callbacks = create_ppo_callbacks(...)
  ├── callbacks.init_callback(self)
  ├── callbacks.on_training_start()
  │
  ├── for episode in range(num_episodes):
  │     │
  │     ├── ... 对战、收集数据、PPO update ...
  │     │
  │     └── callbacks.on_step()          # 每个 episode 后调用
  │           │
  │           ├── CheckpointCallback: if episode % save_interval == 0 → save()
  │           ├── MetricsCallback:      if episode % log_interval == 0 → log()
  │           └── TensorBoardCallback:  if episode % log_interval == 0 → write()
  │
  └── callbacks.on_training_end()        # 训练结束时调用
```

---

## 6. 对外交互

### 6.1 被依赖关系

| 调用方 | 场景 | 使用方法 |
|--------|------|---------|
| **SelfPlayTrainer** (模块4) | 训练循环中的钩子调用 | `callbacks.init_callback()`, `on_training_start()`, `on_step()`, `on_training_end()` |

### 6.2 对下游模块的依赖

| 依赖模块 | 方式 |
|---------|------|
| **模块3（PPOTrainer）** | `init_callback()` 注入 trainer 引用，`get_last_metrics()` 获取指标 |
| **模块8（基础设施）** | `PathConfig` 用于确定检查点目录和 TensorBoard 日志目录 |

---

## 7. 配置参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `save_interval` | 检查点保存间隔（episode 数） | 50-200 |
| `log_interval` | 指标输出间隔（episode 数） | 10 |
| `keep_last_n` | 保留最近 N 个检查点 | 5 |
| `enable_checkpoint` | 是否启用检查点回调 | `True` |
| `enable_metrics_logging` | 是否启用指标日志回调 | `True` |
| `enable_tensorboard` | 是否启用 TensorBoard 回调 | `False` |
| `tensorboard_interval` | TensorBoard 写入间隔 | 10 |

---

## 8. 常见问题与注意事项

### 8.1 回调的设计原则

每个回调应该是 **无状态的**（除必要的计数器外），不修改训练器的核心状态。回调只做 "观察 + 副作用"（保存文件、打印日志、写入指标），不干预训练逻辑。

### 8.2 检查点清理策略

`keep_last_n=5` 意味着训练过程中最多保留 5 个最近检查点。早期的检查点会被自动删除以节省磁盘空间。如果需要所有历史检查点，可设置 `keep_last_n` 为极大值。

### 8.3 回调的执行顺序

回调按 `CallbackList` 中的添加顺序执行。`on_step()` 使用短路逻辑——如果某个回调返回 `False`，后续回调不再执行。但默认实现都返回 `True`，除非有特殊需求。

### 8.4 与 LoggingSubsystem 的区别

回调系统（模块6）和监控系统（模块7）都涉及日志/指标输出，但职责不同：
- **回调系统**：训练循环中的 **同步钩子**，在训练主线程中执行，输出简洁摘要
- **监控系统**：独立的 **数据持久化层**，负责结构化数据（JSONL）的写入，可能涉及后台线程（系统监控）

### 8.5 扩展自定义回调

要添加自定义回调，只需：
1. 继承 `BasePPOCallback`
2. 重写 `_on_step()` 方法
3. 通过 `get_last_metrics()` 获取最新训练指标
4. 加入 `CallbackList`
