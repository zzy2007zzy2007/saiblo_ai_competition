# PPO 训练引擎重构方案

> 本文档基于 `ppo_v2/src/ppo_antwar/` 中 PPO 训练引擎相关代码的分析与问题诊断，提供具体、可执行的重构方案。
> 涉及文件范围：`trainer/`、`callbacks/`、`monitor/` 中与训练指标/回调/训练引擎相关的代码。

---

## 目录

- [1. CheckpointCallback._step_count 未初始化 Bug 修复](#1-checkpointcallback_step_count-未初始化-bug-修复)
- [2. 引入 MetricsSchema 统一指标定义](#2-引入-metricsschema-统一指标定义)
- [3. BasePPOCallback 暴露正式 get_last_metrics 接口](#3-baseppocallback-暴露正式-get_last_metrics-接口)
- [4. 修正 loss 指标命名与计算](#4-修正-loss-指标命名与计算)
- [5. MetricsCache 废弃与 MetricsLoggingCallback 精简](#5-metricscache-废弃与-metricsloggingcallback-精简)
- [6. PPOTrainer 中 LRScheduler 抽取](#6-ppotrainer-中-lrscheduler-抽取)
- [附录：影响范围评估](#附录影响范围评估)

---

## 1. CheckpointCallback._step_count 未初始化 Bug 修复

### 问题分析

[checkpoint_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/checkpoint_callback.py) 中：

```python
# __init__ 中未定义 _step_count

def _on_step(self) -> bool:
    episode = getattr(self, '_step_count', 0)  # 未初始化，getattr 默认返回 0
    if episode % self.save_interval == 0:      # 0 % N == 0 恒为 True，首次调用必然触发保存
        ...
    self._step_count = episode + 1              # 在第 1 次使用之后才初始化
```

**两个具体 Bug**：
1. **首次误保存**：`_step_count` 初始为 0，`0 % save_interval == 0` 恒成立，导致第一步即触发检查点保存。
2. **属性未初始化**：`_step_count` 作为实例变量未在 `__init__` 中声明，依赖 `getattr` 的默认值容错，违反最小惊讶原则。

对比 [metrics_callback.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/metrics_callback.py#L17)（正确的做法）：
```python
def __init__(self, ...):
    super().__init__()
    self._step_count = 0   # 显式初始化
```

### 方案

**目标文件**：`callbacks/checkpoint_callback.py`

**改动**：
1. 在 `__init__` 中新增 `self._step_count = 0`
2. 将 `_on_step` 中的 `getattr(self, '_step_count', 0)` 替换为 `self._step_count`
3. 移除注释中的 `# 被 trainer 调用时确保已设置，用默认值 0 避免 AttributeError`

```python
class CheckpointCallback(BasePPOCallback):
    def __init__(self, save_interval, checkpoint_dir=None, path_config=None, keep_last_n=5):
        super().__init__()
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.path_config = path_config
        self.keep_last_n = keep_last_n
        self._step_count = 0                         # ← 新增
        self._saved_checkpoints: List[str] = []

    def _on_step(self) -> bool:
        if self.checkpoint_dir is not None and self._trainer_ref is not None:
            episode = self._step_count                # ← 替换 getattr
            if episode % self.save_interval == 0:
                ...
            self._step_count = episode + 1
        return True
```

### 影响范围

- **直接文件**：仅 `callbacks/checkpoint_callback.py`
- **无 API 变更**：无外部依赖，纯内部修复

---

## 2. 引入 MetricsSchema 统一指标定义

### 问题分析

当前训练指标 key 在三处独立硬编码：

| 位置 | 代码形式 | key 数量 |
|------|---------|---------|
| `trainer/ppo_trainer.py` `_aggregate_metrics()` | 手动组装 dict | ~25 个 |
| `monitor/batch_metrics_writer.py` | `_FLOAT_KEYS_6` / `_FLOAT_KEYS_4` / `_INT_KEYS` / `_DIRECT_KEYS` | ~20 个 |
| `callbacks/tensorboard_callback.py` | `scalar_tags` 列表 | ~14 个 |

三者互有重叠但不一致。例如 `learning_rate` 和 `entropy_coef` 被 `batch_metrics_writer` 写入但未被 `tensorboard_callback` 记录。当新增指标时需手动同步三处，极易遗漏。

### 方案

**引入 `MetricsSchema` 统一元数据定义层**，定义每个指标的 name、type、格式化精度、是否写入 JSONL、是否写入 TensorBoard 等属性。

#### 2.1 新增文件 `trainer/metrics_schema.py`

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricDef:
    """单个训练指标的定义元信息"""
    name: str                           # 指标名称（dict key）
    dtype: str                          # 'float' | 'int' | 'str'
    precision: Optional[int] = None     # 浮点精度（如 6 表示 round_6）；None 表示原始值
    log_jsonl: bool = True             # 是否写入 batch_metrics.jsonl
    log_tensorboard: bool = True       # 是否写入 TensorBoard
    description: str = ""              # 可读说明


class MetricsSchema:
    """训练指标 Schema - 所有训练指标的单一事实来源"""

    # ── 训练损失类 ──────────────────────────────────────────────────────
    POLICY_LOSS = MetricDef('policy_loss', 'float', precision=6, description='PPO policy clip loss')
    VALUE_LOSS = MetricDef('value_loss', 'float', precision=6, description='PPO value loss (unweighted)')
    ENTROPY = MetricDef('entropy', 'float', precision=6, description='Policy entropy (type + target)')
    ENTROPY_TYPE = MetricDef('entropy_type', 'float', precision=6, log_tensorboard=False)
    ENTROPY_TARGET = MetricDef('entropy_target', 'float', precision=6, log_tensorboard=False)
    CLIP_FRACTION = MetricDef('clip_fraction', 'float', precision=6)
    APPROX_KL = MetricDef('approx_kl', 'float', precision=6, log_tensorboard=False)

    # ── 辅助任务损失 ─────────────────────────────────────────────────────
    AUX_TOWER_LOSS = MetricDef('aux_tower_loss', 'float', precision=6)
    AUX_GOLD_LOSS = MetricDef('aux_gold_loss', 'float', precision=6)
    AUX_ENEMY_TOWER_LOSS = MetricDef('aux_enemy_tower_loss', 'float', precision=6)
    AUX_ENEMY_GOLD_LOSS = MetricDef('aux_enemy_gold_loss', 'float', precision=6)

    # ── 梯度类 ──────────────────────────────────────────────────────────
    GRAD_NORM = MetricDef('grad_norm', 'float', precision=6)
    GRAD_NORM_POLICY = MetricDef('grad_norm_policy', 'float', precision=6)
    GRAD_NORM_VALUE = MetricDef('grad_norm_value', 'float', precision=6)
    GRAD_NORM_MAX_LAYER = MetricDef('grad_norm_max_layer', 'float', precision=6, log_tensorboard=False)

    # ── 奖励/价值统计 ───────────────────────────────────────────────────
    REWARD_MEAN = MetricDef('reward_mean', 'float', precision=4)
    REWARD_STD = MetricDef('reward_std', 'float', precision=4)
    REWARD_MIN = MetricDef('reward_min', 'float', precision=4, log_tensorboard=False)
    REWARD_MAX = MetricDef('reward_max', 'float', precision=4, log_tensorboard=False)
    RETURN_MEAN = MetricDef('return_mean', 'float', precision=4)
    RETURN_STD = MetricDef('return_std', 'float', precision=4)
    ADVANTAGES_STD = MetricDef('advantages_std', 'float', precision=4)

    # ── Value 统计 ──────────────────────────────────────────────────────
    VALUE_INPUT_MEAN = MetricDef('value_input_mean', 'float', precision=6, log_tensorboard=False)
    VALUE_INPUT_STD = MetricDef('value_input_std', 'float', precision=6, log_tensorboard=False)
    VALUE_PRED_MEAN = MetricDef('value_pred_mean', 'float', precision=6, log_tensorboard=False)
    VALUE_PRED_STD = MetricDef('value_pred_std', 'float', precision=6, log_tensorboard=False)
    RATIO_MEAN = MetricDef('ratio_mean', 'float', precision=6, log_tensorboard=False)
    RATIO_STD = MetricDef('ratio_std', 'float', precision=6, log_tensorboard=False)
    TD_ERROR_MEAN = MetricDef('td_error_mean', 'float', precision=6, log_tensorboard=False)
    TD_ERROR_STD = MetricDef('td_error_std', 'float', precision=6, log_tensorboard=False)

    # ── Action 统计 ─────────────────────────────────────────────────────
    VALID_ACTIONS_MEAN = MetricDef('valid_actions_mean', 'float', precision=6, log_tensorboard=False)
    VALID_ACTIONS_MIN = MetricDef('valid_actions_min', 'float', precision=6, log_tensorboard=False)
    BATCH_SIZE = MetricDef('batch_size', 'int', log_tensorboard=False)
    NUM_COLLECTED = MetricDef('num_collected', 'int', log_tensorboard=False)
    NAN_SKIP_COUNT = MetricDef('nan_skip_count', 'int', log_tensorboard=False)

    # ── 超参类（不含 precision，直接记录原始值）────────────────────────
    LEARNING_RATE = MetricDef('learning_rate', 'float', log_jsonl=True)
    ENTROPY_COEF = MetricDef('entropy_coef', 'float', log_jsonl=True)

    # ── 全局注册表 ──────────────────────────────────────────────────────
    _REGISTRY: Dict[str, MetricDef] = {}

    @classmethod
    def _build_registry(cls):
        if cls._REGISTRY:
            return
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, MetricDef):
                cls._REGISTRY[attr.name] = attr

    @classmethod
    def get(cls, name: str) -> Optional[MetricDef]:
        cls._build_registry()
        return cls._REGISTRY.get(name)

    @classmethod
    def get_jsonl_keys(cls) -> List[str]:
        cls._build_registry()
        return [d.name for d in cls._REGISTRY.values() if d.log_jsonl]

    @classmethod
    def get_tensorboard_keys(cls) -> List[str]:
        cls._build_registry()
        return [d.name for d in cls._REGISTRY.values() if d.log_tensorboard]

    @classmethod
    def format_value(cls, name: str, value) -> object:
        """按 schema 定义格式化单个指标值"""
        metric_def = cls.get(name)
        if metric_def is None:
            return value
        if metric_def.dtype == 'int':
            return int(value)
        if metric_def.dtype == 'float' and metric_def.precision is not None:
            return round(float(value), metric_def.precision)
        return value
```

#### 2.2 修改 `monitor/batch_metrics_writer.py`

**重构后**：消除全部硬编码 key 列表，从 `MetricsSchema` 派生需要落盘的字段。

```python
class BatchMetricsWriter:
    """Batch 级训练指标写入器"""

    def __init__(self, filepath: str):
        self._filepath = Path(filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        # 预计算落盘 key 集合（类型分组）
        self._jsonl_keys = MetricsSchema.get_jsonl_keys()
        self._type_rules = self._build_type_rules()

    @staticmethod
    def _build_type_rules() -> Dict[str, str]:
        rules = {}
        for key in MetricsSchema.get_jsonl_keys():
            defn = MetricsSchema.get(key)
            if defn:
                rules[key] = defn.dtype
        return rules

    def write_batch(self, metrics: Dict[str, Any]) -> None:
        record = {}
        for key in self._jsonl_keys:
            if key in metrics:
                record[key] = MetricsSchema.format_value(key, metrics[key])

        # 额外处理：type_ratios / type_probs（动态生成，不在 schema 中）
        type_ratios = {k: round(v, 6) for k, v in metrics.items()
                       if k.startswith('type_') and k.endswith('_ratio')}
        if type_ratios:
            record["type_ratios"] = type_ratios
        type_probs = {k: round(v, 6) for k, v in metrics.items()
                      if k.startswith('type_') and k.endswith('_prob')}
        if type_probs:
            record["type_probs"] = type_probs

        with open(self._filepath, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
```

#### 2.3 修改 `callbacks/tensorboard_callback.py`

```python
class TensorBoardCallback(BasePPOCallback):
    def _on_step(self) -> bool:
        ...
        if not metrics:
            return True

        writer = self._tb_writer
        for key in MetricsSchema.get_tensorboard_keys():
            if key in metrics:
                writer.add_scalar(key, metrics[key], self._step_count)
        return True
```

#### 2.4 修改 `trainer/ppo_trainer.py`

`_aggregate_metrics` 保持指标键名不变（作为指标的**生产方**，它仍然需要组装 dict），但不再需要与消费者对齐键名——消费者通过 `MetricsSchema` 自动发现需要消费哪些键。

需新增一个 `training_loss` 指标（详见第 4 节）：

```python
# 在 _aggregate_metrics 中新增：
result['training_loss'] = np.mean(metrics.get('policy_loss', [0])) + \
    vf_coef * np.mean(metrics.get('value_loss', [0])) - \
    ent_coef * np.mean(metrics.get('entropy', [0]))
```

#### 2.5 修改 `callbacks/metrics_callback.py` 和 `trainer/selfplay.py`

这两处使用 `metrics.get('loss', 0)` 应改为 `metrics.get('training_loss', 0)`（见第 4 节）。

### 影响范围

| 文件 | 改动类型 |
|------|---------|
| `trainer/metrics_schema.py` | **新增** |
| `monitor/batch_metrics_writer.py` | 重写 `__init__` 和 `write_batch`，消除硬编码 key 列表 |
| `callbacks/tensorboard_callback.py` | 将 `scalar_tags` 替换为 `MetricsSchema.get_tensorboard_keys()` |
| `trainer/ppo_trainer.py` | `_aggregate_metrics` 中新增 `training_loss`；保留向后兼容的 `loss` 作为过渡 |
| `callbacks/metrics_callback.py` | 字符串替换：`metrics.get('loss')` → `metrics.get('training_loss')` |
| `trainer/selfplay.py` | 字符串替换：`metrics.get('loss')` → `metrics.get('training_loss')` |

---

## 3. BasePPOCallback 暴露正式 get_last_metrics 接口

### 问题分析

3 个回调类均通过 `getattr(self._trainer_ref, '_last_metrics', None)` 访问 `PPOTrainer` 的私有属性：

- [metrics_callback.py:26](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/metrics_callback.py#L26)
- [tensorboard_callback.py:34](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/tensorboard_callback.py#L34)

`_last_metrics` 是 `PPOTrainer` / `SelfPlayTrainer` 的内部实现细节。回调作为外部扩展点依赖私有属性是**接口层面的耦合**。

### 方案

#### 3.1 修改 `BasePPOCallback`

在基类中定义受控的访问方式：

```python
class BasePPOCallback(BaseCallback):
    def __init__(self):
        self._trainer_ref: Optional[Any] = None
        self._last_metrics_provider: Optional[callable] = None

    def init_callback(self, trainer: Any) -> None:
        self._trainer_ref = trainer
        # 若 trainer 实现了 get_last_metrics()，注入访问方法
        if hasattr(trainer, 'get_last_metrics'):
            self._last_metrics_provider = trainer.get_last_metrics

    def get_last_metrics(self) -> Optional[Dict]:
        """获取最近一次 PPO update 的训练指标字典"""
        if self._last_metrics_provider is not None:
            return self._last_metrics_provider()
        # 向后兼容（在 Trainer 未实现接口时降级）
        if self._trainer_ref is not None:
            return getattr(self._trainer_ref, '_last_metrics', None)
        return None
```

#### 3.2 修改 `PPOTrainer` 和 `SelfPlayTrainer`

在 `PPOTrainer` 和 `SelfPlayTrainer` 中新增公开方法：

```python
class PPOTrainer:
    def get_last_metrics(self) -> Optional[Dict]:
        """公开最近一次 PPO update 的指标"""
        return getattr(self, '_last_metrics', None)
```

`SelfPlayTrainer` 也需要同样的方法（它在 `_flush_batch_to_update` 中写入 `self._trainer._last_metrics`）。

#### 3.3 修改 `MetricsLoggingCallback` 和 `TensorBoardCallback`

将：
```python
metrics = getattr(self._trainer_ref, '_last_metrics', None)
```
替换为：
```python
metrics = self.get_last_metrics()
```

### 影响范围

| 文件 | 改动类型 |
|------|---------|
| `callbacks/base_ppo_callback.py` | 新增 `init_callback` 中的 provider 注入；新增 `get_last_metrics()` |
| `trainer/ppo_trainer.py` | 新增 `get_last_metrics()` 公开方法 |
| `trainer/selfplay.py` | 新增 `get_last_metrics()` 公开方法 |
| `callbacks/metrics_callback.py` | `getattr` → `self.get_last_metrics()` |
| `callbacks/tensorboard_callback.py` | `getattr` → `self.get_last_metrics()` |

---

## 4. 修正 loss 指标命名与计算

### 问题分析

当前 `loss` 指标定义（[ppo_trainer.py:562](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py#L562)）：

```python
'loss': np.mean(metrics.get('policy_loss', [0])) + np.mean(metrics.get('value_loss', [0]))
```

而实际优化目标为：

```
total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy + aux_coefs * aux_losses
```

**差异**：
- `loss` 未乘以 `vf_coef`（通常 0.5），value_loss 权重实际为优化时的 2 倍
- `loss` 不含 `entropy` 和 `aux_losses`，而这些在总损失中占非零权重
- 当前命名 `loss` 暗示"总损失"，实际既不是总损失也不是纯 policy_loss

### 方案

#### 4.1 重命名与修正计算

在 `_aggregate_metrics` 中：

```python
class PPOTrainer:
    def _aggregate_metrics(self, metrics, reward_stats, type_metrics,
                           advantages, returns, td_error_mean, td_error_std,
                           num_collected) -> Dict[str, Any]:
        # ... 现有代码 ...

        ppo_cfg = self.config.get('ppo', {})
        vf_coef = ppo_cfg.get('vf_coef')
        ent_coef = ppo_cfg.get('ent_coef')

        # 新增：精确反映优化目标的 total loss
        result['training_loss'] = (
            np.mean(metrics.get('policy_loss', [0]))
            + vf_coef * np.mean(metrics.get('value_loss', [0]))
            - ent_coef * np.mean(metrics.get('entropy', [0]))
        )

        # 向后兼容：保留 loss 但标记为 deprecated
        result['loss'] = np.mean(metrics.get('policy_loss', [0])) + np.mean(metrics.get('value_loss', [0]))
```

#### 4.2 更新 MetricsSchema

在 `MetricsSchema` 中：

```python
# 新增
TRAINING_LOSS = MetricDef('training_loss', 'float', precision=6, description='Total training loss = policy_loss + vf_coef*value_loss - ent_coef*entropy')

# loss 保留但不写入 TensorBoard（作为过渡），标记不推荐
LEGACY_LOSS = MetricDef('loss', 'float', precision=6, log_tensorboard=False, log_jsonl=True)
```

#### 4.3 更新消费者

- [metrics_callback.py:35](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/metrics_callback.py#L35)：`metrics.get('loss')` → `metrics.get('training_loss')`
- [selfplay.py:413](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/selfplay.py#L413)：同上
- [tensorboard_callback.py:41](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/callbacks/tensorboard_callback.py#L41)：`'loss'` 从 `scalar_tags` 移除，由 `MetricsSchema.get_tensorboard_keys()` 自动包含 `training_loss` 而非 `loss`

### 影响范围

| 文件 | 改动 |
|------|------|
| `trainer/ppo_trainer.py` | `_aggregate_metrics` 中新增 `training_loss` 计算 |
| `trainer/metrics_schema.py` | 新增 `TRAINING_LOSS` 和 `LEGACY_LOSS` 定义 |
| `callbacks/metrics_callback.py` | `metrics.get('loss')` → `metrics.get('training_loss')` |
| `trainer/selfplay.py` | `metrics.get('loss')` → `metrics.get('training_loss')` |

---

## 5. MetricsCache 废弃与 MetricsLoggingCallback 精简

### 使用情况验证

经全代码库搜索：

| 位置 | 使用情况 |
|------|---------|
| `ppo_v2/src/ppo_antwar/monitor/metrics_cache.py` | **定义** |
| `ppo_v2/src/ppo_antwar/__init__.py:47` | 仅 re-export（`from .monitor.metrics_cache import MetricsCache`） |
| `ppo_v2/src/ppo_antwar/` 中其他 .py 文件 | **无任何 import 或使用** |
| `ppo_v1/src/ppo_antwar/monitor/logger.py` | ppo_v1 中使用（`self._metrics_cache = MetricsCache()`），ppo_v2 已不引用 |

**结论**：`MetricsCache` 在 ppo_v2 中是死代码（仅被 `__init__.py` 导出但未被任何业务代码使用），不属于 "与 MetricsLoggingCallback 功能重叠需要合并" 的情况。正确的做法是**直接废弃**。

### `MetricsLoggingCallback` 现状

`MetricsLoggingCallback` 内部维护了 `_metrics_history: List[dict]`（cap 10000），但该数据**仅用于追加存储**，没有任何查询/读取接口——不是缓存，而是一个单向的日志历史记录。它没有 `get_recent()` / `get_window_stats()` 等查询方法，与 `MetricsCache` 的功能交集实际为零。

### 方案

#### 5.1 废弃 MetricsCache（直接移除）

```python
# __init__.py 中移除：
# from .monitor.metrics_cache import MetricsCache
# 从 __all__ 中移除 'MetricsCache'
```

文件 `monitor/metrics_cache.py` 保留但标记 `# DEPRECATED: no longer used`，在下次大版本清理时删除。

#### 5.2 MetricsLoggingCallback 精简（可选）

当前 `_metrics_history` 在 callback 中仅追加不查询，可考虑移除以减少内存占用：

```python
class MetricsLoggingCallback(BasePPOCallback):
    def __init__(self, log_interval: int = 10, window_size: int = 100):
        super().__init__()
        self.log_interval = log_interval
        self._step_count = 0
        self._last_log_step = 0
        # _metrics_history 已移除 — 纯追加不查询，无实际用途
```

但同时需移除第 30 行的 `self._metrics_history.append(...)` 以及相关的 cap 检查逻辑。如果保留历史记录对调试有价值（例如训练结束时 dump 历史），可以保留但应明确用途。

### 影响范围

| 文件 | 改动 |
|------|------|
| `monitor/metrics_cache.py` | 文件头部加 `# DEPRECATED` 标记 |
| `__init__.py` | 移除 import 和 `__all__` 中的 `MetricsCache` |

---

## 6. PPOTrainer 中 LRScheduler 抽取

### 问题分析

[update_lr_schedule](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py#L796-L818) 约 35 行，实现了三种调度策略（warmup、cosine decay、fixed），包含完整的条件分支逻辑。该方法是纯函数式的——输入 `(episode, total_episodes, config)` 即可输出 `(new_lr, new_lr_vf)`，不依赖 trainer 的内部状态。

### 方案

#### 6.1 新增文件 `trainer/lr_scheduler.py`

```python
import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class LRScheduleConfig:
    """学习率调度配置"""
    lr_base: float
    lr_vf_base: float
    warmup_episodes: int = 0
    cosine_decay: bool = True


class LRScheduler:
    """学习率调度器 - 支持 warmup + cosine decay + fixed 三种策略"""

    def __init__(self, config: LRScheduleConfig):
        self._config = config

    @classmethod
    def from_ppo_config(cls, ppo_cfg: Dict) -> "LRScheduler":
        return cls(LRScheduleConfig(
            lr_base=ppo_cfg.get('lr'),
            lr_vf_base=ppo_cfg.get('lr_vf'),
            warmup_episodes=ppo_cfg.get('lr_warmup_episodes') or 0,
            cosine_decay=ppo_cfg.get('lr_cosine_decay', False),
        ))

    def compute(self, episode: int, total_episodes: int) -> Tuple[float, float]:
        """计算当前 episode 的策略网络和价值网络学习率"""
        cfg = self._config

        if cfg.warmup_episodes > 0 and episode <= cfg.warmup_episodes:
            ratio = episode / max(cfg.warmup_episodes, 1)
            return cfg.lr_base * ratio, cfg.lr_vf_base * ratio

        if cfg.cosine_decay:
            progress = episode / max(total_episodes, 1)
            factor = (1.0 + math.cos(math.pi * progress)) / 2.0
            return cfg.lr_base * factor, cfg.lr_vf_base * factor

        return cfg.lr_base, cfg.lr_vf_base

    @staticmethod
    def apply(optimizer, value_optimizer, lr: float, lr_vf: float) -> None:
        """将计算出的学习率应用到优化器"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in value_optimizer.param_groups:
            param_group['lr'] = lr_vf
```

#### 6.2 修改 `PPOTrainer`

```python
class PPOTrainer:
    def __init__(self, ...):
        ...
        self._lr_scheduler = LRScheduler.from_ppo_config(self.config.get('ppo', {}))

    def update_lr_schedule(self, episode: int, total_episodes: int):
        """更新学习率"""
        lr, lr_vf = self._lr_scheduler.compute(episode, total_episodes)
        LRScheduler.apply(self.optimizer, self.value_optimizer, lr, lr_vf)

        if episode == 1 or episode % self.LR_LOG_INTERVAL == 0 or episode >= total_episodes:
            logger.info(
                f"[LR] episode={episode}/{total_episodes} "
                f"lr_policy={lr:.2e}, lr_value={lr_vf:.2e}"
            )
```

### 影响范围

| 文件 | 改动 |
|------|------|
| `trainer/lr_scheduler.py` | **新增** |
| `trainer/ppo_trainer.py` | `update_lr_schedule` 委派给 `LRScheduler`；`__init__` 中创建 `_lr_scheduler` |

---

## 附录：影响范围评估

### 总览

| 项目 | 受影响文件数 | 新增文件数 | 涉及模块 |
|------|------------|-----------|---------|
| ① CheckpointCallback bug 修复 | 1 | 0 | callbacks |
| ② MetricsSchema 统一指标 | 6 | 1 | trainer / monitor / callbacks |
| ③ BasePPOCallback 接口完善 | 5 | 0 | trainer / callbacks |
| ④ loss 指标修正 | 4 | 0 | trainer / callbacks |
| ⑤ MetricsCache 废弃 | 2 | 0 | monitor |
| ⑥ LRScheduler 抽取 | 2 | 1 | trainer |

### 兼容性保障

1. `loss` 指标保留旧定义作为过渡，新指标为 `training_loss`，消费者逐步迁移
2. `BasePPOCallback.get_last_metrics()` 保留 `getattr` 降级路径，确保外部自定义回调不受影响
3. 所有重构项均为**纯内部重构**，不涉及对外部调用方的 API 变更
