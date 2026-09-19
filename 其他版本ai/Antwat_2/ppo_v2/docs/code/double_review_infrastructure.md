# 基础设施模块二次审查评估报告

## 评估概要
- 评估问题数: 11
- 问题真实存在: 8 | 问题不存在: 2 | 问题部分存在: 1
- 级别调整: Issue 1 (CRITICAL→HIGH), Issue 4 (HIGH→MEDIUM), Issue 8 (LOW→不修复), Issue 10 (HIGH→LOW)

## 逐项评估

### Issue 1: CheckpointCallback._on_step 在不保存检查点时不会递增 _step_count
- **真实性**: **确认存在**。源码 `checkpoint_callback.py` 第 31-44 行，`self._step_count = episode + 1` 位于 `if self.checkpoint_dir is not None and self._trainer_ref is not None:` 条件块内部（第 44 行）。当 `checkpoint_dir is None` 或 `_trainer_ref is None` 时，`_step_count` 永远不会递增。
- **级别评估**: CRITICAL → **HIGH**。原报告定为 CRITICAL 有所夸张——当 `checkpoint_dir is None` 时本就不打算保存检查点，`_step_count` 不递增的影响仅限于后续若条件恢复时的步数跳跃。但在正常训练流程中，`callback_factory.py` 总是同时传入 `checkpoint_dir` 和 `path_config`，且 `init_callback` 会在训练开始时注入 `_trainer_ref`。因此在实际使用中此 bug 几乎不会触发。不过代码逻辑确实有误，应为 HIGH。
- **影响评估**: 原影响描述基本准确，但遗漏了一个关键点：原报告说"episode 0 时 `0 % save_interval == 0` 为 True，会在第 0 步就保存检查点"——这是正确的观察，但与 _step_count 不递增是两个独立问题。
- **建议评估**: 原建议"将 `self._step_count += 1` 移到方法最外层"是正确的。关于"排除 episode == 0"的建议需要商榷——如果在第 0 步保存检查点不期望，应该在 `save_interval` 的语义上处理，而非特殊判断 episode==0。
- **详细修复方案**:
  ```python
  # checkpoint_callback.py 第 31-45 行，修改为：
  def _on_step(self) -> bool:
      """在训练步骤触发时检查是否需要保存检查点"""
      self._step_count += 1
      if self.checkpoint_dir is not None and self._trainer_ref is not None:
          if self._step_count % self.save_interval == 0:
              checkpoint_path = str(
                  Path(self.checkpoint_dir) / f"checkpoint_ep{self._step_count}.pt"
              )
              Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
              self._trainer_ref.save_checkpoint(checkpoint_path, self._step_count, {})
              self._saved_checkpoints.append(checkpoint_path)
              logger.info(f"Checkpoint saved at episode {self._step_count}")
              self._cleanup_old_checkpoints()
      return True
  ```
  变更要点：(1) `_step_count += 1` 无条件执行；(2) 直接用 `self._step_count` 判断，去掉中间变量 `episode`；(3) 因为 `_step_count` 先递增再判断，`save_interval=100` 时首次保存在 step 100，避免了 episode 0 就保存的问题。
- **修复代价**: **低**。改动仅涉及一个方法的 3-4 行代码。
- **修复收益**: **中**。修正了步数计数逻辑的 bug，使代码行为正确且不依赖外部调用方式。

---

### Issue 2: int 类型转换会将 float 静默截断
- **真实性**: **确认存在**。源码 `config_parser.py` 第 150-154 行，`int(value)` 对 float 值（如 3.7）会静默截断为 3，无任何警告。
- **级别评估**: HIGH → **HIGH**。配置参数被静默修改确实危险，尤其 `batch_size`、`ppo_epochs` 等关键参数若被截断可能导致训练行为异常且难以排查。
- **影响评估**: 原影响描述准确。YAML 解析器对 `3.0` 会解析为 int，但对 `3.7` 会解析为 float，后者被截断后用户完全不知情。
- **建议评估**: 原建议合理。对于 int 类型转换，当输入为 float 且不为整数时应发出警告。
- **详细修复方案**:
  ```python
  # config_parser.py 第 150-154 行，替换为：
  elif expected_type is int:
      if isinstance(value, float) and value != int(value):
          logger.warning(
              f"类型转换警告: {current_path} 期望 int, 但值 {value} 为非整数 float, 截断为 {int(value)}"
          )
      result[key] = int(value)
  ```
  注意：这里不拒绝覆盖（不 `continue`），因为截断后的值仍有意义，但需要发出警告让用户知晓。符合项目原则"让错误尽早暴露"。
- **修复代价**: **低**。仅增加 3 行条件判断和警告日志。
- **修复收益**: **中**。防止配置被静默修改而不自知，帮助用户尽早发现配置错误。

---

### Issue 3: bool 类型转换对非字符串非布尔值的行为不正确
- **真实性**: **确认存在**。源码 `config_parser.py` 第 167-168 行，`else: result[key] = bool(value)` 对非字符串值直接用 `bool()` 转换。`bool(2)` 为 `True`，这在配置场景下不符合预期。
- **级别评估**: HIGH → **MEDIUM**。实际场景中，CLI 传入 bool 参数通常为字符串（如 `"true"`/`"false"`），YAML 解析的 bool 也是原生 bool。非 0/1 的整数传入 bool 配置项是极低概率场景。不过 `bool(2)` 为 True 确实不符合"2 代表 True"的直觉，应视为潜在问题。
- **影响评估**: 原影响描述准确，但概率较低。在当前 `_TYPE_MAPPING` 中，bool 类型只有 `ppo.lr_cosine_decay` 和 `ppo.enable_auxiliary`，这两个参数的输入几乎不可能是整数 2。
- **建议评估**: 原建议合理。增加对整数 0/1 的显式处理，对其他非预期输入发出警告。
- **详细修复方案**:
  ```python
  # config_parser.py 第 155-168 行，替换为：
  elif expected_type is bool:
      if isinstance(value, str):
          lowered = value.strip().lower()
          if lowered in ("true", "1"):
              result[key] = True
          elif lowered in ("false", "0"):
              result[key] = False
          else:
              logger.warning(
                  f"类型转换失败: {current_path} 期望 bool, 但值不可解析 '{value}', 跳过覆盖"
              )
              continue
      elif isinstance(value, bool):
          result[key] = value
      elif isinstance(value, int):
          if value in (0, 1):
              result[key] = bool(value)
          else:
              logger.warning(
                  f"类型转换警告: {current_path} 期望 bool, 但整数值 {value} 非 0/1, 视为 {bool(value)}, 跳过覆盖"
              )
              continue
      else:
          logger.warning(
              f"类型转换失败: {current_path} 期望 bool, 但类型为 {type(value).__name__}, 跳过覆盖"
          )
          continue
  ```
  变更要点：(1) 增加对原生 `bool` 类型的直接处理；(2) 对 `int` 只接受 0/1，其他整数拒绝覆盖；(3) 其他类型一律拒绝。
- **修复代价**: **低**。改动约 10 行，逻辑清晰。
- **修复收益**: **中**。使 bool 类型转换更健壮，避免非预期输入静默通过。

---

### Issue 4: MetricsLoggingCallback 使用 `or 0` 导致合法值 0.0 被替换
- **真实性**: **确认存在**。源码 `metrics_callback.py` 第 22-26 行，`metrics.get('training_loss') or 0` 模式确实会将 `0.0` 替换为 `0`。
- **级别评估**: HIGH → **MEDIUM**。原报告定为 HIGH 过高。这段代码仅用于日志输出（`logger.info`），且 `0.0 or 0` 的结果在 `:.4f` 格式化后输出为 `0.0000`，无论类型是 float 还是 int 都一样。0.0 与 0 在日志输出场景中无实际差异。唯一的问题是 `0.0`（合法指标值）与 `None`（缺失指标）无法区分，但缺失指标时 log 输出 0.0000 也是合理行为。
- **影响评估**: 原影响描述中"类型从 float 变成 int"准确但影响微乎其微——这些值仅用于日志格式化输出，不参与后续计算。"如果后续逻辑依赖类型的精确性可能出问题"这一说法不成立，因为这里没有后续逻辑。
- **建议评估**: 原建议方向正确，使用 `dict.get(key, default)` 更规范。
- **详细修复方案**:
  ```python
  # metrics_callback.py 第 22-26 行，替换为：
  training_loss = metrics.get('training_loss', 0)
  policy_loss = metrics.get('policy_loss', 0)
  value_loss = metrics.get('value_loss', 0)
  entropy = metrics.get('entropy', 0)
  reward_mean = metrics.get('reward_mean', 0)
  ```
  变更要点：用 `dict.get(key, default)` 替代 `dict.get(key) or default`，语义更准确——仅在 key 不存在时使用默认值，而非值为 falsy 时。
- **修复代价**: **低**。5 行简单替换。
- **修复收益**: **低**。仅日志输出场景，实际行为无可见变化，但代码语义更准确。

---

### Issue 5: EpisodeBatchWriter 窗口判断仅基于 episode 计数
- **真实性**: **部分存在**。源码 `episode_batch_writer.py` 第 64 行，`on_batch_complete` 中仅通过 `self._episode_count_in_window >= self._window_size` 判断是否刷新窗口。`_batch_count_in_window` 确实未参与判断。
- **级别评估**: MEDIUM → **LOW**。原报告的担忧"如果训练过程中只调用 `on_batch_complete` 而不调用 `on_episode_complete`，窗口永远不会刷新"在理论上成立，但在实际训练流程中，`on_episode_complete` 和 `on_batch_complete` 是成对调用的——每个 episode 完成后都会触发 `on_episode_complete`。窗口大小的语义本身就是基于 episode 数量，而非 batch 数量。`_batch_count_in_window` 的存在仅是为了记录窗口内的 batch 数，供聚合统计使用，而非作为触发条件。这是合理的设计选择。
- **影响评估**: 原影响描述的极端场景（只调 batch 不调 episode）在实际中不会发生。窗口按 episode 计数触发是符合业务语义的。
- **建议评估**: 原建议"基于 batch 计数或混合计数来触发窗口刷新"不合理。EpisodeBatch 的语义本身就是以 episode 为粒度，batch 是 episode 内的子操作。不应改变触发条件。
- **详细修复方案**: 无需修复。但如果要消除 `_batch_count_in_window` 不参与判断的"困惑"，可以删除该字段，因为它从未在判断中使用，仅在 `_reset_window` 中重置。不过 `_batch_count_in_window` 在聚合统计中可能有用（`num_batches` 字段），所以保留也是合理的。当前代码逻辑是正确的，无需改动。
- **修复代价**: 不适用
- **修复收益**: 不适用

---

### Issue 6: SystemMetricsSampler._flush 不清理 deque 导致重复记录
- **真实性**: **确认存在**。源码 `system_metrics_sampler.py` 第 129-153 行。`_flush` 将 `list(self._samples)` 写入 `system_metrics.json`（覆盖写，没问题），同时将 `samples_list[-self._samples_per_write:]`（最近 6 条）追加到 `.json.log`。但由于 deque 不清理，下次 flush 时 deque 仍包含上次已写入的样本，导致 `.json.log` 中出现重复记录。
  具体分析：`_maybe_flush`（第 122-127 行）在 `len(self._samples) >= self._samples_per_write and len(self._samples) % self._samples_per_write == 0` 时触发。假设 `SAMPLES_PER_WRITE=6`：
  - 第 6 个样本到达，触发 flush，写入最近 6 条到 `.json.log`
  - 第 12 个样本到达，触发 flush，写入最近 6 条（第 7-12 条）到 `.json.log`——此时无重复
  - 但如果在两次 flush 之间调用了 `stop()`→`_flush()`，deque 未清理，下次 start 后的新样本可能与之前的旧样本混合
  - 更关键的是：如果 `_flush` 被外部调用（如 `stop()` 中），后续的 `_maybe_flush` 可能再次写入已写入的样本

  实际上仔细推演：`_maybe_flush` 每 6 个样本触发一次，且写入的是 `[-6:]`，由于样本是顺序追加的，每次 flush 写入的 6 条恰好是新增的 6 条，不会重复。**但 `_flush` 中写入 `.json.log` 的逻辑是 `samples_list[-self._samples_per_write:]`，在 `_maybe_flush` 触发时 deque 中有 6/12/18... 个样本，`[-6:]` 确实是最新 6 条，不重复。**

  然而，`stop()` 方法（第 58-70 行）调用了 `_flush()`，此时如果 deque 中已有 18 个样本但上次 flush 是在第 12 个样本时，那么 `[-6:]` 会是第 13-18 条，仍不重复。

  **重新评估**：在 `_maybe_flush` 的触发逻辑下，flush 严格在样本数为 6 的倍数时触发，每次写最近 6 条，确实不会重复。问题仅出现在非正常流程中（如手动调用 `_flush`）。但 `stop()` 中的 `_flush` 也只写 `[-6:]`，而此时 deque 中的样本数可能不是 6 的倍数，导致部分样本被遗漏而非重复。

  实际上，**真正的问题是 `system_metrics.json`（覆盖写）中始终包含全部 deque 内容（最多 1000 条）的统计信息，这是合理的。但 `.json.log` 仅写入最近 6 条，可能遗漏 stop 时 deque 中间部分的样本。**

  结论：原报告说的"重复记录"在实际正常运行流程中不会发生，但存在"遗漏记录"的潜在问题（stop 时 deque 中的中间样本未被追加到 `.json.log`）。
- **级别评估**: MEDIUM → **LOW**。正常运行流程中不会出现重复记录。仅存在 stop 时的遗漏问题，且系统指标采样本身非关键功能。
- **影响评估**: 原影响描述"`.json.log` 中出现重复记录"在正常运行流程中不准确，实际风险是 stop 时可能遗漏部分样本记录。
- **建议评估**: 原建议"清理已写入的样本"方向正确但实现需谨慎——`system_metrics.json` 依赖全部 deque 数据计算统计信息。更好的方案是在 `_flush` 中记录已写入 `.json.log` 的位置。
- **详细修复方案**:
  ```python
  # system_metrics_sampler.py，在 __init__ 中增加：
  self._last_flushed_index = 0

  # _flush 方法中，替换第 151-153 行：
  with open(log_file, "a") as f:
      new_samples = samples_list[self._last_flushed_index:]
      for s in new_samples:
          f.write(json.dumps(s) + "\n")
  self._last_flushed_index = len(samples_list)
  ```
  变更要点：用索引记录已写入位置，仅追加新增样本，避免重复和遗漏。
- **修复代价**: **低**。增加 1 个实例属性和 3 行逻辑代码。
- **修复收益**: **低**。正常运行流程中不会重复，仅修正了 stop 场景下的遗漏问题。

---

### Issue 7: EvalBattleWriter._read_eval_records 是死代码
- **真实性**: **确认存在**。源码 `eval_battle_writer.py` 第 177-193 行，`_read_eval_records` 方法在类中定义但从未被调用。`on_episode_batch_complete` 已改用内存缓冲区 `self._eval_buffer`。
- **级别评估**: LOW → **LOW**。死代码，不影响功能，但应清理。
- **影响评估**: 原影响描述准确。该方法是从文件回读的旧实现，已被内存缓冲方案替代。
- **建议评估**: 原建议"删除 `_read_eval_records` 方法"正确。根据项目原则"不做向前兼容，不用考虑历史版本的兼容性"，死代码应当删除。
- **详细修复方案**:
  ```python
  # eval_battle_writer.py，删除第 177-193 行（整个 _read_eval_records 方法）
  ```
  即删除：
  ```python
      def _read_eval_records(self, ep_start: int, ep_end: int) -> List[Dict]:
          if not self._battle_log_path.exists():
              return []
          records = []
          with open(self._battle_log_path, "r") as f:
              for line in f:
                  line = line.strip()
                  if not line:
                      continue
                  try:
                      rec = json.loads(line)
                      ep = rec.get("episode", -1)
                      if ep_start <= ep <= ep_end:
                          records.append(rec)
                  except (json.JSONDecodeError, KeyError):
                      continue
          return records
  ```
- **修复代价**: **低**。删除 17 行代码。
- **修复收益**: **低**。减少代码噪音，避免维护者困惑。

---

### Issue 8: config_parser.load_yaml 未指定文件编码
- **真实性**: **确认存在**。源码 `config_parser.py` 第 71 行，`open(yaml_path, "r")` 未指定 `encoding` 参数。
- **级别评估**: LOW → **不修复**。项目原则明确"优先考虑 Linux，不用考虑 Windows"。在 Linux/macOS 上默认编码为 UTF-8，不会出问题。YAML 配置文件通常也不含非 ASCII 字符（配置项为英文 key + 数值）。
- **影响评估**: 原影响描述的"Windows 上可能 UnicodeDecodeError"在项目原则下不适用。
- **建议评估**: 原建议在通用场景下合理，但根据项目原则"优先考虑 Linux，不用考虑 Windows"，此问题不应修复。加上 `encoding="utf-8"` 虽然简单，但属于过度防护，与"不做容错，让错误尽早暴露"原则一致——如果文件编码有问题，应该在读取时直接报错而非静默处理。
- **详细修复方案**: 不修复。
- **修复代价**: 不适用
- **修复收益**: 不适用

---

### Issue 9: CheckpointCallback 未使用 path_config 参数
- **真实性**: **确认存在**。源码 `checkpoint_callback.py` 第 20 行接受 `path_config` 参数，第 26 行存储为 `self.path_config`，但在 `_on_step` 中完全未使用。`callback_factory.py` 第 34-35 行同时传入 `checkpoint_dir` 和 `path_config`。
- **级别评估**: MEDIUM → **MEDIUM**。`path_config` 是公开参数但未被使用，属于接口欺骗。调用者可能认为传了 `path_config` 就能工作，但实际只依赖 `checkpoint_dir`。不过由于 `callback_factory.py` 总是同时传入两者，当前不会造成实际问题。
- **影响评估**: 原影响描述准确。`path_config` 参数目前是无用的，但接口上暗示它有意义。
- **建议评估**: 原建议给出了两个方向。根据项目原则"不做向前兼容，不用考虑将来的修改而增加复杂性"和"不做过度设计"，应选择**移除 `path_config` 参数**，而非增加从 `path_config` 获取检查点目录的逻辑。当前 `callback_factory.py` 在调用时已经显式传入 `checkpoint_dir`（从 `path_config.get_checkpoint_dir(run_id)` 获取），`CheckpointCallback` 不需要关心 `path_config`。
- **详细修复方案**:
  ```python
  # checkpoint_callback.py，修改为：
  from pathlib import Path
  from typing import List, Optional

  from loguru import logger

  from .base_ppo_callback import BasePPOCallback


  class CheckpointCallback(BasePPOCallback):
      """检查点保存回调 - 定期保存模型检查点

      对应 OpenRL 的 CheckpointCallback，扩展了旧检查点清理逻辑。
      """

      def __init__(
          self,
          save_interval: int,
          checkpoint_dir: Optional[str] = None,
          keep_last_n: int = 5,
      ):
          super().__init__()
          self.save_interval = save_interval
          self.checkpoint_dir = checkpoint_dir
          self.keep_last_n = keep_last_n
          self._step_count = 0
          self._saved_checkpoints: List[str] = []
  ```
  同时修改 `callback_factory.py`：
  ```python
  # callback_factory.py 第 13-17 行，移除 path_config 参数：
  def create_ppo_callbacks(
      save_interval: int,
      log_interval: int = 10,
      checkpoint_dir: Optional[str] = None,
      enable_checkpoint: bool = True,
      enable_metrics_logging: bool = True,
      enable_tensorboard: bool = False,
      tensorboard_dir: Optional[str] = None,
      tensorboard_interval: int = 10,
  ) -> Optional[CallbackList]:

  # 第 32-35 行，移除 path_config 传参：
      CheckpointCallback(
          save_interval=save_interval,
          checkpoint_dir=checkpoint_dir,
      )
  ```
  同时移除 `callback_factory.py` 中对 `PathConfig` 的 import（如果不再使用）。
- **修复代价**: **低**。修改 2 个文件，删除相关参数和 import。
- **修复收益**: **中**。消除接口误导，使代码更简洁。

---

### Issue 10: BaseCallback 缺少方法定义
- **真实性**: **确认存在**。源码 `base_callback.py` 第 1-5 行，`BaseCallback` 仅有文档字符串，无任何方法定义。`CallbackList` 通过 `cb.on_step()`、`cb.init_callback()` 等方式调用，依赖鸭子类型。
- **级别评估**: HIGH → **LOW**。原报告定为 HIGH 过高。实际上：
  1. `CallbackList` 类型注解为 `List[BaseCallback]`，但实际使用的回调都继承自 `BasePPOCallback`，后者已定义了所有必要方法
  2. 项目原则"不做容错，让错误尽早暴露尽早解决"——如果有人直接继承 `BaseCallback` 且忘记实现方法，`AttributeError` 正是期望的快速失败行为
  3. 使用 `abc.ABC` + `abstractmethod` 会强制所有子类实现所有方法，但 `BasePPOCallback` 已经提供了默认空实现，`ABC` 反而增加复杂性
  4. 当前只有 `BasePPOCallback` 一个直接子类被使用，没有多态继承的风险
- **影响评估**: 原影响描述"直接继承 `BaseCallback` 而非 `BasePPOCallback`，忘记实现方法会抛出 `AttributeError`"——这恰好符合项目"快速失败"原则，不需要额外防护。
- **建议评估**: 原建议使用 `abc.ABC` + `abstractmethod` 与项目原则冲突。更好的方案是给 `BaseCallback` 添加与 `BasePPOCallback` 一致的默认空实现，使其成为合理的基类。但这不是紧急问题。
- **详细修复方案**:
  ```python
  # base_callback.py，修改为：
  class BaseCallback:
      """基础回调接口"""

      def init_callback(self, trainer) -> None:
          pass

      def on_training_start(self) -> None:
          pass

      def on_step(self) -> bool:
          return True

      def on_training_end(self) -> None:
          pass


  __all__ = ["BaseCallback"]
  ```
  变更要点：提供默认空实现（非 abstractmethod），使 `BaseCallback` 成为合理的基类。子类可以按需重写。这与 `BasePPOCallback` 的默认实现一致，不引入 ABC 的复杂性。
- **修复代价**: **低**。增加 12 行方法定义。
- **修复收益**: **低**。使接口更明确，但当前无实际使用场景会因此受益。属于代码规范改善。

---

### Issue 11: EpisodeBatchWriter _aggregate_battle 空列表防护与调用方不一致
- **真实性**: **确认存在**。源码 `episode_batch_writer.py` 第 82-83 行，`_flush_battle_window` 已检查 `if not self._battle_buffer: return`；第 103-105 行，`_aggregate_battle` 又检查 `if n == 0: return {}`。
- **级别评估**: LOW → **LOW**。双重防护不会造成 bug。原报告提出的"battle_stats 文件缺少对应窗口记录导致数据间隙"是一个合理的观察，但需要商榷其严重性——如果一个窗口内所有 episode 都没有 battle_record，写入一条 `num_episodes=0` 的占位记录是否比不写入更好，取决于消费方的需求。如果消费方只关心有数据的窗口，不写入更合理。
- **影响评估**: 原影响描述的"数据间隙"问题存在但影响有限。聚合数据消费方（如可视化工具）需要处理可能缺失的窗口，但这在数据聚合场景中是常见模式。
- **建议评估**: 原建议"即使 buffer 为空也写入一条占位记录"需要根据实际消费方需求决定。在当前场景下，没有 battle_record 的窗口写入占位记录没有实际价值——缺少对战数据就意味着该窗口没有可聚合的内容。不建议增加占位记录。
- **详细修复方案**: 无需修复。当前的防护逻辑是合理的：调用方前置检查避免不必要的聚合调用，`_aggregate_battle` 内部检查作为防御性编程（静态方法可能被其他地方调用）。两者职责不同，不是"不一致"。
- **修复代价**: 不适用
- **修复收益**: 不适用

## 总体结论

11 个问题中，**8 个确认存在且值得修复**，2 个不存在或不需修复（Issue 5 窗口设计合理、Issue 8 项目原则下不需处理），1 个部分存在（Issue 6 重复记录实际不会发生，但有遗漏风险）。

### 建议修复优先级排序

| 优先级 | Issue | 修复代价 | 修复收益 |
|--------|-------|---------|---------|
| 1 | Issue 1: _step_count 不递增 | 低 | 中 |
| 2 | Issue 2: int 静默截断 float | 低 | 中 |
| 3 | Issue 9: 移除无用 path_config 参数 | 低 | 中 |
| 4 | Issue 3: bool 类型转换不健壮 | 低 | 中 |
| 5 | Issue 4: `or 0` 替换为 `.get(key, 0)` | 低 | 低 |
| 6 | Issue 10: BaseCallback 增加默认方法 | 低 | 低 |
| 7 | Issue 7: 删除死代码 | 低 | 低 |
| 8 | Issue 6: _flush 遗漏记录问题 | 低 | 低 |

### 不修复项

| Issue | 原因 |
|-------|------|
| Issue 5 | 窗口按 episode 计数触发是合理的业务语义设计 |
| Issue 8 | 项目原则"优先考虑 Linux"，不需要为 Windows 兼容性添加编码参数 |
| Issue 11 | 当前防护逻辑合理，不需要写入空占位记录 |

### 关键级别调整

- **Issue 1**: CRITICAL → HIGH（实际使用中几乎不触发，但逻辑确实有误）
- **Issue 3**: HIGH → MEDIUM（非 0/1 整数传入 bool 配置项是极低概率场景）
- **Issue 4**: HIGH → MEDIUM（仅影响日志输出，0.0 和 0 在格式化后无可见差异）
- **Issue 6**: MEDIUM → LOW（正常运行流程中不会重复，仅有 stop 场景遗漏风险）
- **Issue 10**: HIGH → LOW（当前实际使用中不会触发问题，且快速失败符合项目原则）
