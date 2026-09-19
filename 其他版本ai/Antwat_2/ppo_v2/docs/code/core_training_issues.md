# PPO V2 核心训练模块代码审查报告

审查日期：2026-06-10
审查范围：ppo_v2/src/ppo_antwar/trainer/ 及 utils/ 下 13 个核心文件

---

## 问题 1：合并批次中截断 Episode 的 GAE 计算错误

**严重程度：CRITICAL**
**所在文件及行号：**
- `batch.py` 第 130-131 行（`merge` 方法）
- `gae.py` 第 42-50 行（`compute_gae` 函数）
- `ppo_trainer.py` 第 116-121 行（`_compute_gae` 调用）
- `episode_collector.py` 第 107 行（`add_step` 记录 `done=False`）

**问题描述：**

当多个 episode batch 被合并（`EpisodeBatch.merge`）后，GAE 计算会产生错误的优势估计。具体原因：

1. **`final_value` 丢失**：`batch.py` 第 131 行只保留最后一个 batch 的 `final_value`（`merged.final_value = batches[-1].final_value`），中间被截断 episode 的 bootstrap value 丢失。虽然第 130 行存储了 `_final_values = [b.final_value for b in batches]`，但该列表从未被 GAE 计算消费。

2. **`done` 标志不一致**：`episode_collector.py` 第 107 行在 `add_step` 时记录的 `done` 值来自 `terminated or truncated`（第 105 行），此时对于因 `max_steps` 截断的 episode，env 并未返回 terminated/truncated，因此 `done=False`。第 185-186 行虽然后续将 `done` 设为 `True` 以退出循环，但 batch 中已记录 `done=False`。

3. **GAE 计算错误传播**：在 `gae.py` 第 47 行，对于中间步骤 `next_value = values_t[t + 1] * (1.0 - dones_t[t])`。当截断 episode 末步 `dones_t[t]=0` 时，`next_value = values_t[t+1]`，即下一个 episode 首步的价值——这是完全错误的。GAE 会将跨 episode 的优势值错误地传播，相当于把两个独立 episode 当成了连续轨迹处理。

**示例场景：**
```
合并后的 batch: [E1(自然终止), E2(截断, done=False), E3(自然终止)]
- E1 末步: done=True → GAE 正确重置（next_value=0）
- E2 末步: done=False → next_value = E3首步的V(s)，而非E2的final_value → 错误！
- GAE 优势从 E2 末尾错误传播到 E3 开头
```

**修复建议：**

方案 A（推荐）：修改 `compute_gae` 接受逐 episode 的 `final_values` 和 episode 边界信息，在每个 episode 边界正确使用对应的 bootstrap value：

```python
def compute_gae(
    rewards, values, dones, gamma, gae_lambda, device,
    final_values=None,  # List[float]，每个 episode 的 bootstrap value
    episode_boundaries=None,  # List[int]，每个 episode 在合并 batch 中的结束索引
):
    ...
    for t in reversed(range(T)):
        if t == T - 1:
            # 使用最后一个 episode 的 final_value
            next_value = 0.0 if dones_t[t] > 0.5 else (final_values[-1] if final_values else 0.0)
        elif dones_t[t] > 0.5:
            next_value = 0.0  # 自然终止
        elif t in episode_boundary_set:
            # 截断 episode 边界：使用对应的 final_value
            ep_idx = episode_boundary_map[t]
            next_value = final_values[ep_idx]
        else:
            next_value = values_t[t + 1]
        ...
```

方案 B（更简单但破坏现有语义）：在 `episode_collector.py` 中，对截断 episode 的末步也记录 `done=True`，并在 GAE 中引入单独的 `truncated` 标记数组来区分自然终止和截断。

---

## 问题 2：NaN 恢复返回值被忽略，训练未正确停止

**严重程度：HIGH**
**所在文件及行号：** `selfplay.py` 第 456-458 行

**问题描述：**

`_flush_batch_to_update` 中检测到 NaN/Inf 后调用 `self._trainer.handle_nan_recovery(self._episode_count)`，但完全忽略了其返回值。`NanRecoveryHandler.on_nan_detected` 返回 `False` 时表示 NaN 次数已达阈值、应停止训练，但当前代码无论返回什么都继续训练循环。

```python
if has_critical:
    self._trainer.handle_nan_recovery(self._episode_count)  # 返回值被忽略！
self._batch_pool = []
return  # 继续训练循环
```

**修复建议：**

```python
if has_critical:
    should_continue = self._trainer.handle_nan_recovery(self._episode_count)
    if not should_continue:
        logger.error("NaN threshold exceeded, stopping training")
        raise RuntimeError("Training aborted: NaN count exceeded threshold")
self._batch_pool = []
return
```

---

## 问题 3：`opponent_agent` 为 None 时 `EpisodeCollector.collect` 崩溃

**严重程度：HIGH**
**所在文件及行号：**
- `episode_collector.py` 第 94 行（直接调用 `opponent_agent.act`）
- `episode_collector.py` 第 233-238 行（`_get_opponent_move` 已实现 None 处理但未被调用）
- `selfplay.py` 第 283-303 行（`_select_opponent_agent` 可返回 `opponent_agent=None`）

**问题描述：**

`_select_opponent_agent` 在对手加载失败时返回 `opponent_agent=None`。该值传递到 `EpisodeCollector.collect` 后，第 94 行直接调用 `opponent_agent.act(obs_opponent)` 会抛出 `AttributeError: 'NoneType' object has no attribute 'act'`。

值得注意的是，`_get_opponent_move` 方法（第 233-238 行）已实现了 `opponent_agent is None` 的回退逻辑（随机选择合法动作），但 `collect` 方法从未调用它。

**修复建议：**

将第 94 行替换为：
```python
action_opponent = self._get_opponent_move(opponent_agent, obs_opponent)
```

---

## 问题 4：`best_avg_reward_episode` 在滑动窗口截断后索引错误

**严重程度：MEDIUM**
**所在文件及行号：** `selfplay.py` 第 674-678 行

**问题描述：**

`self._episode_rewards` 是一个滑动窗口列表（第 334-337 行在超过 `_max_episode_stats` 时截断旧数据）。第 674 行使用 `self._episode_rewards.index(max(self._episode_rewards))` 获取最大奖励的索引，但该索引是窗口内的相对位置，不是实际的 episode 编号。加 1 不等于实际 episode 编号。

例如：`_max_episode_stats=1000`，当前 episode=5000，窗口包含 episode 4001-5000 的奖励。如果最大奖励在窗口 index=0 处（对应 episode 4001），代码报告 `0+1=1`，实际应为 4001。

**修复建议：**

维护一个与 `_episode_rewards` 同步的 episode 编号列表，或记录全局最佳奖励及对应的 episode 编号：

```python
# 在 __init__ 中添加
self._best_reward = float('-inf')
self._best_reward_episode = 0

# 在 _collect_and_update_payoff 中更新
if reward > self._best_reward:
    self._best_reward = reward
    self._best_reward_episode = self._episode_count

# 在 _finalize_training 中使用
"best_avg_reward": self._best_reward,
"best_avg_reward_episode": self._best_reward_episode,
```

---

## 问题 5：`checkpoint_path` 在异常处理中可能未定义

**严重程度：MEDIUM**
**所在文件及行号：** `selfplay.py` 第 297-300 行

**问题描述：**

`_select_opponent_agent` 方法中，如果 `select_opponent()`（第 287 行）抛出异常，则 `checkpoint_path`（第 288 行）尚未赋值，但第 299 行的 except 块引用了 `checkpoint_path`，会触发 `NameError: name 'checkpoint_path' is not defined`，导致原始异常信息丢失。

```python
try:
    opponent_id = self._selfplay_manager.select_opponent()
    checkpoint_path = self._selfplay_manager.get_opponent_checkpoint_path(...)  # 可能执行不到
    ...
except Exception as e:
    logger.error(f"Failed to load opponent {opponent_id} from {checkpoint_path}: {e}")
    #                                                    ^^^^^^^^^^^^^^^^ 可能未定义
```

**修复建议：**

```python
checkpoint_path = "unknown"
try:
    opponent_id = self._selfplay_manager.select_opponent()
    checkpoint_path = self._selfplay_manager.get_opponent_checkpoint_path(opponent_id)
    ...
except Exception as e:
    logger.error(f"Failed to load opponent {opponent_id} from {checkpoint_path}: {e}")
    opponent_id = None
```

---

## 问题 6：日志中的 `value_pred_mean`/`value_pred_std` 与损失计算使用的值不一致

**严重程度：LOW**
**所在文件及行号：**
- `ppo_trainer.py` 第 446-448 行（`new_values` 被 clamp）
- `ppo_trainer.py` 第 543-544 行（`_collect_step_metrics` 使用未 clamp 的 `new_values`）

**问题描述：**

在 `_compute_losses` 中，`new_values` 被 `clamp(values_min, values_max)` 后用于计算 value loss。但 `_process_minibatch` 中传递给 `_collect_step_metrics` 的 `new_values` 是 `clean_tensor` 处理后的原始值（第 362 行），不含 clamp。因此日志记录的 `value_pred_mean`/`value_pred_std` 反映的是 clamp 前的值，与实际参与损失计算的值不同，可能导致调试时产生误导。

**修复建议：**

在 `_process_minibatch` 中，将 clamp 后的 `new_values` 传递给 `_collect_step_metrics`，或单独记录 clamp 前后的值：

```python
# 方案：在 _compute_losses 返回 clamped new_values，供 metrics 使用
new_values_clamped = new_values.clamp(...)
# ... 使用 new_values_clamped 计算 loss ...
return total_loss, aux_metrics, clip_fraction, ratio, log_ratio, policy_loss_scalar, value_loss_scalar, new_values_clamped
```
