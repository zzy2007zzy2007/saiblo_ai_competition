# PPO V2 核心训练模块 — 二次审查评估报告

二次审查日期：2026-06-10
基于：`ppo_v2/docs/code/core_training_issues.md`
源代码根目录：`ppo_v2/src/ppo_antwar/`
项目原则：不做容错让错误尽早暴露、不做过度设计、解决问题从根源出发

---

## 问题 1：合并批次中截断 Episode 的 GAE 计算错误

### 1. 问题真实性：**真实存在，但影响范围需修正**

**源代码验证：**

- `batch.py` 第 129-131 行确认：`merge` 保存了 `_final_values = [b.final_value for b in batches]`，但只将 `batches[-1].final_value` 赋给 `merged.final_value`。`_final_values` 列表从未被 `compute_gae` 消费。

- `episode_collector.py` 第 105 行 `done = terminated or truncated`，第 107 行 `batch.add_step(..., done)` —— 注意这里 `done` 是 env 返回的 `terminated or truncated`。**但关键点是**：第 185-186 行 `if episode_length >= self._max_steps: done = True` 只影响循环退出，不影响 batch 中已记录的 `done` 值。实际上对于因 `max_steps` 截断的 episode，env 的 `truncated` 标志本身可能为 `True`（Gymnasium API 中 `truncated` 在达到 `max_steps` 时会为 `True`），因此第 105 行的 `done = terminated or truncated` **可能已经是 `True`**。

- 但如果 env 未设置 `max_episode_steps`，则 `truncated=False`，此时截断 episode 的末步 `done=False`。

- `gae.py` 第 47 行：`next_value = values_t[t + 1] * (1.0 - dones_t[t])`。当截断 episode 末步 `dones_t[t]=0` 时，`next_value = values_t[t+1]`，即下一个 episode 首步的 value —— 这确实是跨 episode 的错误传播。

- `ppo_trainer.py` 第 116-121 行：`_compute_gae` 只传入 `final_value=batch.final_value`，即只有最后一个 batch 的 final_value，中间 episode 的 final_value 丢失。

- `episode_collector.py` 第 192-194 行确认了截断检测逻辑：`was_truncated = (episode_length >= self._max_steps) and not (terminated or truncated)`，此时设置 `batch.final_value = self._self_agent.get_value(obs_self)`。但合并后此值丢失。

**修正**：原审查的示例场景中"E2(截断, done=False)"不一定总是成立——取决于 env 是否在 `max_steps` 时返回 `truncated=True`。如果 env 返回 `truncated=True`，则 `done=True`，GAE 在该步会正确重置为 `next_value=0`，只是 bootstrap value 不准确（应为 `V(s_T)` 而非 0）。**真正的问题核心**：合并 batch 后中间 episode 的 `final_value` 丢失，导致截断 episode 的 GAE bootstrap value 错误。

### 2. 级别准确性：**CRITICAL 合理，但需分情况**

- 如果 env 设置了 `max_episode_steps`（Gymnasium 标准做法），截断时 `truncated=True` → `done=True` → GAE 不会跨 episode 传播，但 bootstrap value 为 0 而非 `V(s_T)`，导致截断步的优势值偏负。影响中等。
- 如果 env 未设置 `max_episode_steps`，截断时 `done=False` → GAE 跨 episode 传播 → 优势值严重错误。影响严重。

**结论**：级别 CRITICAL 合理，但应明确触发条件。

### 3. 影响准确性：**基本准确，略有夸大**

- "GAE 优势从 E2 末尾错误传播到 E3 开头"只有在 env 不返回 `truncated=True` 时才成立。
- 但即使 `done=True`，截断 episode 末步的 bootstrap value 为 0（而非 `V(s_T)`）也会导致该步 GAE 偏差，这是确定存在的问题。

### 4. 建议科学性：**方案 A 方向正确但过度设计；方案 B 更符合项目原则**

方案 A 需要引入 `episode_boundaries` 和 `final_values` 列表，修改 `compute_gae` 接口签名，属于过度设计。

方案 B 更简单：在截断 episode 末步设置 `done=True`，并使用 `final_value` 做 bootstrap。但方案 B 的描述"引入单独的 `truncated` 标记数组"也不必要——GAE 只需区分"自然终止（done=True, final_value=0）"和"截断终止（done=True, final_value=V(s_T)）"，而 `final_value` 已能表达这一语义。

### 5. 详细修复方案

**修复策略**：从根源出发——保证每个 episode 的末步 `done=True`，同时保留每个 episode 的 `final_value` 供 GAE 使用。

**步骤 1**：修改 `episode_collector.py`，确保截断 episode 末步也记录 `done=True`

文件：`ppo_v2/src/ppo_antwar/trainer/episode_collector.py`

```python
# 修改前（第 105-107 行）：
done = terminated or truncated
batch.add_step(obs_self, action_self, reward_self, value, log_prob, done)

# 修改后：
env_done = terminated or truncated
batch.add_step(obs_self, action_self, reward_self, value, log_prob, env_done)
```

然后在循环退出后，如果是截断而非自然终止，修正末步的 done：

```python
# 在第 185-186 行 while 循环结束后，添加：
# 如果因 max_steps 截断但 env 未标记 truncated，修正末步 done
if episode_length >= self._max_steps and not (terminated or truncated):
    batch.dones[-1] = True
```

**步骤 2**：修改 `batch.py` 的 `merge` 方法，将 `final_value` 改为列表

文件：`ppo_v2/src/ppo_antwar/trainer/batch.py`

```python
# 修改前（第 33-34 行）：
self.final_value: float = 0.0
self._final_values: list = []

# 修改后：
self.final_values: List[float] = []  # 每个 episode 的 bootstrap value
```

```python
# 修改前（第 129-131 行）：
merged._final_values = [b.final_value for b in batches]
merged.final_value = batches[-1].final_value if batches else 0.0

# 修改后：
# 收集各 batch 的 final_values（单个 batch 只有一个 episode，所以 final_values 长度为 1）
for b in batches:
    if b.final_values:
        merged.final_values.extend(b.final_values)
    else:
        merged.final_values.append(b.final_value)
```

单 episode batch 的 `final_values` 初始化：

```python
# 在 EpisodeCollector.collect 末尾（第 193-194 行后）：
if was_truncated:
    batch.final_values = [self._self_agent.get_value(obs_self)]
else:
    batch.final_values = [0.0]
```

**步骤 3**：修改 `compute_gae` 使用逐 episode 的 `final_values`

文件：`ppo_v2/src/ppo_antwar/utils/gae.py`

```python
def compute_gae(
    rewards, values, dones, gamma, gae_lambda, device,
    final_value=0.0,
    final_values=None,  # List[float]，每个 episode 的 bootstrap value
):
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    values_t = torch.tensor(values, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    T = len(rewards)
    advantages = torch.zeros(T, device=device)
    gae = 0.0

    # 构建 episode 边界 map：done=True 的位置即为 episode 结束
    # 为每个 done=True 的位置分配对应的 final_value
    episode_final_values = []
    if final_values is not None:
        episode_final_values = list(final_values)
    else:
        # 兼容旧接口：所有 episode 共用一个 final_value（仅最后一个 episode）
        done_indices = (dones_t > 0.5).nonzero(as_tuple=True)[0].tolist()
        for i in range(len(done_indices)):
            if i == len(done_indices) - 1:
                episode_final_values.append(final_value)
            else:
                episode_final_values.append(0.0)  # 自然终止
        # 如果没有 done=True（不应该发生），使用 final_value
        if not done_indices:
            episode_final_values.append(final_value)

    ep_idx = len(episode_final_values) - 1  # 从最后一个 episode 开始倒推

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0 if dones_t[t] > 0.5 else final_value
        elif dones_t[t] > 0.5:
            # episode 边界：使用该 episode 的 bootstrap value
            if ep_idx >= 0 and ep_idx < len(episode_final_values):
                next_value = episode_final_values[ep_idx]
            else:
                next_value = 0.0
            ep_idx -= 1
        else:
            next_value = values_t[t + 1]

        delta = rewards_t[t] + gamma * next_value - values_t[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones_t[t]) * gae
        advantages[t] = gae

    returns = advantages + values_t
    return advantages, returns
```

**步骤 4**：修改 `ppo_trainer.py` 的 `_compute_gae` 传入 `final_values`

文件：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py`

```python
# 修改前（第 775-783 行）：
advantages, returns = compute_gae(
    rewards=rewards,
    values=values,
    dones=dones,
    gamma=ppo_cfg.get("gamma"),
    gae_lambda=ppo_cfg.get("gae_lambda"),
    device=self.device,
    final_value=final_value,
)

# 修改后：
advantages, returns = compute_gae(
    rewards=rewards,
    values=values,
    dones=dones,
    gamma=ppo_cfg.get("gamma"),
    gae_lambda=ppo_cfg.get("gae_lambda"),
    device=self.device,
    final_value=final_value,
    final_values=getattr(batch, 'final_values', None),
)
```

同时修改 `_ppo_update` 方法签名以传入 batch：

```python
# 修改前（第 116-121 行）：
advantages, returns = self._compute_gae(
    tensors["rewards"].tolist(),
    tensors["values"].tolist(),
    tensors["dones"].tolist(),
    final_value=batch.final_value,
)

# 修改后：
advantages, returns = self._compute_gae(
    tensors["rewards"].tolist(),
    tensors["values"].tolist(),
    tensors["dones"].tolist(),
    final_value=batch.final_value,
    batch=batch,
)
```

### 6. 修复代价：**中**

需要修改 4 个文件（batch.py、episode_collector.py、gae.py、ppo_trainer.py），接口变更有向后兼容性（`final_values` 参数可选），但逻辑较复杂，需要仔细测试 episode 边界对应关系。

### 7. 修复收益：**高**

截断 episode 的 GAE 计算是训练正确性的根本保障。如果 env 不返回 `truncated=True`，当前代码会产生严重错误的优势估计；即使 env 返回 `truncated=True`，bootstrap value 为 0 也会导致截断步的价值估计偏差。修复后训练将产生正确的梯度信号。

---

## 问题 2：NaN 恢复返回值被忽略，训练未正确停止

### 1. 问题真实性：**真实存在**

**源代码验证：**

- `selfplay.py` 第 456-458 行：
  ```python
  if has_critical:
      self._trainer.handle_nan_recovery(self._episode_count)
  self._batch_pool = []
  return
  ```
  返回值确实被忽略。

- `safety.py` 第 60-81 行确认 `on_nan_detected` 返回 `bool`：`True` 表示继续训练，`False` 表示应停止训练。

- `ppo_trainer.py` 第 851-858 行确认 `handle_nan_recovery` 转发 `on_nan_detected` 的返回值。

### 2. 级别准确性：**HIGH 合理**

NaN 达到阈值后不停止训练，会导致模型在已损坏的权重上继续训练，产生更多 NaN，浪费算力且可能无法恢复。

### 3. 影响准确性：**准确**

当 NaN 计数达到阈值时，`on_nan_detected` 返回 `False`，但调用方忽略了该信号，训练继续进行，权重持续恶化。

### 4. 建议科学性：**建议合理**

原审查建议检查返回值并在 `False` 时抛出 `RuntimeError`，符合"让错误尽早暴露"的原则。

### 5. 详细修复方案

文件：`ppo_v2/src/ppo_antwar/trainer/selfplay.py`

```python
# 修改前（第 456-458 行）：
if has_critical:
    self._trainer.handle_nan_recovery(self._episode_count)
self._batch_pool = []
return

# 修改后：
if has_critical:
    should_continue = self._trainer.handle_nan_recovery(self._episode_count)
    if not should_continue:
        raise RuntimeError(
            f"NaN count exceeded threshold at episode {self._episode_count}, aborting training"
        )
self._batch_pool = []
return
```

### 6. 修复代价：**低**

仅需修改 1 个文件的 3 行代码。

### 7. 修复收益：**高**

防止在权重已损坏的情况下继续浪费训练资源，确保 NaN 恢复机制的完整性。

---

## 问题 3：opponent_agent 为 None 时 EpisodeCollector.collect 崩溃

### 1. 问题真实性：**真实存在**

**源代码验证：**

- `episode_collector.py` 第 94 行：`action_opponent = opponent_agent.act(obs_opponent)` —— 直接调用 `opponent_agent.act()`，无 None 检查。

- `episode_collector.py` 第 232-238 行：`_get_opponent_move` 方法已实现 None 处理逻辑（随机选择合法动作），但 `collect` 方法从未调用它。

- `selfplay.py` 第 283-303 行：`_select_opponent_agent` 在 `except` 块中设置 `opponent_id = None` 但返回 `(None, None)`。第 572 行 `opponent_id, opponent_agent = self._select_opponent_agent()`，如果返回 None，则传入 `collect`。

- `selfplay.py` 第 706-715 行：`_collect_episode_with_swap` 直接将 `opponent_agent` 传递给 `EpisodeCollector.collect`，无 None 检查。

### 2. 级别准确性：**HIGH 合理**

`opponent_agent` 为 None 时必定崩溃，属于运行时必现错误。

### 3. 影响准确性：**准确**

`AttributeError: 'NoneType' object has no attribute 'act'` 会导致当前 episode 收集失败。但根据项目原则"不做容错让错误尽早暴露"，崩溃本身是合理的——问题在于 `_get_opponent_move` 这种容错式回退逻辑是否应该存在。

### 4. 建议科学性：**部分合理，但需重新审视**

原审查建议将第 94 行替换为调用 `_get_opponent_move`，这等价于"对手加载失败时用随机策略代替"。这属于容错逻辑，与项目原则冲突。

**更符合项目原则的方案**：对手加载失败时应该让错误尽早暴露，而不是悄悄回退到随机策略。应该在 `_select_opponent_agent` 层面确保不会返回 `None`，或者在调用链中尽早抛出明确异常。

但考虑到自对弈训练的连续性需求，如果每次对手加载失败都中断训练也不合理。因此需要在"尽早暴露"和"训练连续性"之间平衡。

### 5. 详细修复方案

**方案（推荐）**：在 `_select_opponent_agent` 中，如果加载失败则抛出异常让调用方决定；同时在 `_run_single_episode` 中 catch 该异常，记录日志后 continue 跳过此 episode。

文件：`ppo_v2/src/ppo_antwar/trainer/selfplay.py`

```python
# 修改 _select_opponent_agent（第 283-303 行）：
def _select_opponent_agent(self) -> Tuple[str, OpponentAgent]:
    """选择对手 Agent。加载失败时抛出 RuntimeError。"""
    opponent_id = self._selfplay_manager.select_opponent()
    checkpoint_path = self._selfplay_manager.get_opponent_checkpoint_path(opponent_id)
    try:
        opponent_agent = OpponentAgent.from_checkpoint(
            checkpoint_path,
            device=self._trainer.device,
            hidden_dim=self._config.get("network", {}).get("hidden_dim"),
            enable_auxiliary=self._config.get("ppo", {}).get("enable_auxiliary"),
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load opponent {opponent_id} from {checkpoint_path}: {e}"
        ) from e
    return opponent_id, opponent_agent
```

```python
# 修改 _run_single_episode（第 560-602 行），在 _select_opponent_agent 调用处加 try：
def _run_single_episode(self, env_factory):
    # ... 省略前面的代码 ...
    try:
        opponent_id, opponent_agent = self._select_opponent_agent()
    except RuntimeError as e:
        logger.error(f"Skipping episode {self._episode_count}: {e}")
        return {}, 0, 0, "draw"
    # ... 省略后面的代码 ...
```

这样修改同时解决了问题 5（`checkpoint_path` 未定义），因为 `checkpoint_path` 在 try 块外已被赋值，且异常被显式捕获。

### 6. 修复代价：**低**

修改 2 个方法，约 10 行代码变更。

### 7. 修复收益：**高**

避免运行时必现崩溃，同时保持错误可见性（日志记录），不违背"尽早暴露"原则。

---

## 问题 4：best_avg_reward_episode 在滑动窗口截断后索引错误

### 1. 问题真实性：**真实存在**

**源代码验证：**

- `selfplay.py` 第 332-337 行：`_episode_rewards` 超过 `_max_episode_stats` 时截断旧数据，保留最新 1000 条。
- `selfplay.py` 第 674-678 行：
  ```python
  "best_avg_reward_episode": self._episode_rewards.index(
      max(self._episode_rewards)
  ) + 1 if self._episode_rewards else 0,
  ```
  `.index()` 返回的是窗口内的相对索引，而非实际 episode 编号。

### 2. 级别准确性：**MEDIUM 合理**

该值仅用于训练结束时的 summary 日志，不影响训练过程本身。但在长期训练中，summary 中的 episode 编号会严重失真。

### 3. 影响准确性：**准确**

原审查的示例完全正确。例如窗口包含 episode 4001-5000，最大奖励在 index=0（即 episode 4001），代码报告 `0+1=1`，实际应为 4001。

### 4. 建议科学性：**合理**

原审查建议维护全局的 `_best_reward` 和 `_best_reward_episode`，简单直接，无过度设计。

### 5. 详细修复方案

文件：`ppo_v2/src/ppo_antwar/trainer/selfplay.py`

```python
# 在 __init__ 中添加（约第 73 行后）：
self._best_reward = float('-inf')
self._best_reward_episode = 0
```

```python
# 在 _collect_and_update_payoff 中（约第 332 行后）添加：
self._episode_rewards.append(reward)
self._episode_lengths.append(length)
# 维护全局最佳奖励
if reward > self._best_reward:
    self._best_reward = reward
    self._best_reward_episode = self._episode_count
```

```python
# 修改 _finalize_training 中的 summary（第 668-679 行）：
# 修改前：
"best_avg_reward": max(self._episode_rewards) if self._episode_rewards else 0,
"best_avg_reward_episode": self._episode_rewards.index(max(self._episode_rewards)) + 1 if self._episode_rewards else 0,

# 修改后：
"best_avg_reward": self._best_reward if self._best_reward > float('-inf') else 0,
"best_avg_reward_episode": self._best_reward_episode,
```

### 6. 修复代价：**低**

修改 1 个文件，添加 2 个属性 + 3 行更新逻辑 + 替换 2 行 summary。

### 7. 修复收益：**中**

修正训练 summary 中的关键指标，避免调试时被错误信息误导。不影响训练过程。

---

## 问题 5：checkpoint_path 在异常处理中可能未定义

### 1. 问题真实性：**真实存在**

**源代码验证：**

- `selfplay.py` 第 286-302 行：
  ```python
  try:
      opponent_id = self._selfplay_manager.select_opponent()   # 第 287 行
      checkpoint_path = self._selfplay_manager.get_opponent_checkpoint_path(...)  # 第 288 行
      opponent_agent = OpponentAgent.from_checkpoint(...)      # 第 291 行
  except Exception as e:
      logger.error(f"Failed to load opponent {opponent_id} from {checkpoint_path}: {e}")  # 第 299 行
  ```
  如果第 287 行 `select_opponent()` 抛出异常，则 `checkpoint_path` 未赋值，第 299 行引用它将触发 `NameError`。

### 2. 级别准确性：**MEDIUM 合理**

原始异常信息会被 `NameError` 覆盖，增加调试难度。但不会导致训练崩溃——except 块捕获了 `NameError`，只是日志信息不完整。

### 3. 影响准确性：**准确**

原审查描述正确。`NameError` 会导致原始异常 `e` 的信息被丢失，日志中只显示 `name 'checkpoint_path' is not defined`。

### 4. 建议科学性：**基本合理，但可以更简洁**

原审查建议在 try 之前初始化 `checkpoint_path = "unknown"`，这是最简单的修法，但不够彻底。更好的方案是与问题 3 一并修复——将 `_select_opponent_agent` 改为显式抛出 `RuntimeError`（见问题 3 的修复方案），这样 `checkpoint_path` 已在 try 块内赋值，且异常被显式构造，不再依赖 except 块中的 `checkpoint_path`。

### 5. 详细修复方案

**方案 A（独立修复）**：

文件：`ppo_v2/src/ppo_antwar/trainer/selfplay.py`

```python
# 修改前（第 283-303 行）：
def _select_opponent_agent(self) -> Tuple[Optional[str], Optional[OpponentAgent]]:
    opponent_id = None
    opponent_agent = None
    try:
        opponent_id = self._selfplay_manager.select_opponent()
        checkpoint_path = self._selfplay_manager.get_opponent_checkpoint_path(opponent_id)
        opponent_agent = OpponentAgent.from_checkpoint(...)
    except Exception as e:
        logger.error(f"Failed to load opponent {opponent_id} from {checkpoint_path}: {e}")
        opponent_id = None
    return opponent_id, opponent_agent

# 修改后：
def _select_opponent_agent(self) -> Tuple[str, OpponentAgent]:
    opponent_id = self._selfplay_manager.select_opponent()
    checkpoint_path = self._selfplay_manager.get_opponent_checkpoint_path(opponent_id)
    try:
        opponent_agent = OpponentAgent.from_checkpoint(
            checkpoint_path,
            device=self._trainer.device,
            hidden_dim=self._config.get("network", {}).get("hidden_dim"),
            enable_auxiliary=self._config.get("ppo", {}).get("enable_auxiliary"),
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load opponent {opponent_id} from {checkpoint_path}: {e}"
        ) from e
    return opponent_id, opponent_agent
```

这样 `checkpoint_path` 在 `try` 之前已赋值，`RuntimeError` 中可以正确包含其值。

**方案 B（与问题 3 联合修复）**：见问题 3 的修复方案，已一并解决。

### 6. 修复代价：**低**

修改 1 个方法，约 5 行变更。

### 7. 修复收益：**中**

避免异常信息丢失，提高调试效率。且与问题 3 联合修复后，整体代码更清晰。

---

## 问题 6：日志中的 value_pred_mean/value_pred_std 与损失计算使用的值不一致

### 1. 问题真实性：**真实存在**

**源代码验证：**

- `ppo_trainer.py` 第 446-449 行：`new_values = new_values.clamp(values_min, values_max)` —— `new_values` 被 in-place clamp 后用于计算 value loss。
- `ppo_trainer.py` 第 362 行：`new_values = clean_tensor(new_values)` —— 这是 clamp 前的处理。
- `ppo_trainer.py` 第 393-408 行：`_collect_step_metrics` 接收的 `new_values` 是第 362 行 `clean_tensor` 后的值，但此时已被 `_compute_losses` 中的 clamp 修改了吗？

**关键分析**：`_compute_losses` 第 446 行 `new_values = new_values.clamp(...)` 创建了一个新张量（`clamp` 不是 in-place 操作），只修改了函数局部变量 `new_values` 的绑定，不影响调用方 `_process_minibatch` 中的同名变量。

调用链：
1. `_process_minibatch` 第 362 行：`new_values = clean_tensor(new_values)` —— 此时 `new_values` 是 clean 后的值
2. 第 365-380 行：将 `new_values` 传入 `_compute_losses`
3. `_compute_losses` 第 446 行：`new_values = new_values.clamp(...)` —— 函数参数的局部绑定，**不影响**调用方的 `new_values`
4. `_process_minibatch` 第 393-408 行：将 `new_values` 传入 `_collect_step_metrics` —— 此 `new_values` 仍是第 362 行 clean 后的值

**因此日志中的 `value_pred_mean`/`value_pred_std` 确实是 clamp 前的值**，与损失计算使用的 clamp 后的值不一致。原审查描述正确。

### 2. 级别准确性：**LOW 合理**

这只是日志指标与实际计算值的细微差异。clamp 范围为 `[-500, 500]`，正常训练中 value 预测很少超出此范围，因此大多数情况下 clamp 前后值相同。只有 value 预测极端时才会出现差异。

### 3. 影响准确性：**准确但不严重**

可能导致调试时误判 value 网络的输出分布，但实际影响很小。

### 4. 建议科学性：**方向正确但代价不值得**

原审查建议将 clamp 后的 `new_values` 传出给 metrics，需要修改 `_compute_losses` 的返回值，增加了接口复杂度。对于一个 LOW 级别的日志一致性问题，这个代价不值得。

更简单的方案：在 `_collect_step_metrics` 中直接 clamp 一次即可。

### 5. 详细修复方案

文件：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py`

```python
# 修改 _collect_step_metrics（第 542-543 行）：
# 修改前：
"value_pred_mean": new_values.mean().item(),
"value_pred_std": new_values.std().item(),

# 修改后：
"value_pred_mean": new_values.clamp(
    float(clip_values.get("values_min", -500)),
    float(clip_values.get("values_max", 500)),
).mean().item(),
"value_pred_std": new_values.clamp(
    float(clip_values.get("values_min", -500)),
    float(clip_values.get("values_max", 500)),
).std().item(),
```

但这需要 `_collect_step_metrics` 接收 `clip_values` 参数。查看方法签名（第 508-524 行），当前已有 `clip_eps` 和 `clip_eps_vf` 参数但没有 `clip_values`。添加参数会增加接口负担。

**更合理的方案**：在 `_process_minibatch` 中，将 clamp 后的值单独传给 `_collect_step_metrics`：

```python
# 在 _process_minibatch 第 362 行后添加：
new_values_for_metrics = new_values.clamp(
    float(clip_values.get("values_min", -500)),
    float(clip_values.get("values_max", 500)),
)
```

然后将 `new_values_for_metrics` 传给 `_collect_step_metrics` 而非 `new_values`。

但考虑到这是 LOW 级别问题，且正常训练中 clamp 几乎不生效，**建议暂不修复**，仅在文档中标注此已知差异。

### 6. 修复代价：**低**

约 5 行代码变更。

### 7. 修复收益：**低**

正常训练中 value 预测极少超出 `[-500, 500]`，日志差异几乎不会出现。修复收益极低。

---

## 综合评估总结

| 问题 | 真实性 | 原级别 | 建议级别 | 修复优先级 | 修复代价 | 修复收益 |
|------|--------|--------|----------|------------|----------|----------|
| 1. GAE 计算错误 | 真实 | CRITICAL | CRITICAL | P0 | 中 | 高 |
| 2. NaN 返回值忽略 | 真实 | HIGH | HIGH | P1 | 低 | 高 |
| 3. opponent_agent None 崩溃 | 真实 | HIGH | HIGH | P1 | 低 | 高 |
| 4. best_avg_reward_episode 索引错误 | 真实 | MEDIUM | MEDIUM | P2 | 低 | 中 |
| 5. checkpoint_path 未定义 | 真实 | MEDIUM | MEDIUM | P2 | 低 | 中 |
| 6. value_pred 日志不一致 | 真实 | LOW | LOW | P3 | 低 | 低 |

**修复建议优先级排序**：
1. 问题 2（NaN 返回值）+ 问题 5（checkpoint_path）+ 问题 3（opponent_agent None）可联合修复，代价低收益高
2. 问题 1（GAE 计算错误）是核心算法正确性问题，必须修复但需仔细测试
3. 问题 4（索引错误）简单修复即可
4. 问题 6（日志不一致）建议暂不修复，仅标注已知差异
