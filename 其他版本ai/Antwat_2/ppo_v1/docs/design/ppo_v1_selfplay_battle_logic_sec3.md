# 3. SelfPlay 训练流程

> 本章描述 `SelfPlayTrainer` 的完整训练循环，涵盖对手选择、数据收集、PPO 更新、对手池管理、Payoff 更新和 Baseline Battle 评估。

---

## 3.1 整体训练循环

`SelfPlayTrainer.train()` 是 SelfPlay 训练的主入口，位于 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L669-L798)。每个 episode 执行以下 8 个步骤：

| 步骤 | 操作 | 关键方法 |
|------|------|----------|
| 1 | **训练进度更新** | `selfplay_manager.update_progress(episode)` |
| 2 | **对手选择** | `selfplay_manager.select_opponent()` |
| 3 | **加载对手 Agent** | `_load_opponent_agent(opponent_id)` |
| 4 | **先后手交替** | `swap_positions = (battle_count % 2 == 1)` |
| 5 | **单局数据收集** | `_collect_episode_with_swap(env, opponent_agent, swap_positions, ...)` |
| 6 | **PPO 更新** | `trainer._ppo_update(batch)` |
| 7 | **Payoff 更新** | `selfplay_manager.update_payoff(opponent_id, result, rounds)` |
| 8 | **定期 Checkpoint 与 Baseline Battle** | `opponent_update_interval` / `battle_interval` |

核心流程伪代码：

```python
def train(self, num_episodes=100000, opponent_update_interval=500, ...):
    env = self.trainer.env_factory()

    for episode in range(num_episodes):
        # 步骤1: 更新训练进度（用于动态探索/利用调整）
        self.selfplay_manager.update_progress(episode)

        # 步骤2: 对手选择
        opponent_id = self.selfplay_manager.select_opponent()

        # 步骤3: 加载对手 Agent
        opponent_agent = self._load_opponent_agent(opponent_id) if opponent_id else None

        # 步骤4: 先后手交替
        swap_positions = (self.battle_count % 2 == 1)

        # 步骤5: 单局数据收集
        batch, battle_details = self._collect_episode_with_swap(
            env, opponent_agent, swap_positions, opponent_id, episode
        )
        self.battle_count += 1

        # 步骤6: PPO 更新
        if len(batch) > 0:
            ppo_metrics = self.trainer._ppo_update(batch)
            self.trainer.episode_count += 1
            # 记录指标: episode_reward, episode_length, total_steps

        # 步骤7: Payoff / 对手池更新
        if opponent_id is not None:
            self.selfplay_manager.update_payoff(opponent_id, result, rounds)

        # 步骤8a: 定期保存新对手 checkpoint（每 opponent_update_interval 个 episode）
        if episode % opponent_update_interval == 0 and episode > 0:
            checkpoint_path = self.trainer.save_checkpoint(...)
            self.selfplay_manager.add_opponent(checkpoint_path)
            self._save_league_state()

        # 步骤8b: Baseline Battle 评估（每 battle_interval 个 episode）
        if self.battle_coordinator and episode % self.battle_interval == 0:
            results = self.battle_coordinator.run_ppo_evaluation_with_agent(...)

        # 定期衰减历史战绩
        self._maybe_decay(episode)
```

**关键配置参数**（来自 [ppo_antwar.yaml](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/configs/ppo_antwar.yaml)）：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `total_episodes` | 150000 | 训练总 episode 数 |
| `opponent_update_interval` | 500 | 每多少 episode 保存一个对手 checkpoint |
| `battle_interval` | 1000 | 每多少 episode 触发一次 Baseline Battle |
| `n_battles` | 5 | Baseline Battle 的对战数 |
| `max_steps_per_episode` | 512 | 单局最大步数 |
| `decay_interval` | 1000 | 历史战绩衰减的间隔 |

---

## 3.2 对手选择机制（OpponentSelector）

`OpponentSelector` 负责从对手池中智能选择对手，位于 [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L9-L131)。

### 3.2.1 exploit vs explore 概率

每次选择对手时，以概率 `exploit_prob` 进入 **exploit 模式**，否则进入 **explore 模式**。

```python
exploit_prob = self._get_adaptive_exploit_prob()
if random.random() < exploit_prob:
    selected = self._exploit_select(player_id, opponent_candidates)
    if selected is not None:
        return selected
return self._explore_select(opponent_candidates)
```

### 3.2.2 自适应 exploit_prob 增长

`exploit_prob` 随训练进度自适应增长，从 **0.3** 逐步增加到 **0.9**，支持三种调度方式：

**线性调度（linear）**：

```
exploit_prob = 0.3 + 0.6 × progress
```

其中 `progress = min(1.0, total_episodes / 10000)`，即前 10000 个 episode 内线性增长。

**Sigmoid 调度（sigmoid）**：

```
exploit_prob = 0.3 + 0.6 × 1/(1 + exp(-10 × (progress - 0.5)))
```

**指数调度（exp）**：

```
exploit_prob = 0.3 + 0.6 × (1 - exp(-5 × progress))
```

### 3.2.3 exploit 模式：选择最强的 confident 对手

```python
def _exploit_select(self, player_id, opponent_candidates):
    # 1. 筛选 confidence 对手（sigma < 5.0）
    confident_opponents = [
        opp_id for opp_id in opponent_candidates
        if self._is_confident_opponent(opp_id)
    ]

    # 2. 如果有 confident 对手，选 TrueSkill mu 最高的前 3 个中随机
    if confident_opponents:
        rated_opponents.sort(key=lambda x: x[1], reverse=True)
        top_opponents = rated_opponents[:min(3, len(rated_opponents))]
        return random.choice(top_opponents)

    # 3. 如果没有 confident 对手，选 sigma 最小的（最有潜力的）
    all_ratings.sort(key=lambda x: x[1])
    return all_ratings[0][0]
```

### 3.2.4 explore 模式：随机选择一个较成熟的对手

```python
def _explore_select(self, opponent_candidates):
    # 过滤掉 sigma 过大的（不确定性太高的对手）
    filtered = [
        opp_id for opp_id in opponent_candidates
        if not self.payoff.get_trueskill_rating(opp_id) or
           self.payoff.get_trueskill_rating(opp_id).sigma < 7.5  # min_confidence_sigma * 1.5
    ]
    if not filtered:
        filtered = opponent_candidates
    return random.choice(filtered)
```

### 3.2.5 confident 判断

```python
def _is_confident_opponent(self, opponent_id):
    rating = self.payoff.get_trueskill_rating(opponent_id)
    if rating is None:
        return False
    return rating.sigma < 5.0  # _min_confidence_sigma
```

---

## 3.3 先后手交替（Swap Positions）

偶数局 Agent 先手、奇数局 Agent 后手：

```python
swap_positions = (self.battle_count % 2 == 1)
```

| `battle_count` | `swap_positions` | Agent 角色 |
|:---:|:---:|:---|
| 0 | `False` | Agent 先手 (player_0)，对手后手 (player_1) |
| 1 | `True` | Agent 后手 (player_1)，对手先手 (player_0) |
| 2 | `False` | Agent 先手 |
| 3 | `True` | Agent 后手 |
| ... | ... | 交替 |

这确保了 Agent 在两种角色下都得到充分训练，避免位置偏好。

---

## 3.4 单局对战数据收集（_collect_episode_with_swap）

这是 SelfPlay 最核心的数据收集函数，位于 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L423-L667)。它完成一整局对战的观察获取、动作选择、step 执行、奖励获取和结果判定。

### 3.4.1 数据结构：EpisodeBatch

数据收集使用 `EpisodeBatch` 数据结构，定义于 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L33-L57)：

```python
@dataclass
class EpisodeBatch:
    observations_board: List[np.ndarray]   # (T,) 每个元素 shape=(28,19,19)
    observations_global: List[np.ndarray]  # (T,) 每个元素 shape=(33,)
    observations_mask: List[np.ndarray]    # (T,) 每个元素 shape=(96,)
    actions: List[np.ndarray]              # (T,) 标量 action_id
    rewards: List[np.ndarray]              # (T,) 标量 reward
    values: List[np.ndarray]               # (T,) 标量 value 估计
    log_probs: List[np.ndarray]            # (T,) 标量 log_prob
    dones: List[np.ndarray]                # (T,) 0.0 或 1.0

    def __len__(self):
        return len(self.actions)

    def to_tensors(self, device):
        return {
            'board': torch.FloatTensor(np.array(self.observations_board)).to(device),
            'global': torch.FloatTensor(np.array(self.observations_global)).to(device),
            'action_mask': torch.FloatTensor(np.array(self.observations_mask)).to(device),
            'actions': torch.LongTensor(np.array(self.actions)).to(device),
            'rewards': torch.FloatTensor(np.array(self.rewards)).to(device),
            'values': torch.FloatTensor(np.array(self.values)).to(device),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)).to(device),
            'dones': torch.FloatTensor(np.array(self.dones)).to(device),
        }
```

### 3.4.2 完整流程步骤

每局对战的循环体内，每一步执行以下操作：

```
初始化: env.reset() → obs

循环 (while not done and steps < max_steps):
  ┌─────────────────────────────────────────────┐
  │ 1. 获取 Agent 观察                          │
  │    swap_positions=False: player_obs = obs["player_0"]   │
  │    swap_positions=True:  player_obs = obs["player_1"]   │
  ├─────────────────────────────────────────────┤
  │ 2. Agent 动作选择 (trainer._select_action)   │
  │    → action, log_prob, value                │
  ├─────────────────────────────────────────────┤
  │ 3. 对手动作选择（3 种情况）                   │
  │    ① 有对手 agent: opponent_agent.act()     │
  │    ② 无对手 agent: 随机合法动作              │
  ├─────────────────────────────────────────────┤
  │ 4. 双 step 执行                             │
  │    env.step(player_0_action)                │
  │    env.step(player_1_action) → obs, rewards │
  ├─────────────────────────────────────────────┤
  │ 5. 奖励获取                                 │
  │    swap=False: reward = rewards["player_0"] │
  │    swap=True:  reward = rewards["player_1"] │
  ├─────────────────────────────────────────────┤
  │ 6. 记录到 EpisodeBatch                      │
  │    追加 observation、action、log_prob、     │
  │    value、reward、done                      │
  ├─────────────────────────────────────────────┤
  │ 7. 终止判定                                 │
  │    done = terminated or truncated           │
  │    done→1.0 时 batch.dones[-1]=1.0         │
  └─────────────────────────────────────────────┘
```

### 3.4.3 第一步：获取 Agent 观察

根据 `swap_positions` 决定读取哪个玩家的观察：

```python
player = 1 if swap_positions else 0  # Agent 的玩家编号

if not swap_positions:
    player_obs = obs["player_0"]
else:
    player_obs = obs["player_1"]
```

`player_obs` 结构为 `{'board': (28,19,19), 'global': (33,), 'action_mask': (96,)}`。

### 3.4.4 第二步：Agent 动作选择

调用 `trainer._select_action(player_obs)`，位于 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L628-L660)：

```python
def _select_action(self, observation):
    board = torch.FloatTensor(observation['board']).unsqueeze(0).to(self.device)
    global_obs = torch.FloatTensor(observation['global']).unsqueeze(0).to(self.device)
    action_mask = torch.FloatTensor(observation['action_mask']).unsqueeze(0).to(self.device)

    # NaN 输入检测和清理
    if torch.isnan(board).any() or torch.isnan(global_obs).any():
        board = torch.nan_to_num(board, nan=0.0)
        global_obs = torch.nan_to_num(global_obs, nan=0.0)

    with torch.no_grad():
        action, log_prob, value = self.policy.get_action(
            board, global_obs, action_mask, deterministic=False  # 训练中采样
        )

    # NaN 输出检测和回退
    if torch.isnan(action) or torch.isnan(log_prob) or torch.isnan(value):
        action = torch.tensor(0, device=self.device)
        log_prob = torch.tensor(0.0, device=self.device)
        value = torch.tensor(0.0, device=self.device)

    return action.cpu().item(), log_prob.cpu().item(), value.cpu().item()
```

`deterministic=False` 表示训练时使用概率采样（`Categorical(probs).sample()`），而非 argmax。

### 3.4.5 第三步：对手动作选择（3 种情况）

**情况①：有对手 Agent（正常 SelfPlay）**

对手是历史 checkpoint 保存的 PPO Agent，加载后通过 `opponent_agent.act()` 获取动作：

```python
if opponent_agent is not None:
    if not swap_positions:
        opponent_obs = obs["player_1"]
    else:
        opponent_obs = obs["player_0"]
    opponent_action, _, _ = opponent_agent.act(opponent_obs)
```

`PPOAgent.act()` 使用 `deterministic=False`（采样），返回 `(action, log_prob, value)`。

**情况②：无对手 Agent（随机合法动作）**

训练初期对手池为空或对手加载失败时，使用随机合法动作：

```python
else:
    opponent_mask = torch.FloatTensor(opponent_obs["action_mask"]).to(self.trainer.device)
    valid_actions = torch.where(opponent_mask > 0)[0]
    if len(valid_actions) > 0:
        opponent_action = int(np.random.choice(valid_actions.cpu().numpy()))
    else:
        opponent_action = 0  # fallback to NO_OP
```

这保证了训练初期也有基本的对抗性。

**情况③：对手同样需要 swap 的观察**

对手观察的获取同样取决于 `swap_positions`：

| `swap_positions` | Agent 观察 | 对手观察 |
|:---:|:---|:---|
| `False` | `obs["player_0"]` | `obs["player_1"]` |
| `True` | `obs["player_1"]` | `obs["player_0"]` |

### 3.4.6 第四步：双 step 执行

AntWar 的回合制要求两个玩家各执行一次 `step` 才构成一个完整回合。`env.step()` 与 `swap_positions` 的关系：

```python
# 将动作映射到 player_0 和 player_1
if not swap_positions:
    player_0_action = action          # Agent 是 player_0
    player_1_action = opponent_action  # 对手是 player_1
else:
    player_0_action = opponent_action  # 对手是 player_0
    player_1_action = action          # Agent 是 player_1

# 先手 step → 后手 step（构成一个完整回合）
obs, _, _, _, _ = env.step(player_0_action)
obs, rewards, terminated, truncated, info = env.step(player_1_action)
```

**关键**：奖励只在后手 `env.step(player_1_action)` 后发放。

### 3.4.7 第五步：奖励获取

```python
if not swap_positions:
    reward = rewards.get("player_0", 0.0)
else:
    reward = rewards.get("player_1", 0.0)
batch.rewards.append(reward)
total_reward += reward
```

### 3.4.8 第六步：记录到 EpisodeBatch

每一步追加 Agent 的 `board`、`global`、`action_mask`、`action`、`log_prob`、`value` 到 `EpisodeBatch`：

```python
batch.observations_board.append(player_obs['board'])
batch.observations_global.append(player_obs['global'])
batch.observations_mask.append(player_obs['action_mask'])
batch.actions.append(action)
batch.values.append(value)
batch.log_probs.append(log_prob)
batch.dones.append(0.0)  # 中间步 dones=0
```

### 3.4.9 第七步：终止判定与结果判定

```python
done = terminated or truncated
if done:
    batch.dones[-1] = 1.0  # 最后一步 done=1
```

对局结束后，通过比较双方基地的最终 HP 判定结果：

```python
our_hp = game_state.bases[player].hp
enemy_hp = game_state.bases[1 - player].hp

if enemy_hp <= 0 and our_hp > 0:
    result = 'win'
elif our_hp <= 0 and enemy_hp > 0:
    result = 'loss'
else:
    result = 'draw'
```

同时记录先后手胜率（`first_move_wins`、`second_move_wins` 等）。

### 3.4.10 Battle Details 记录

整个对局过程中，`_collect_episode_with_swap` 还维护了一个 `battle_details` 字典，记录：

- 每回合的 HP、金币、塔状态（`round_details`）
- 每步的动作、log_prob、value、reward（`actions`）
- 动作分布统计（`action_counts`）
- 最终 HP、结果、总奖励、回合数、耗时

这些详情最终通过 `_save_selfplay_battle_details()` 保存到 `selfplay_battles/detailed_battles/` 目录下的 JSON 文件。

---

## 3.5 PPO 更新

PPO 更新由 `PPOTrainer._ppo_update()` 执行，位于 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L662-L823)。

### 3.5.1 GAE 计算

GAE（Generalized Advantage Estimation）由 `_compute_gae()` 实现，位于 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L825-L847)：

```python
def _compute_gae(self, rewards, values, dones):
    advantages = torch.zeros_like(rewards)
    running_advantage = 0.0
    gamma = self.ppo_config.gamma           # 0.99
    gae_lambda = self.ppo_config.gae_lambda  # 0.95

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        running_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * running_advantage
        advantages[t] = running_advantage

    returns = advantages + values.detach()
    return advantages, returns
```

公式：
- **TD 误差**: `δ_t = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)`
- **GAE Advantage**: `A_t = δ_t + γ · λ · (1 - done_t) · A_{t+1}`
- **Returns**: `R_t = A_t + V(s_t)`

返回的 `advantages` 和 `returns` 会被 clamp 到安全范围：
- `returns`: `[-1e6, 1e6]`
- `advantages`: `[-1e4, 1e4]`

### 3.5.2 多 Epoch 小批量更新

```python
indices = np.arange(batch_size)
np.random.shuffle(indices)  # 每个 epoch 前打乱数据

for _ in range(self.ppo_config.ppo_epochs):       # ppo_epochs = 4
    for start in range(0, batch_size, self.ppo_config.batch_size):  # batch_size = 64
        end = min(start + self.ppo_config.batch_size, batch_size)
        batch_indices = indices[start:end]
        # ... 对该 mini-batch 执行一次更新
```

每个 episode 的数据被分成多个 mini-batch，并在 4 个 epoch 内重复使用（每次重新 shuffle）。

### 3.5.3 Policy Loss：Clipped PPO Surrogate Objective

```python
# 计算新旧策略的比率
ratio = torch.exp(action_log_probs - old_log_probs)
ratio = torch.clamp(ratio, min=1e-5, max=1e5)  # 数值安全

# Clipped surrogate objective
surr1 = ratio * advantages.detach()
surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages.detach()

policy_loss = -torch.min(surr1, surr2).mean()
```

其中 `clip_eps = 0.2`，即比率被裁剪到 `[0.8, 1.2]` 范围。

### 3.5.4 Value Loss：Value Clipping 版 MSE

```python
# 价值函数裁剪 (Value Clipping)
values_clipped = old_values + torch.clamp(
    values - old_values,
    -clip_eps_vf,   # 0.2
    clip_eps_vf     # 0.2
)
value_loss_unclipped = F.mse_loss(values, returns.detach(), reduction='none')
value_loss_clipped = F.mse_loss(values_clipped, returns.detach(), reduction='none')
value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
```

Value Clipping 防止价值函数更新过大，与 Policy Clipping 理念一致。

### 3.5.5 Entropy Bonus

```python
entropy_loss = -entropy.mean()
```

Entropy Bonus 鼓励策略保持探索性，防止过早收敛。系数由熵退火动态调整（见 §3.6）。

### 3.5.6 总 Loss

```python
loss = policy_loss + vf_coef * value_loss + current_ent_coef * entropy_loss
```

其中 `vf_coef = 0.02`，`current_ent_coef` 由熵退火决定（初始值为 `ent_coef = 0.15`）。

### 3.5.7 梯度分别裁剪

策略网络和价值网络的参数梯度分开裁剪，使用不同的阈值：

```python
# 按参数名分组
for name, param in self.policy.named_parameters():
    if 'value_head' in name:
        value_params.append(param)   # 价值网络参数
    else:
        policy_params.append(param)  # 策略网络参数

# 分别裁剪
nn.utils.clip_grad_norm_(policy_params, max_grad_norm)     # max_grad_norm = 0.5
nn.utils.clip_grad_norm_(value_params, max_grad_norm_vf)    # max_grad_norm_vf = 0.2
self.optimizer.step()
```

分离梯度裁剪的原因是策略网络和价值网络的梯度规模可能差异很大。价值网络的梯度裁剪阈值更小（0.2 vs 0.5），防止价值函数剧烈震荡。

### 3.5.8 NaN 鲁棒性机制

整个 PPO 更新过程中贯穿了大量 NaN 检测和清理逻辑：

1. **输入数据清理**：`old_values`、`returns`、`advantages` 在计算前通过 `_clean_tensor()` 清理 NaN
2. **网络输出清理**：`action_log_probs`、`values`、`entropy` 检测并清理 NaN
3. **Loss 清理**：`policy_loss`、`value_loss`、`entropy_loss` 单独清理，再组合成总 loss
4. **NaN 阈值**：连续遇到 `nan_threshold=5` 次 NaN 后跳过该 batch（`_on_nan_detected()` 返回 True 则 `continue`，否则返回空字典）
5. **数值 clamp**：所有中间张量被 clamp 到安全范围（`VALUES_MIN=-10`、`VALUES_MAX=10`、`RATIO_MIN=1e-5`、`RATIO_MAX=1e5` 等）

### 3.5.9 返回值

```python
return {
    'policy_loss': total_policy_loss / update_count,
    'value_loss': total_value_loss / update_count,
    'entropy': total_entropy / update_count,
    'loss': total_loss / update_count,
}
```

---

## 3.6 学习率与熵退火

### 3.6.1 学习率调度

`get_learning_rate()` 位于 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L193-L211)，包含两个阶段：

**第一阶段：预热（Warmup）**——前 `lr_warmup_episodes=1000` 个 episode：

```python
if episode < warmup_episodes:
    alpha = episode / warmup_episodes
    return warmup_init + (self.base_lr - warmup_init) * alpha
```

`lr` 从 `warmup_init=1e-5` 线性增长到 `base_lr=1e-4`。

**第二阶段：可选余弦退火（Cosine Decay）**——预热之后：

```python
elif use_cosine_decay:
    progress = (episode - warmup_episodes) / (total_episodes - warmup_episodes)
    return base_lr * (1 + cos(π * progress)) / 2
```

`lr` 从 `base_lr=1e-4` 余弦衰减到 0。如果 `lr_cosine_decay=False`，则保持 `base_lr` 不变。

学习率图：

```
lr
│
│  1e-4 ┤         ╲
│       │        ╱  ╲______  (cosine decay)
│       │      ╱
│  1e-5 ┤____╱
│
└──────┼──────────────────────→ episode
       0   1000              150000
       (warmup)
```

### 3.6.2 熵退火

`get_entropy_coef()` 位于 [ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L180-L191)：

```python
def get_entropy_coef(self, episode):
    ent_coef_initial = self.ppo_config.ent_coef      # 0.15
    ent_coef_final = self.ppo_config.get('ent_coef_final', 0.001)
    anneal_power = self.ppo_config.get('ent_anneal_power', 1.0)

    progress = episode / self.total_episodes
    annealed = ent_coef_initial * max(0, 1 - progress) ** anneal_power
    return max(ent_coef_final, annealed)
```

公式：

```
ent_coef = max(ent_coef_final, ent_coef_initial × (1 - progress)^anneal_power)
```

其中 `progress = episode / total_episodes ∈ [0, 1]`。默认为线性退火（`anneal_power=1.0`），从 `0.15` 衰减到 `0.001`。

学习率和熵系数在每个 episode 的 PPO 更新前被更新：

```python
self.current_ent_coef = self.get_entropy_coef(self.episode_count)
self.current_lr = self.get_learning_rate(self.episode_count)
for param_group in self.optimizer.param_groups:
    param_group['lr'] = self.current_lr
```

---

## 3.7 对手池管理（OpponentPool）

`OpponentPool` 管理历史对手的存储和淘汰，位于 [opponent_selector.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L142-L267)。

### 3.7.1 核心参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `max_size` | 10 | 对手池最大容量 |
| `min_games_threshold` | 8 | 最小对战场次阈值（低于此值的对手受到保护） |

### 3.7.2 添加对手

```python
def add(self, opponent_id, checkpoint_path=None):
    if opponent_id not in self._opponents:
        self._opponents.append(opponent_id)
        self._added_timestamps[opponent_id] = 0
        self._games_played[opponent_id] = 0
        if checkpoint_path:
            self._checkpoint_paths[opponent_id] = checkpoint_path
        if len(self._opponents) > self.max_size:
            self._evict_worst()
```

每个对手由一个唯一的 `opponent_id`（如 `"opp_a1b2c3d4"`）标识，关联一个 `checkpoint_path`。

### 3.7.3 淘汰策略

当池满时（超过 `max_size=10`），触发淘汰：

```python
def _evict_worst(self):
    # 1. 筛选对战场次 ≥ min_games_threshold(8) 的对手
    candidates = [opp_id for opp_id in self._opponents
                  if self._games_played.get(opp_id, 0) >= self.min_games_threshold]

    # 2. 如果没有符合条件的对手，淘汰最旧的（_opponents[0]）
    if not candidates:
        victim = self._opponents[0]
    else:
        # 3. 计算淘汰分数：mu - 保护加成
        def eviction_score(opp_id):
            games = self._games_played.get(opp_id, 0)
            rating = self._payoff.get_trueskill_rating(opp_id)
            mu = rating.mu if rating else 25.0

            # 低场次保护：场次越少，保护加成越大
            if games < self.min_games_threshold:
                protection_bonus = (self.min_games_threshold - games) * 0.5
            else:
                protection_bonus = 0
            return mu - protection_bonus

        victim = min(candidates, key=eviction_score)  # 淘汰分数最低的

    self._remove_opponent(victim)
```

**淘汰逻辑**：在满足最低场次要求的对手中，选 `eviction_score = mu - protection_bonus` 最低的。`protection_bonus` 给予对战场次较少的对手额外保护（每少打一场 +0.5 分）。

### 3.7.4 对战场次追踪

每场对战后调用 `increment_games()`：

```python
def increment_games(self, opponent_id):
    if opponent_id in self._games_played:
        self._games_played[opponent_id] += 1
```

### 3.7.5 初始对手创建

训练开始时，如果对手池为空（即没有从持久化状态恢复），创建 `initial_opponents=3` 个随机权重的对手：

```python
def _create_initial_opponents(self):
    for i in range(self.initial_opponents):
        checkpoint_path = self._create_random_agent_checkpoint()
        new_opponent_id = self.selfplay_manager.add_opponent(checkpoint_path)

def _create_random_agent_checkpoint(self):
    policy = AntWarPolicy(...)  # 随机初始化的网络
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'config': {'network': self.trainer.network_config}
    }
    torch.save(checkpoint, checkpoint_path)
```

这些随机策略对手保证了训练初期的对抗多样性。

### 3.7.6 持久化

`OpponentPool` 支持保存/加载状态，通过 JSON 序列化：

```python
def save_state(self, filepath):
    state = {
        'max_size': self.max_size,
        'min_games_threshold': self.min_games_threshold,
        'opponent_ids': self._opponents,
        'checkpoint_paths': self._checkpoint_paths,
        'added_timestamps': self._added_timestamps,
        'games_played': self._games_played,
    }
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)

@classmethod
def load_state(cls, filepath):
    with open(filepath, 'r') as f:
        state = json.load(f)
    pool = cls(max_size=..., min_games_threshold=...)
    pool._opponents = state.get('opponent_ids', [])
    pool._checkpoint_paths = state.get('checkpoint_paths', {})
    pool._added_timestamps = state.get('added_timestamps', {})
    pool._games_played = state.get('games_played', {})
    return pool
```

持久化文件保存在 `{checkpoint_dir}/league_state/` 目录下：
- `opponent_pool.json`：对手池状态
- `payoff.json`：Payoff 状态

### 3.7.7 定期添加对手的触发

在训练循环中，每 `opponent_update_interval=500` 个 episode：

```python
if episode % opponent_update_interval == 0 and episode > 0:
    checkpoint_path = self.trainer.save_checkpoint(
        str(self.checkpoint_dir / f"model_{episode}.pt")
    )
    new_opponent_id = self.selfplay_manager.add_opponent(checkpoint_path)
    self._save_league_state()  # 同步持久化
```

---

## 3.8 Payoff 更新与 TrueSkill 评分

### 3.8.1 SelfPlayManager.update_payoff()

每局对战后，调用此方法更新 Payoff，位于 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L79-L109)：

```python
def update_payoff(self, opponent_id, result, rounds=0):
    # 1. 更新 BattleSharedPayoff 的战绩和 TrueSkill 评分
    self.payoff.update(
        home=self.current_player_id,  # "current_agent"
        away=opponent_id,
        result=result,  # 1=win, 0=draw, -1=loss
    )

    # 2. 更新对手的对战场次
    self.opponent_pool.increment_games(opponent_id)

    # 3. 更新内部统计字典
    if result == 1:
        self.games_against_opponents[opponent_id]['wins'] += 1
    elif result == 0:
        self.games_against_opponents[opponent_id]['draws'] += 1
    else:
        self.games_against_opponents[opponent_id]['losses'] += 1
```

### 3.8.2 BattleSharedPayoff.update()

这是 Payoff 更新的核心方法，位于 [payoff.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L76-L119)：

```python
def update(self, home, away, result):
    self.add_player(home)
    self.add_player(away)

    # --- 双向战绩记录 ---
    # 正向记录 (home vs away)
    key = f'{home}-{away}'
    record = self._data[key]
    record['games'] += 1
    if result == 0:      record['draws'] += 1
    elif result == 1:    record['wins'] += 1
    else:                record['losses'] += 1

    # 反向记录 (away vs home)
    reverse_key = f'{away}-{home}'
    reverse_record = self._data[reverse_key]
    reverse_record['games'] += 1
    if result == 0:      reverse_record['draws'] += 1
    elif result == 1:    reverse_record['losses'] += 1
    else:                reverse_record['wins'] += 1

    # --- TrueSkill 评分更新 ---
    home_rating = self._get_or_create_rating(home)
    away_rating = self._get_or_create_rating(away)

    ts_home = home_rating.to_trueskill()
    ts_away = away_rating.to_trueskill()

    if result == 1:
        new_home, new_away = trueskill.rate_1vs1(ts_home, ts_away)
    elif result == -1:
        new_away, new_home = trueskill.rate_1vs1(ts_away, ts_home)
    else:  # draw
        new_home, new_away = trueskill.rate_1vs1(ts_home, ts_away, drawn=True)

    self._trueskill_ratings[home] = TrueSkillRating.from_trueskill(
        new_home, games_played=home_rating.games_played + 1
    )
    self._trueskill_ratings[away] = TrueSkillRating.from_trueskill(
        new_away, games_played=away_rating.games_played + 1
    )
```

### 3.8.3 TrueSkillRating 数据结构

```python
@dataclass
class TrueSkillRating:
    mu: float = 25.0        # 技能均值，初始 25.0
    sigma: float = 8.33     # 技能不确定性，初始 8.33
    games_played: int = 0   # 已进行对战数

    @property
    def is_confident(self) -> bool:
        return self.sigma < 3.0

    def to_trueskill(self):
        return trueskill.Rating(mu=self.mu, sigma=self.sigma)

    @classmethod
    def from_trueskill(cls, rating, games_played=0):
        return cls(mu=rating.mu, sigma=rating.sigma, games_played=games_played)
```

### 3.8.4 TrueSkill 更新原理

使用 `trueskill.rate_1vs1()` 进行贝叶斯评分更新：

- 每次对战后，根据结果（win/loss/draw）更新双方的 `(mu, sigma)`
- `mu` 代表估计技能水平
- `sigma` 代表估计不确定性，随对战次数增加而降低
- 对手池管理中用 `sigma < 5.0` 作为 "confident" 的判断阈值

### 3.8.5 胜率计算

`get_win_rate()` 使用贝叶斯平滑（Beta 分布 + confidence weight）：

```python
def get_win_rate(self, home, away):
    record = self._data[key]
    alpha = 1 + wins + draws * 0.5   # Beta 先验: α=1
    beta = 1 + losses + draws * 0.5   # Beta 先验: β=1
    win_rate = alpha / (alpha + beta)

    # confidence_weight: 场次越少，越接近 0.5（无信息先验）
    confidence_weight = min(1.0, games / self._min_win_rate_games)
    win_rate = 0.5 + confidence_weight * (win_rate - 0.5)
    return win_rate
```

### 3.8.6 可利用性计算

`get_exploitability()` 衡量当前 Agent 相对于对手池的强弱：

```python
def get_exploitability(self):
    current = self._trueskill_ratings.get("current_agent")
    avg_mu = mean of opponent ratings
    avg_sigma = mean of opponent sigmas

    if current.mu <= avg_mu:
        advantage = (avg_mu - current.mu) / (current.sigma + avg_sigma)
        return min(1.0, 0.5 + advantage * 0.5)  # 越弱 → 越接近 1.0
    else:
        advantage = (current.mu - avg_mu) / (current.sigma + avg_sigma)
        return max(0.0, 0.5 - advantage * 0.5)  # 越强 → 越接近 0.0
```

**可利用性解读**：
- `= 1.0`：当前 Agent 远弱于对手池
- `= 0.5`：当前 Agent 与对手池平均水平相当
- `= 0.0`：当前 Agent 远强于对手池

### 3.8.7 历史衰减

每 `decay_interval=1000` 个 episode，对历史战绩进行衰减：

```python
def decay_all(self):
    for key in self._data:
        self._data[key] = self._data[key] * self._decay  # decay=0.99
```

`BattleRecordDict.__mul__` 将 `wins`、`draws`、`losses`、`games` 四个字段同时乘以 `decay=0.99`，使旧战绩的权重随时间指数衰减，让评分系统更关注近期表现。

### 3.8.8 Payoff 持久化

`BattleSharedPayoff` 同样支持保存/加载：

```python
def save_state(self, filepath):
    state = {
        'players': self._players,
        'players_ids': self._players_ids,
        'trueskill_ratings': {
            pid: {'mu': r.mu, 'sigma': r.sigma, 'games_played': r.games_played}
            for pid, r in self._trueskill_ratings.items()
        },
        'battle_records': dict(self._data),
        'decay': self._decay,
        'min_win_rate_games': self._min_win_rate_games,
    }
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=2)
```

---

## 3.9 SelfPlayManager 整合

