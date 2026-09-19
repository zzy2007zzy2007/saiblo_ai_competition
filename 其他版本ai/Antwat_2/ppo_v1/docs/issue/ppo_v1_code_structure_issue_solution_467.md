# 问题 4、6、7 技术解决方案

## 目录

1. [问题 4：`_save_training_status()` 重复](#1-问题-4_save_training_status-重复)
2. [问题 6：`_get_opponent_move()` hasattr 链](#2-问题-6_get_opponent_move-hasattr-链)
3. [问题 7：`train()` 中 context 110 行搬运](#3-问题-7train-中-context-110-行搬运)
4. [集成验证](#4-集成验证)

---

## 1. 问题 4：`_save_training_status()` 重复

### 1.1 现状

两个类中各有几乎相同的方法：

- [ppo_trainer.py:L279-L368](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L279-L368)
- [selfplay.py:L326-L419](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L326-L419)

两者 90% 的逻辑相同，差异仅在于：

| 差异点 | ppo_trainer.py | selfplay.py |
|---|---|---|
| empty check | `if not self.logger or not self.time_tracker: return` | `if self.trainer.time_tracker and hasattr(...)` 链 |
| 属性访问 | 直接访问 `self.xxx` | 通过 `self.trainer.xxx` 间接访问 |
| battle_data | `self.battle_manager.get_battle_stats()` | 传进来的参数，始终为空 `{}` |
| 异常处理 | `logger.warning(f"⚠ ...")` | `logger.error(f"⚠ ...", exc_info=True)` + `log_exception` |
| 日志记录 | 无 start/end log | `logger.info` 打印开始/完成 |

### 1.2 方案：提取公共方法到 PPOTrainer

**策略**：将唯一实现放在 PPOTrainer 中，SelfPlayTrainer 转发调用。

#### 步骤 1：将 PPOTrainer 的 `_save_training_status` 升级为公共方法

将方法从 `_save_training_status` 重命名为 `save_training_status`（去掉下划线前缀），并使其接受可选的 `battle_data` 参数（由调用方传入）：

在 [ppo_trainer.py:L279-L368](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L279-L368) 位置修改：

```python
def save_training_status(self, episode: int,
                         battle_result: Optional[Tuple[int, int, int]] = None,
                         battle_data: Optional[Dict] = None) -> None:
    """保存训练状态和时间统计（公共方法）

    Args:
        episode: 当前训练轮次
        battle_result: 对战结果 (wins, losses, draws)，可选
        battle_data: 对战统计数据，可选。由调用方（SelfPlayTrainer）传入。
    """
    if not self.logger or not self.time_tracker:
        return

    try:
        # ---- 训练速度 ----
        if self.time_tracker.training_episodes_count > 0:
            training_speed = (
                self.time_tracker.training_episodes_count
                / (self.time_tracker.training_time_total / 60
                   if self.time_tracker.training_time_total > 0 else 0)
            )
        else:
            training_speed = 0

        # ---- 统计指标 ----
        history = self.training_metrics_cache.get('history', [])
        recent_100 = history[-HISTORICAL_METRICS_WINDOW:] if len(history) >= HISTORICAL_METRICS_WINDOW else history
        recent_10 = history[-RECENT_METRICS_WINDOW:] if len(history) >= RECENT_METRICS_WINDOW else history

        avg_reward_last_100 = np.mean([m['avg_reward'] for m in recent_100]) if recent_100 else 0
        avg_reward_last_10 = np.mean([m['avg_reward'] for m in recent_10]) if recent_10 else 0

        loss_values = [
            m['avg_loss'] for m in recent_100
            if m.get('avg_loss') is not None and not np.isnan(m['avg_loss'])
        ]
        avg_loss_last_100 = np.mean(loss_values) if loss_values else 0

        avg_rounds = np.mean(self.episode_lengths[-HISTORICAL_METRICS_WINDOW:]) if self.episode_lengths else 0

        # ---- 对战结果 ----
        win_rate = 0
        total_wins = 0
        total_losses = 0
        total_draws = 0
        if battle_result:
            wins, losses, draws = battle_result
            total = wins + losses + draws
            win_rate = wins / total if total > 0 else 0
            total_wins = wins
            total_losses = losses
            total_draws = draws

        # ---- 稳定性 ----
        reward_std = np.std([m['avg_reward'] for m in recent_100]) if len(recent_100) > 1 else 0
        loss_std = np.std(loss_values) if len(loss_values) > 1 else 0
        training_stability = {'reward_std': float(reward_std), 'loss_std': float(loss_std)}

        # ---- 对战数据 ----
        bd = battle_data if battle_data is not None else {}
        if not bd and self.battle_manager and hasattr(self.battle_manager, 'get_battle_stats'):
            bd = self.battle_manager.get_battle_stats()

        # ---- 写入 ----
        self.logger.save_training_status(
            episode=episode,
            total_steps=self.total_steps,
            best_avg_reward=self.best_avg_reward,
            avg_reward_last_100=avg_reward_last_100,
            avg_reward_last_10=avg_reward_last_10,
            avg_loss_last_100=avg_loss_last_100,
            avg_rounds=avg_rounds,
            total_wins=total_wins,
            total_losses=total_losses,
            total_draws=total_draws,
            win_rate=win_rate,
            training_speed=training_speed,
            ppo_metrics={},
            training_stability=training_stability,
            battle_data=bd,
            recent_battle_data=bd,
            start_time=self.training_start_time,
            current_time=self.time_tracker.last_training_end_time,
        )

        if bd:
            self.logger.save_battle_stats(bd)

        self.time_tracker.save_time_statistics()

    except Exception as e:
        logger.warning(f"⚠ 保存训练状态异常: {e}")
```

**变更要点**：

- 方法名去下划线前缀（`_save_training_status` → `save_training_status`）
- 新增参数 `battle_data: Optional[Dict] = None`
- 对战数据逻辑：优先用传入的 `battle_data`，fallback 到 `self.battle_manager.get_battle_stats()`
- 保留简洁的异常处理（`logger.warning`），不引入额外的 error log（调用方有需要可自己加）

#### 步骤 2：删除 SelfPlayTrainer 中的 `_save_training_status`

删除 [selfplay.py:L326-L419](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L326-L419) 整个方法。

#### 步骤 3：修改 SelfPlayTrainer 的调用点

在 `train()` 方法中（当前 L1061），将：

```python
self._save_training_status(episode + batch_size, battle_result)
```

改为：

```python
try:
    self.trainer.save_training_status(episode + batch_size, battle_result)
except Exception as e:
    logger.error(f"⚠ 保存训练状态异常: {e}", exc_info=True)
    if self.trainer.logger:
        self.trainer.logger.log_exception(e, {
            'event': 'save_training_status_error',
            'episode': episode + batch_size,
        })
```

如果需要在 SelfPlay 层保留额外的日志（开始/完成 log），可以在调用前后加入：

```python
logger.info(f"[save_training_status] 开始: episode={episode + batch_size}, "
            f"battle_result={battle_result}")
self.trainer.save_training_status(episode + batch_size, battle_result)
logger.info(f"[save_training_status] 完成")
```

### 1.3 改动范围

| 文件 | 改动 | 行数变化 |
|---|---|---|
| `ppo_trainer.py` | 方法重命名 + 新增 `battle_data` 参数 | ~3 行 |
| `selfplay.py` | 删除 `_save_training_status`，修改调用点 | -94 +5 |

---

## 2. 问题 6：`_get_opponent_move()` hasattr 链

### 2.1 现状

**位置**：[selfplay.py:L716-L749](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L716-L749)

当前通过 `hasattr` 判断 4 种对手类型，缺少统一接口：

```python
def _get_opponent_move(self, opponent_agent, env, obs, swap_positions: bool):
    if opponent_agent is None:
        opponent_action_id = self._random_action_from_mask(obs, swap_positions)
    elif hasattr(opponent_agent, 'act'):
        opponent_action_id, _, _ = opponent_agent.act(opponent_obs)
    elif hasattr(opponent_agent, 'choose_operations'):
        ops = opponent_agent.choose_operations(state, opponent_player)
        opponent_ops = ops if ops else []
    elif hasattr(opponent_agent, 'choose_bundle'):
        result = opponent_agent.choose_bundle(state, opponent_player)
        opponent_ops = list(result.operations) if hasattr(result, 'operations') else (result if result else [])
    else:
        opponent_action_id = self._random_action_from_mask(obs, swap_positions)

    return opponent_action_id, opponent_ops
```

调用点：[selfplay.py:L557](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L557)

```python
opponent_action_id, opponent_ops = self._get_opponent_move(
    opponent_agent, env, obs, swap_positions
)
```

### 2.2 方案：定义 Agent Protocol + 适配器模式

**策略**：不改变现有 Agent 类（PPOAgent / 基线 AI），而是引入一层薄适配，将各类 Agent 统一到一个协商接口。

#### 步骤 1：定义 Agent Protocol

新建文件 `ppo/src/ppo_antwar/trainer/agent_protocol.py`：

```python
"""对手 Agent 的统一接口定义"""

from typing import Protocol, Tuple, Optional, Any

from gymnasium.spaces import Space


class OpponentAgent(Protocol):
    """对手 Agent 的协商接口。

    所有可用于 SelfPlay 对战的 Agent 都应符合此协议。
    """

    def get_action(self, observation: dict, state: Any,
                   player: int, action_space: Space,
                   rng: Any) -> Tuple[Optional[int], Optional[list]]:
        """获取对手在当前局面的动作。

        Args:
            observation: 当前观测（action_id agent 使用）
            state: 游戏运行时状态（operations agent 使用）
            player: 对手的玩家编号（0 或 1）
            action_space: 动作空间（用于 mask 采样）
            rng: 随机数生成器（np.random.RandomState 或 np.random）

        Returns:
            (action_id, operations): action_id 和 operations 至少有一个非 None。
        """
        ...
```

#### 步骤 2：为各 Agent 类型创建适配器

在同一个文件中，或放在适合的模块中：

```python
import numpy as np
import torch


class RandomAgent:
    """无对手时使用的随机 Agent。"""

    def get_action(self, observation: dict, state: Any,
                   player: int, action_space, rng) -> Tuple[Optional[int], Optional[list]]:
        opponent_obs = observation["player_1"] if player == 1 else observation["player_0"]
        mask = torch.FloatTensor(opponent_obs["action_mask"])
        valid_actions = torch.where(mask > 0)[0]
        if len(valid_actions) > 0:
            action_id = int(rng.choice(valid_actions.cpu().numpy()))
        else:
            action_id = 0
        return action_id, None


class PPOAgentAdapter:
    """包装 PPOAgent（有 act 方法）。"""

    def __init__(self, agent):
        self._agent = agent

    def get_action(self, observation: dict, state: Any,
                   player: int, action_space, rng) -> Tuple[Optional[int], Optional[list]]:
        opponent_obs = observation["player_1"] if player != 0 else observation["player_0"]
        action_id, _, _ = self._agent.act(opponent_obs)
        return action_id, None


class OperationsAgentAdapter:
    """包装 choose_operations 风格的 Agent（如基线 AI）。"""

    def __init__(self, agent):
        self._agent = agent

    def get_action(self, observation: dict, state: Any,
                   player: int, action_space, rng) -> Tuple[Optional[int], Optional[list]]:
        ops = self._agent.choose_operations(state, player)
        return None, ops if ops else []


class BundleAgentAdapter:
    """包装 choose_bundle 风格的 Agent。"""

    def __init__(self, agent):
        self._agent = agent

    def get_action(self, observation: dict, state: Any,
                   player: int, action_space, rng) -> Tuple[Optional[int], Optional[list]]:
        result = self._agent.choose_bundle(state, player)
        if hasattr(result, 'operations'):
            return None, list(result.operations)
        return None, result if result else []
```

#### 步骤 3：在数据收集前统一包装对手

在 `train()` 中（L787-L793 附近），将对手加载后立即包装：

```python
# 原代码
if opponent_id is not None:
    opponent_agent = self._load_opponent_agent(opponent_id)
else:
    opponent_agent = None

# 改为
if opponent_id is not None:
    raw_agent = self._load_opponent_agent(opponent_id)
    opponent_agent = self._wrap_agent(raw_agent)
else:
    opponent_agent = RandomAgent()
```

新增 `_wrap_agent` 方法：

```python
def _wrap_agent(self, agent) -> OpponentAgent:
    """将各类 Agent 包装为统一的 OpponentAgent 接口。"""
    if isinstance(agent, OpponentAgent):
        return agent
    if hasattr(agent, 'act'):
        return PPOAgentAdapter(agent)
    if hasattr(agent, 'choose_operations'):
        return OperationsAgentAdapter(agent)
    if hasattr(agent, 'choose_bundle'):
        return BundleAgentAdapter(agent)
    return RandomAgent()
```

> **注意**：`_wrap_agent` 中的 hasattr 判断是**一次性**的（在加载对手时调用），而原始问题中的 hasattr 链是**每步都调用**（在 while 循环内）。一次性判断是可接受的。

#### 步骤 4：简化 `_get_opponent_move` 为一行调用

```python
def _get_opponent_move(self, opponent_agent: OpponentAgent, env, obs,
                       swap_positions: bool):
    player = 1 if not swap_positions else 0
    state = env._runtime.state if hasattr(env, '_runtime') and env._runtime else None
    return opponent_agent.get_action(
        observation=obs,
        state=state,
        player=player,
        action_space=None,
        rng=np.random,
    )
```

### 2.3 改动范围

| 文件 | 改动 | 行数 |
|---|---|---|
| `agent_protocol.py`（新） | Protocol + 4 个适配器类 | ~80 行 |
| `selfplay.py` | 新增 `_wrap_agent`（~15 行），简化 `_get_opponent_move`（40→10），修改调用点 | -30 +15 |
| 对手加载处（1 处） | `opponent_agent` 赋值为包装后的对象 | ~2 行 |

---

## 3. 问题 7：`train()` 中 context 110 行搬运

### 3.1 现状

**位置**：[selfplay.py:L871-L998](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L871-L998)

这段代码在 `train()` 方法内部做了以下事情：

| 子任务 | 行数 | 说明 |
|---|---|---|
| 金币均值计算 | L854-L869 | 从 details_pool 遍历取 our/enemy coins |
| 金币分桶 | L871-L888 | 内联函数 `compute_coin_buckets` |
| PPO 指标映射 | L890-L925 | `ppo_metrics.get('xxx', 0)` × 35 行 |
| reward 来源分解 | L935-L951 | 从 details_pool 汇总 rw_* 字段 |
| action reward/count | L953-L969 | 按动作类型汇总 actrw_*/actcnt_* |
| opponent 维度指标 | L973-L998 | 按 opponent_id 汇总 win rate / avg reward |

### 3.2 方案：提取为 `MetricsCollector` 类

**策略**：将所有指标收集逻辑提取到独立的 `MetricsCollector` 类中，该类接收 `ppo_metrics` 和 `details_pool`，输出 `context` 字典。

#### 步骤 1：创建 `MetricsCollector`

新建文件 `ppo/src/ppo_antwar/trainer/metrics_collector.py`：

```python
"""训练指标收集器 —— 将 ppo_metrics + details_pool 汇总为 context 字典"""

from typing import Dict, List, Any, Optional
import numpy as np


class MetricsCollector:
    """从 PPO 更新结果和战局详情中收集训练指标。

    用法:
        collector = MetricsCollector()
        context = collector.collect(ppo_metrics, details_pool)
    """

    # ---- PPO 指标名列表（从 ppo_metrics 直接透传的键） ----
    _PPO_METRIC_KEYS = [
        'policy_loss', 'value_loss', 'entropy',
        'reward_mean', 'reward_std', 'reward_min', 'reward_max',
        'advantages_mean', 'advantages_std',
        'return_mean', 'return_std',
        'clip_fraction', 'gradient_norm', 'hard_entropy_penalty',
        'value_input_mean', 'value_input_std',
        'ratio_mean', 'ratio_std',
        'grad_norm_policy', 'grad_norm_value', 'grad_norm_max_layer',
        'nan_skip_count',
        'valid_actions_mean', 'valid_actions_min',
        'td_error_mean', 'td_error_std', 'batch_size',
    ]

    @staticmethod
    def _compute_coin_buckets(coin_list: List[float]) -> Dict[str, float]:
        """将金币列表分桶（按百分比返回）。"""
        buckets = {'lt_30': 0, 'ge_30_lt_60': 0, 'ge_60_lt_100': 0, 'ge_100': 0}
        for c in coin_list:
            if c < 30:
                buckets['lt_30'] += 1
            elif c < 60:
                buckets['ge_30_lt_60'] += 1
            elif c < 100:
                buckets['ge_60_lt_100'] += 1
            else:
                buckets['ge_100'] += 1
        total = len(coin_list)
        if total == 0:
            return {f"coins_{k}": 0.0 for k in buckets}
        return {f"coins_{k}": v / total * 100.0 for k, v in buckets.items()}

    @staticmethod
    def _extract_ppo_metrics(ppo_metrics: Optional[Dict]) -> Dict[str, float]:
        """从 ppo_metrics 中提取标准指标。"""
        ctx = {}
        if ppo_metrics is None:
            return ctx
        for key in MetricsCollector._PPO_METRIC_KEYS:
            ctx[key] = float(ppo_metrics.get(key, 0))
        # 动态键：type_*, logit_*, prob_*
        for k, v in ppo_metrics.items():
            if k.startswith(('type_', 'logit_', 'prob_')) and isinstance(v, (int, float)):
                ctx[k] = float(v)
        return ctx

    @staticmethod
    def _extract_coin_metrics(details_pool: List[Dict]) -> Dict[str, float]:
        """从战局详情中提取金币相关指标。"""
        all_our_coins = []
        all_enemy_coins = []
        all_our_cumulative = []
        all_enemy_cumulative = []
        for d in details_pool:
            rds = d.get('round_details', [])
            for r in rds:
                all_our_coins.append(r.get('our_coins', 0))
                all_enemy_coins.append(r.get('enemy_coins', 0))
            all_our_cumulative.append(d.get('our_cumulative_coins', 50))
            all_enemy_cumulative.append(d.get('enemy_cumulative_coins', 50))

        n = len(all_our_coins)
        nc = len(all_our_cumulative)
        ctx = {
            'avg_our_coins': sum(all_our_coins) / n if n else 0,
            'avg_enemy_coins': sum(all_enemy_coins) / n if n else 0,
            'avg_our_cumulative_coins': sum(all_our_cumulative) / nc if nc else 50,
            'avg_enemy_cumulative_coins': sum(all_enemy_cumulative) / nc if nc else 50,
        }
        ctx.update(MetricsCollector._compute_coin_buckets(all_our_coins))
        ctx.update(MetricsCollector._compute_coin_buckets(all_enemy_coins))
        return ctx

    @staticmethod
    def _extract_reward_sources(details_pool: List[Dict]) -> Dict[str, float]:
        """从战局详情中汇总奖励来源。"""
        sources = {
            'rw_hp_attack': 0.0, 'rw_coin_gain': 0.0,
            'rw_tower_survival': 0.0, 'rw_tech_bonus': 0.0,
            'rw_die_penalty': 0.0, 'rw_end_reward': 0.0,
            'rw_speed_bonus': 0.0,
        }
        for d in details_pool:
            for rd_key in ('round_details', 'actions'):
                actions_list = d.get(rd_key, [])
                if not isinstance(actions_list, list):
                    continue
                for step in actions_list:
                    rd = step.get('reward_detail', {})
                    if not isinstance(rd, dict):
                        continue
                    for src_key in sources:
                        bare_key = src_key[3:] if src_key.startswith('rw_') else src_key
                        sources[src_key] += rd.get(bare_key, 0.0)
        return sources

    @staticmethod
    def _extract_action_rewards(details_pool: List[Dict],
                                action_space_config: Dict) -> Dict[str, float]:
        """从战局详情中按动作类型汇总 reward 和 count。"""
        action_rewards: Dict[str, float] = {}
        action_counts: Dict[str, int] = {}
        result: Dict[str, float] = {}

        for d in details_pool:
            for action_rec in d.get('actions', []):
                action_id = action_rec.get('action', 0)
                rd = action_rec.get('reward_detail', {})
                action_reward_val = rd.get('action_reward', 0.0) if isinstance(rd, dict) else 0.0
                type_name = 'noop'
                for tname, tcfg in action_space_config.items():
                    if tcfg['start'] <= action_id < tcfg['end']:
                        type_name = tname
                        break
                action_rewards[type_name] = action_rewards.get(type_name, 0.0) + action_reward_val
                action_counts[type_name] = action_counts.get(type_name, 0) + 1

        for type_name in action_rewards:
            result[f'actrw_{type_name}'] = action_rewards[type_name]
            result[f'actcnt_{type_name}'] = float(action_counts[type_name])
        return result

    @staticmethod
    def _extract_opponent_metrics(per_episode_buffer: List[Dict]) -> Dict[str, float]:
        """从逐局统计中汇总对手维度指标。"""
        opponent_stats = {}
        for s in per_episode_buffer:
            oid = s['opponent_id']
            if oid not in opponent_stats:
                opponent_stats[oid] = {'rewards': [], 'wins': 0, 'losses': 0, 'draws': 0}
            opponent_stats[oid]['rewards'].append(s['reward'])
            if s['result'] == 'win':
                opponent_stats[oid]['wins'] += 1
            elif s['result'] == 'loss':
                opponent_stats[oid]['losses'] += 1
            elif s['result'] == 'draw':
                opponent_stats[oid]['draws'] += 1

        ctx = {}
        for oid, stats in opponent_stats.items():
            total_battles = stats['wins'] + stats['losses'] + stats['draws']
            win_rate = stats['wins'] / total_battles if total_battles > 0 else 0
            avg_reward = np.mean(stats['rewards']) if stats['rewards'] else 0
            ctx[f'opponent_win_rate_{oid}'] = float(win_rate)
            ctx[f'opponent_reward_{oid}'] = float(avg_reward)
        return ctx

    def collect(self, ppo_metrics: Optional[Dict], details_pool: List[Dict],
                per_episode_buffer: List[Dict],
                action_space_config: Optional[Dict] = None) -> Dict[str, Any]:
        """汇总所有指标到一个 context 字典。

        Args:
            ppo_metrics: _ppo_update() 返回的指标字典
            details_pool: 本批战局详情列表
            per_episode_buffer: 逐局统计列表

        Returns:
            扁平化的 context 字典，可直接传给 Logger.save_training_metrics()
        """
        context = {}

        context.update(self._extract_ppo_metrics(ppo_metrics))
        context.update(self._extract_coin_metrics(details_pool))
        context.update(self._extract_reward_sources(details_pool))

        if action_space_config:
            context.update(self._extract_action_rewards(details_pool, action_space_config))

        context.update(self._extract_opponent_metrics(per_episode_buffer))

        return context
```

#### 步骤 2：在 `train()` 中替换原有的 110 行

原来的代码块（L845-L998）替换为：

```python
# ---- 指标收集 ----
context = self.metrics_collector.collect(
    ppo_metrics=ppo_metrics,
    details_pool=details_pool,
    per_episode_buffer=per_episode_buffer,
    action_space_config=ACTION_SPACE_CONFIG,
)

context['entropy_coef'] = self.trainer.current_ent_coef
context['learning_rate'] = self.trainer.current_lr
context['num_collected'] = len(batch_pool)
```

在 `SelfPlayTrainer.__init__` 中初始化：

```python
self.metrics_collector = MetricsCollector()
```

### 3.3 改动范围

| 文件 | 改动 | 行数变化 |
|---|---|---|
| `metrics_collector.py`（新） | MetricsCollector 类完整实现 | ~150 行 |
| `selfplay.py` | 减少 context 构建代码，新增 1 行初始化 + 5 行调用 | -110 +6 |

---

## 4. 集成验证

### 4.1 三个方案的相互影响

三个方案**独立**，可在不同时间点分别实施：

```
问题 7（context）         问题 4（status 重复）      问题 6（hasattr 链）
    │                          │                         │
    ▼                          ▼                         ▼
新增 metrics_collector.py    修改 ppo_trainer.py        新增 agent_protocol.py
修改 selfplay.py: 替换       删除 selfplay.py: 方法      修改 selfplay.py: 简化
  110 行 context 构建          修改调用点（3 行）          _get_opponent_move
```

**无相互影响**：没有文件在同一位置被多个方案修改。

### 4.2 验证步骤

实施每个方案后，依次验证：

1. **问题 7**：运行 `_collect_batch_parallel` 或直接在现有训练中验证 context 字典的键集合与改造前一致
2. **问题 4**：检查 `training_status.json` 输出内容与改造前一致
3. **问题 6**：运行一局对战（用不同类型对手），验证动作获取结果一致

### 4.3 回归风险

| 方案 | 风险 | 缓解措施 |
|---|---|---|
| 问题 7 | context 键遗漏 | `_PPO_METRIC_KEYS` 列表显式枚举所有键 |
| 问题 4 | battle_data 为 None 时行为改变 | fallback 逻辑保留 `self.battle_manager.get_battle_stats()` |
| 问题 6 | 适配器引入间接调用开销 | `get_action` 调用在 while 循环内，但适配器仅做转发，零额外分配 |

### 4.4 实施顺序建议

1. **先做问题 7**（低风险，`train()` 立即缩短 110 行）
2. **再做问题 4**（消除重复代码，无行为变化）
3. **最后做问题 6**（适配器引入新文件，需确认 Agent 类型的完整性）
