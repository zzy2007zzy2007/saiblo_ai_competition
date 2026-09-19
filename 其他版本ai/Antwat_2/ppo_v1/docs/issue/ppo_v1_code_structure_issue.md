# SelfPlay 训练代码组织问题审查报告

## 概述

本次审查覆盖以下核心文件：

| 文件 | 行数 |
|---|---|
| `selfplay.py` | 1279 |
| `ppo_trainer.py` | 1252 |
| `opponent_selector.py` | 315 |
| `payoff.py` | 268 |
| `selfplay_logger.py` | 175 |
| `callback_factory.py` | 103 |

共发现 **23 个问题**，其中 **Critical 3 个**、**High 8 个**、**Medium 8 个**、**Low 5 个**。

---

## 一、Critical（3 个）

### 🔴 1. SelfPlayTrainer — 典型 God Class

**位置**：[selfplay.py:L139-L1279](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L139-L1279)（1279 行）

单个类承担至少 **10 种职责**：

| 职责 | 相关代码 |
|---|---|
| 训练循环驱动 | `train()` (L759-L1062) |
| 数据采集（环境交互 + 详情记录） | `_collect_episode_with_swap()` (L421-L680) |
| 对手池管理（创建、淘汰、持久化） | `_create_initial_opponents()` (L1096) |
| League 状态持久化 | `_save_league_state()` / `_load_league_state()` |
| Baseline 对战评估 | `train()` 内联 (L1026-L1056) |
| 训练状态快照（与 PPOTrainer 重复） | `_save_training_status()` (L326-L419) |
| SelfPlay 对战详情日志 | `_save_selfplay_battle_details()` / `_flush_selfplay_battle_cache()` |
| 每局统计输出 | `_flush_per_episode_stats()` (L1083) |
| 权重快照 | `_save_weight_stats()` (L1064) |
| 模型评估 | `evaluate()` / `_play_evaluation_game()` |

**问题影响**：任何单一职责的变更都需要触碰这个巨型类；测试其中一个行为需要初始化整个对象链；违反单一职责原则（SRP）。

---

### 🔴 2. `train()` — 304 行的 Monster Method

**位置**：[selfplay.py:L759-L1062](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L759-L1062)

该方法在一个巨型循环体内混合了以下所有逻辑（嵌套最深达 5 层）：

```
for episode in range(num_episodes):     # L777
  ├── 时间追踪 (L782-783)
  ├── 对手选择 (L788-791)
  ├── 数据采集 (L792-793)
  ├── 对战详情保存 (L798-801)
  ├── Payoff 更新 (L803-806)
  ├── 缓存刷新 (L808)
  ├── 每局指标收集 (L813-823)
  ├── PPO 更新循环 (L830-1013)
  │   ├── batch merge (L832)
  │   ├── _ppo_update() (L834)
  │   ├── 学习率/熵退火 (L837-842)
  │   ├── 大量指标提取 (L845-850)
  │   ├── 金币分布计算 (L854-889)
  │   ├── context 构建 (L890-998) — 约 110 行纯数据搬运
  │   └── 日志写入 (L1001-1007)
  ├── 对手定期添加 (L1015-1024)
  ├── Baseline 对战评估 (L1027-1056)
  ├── 衰减检查 (L1058)
  └── 训练状态保存 (L1061-1062)
```

**问题影响**：300+ 行的方法几乎不可能理解和修改；任何一处的变更都影响全局；context 构建部分（L890-L998）有超过 100 行纯数据搬运代码。

---

### 🔴 3. `_collect_episode_with_swap()` — 260 行的 Monster Method

**位置**：[selfplay.py:L421-L680](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L421-L680)

在单个方法中同时完成：环境步进循环、回合详情记录（HP/金币/塔/蚂蚁种类/科技等级）、动作选择、对手动作获取、奖励分解、胜负判定、时间追踪。

关键段落在 [L484-L621](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L484-L621)（约 137 行的 while 循环），其中嵌套的 try/except 块有 **3 层**：while → try → try/if-hasattr。

**问题影响**：260 行的方法无法单独测试数据采集逻辑或日志逻辑；底层游戏状态访问被内联到方法中，导致环境耦合极深；战斗详情收集与训练数据收集互相纠缠。

---

## 二、High（8 个）

### 🟠 4. `_save_training_status()` 在两个类间大量重复

**位置**：[selfplay.py:L326-L419](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L326-L419) 和 [ppo_trainer.py:L279-L368](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L279-L368)

约 **90% 逻辑完全相同**，包括：

- 相同的 training_speed 计算逻辑 (L290-297 vs L337-345)
- 相同的 recent_100/recent_10 窗口统计 (L300-302 vs L349-351)
- 相同的 avg_reward_last_100/last_10 计算 (L304-305 vs L353-354)
- 相同的 loss_values 计算 (L307-311 vs L356-360)
- 相同的 avg_rounds 计算 (L313 vs L362)
- 相同的 battle_result 处理 (L320-326 vs L369-375)
- 相同的 reward_std/loss_std 计算 (L329-331 vs L378-379)
- 相同的 save_training_status 调用 (L339-358 vs L385-403)

唯一区别是 selfplay.py 版本通过 `self.trainer.xxx` 间接访问，并增加了 exc_info 和异常日志。

**问题影响**：违反 DRY 原则；两个方法需要同步维护，任何一个修改都可能遗漏另一个。

---

### 🟠 5. SelfPlayTrainer 对 PPOTrainer 内部实现的深层耦合

**位置**：遍布 [selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py)

直接大量访问 `self.trainer` 的内部属性和方法：

```python
# L834: 调用私有方法
self.trainer._ppo_update(combined_batch)

# L837-L842: 直接修改优化器状态
self.trainer.current_ent_coef = self.trainer.get_entropy_coef(episode + 1)
self.trainer.current_lr = self.trainer.get_learning_rate(episode + 1)
for param_group in self.trainer.optimizer.param_groups:
    param_group['lr'] = self.trainer.current_lr
for param_group in self.trainer.value_optimizer.param_groups:
    param_group['lr'] = ...

# L848-L851: 直接修改内部列表
self.trainer.episode_rewards.append(episode_reward)
self.trainer.episode_entropies.append(...)
self.trainer.episode_lengths.append(...)
self.trainer.total_steps += episode_length_total

# L334: 访问 time_tracker
self.trainer.time_tracker and hasattr(self.trainer.time_tracker, ...)

# L632-633: 访问 policy class
self.trainer.policy.__class__(...)
```

**问题影响**：违反迪米特法则（Law of Demeter）；任何对 PPOTrainer 内部的重构都会破坏 SelfPlayTrainer。

---

### 🟠 6. `_get_opponent_move()` 中使用 hasattr/if-elif 链做多态分发

**位置**：[selfplay.py:L713-L749](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L713-L749)

```python
if opponent_agent is None:
    opponent_action_id = self._random_action_from_mask(...)
elif hasattr(opponent_agent, 'act'):
    opponent_action_id, _, _ = opponent_agent.act(opponent_obs)
elif hasattr(opponent_agent, 'choose_operations'):
    ops = opponent_agent.choose_operations(state, opponent_player)
elif hasattr(opponent_agent, 'choose_bundle'):
    result = opponent_agent.choose_bundle(state, opponent_player)
else:
    opponent_action_id = self._random_action_from_mask(...)
```

有 **4 种不同接口的 Agent 类型**（action-id agent / operation agent / bundle agent / None-random），但没有统一接口抽象。

**问题影响**：每增加一种 Agent 类型都需要修改这个 if-elif 链；违反开闭原则（OCP）；类型不安全。

---

### 🟠 7. `train()` 中 context 构建是 110 行纯数据搬运

**位置**：[selfplay.py:L890-L998](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L890-L998)

```python
context = {
    'policy_loss': ppo_metrics.get('policy_loss', 0) if ppo_metrics else 0,
    'value_loss': ppo_metrics.get('value_loss', 0) if ppo_metrics else 0,
    'entropy': ppo_metrics.get('entropy', 0) if ppo_metrics else 0,
    # ... 40+ 钥匙值搬运
}
```

还包含内联定义的 `compute_coin_buckets()` 函数（L871-L885）和奖励来源聚合逻辑（L935-964）。

**问题影响**：没有任何业务逻辑，纯粹是格式转换；应提取到专门的 MetricsFormatter/Collector 类中；每次 `_ppo_update()` 返回结构变化，都需要同步修改此处。

---

### 🟠 8. PPOTrainer 也是一个 God Class

**位置**：[ppo_trainer.py:L82-L1037](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L82-L1037)（955 行）

承担了以下职责：

| 职责类型 | 具体职责 |
|---|---|
| 基础设施 | Logger / TimeTracker / SystemMetricsSampler / Battle 系统 / 回调系统初始化 |
| 算法逻辑 | PPO update / GAE / 双网络策略+价值 |
| 训练调度 | lr warmup + cosine decay / entropy annealing |
| 网络管理 | 网络创建 / 设备管理 / 双优化器管理 |
| 鲁棒性 | NaN 处理 / 配置验证 |
| 动作选择 | _select_action / _select_action_with_exploration / PPOAgent |
| 检查点 | save_checkpoint / load_checkpoint |
| 日志 | training metrics / training status |

**问题影响**：混合了"基础设施"和"算法逻辑"以及"训练调度"；如果继续膨胀将不可维护。

---

### 🟠 9. `_ppo_update()` — 298 行的 Monster Method

**位置**：[ppo_trainer.py:L665-L963](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L665-L963)

该方法包含：奖励统计 (L674-L684) → GAE 计算 (L687-L698) → Returns 归一化 (L704-L707) → 双层循环 (ppo_epochs × mini_batches, L734-L913) → 策略/价值/熵损失计算 → 梯度裁剪 → NaN 检测与处理 (L845-L852) → 指标聚合 (约 50 行, L916-L962) → 自动 checkpoint 保存 (L912-L914)。

**问题影响**：几乎无法独立测试其中任何一个子部分；修改 PPO 算法、NaN 处理、指标收集中的任何一个都会影响其他部分。

---

### 🟠 10. 私有方法越界访问

**位置**：[selfplay.py:L834](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L834)

```python
ppo_metrics = self.trainer._ppo_update(combined_batch)
```

`_ppo_update` 以下划线开头标识为私有方法，却被 SelfPlayTrainer 直接调用。

**问题影响**：打破封装；PPOTrainer 应提供公共的 `update()` 或 `train_step()` 方法。

---

### 🟠 11. 架构分层不足

当前 4 个核心文件约 3100 行代码，没有清晰的接口边界。`PPOTrainer` 的 docstring 声称自己是"纯算子层，不持有训练循环"，但实际上持有大量训练状态（optimizer state、episode_count、total_steps、metrics caches、entropy coef、lr 等），并被 SelfPlayTrainer 当做内部实现来使用。

**问题影响**：`PPOTrainer` 和 `SelfPlayTrainer` 之间的边界模糊；合理的分层应为：

```
Orchestrator (TrainLoop)
├── AlgorithmModule (PPO Update / GAE / Networks)
├── DataCollector (Environment interaction)
├── LeagueManager (OpponentPool + Payoff + Selector)
├── Evaluator (Battle evaluation)
└── Logger (Metrics + Status + Battle details)
```

---

## 三、Medium（8 个）

### 🟡 12. `_random_action_from_mask()` 逻辑重复

**位置**：[selfplay.py:L751-L757](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L751-L757) 和 [ppo_trainer.py:L557-L563](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L557-L563)

从 action_mask 中随机选择合法动作的逻辑在两个文件中重复出现，且都有相同的 `random.choice` + fallback 模式。

---

### 🟡 13. 硬编码魔法数字分散在多处

| 位置 | 硬编码值 | 应读取自 |
|---|---|---|
| [selfplay.py:L480-L481](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L480-L481) | `our_cumulative = 50`, `enemy_cumulative = 50` | 配置文件 |
| [selfplay.py:L1084](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L1084) | `decay_interval: int = 1000` | 配置文件 |
| [opponent_selector.py:L38](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L38) | `self._total_episodes / 10000.0`（progress 分母） | total_episodes 配置 |
| [opponent_selector.py:L24](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L24) | `self._min_confidence_sigma = 5.0` | 配置文件 |

---

### 🟡 14. `_select_action_with_exploration()` 内联调试逻辑

**位置**：[ppo_trainer.py:L621-L663](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L621-L663)

探索分支中重新做了前向传播（L630-L638），而 `_select_action()` 已被调用过一次（L622），意味着探索时做了**两次推理**。调试日志（L646-L652）被硬编码在方法内部，无法运行时关闭。

---

### 🟡 15. PPOAgent 和 AntWarAgent 大量重复

**位置**：[ppo_trainer.py:L1040-L1084](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L1040-L1084) 和 [ppo_trainer.py:L1207-L1252](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L1207-L1252)

`act()` 和 `evaluate()` 方法几乎逐字相同。`PPOAgent.select_action()` (L1073-1084) 也与 `act()` (L1053-1071) 重复（仅 `deterministic` 参数不同）。

`board/global/mask → FloatTensor → unsqueeze → to(device) → net` 模式重复了至少 **6 次**（`_select_action`、`_select_action_with_exploration`、`PPOAgent.act`、`PPOAgent.select_action`、`PPOAgent.evaluate`、`AntWarAgent.act`、`AntWarAgent.evaluate`）。

---

### 🟡 16. OpponentPool 通过隐式引用访问 payoff

**位置**：[opponent_selector.py:L157-L159](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L157-L159)

```python
def set_payoff_reference(self, payoff):
    self._payoff = payoff
```

淘汰逻辑中靠 `if self._payoff is not None` 检查。引用未设置时回退到默认值（mu=25.0），可能导致不合理淘汰。

**问题影响**：时间依赖的脆弱设计；如果在设置 payoff 引用之前发生淘汰，评分计算会走默认分支。

---

### 🟡 17. Callback 系统集成不协调

**位置**：[callback_factory.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/callbacks/callback_factory.py) 和 [selfplay.py:L1009-L1010](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L1009-L1010)

回调通过 `create_ppo_callbacks()` 创建后同时传给 PPOTrainer 和 SelfPlayTrainer，但 `callbacks.on_step()` 仅在 selfplay.py 中被调用。PPOTrainer 的 `init_callback`（L186）也初始化了它们——实际未被训练循环使用。

---

### 🟡 18. `_save_training_status` 内部动态 import

**位置**：[selfplay.py:L332](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L332)

```python
from ppo_antwar.monitor.constants import ...
```

在方法内部进行 import，说明该常量不是类级别的依赖，依赖关系没有正确建模。

---

## 四、Low（5 个）

### 🟢 19. `PPOTrainer.__init__()` try/except 过宽

**位置**：[ppo_trainer.py:L91-L188](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/ppo_trainer.py#L91-L188)

整个初始化逻辑（约 100 行）被包裹在一个 `try/except Exception` 块中，无法精确定位哪一个初始化步骤失败。

---

### 🟢 20. OpponentSelector 进度计算的硬编码参数

**位置**：[opponent_selector.py:L33-L49](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/opponent_selector.py#L33-L49)

```python
progress = self._total_episodes / 10000.0  # 硬编码分母
```

sigmoid schedule 的 `-10 * (progress - 0.5)` 和 exp schedule 的 `-5 * progress` 参数也全部硬编码。

---

### 🟢 21. `"current_agent"` 是跨文件的魔法字符串

**位置**：3 个文件

- [selfplay.py:L41](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L41) `self.current_player_id = "current_agent"`
- [selfplay_logger.py:L33](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay_logger.py#L33) `agent_name="current_agent"`
- [payoff.py:L207-L213](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L207-L213) `pid != "current_agent"`

如果将来需要改变命名约定（如支持多 agent 并行训练），需要修改多处。

---

### 🟢 22. BattleSharedPayoff 中 `_players` 和 `_players_ids` 冗余

**位置**：[payoff.py:L55-L57](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/league/payoff.py#L55-L57)

```python
self._players: list = []
self._players_ids: list = []
```

两个列表的内容始终相同，`add_player()` 方法同时向两者追加相同值。`_players` 列表除了 `get_payoff_matrix()` 中的迭代外没有其他用途，而那时也可以用 `_players_ids`。

---

### 🟢 23. SelfPlayManager 薄包装引入了冗余状态

**位置**：[selfplay.py:L22-L136](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L22-L136)

维护了额外的 `games_against_opponents` 字典（L43）和 `_total_rounds`/`_total_battles` 计数器，但这些信息已可从 `BattleSharedPayoff` 推导出来。

---

## 五、总结

| 严重程度 | 数量 | 核心主题 |
|---|---|---|
| 🔴 Critical | 3 | God Class ×1、Monster Method ×2 |
| 🟠 High | 8 | 代码重复、深层耦合、缺少抽象、架构分层不足 |
| 🟡 Medium | 8 | 硬编码值、逻辑重复、隐式依赖、回调集成不协调 |
| 🟢 Low | 5 | 魔法字符串、冗余状态、try/except 过宽 |

### 最紧迫的改进建议

1. **拆分 `SelfPlayTrainer`**：将数据采集（Collector）、日志（Logger）、评估（Evaluator）独立为单独的类，`train()` 保留为编排循环

2. **消除 `_save_training_status()` 重复**：将公共逻辑提取到 `PPOTrainer` 中作为唯一实现，SelfPlayTrainer 通过公共方法调用

3. **提取 `_ppo_update()` 的指标收集**：将 L916-L962 约 50 行的指标聚合提取为 `PPOMetrics` dataclass，消除 [selfplay.py:L890-L998](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo/src/ppo_antwar/trainer/selfplay.py#L890-L998) 中 110 行 context 字典搬运

4. **为 Agent 类型定义统一接口**：使用 Protocol 或 ABC 定义 Agent 接口，消除 `_get_opponent_move` 中的 hasattr 链

5. **将 PPOTrainer 的直接访问替换为公共 API**：`self.trainer.optimizer.param_groups` → 公共的 `set_learning_rate()` 方法，`self.trainer.episode_rewards.append()` → 公共的 `record_episode_reward()` 方法
