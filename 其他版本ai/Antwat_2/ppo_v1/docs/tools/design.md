# ppo_v2 监控字段补齐 — 技术设计方案

> 本文档描述将 ppo_v2 监控表格恢复到 ppo_v1 最完整字段水平，并实施语义优化的技术方案。

---

## 1. 方案概览

### 1.1 改动范围

本方案涉及 **3 层、7 个文件** 的改动：

| 层次 | 文件 | 改动量级 | 核心职责 |
|------|------|---------|---------|
| 数据生产层 | [ppo_v2/.../trainer/ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py) | 中 | 扩充 `_aggregate_metrics()` 输出字段 |
| 数据生产层 | [ppo_v2/.../trainer/selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/selfplay.py) | 大 | 新增 context 构建：经济、奖励来源、动作统计 |
| 日志聚合层 | [ppo_v2/.../monitor/training_logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/training_logger.py) | 中 | 将新字段透传给 Logger |
| 日志写入层 | [ppo_v2/.../monitor/logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/logger.py) | 小 | 自动展开 `**metrics` |
| 展示层 | [tools/monitor/format_metrics_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_metrics_table.py) | 中 | 新增列 + 语义优化 |
| 展示层 | [tools/monitor/format_action_reward_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_action_reward_table.py) | 中 | Super% 拆分 + 新辅助列 |
| 展示层 | [tools/monitor/format_battle_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_battle_table.py) | 中 | 改进默认输出为累计+逐轮并列 |
| 展示层 | [tools/monitor/scan_anomalies.sh](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/scan_anomalies.sh) | 小 | 新增检测项 |

### 1.2 设计原则

1. **向后兼容**: 所有格式脚本在处理旧 JSONL（无新增字段）时必须正常工作，以 `N/A` 或条件性隐藏处理缺失字段
2. **最小侵入**: 数据生产层的改动优先在已有方法内扩展现有 dict，不新增独立类
3. **分层解耦**: 数据生产层只负责产生字段 dict，展示层只负责渲染表格，互不依赖
4. **字段名统一**: 全部采用 snake_case 命名，与 ppo_v1 一致

---

## 2. 数据生产层改动（ppo_v2）

### 2.1 PPOTrainer._aggregate_metrics() — 扩充输出字段

**文件**: [ppo_v2/.../trainer/ppo_trainer.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py)  
**方法**: `_aggregate_metrics()` (第 484 行)

**改动内容**: 在现有返回 dict 中追加以下字段

```python
# == 新增字段: 价值诊断（从 _process_minibatch 收集）==
# value_input_mean / value_input_std: 来自 tensors['values'] 的统计
#     收集方式: 在 _prepare_minibatch 中记录 mb_data['old_values'] 的 mean/std
# value_pred_mean / value_pred_std: 来自 mb_data 中新 values 的预测统计
#     收集方式: 在 _process_minibatch 中累加 values.mean().item() / values.std().item()

# == 新增字段: 比率诊断 ==
# ratio_mean / ratio_std: 来自 mb_data 中 ratio 的统计
#     收集方式: 在 _process_minibatch 中累加 ratio.mean().item() / ratio.std().item()

# == 新增字段: 有效动作诊断 ==
# valid_actions_mean / valid_actions_min: 来自 action_mask 的统计
#     收集方式: 在 _process_minibatch 中累加 valid_counts.mean().item() / valid_counts.min().item()

# == 新增字段: TD 误差诊断 ==
# td_error_mean / td_error_std: 在 _train_ppo_epochs 中 GAE 计算后统计
#     收集方式: advantages + values - returns 的 mean/std（已在 ppo_v1 中实现）

# == 新增字段: 熵诊断 ==
# entropy_type: 各动作类型熵（已在 evaluate_actions 中计算 type_ent）
#     收集方式: 在 _process_minibatch 中累加 type_ent.mean().item()
# entropy_target: 目标熵（已在 evaluate_actions 中计算 target_ent）
#     收集方式: 在 _process_minibatch 中累加 target_ent.mean().item()

# == 新增字段: num_collected ==
#     收集方式: len(batch) = total_samples
```

#### 具体改动清单

**A. `_process_minibatch()` 返回值扩充**:

```python
# 当前返回值:
return {
    'grad_norm': ..., 'grad_norm_policy': ..., 'grad_norm_value': ...,
    'grad_norm_max_layer': ...,
    'policy_loss': ..., 'value_loss': ..., 'entropy': ..., 'clip_fraction': ...,
    **aux_metrics,  # 包含 aux_tower_loss, aux_gold_loss, aux_enemy_tower_loss, aux_enemy_gold_loss
}

# 改为:
return {
    # ...原有字段不变...
    # 新增:
    'value_pred_mean': values.mean().item(),
    'value_pred_std': values.std().item(),
    'ratio_mean': ratio.mean().item(),
    'ratio_std': ratio.std().item(),
    'entropy_type': type_ent.mean().item(),     # 来自 evaluate_actions 返回值
    'entropy_target': target_ent.mean().item(),   # 来自 evaluate_actions 返回值
    'valid_actions_mean': valid_counts.mean().item(),  # valid_counts = action_mask.sum(dim=-1)
    'valid_actions_min': valid_counts.min().item(),
}
```

**B. `_aggregate_metrics()` 返回值扩充**:

```python
# 当前返回:
return {
    'policy_loss': ..., 'value_loss': ..., 'entropy': ..., 'loss': ...,
    'clip_fraction': ..., 'reward_mean': ..., 'reward_std': ..., 'reward_min': ..., 'reward_max': ...,
    'advantages_std': ..., 'return_mean': ..., 'return_std': ...,
    'batch_size': ..., 'grad_norm': ..., 'grad_norm_policy': ..., 'grad_norm_value': ...,
    'grad_norm_max_layer': ..., 'skipped_minibatches': ...,
    'aux_tower_loss': ..., 'aux_gold_loss': ..., 'aux_enemy_tower_loss': ..., 'aux_enemy_gold_loss': ...,
    **type_metrics,
}

# 改为:（保持现有字段不变，追加）
return {
    # ...原有字段不变...
    # 新增
    'value_pred_mean': np.mean(metrics.get('value_pred_mean', [0])),
    'value_pred_std': np.mean(metrics.get('value_pred_std', [0])),
    'ratio_mean': np.mean(metrics.get('ratio_mean', [0])),
    'ratio_std': np.mean(metrics.get('ratio_std', [0])),
    'entropy_type': np.mean(metrics.get('entropy_type', [0])),
    'entropy_target': np.mean(metrics.get('entropy_target', [0])),
    'valid_actions_mean': np.mean(metrics.get('valid_actions_mean', [0])),
    'valid_actions_min': np.min(metrics.get('valid_actions_min', [0])) if metrics.get('valid_actions_min') else 0.0,
    'td_error_mean': ...,  # 需从 _compute_gae 后的 td_error 获取
    'td_error_std': ...,   # 同上
    'num_collected': len(tensors['rewards']),
}
```

**C. TD 误差计算位置**:

在 `_ppo_update()` 中，`compute_gae` 后可计算 TD 误差：

```python
# _ppo_update 方法中，_compute_gae 调用后追加:
advantages, returns = self._compute_gae(...)
td_errors = returns - tensors['values']  # TD 误差: returns - old_values
td_error_mean = td_errors.mean().item()
td_error_std = td_errors.std().item()
# 将 td_error_mean / td_error_std 传入 _aggregate_metrics()
```

### 2.2 SelfPlayTrainer — 新增上下文构建管道

**文件**: [ppo_v2/.../trainer/selfplay.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/trainer/selfplay.py)

#### 2.2.1 新增常量与方法

在 `SelfPlayTrainer` 类中新增以下方法和常量：

```python
# 在类定义中追加
# (1) PPO 指标键列表（供字段提取使用）
PPO_METRIC_KEYS = [
    'policy_loss', 'value_loss', 'entropy', 'entropy_type', 'entropy_target',
    'aux_tower_loss', 'aux_gold_loss', 'aux_enemy_tower_loss', 'aux_enemy_gold_loss',
    'reward_mean', 'reward_std', 'reward_min', 'reward_max',
    'advantages_std', 'return_mean', 'return_std',
    'clip_fraction', 'grad_norm',
    'value_pred_mean', 'value_pred_std',
    'ratio_mean', 'ratio_std',
    'grad_norm_policy', 'grad_norm_value', 'grad_norm_max_layer',
    'td_error_mean', 'td_error_std',
    'valid_actions_mean', 'valid_actions_min',
    'batch_size', 'num_collected', 'nan_skip_count',
]

# (2) 奖励来源键列表
SOURCE_KEYS = [
    'rw_hp_attack_base', 'rw_hp_attack_tower', 'rw_coin_gain', 'rw_tower_survival',
    'rw_balance', 'rw_tech_bonus', 'rw_die_penalty', 'rw_end_reward',
]
```

#### 2.2.2 新增 _build_training_context() 方法

在 `_flush_batch_to_update()` 之前或之后调用：

```python
def _build_training_context(self, ppo_metrics: Dict, episode_rewards: List,
                            episode_lengths: List) -> Dict[str, Any]:
    """构建完整的训练上下文 dict，与 Logger 的输出字段对齐"""
    context = {}

    # 1) 提取 PPO 指标（仅保留定义在 PPO_METRIC_KEYS 中的键）
    for key in self.PPO_METRIC_KEYS:
        if key in ppo_metrics:
            context[key] = ppo_metrics[key]

    # 2) 超参数快照
    ppo_cfg = self._config.get('ppo', {})
    context['entropy_coef'] = ppo_cfg.get('ent_coef', 0.2)
    context['learning_rate'] = ppo_cfg.get('lr', 0.00005)

    # 3) 批次信息
    context['num_collected'] = len(self._batch_pool)  # 当前 batch pool 中的 episode 数

    # 4) 经济统计: 从 per_episode_stats 的 snapshots 中聚合
    coin_stats = self._aggregate_coin_stats()
    context.update(coin_stats)

    # 5) 奖励来源分解: 从 per_episode_stats 中聚合（需记录 reward_detail）
    #    注意: ppo_v2 当前未记录 reward_detail，需先在 _collect_episode_with_swap 中采集
    reward_sources = self._aggregate_reward_sources()
    context.update(reward_sources)

    # 6) 动作奖励/计数: 从 pending_action_stats 中聚合为扁平字段
    action_stats = self._aggregate_action_reward_stats()
    context.update(action_stats)

    return context
```

#### 2.2.3 新增经济统计方法

```python
def _aggregate_coin_stats(self) -> Dict[str, float]:
    """从 per_episode_stats_buffer 中的 step_snapshots 聚合经济统计"""
    all_our_coins = []
    all_enemy_coins = []
    our_cumulative_list = []
    enemy_cumulative_list = []

    for record in self._per_episode_stats_buffer[-50:]:  # 最近 50 个 episode
        # 注意: 当前 per_episode_stats_buffer 不包含 step_snapshots
        # 需要在 _collect_episode_with_swap 中将 snapshots 存入该 buffer
        snapshots = record.get('snapshots', [])
        for s in snapshots:
            all_our_coins.append(s.get('own_coins', 0))
            all_enemy_coins.append(s.get('enemy_coins', 0))

    # 计算均值
    ...

    # 计算金币分布桶
    our_buckets = self._compute_coin_buckets(all_our_coins, prefix='our_coins')
    enemy_buckets = self._compute_coin_buckets(all_enemy_coins, prefix='enemy_coins')

    result = {
        'avg_our_coins': ...,
        'avg_enemy_coins': ...,
        'avg_our_cumulative_coins': ...,
        'avg_enemy_cumulative_coins': ...,
    }
    result.update(our_buckets)
    result.update(enemy_buckets)
    return result
```

**前置依赖**: `_collect_episode_with_swap()` 已在每步采集 `step_snapshots`（见第 428-434 行），但 snapshot 数据未传入 `per_episode_stats_buffer`。需在 `_per_episode_stats_buffer.append({...})` 中添加 `'snapshots': step_snapshots`。

#### 2.2.4 新增奖励来源聚合方法

**前置依赖**: ppo_v2 的环境 `antwar_env.py` 的 `step()` 返回值 `info` 中是否包含 `reward_detail` 分解？

- 若 `info` 中含有 reward component 分解 → 只需在 `_collect_episode_with_swap()` 中采集
- 若无 → 需要在环境层新增 reward_detail 的输出（在 antwar_env.py 或 reward 模块中）

**采集方式**: 在 `_collect_episode_with_swap()` 中每步追加：

```python
# 在步循环中，env.step() 后
reward_detail = info.get('reward_detail', {})
# 将 reward_detail 存入 step_snapshots 或独立的 list
```

**聚合方法**:

```python
def _aggregate_reward_sources(self) -> Dict[str, float]:
    """从 step_snapshots 聚合奖励来源"""
    result = {k: 0.0 for k in self.SOURCE_KEYS}
    # ... 遍历 step_snapshots 中的 reward_detail，累加各分量 ...
    return result
```

#### 2.2.5 新增动作奖励/计数扁平化方法

```python
def _aggregate_action_reward_stats(self) -> Dict[str, float]:
    """将 pending_action_stats 中的 dict 展开为扁平字段"""
    result = {}
    for stats in self._pending_action_stats:
        for action_type, count in stats.get('action_counts', {}).items():
            key = f'actcnt_{action_type}'
            result[key] = result.get(key, 0) + count
        for action_type, reward in stats.get('action_rewards', {}).items():
            key = f'actrw_{action_type}'
            result[key] = result.get(key, 0.0) + reward
    return result
```

#### 2.2.6 调用链路改造

在 `_flush_batch_to_update()` 方法中（第 217 行），PPO update 后、调用 `_training_logger.log_training_metrics()` 前，插入 context 构建：

```python
def _flush_batch_to_update(self) -> None:
    # ... 原有 batch validation 和 PPO update 代码不变 ...

    metrics = self._trainer._ppo_update(merged, self._episode_count)

    # ==== 新增: 构建上下文 ====
    context = self._build_training_context(metrics,
                                           self._episode_rewards,
                                           self._episode_lengths)
    # 将 context 合并到 metrics 中（metrics 会传入 logger）
    metrics.update(context)
    # ========================

    self._trainer._last_metrics = metrics
    # ... 后续代码不变 ...
```

### 2.3 TrainingLogger — 透传新增字段

**文件**: [ppo_v2/.../monitor/training_logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/training_logger.py)

**改动**: 当前 `log_training_metrics()` 已经将 `metrics` dict 透传给 `Logger.save_training_metrics()`（第 43 行），只需确保新增字段在 metrics dict 中存在即可。无需修改 training_logger.py。

### 2.4 Logger — 无需修改

**文件**: [ppo_v2/.../monitor/logger.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/ppo_v2/src/ppo_antwar/monitor/logger.py)  
**方法**: `save_training_metrics()` (第 77 行)

当前实现已使用 `**metrics` 展开所有键值对写入 JSONL（第 82 行），因此**无需修改**此文件。所有新增字段会自动出现在 JSONL 记录中。

---

## 3. 展示层改动（tools/monitor/）

### 3.1 format_metrics_table.py — 新增列 + 语义优化

**文件**: [tools/monitor/format_metrics_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_metrics_table.py)

#### 3.1.1 条件列逻辑扩展

当前脚本已具备条件性显示的架构（根据字段存在与否动态选择表头）。在此基础上扩展：

| 列名 | 检测字段 | 显示条件 |
|------|---------|---------|
| OurCoin | `avg_our_coins` | 存在则显示 |
| EnCoin | `avg_enemy_coins` | 存在则显示 |
| OurCum | `avg_our_cumulative_coins` | 存在则显示 |
| EnCum | `avg_enemy_cumulative_coins` | 存在则显示 |
| TowerLoss | `aux_tower_loss` | 存在则显示 |
| GoldLoss | `aux_gold_loss` | 存在则显示 |
| EnTowerLoss | `aux_enemy_tower_loss` | 存在则显示（新增） |
| EnGoldLoss | `aux_enemy_gold_loss` | 存在则显示（新增） |
| GradNorm | `grad_norm` | 存在则显示（新增） |
| ClipFrac | `clip_fraction` | 存在则显示（新增） |

#### 3.1.2 表头生成逻辑

```python
# 将当前硬编码的 if/elif/else 重构为表头构建器
def _build_header(unique_samples):
    cols = [('Ep', 6, 'd')]
    cols.append(('Reward', 10, '.2f'))
    # 新增: LastRew - 从 per_episode_stats 的最新值读取
    if any('last_reward' in h for h in unique_samples):
        cols.append(('LastRew', 10, '.2f'))

    # 经济列
    if any('avg_our_coins' in h for h in unique_samples):
        cols.append(('OurCoin', 8, '.1f'))
        cols.append(('EnCoin', 8, '.1f'))
    if any('avg_our_cumulative_coins' in h for h in unique_samples):
        cols.append(('OurCum', 8, '.1f'))
        cols.append(('EnCum', 8, '.1f'))

    # 诊断列
    if any('grad_norm' in h for h in unique_samples):
        cols.append(('GradNorm', 8, '.2f'))
    if any('clip_fraction' in h for h in unique_samples):
        cols.append(('ClipFrac', 8, '.2f'))

    # 辅助损失列
    if any('aux_tower_loss' in h for h in unique_samples):
        cols.append(('TowerLoss', 11, '.6f'))
    if any('aux_gold_loss' in h for h in unique_samples):
        cols.append(('GoldLoss', 11, '.6f'))
    if any('aux_enemy_tower_loss' in h for h in unique_samples):
        cols.append(('EnTowerLoss', 11, '.6f'))
    if any('aux_enemy_gold_loss' in h for h in unique_samples):
        cols.append(('EnGoldLoss', 11, '.6f'))

    # PPO 损失列（始终显示）
    cols.append(('PolicyLoss', 12, '.6f'))
    cols.append(('ValueLoss', 12, '.6f'))
    cols.append(('Entropy', 10, '.4f'))

    # 超参
    cols.append(('EntCoef', 8, '.4f' if any('entropy_coef' in h for h in unique_samples) else None))
    cols.append(('LR', 14, '.10f' if any('learning_rate' in h for h in unique_samples) else None))

    # 轮次
    cols.append(('Rounds', 8, '.0f'))

    return cols
```

#### 3.1.3 新增 LastRew 列的支持

从 `per_episode_stats.jsonl` 读取最新记录的 reward，写入 JSONL 的 `last_reward` 字段。

**实现方式**: 在 `_build_training_context()` 中添加：

```python
context['last_reward'] = self._episode_rewards[-1] if self._episode_rewards else 0.0
```

#### 3.1.4 表底统计行增强

在当前 Reward 统计行（第 156 行）基础上，添加：

```python
# 新增: 打印表头列的详细统计
if rewards:
    print(f"Reward  mean: {mean:.2f}  std: {std:.2f}  min: {min_r:.2f}  max: {max_r:.2f}")
    # 若存在 Economic 字段，添加经济统计行
    if show_coins:
        our_coins = [h.get('avg_our_coins', 0) for h in unique if 'avg_our_coins' in h]
        if our_coins:
            print(f"OurCoin mean: {sum(our_coins)/len(our_coins):.1f}  "
                  f"EnCoin mean: {sum(en_coins)/len(en_coins):.1f}" if en_coins else "")
```

### 3.2 format_action_reward_table.py — 列优化

**文件**: [tools/monitor/format_action_reward_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_action_reward_table.py)

#### 3.2.1 改动清单

| 改动项 | 当前行为 | 目标行为 |
|--------|---------|---------|
| Super% 拆分 | 聚合 4 类超级武器为一个 Super% | 拆分为 LS%、Emp%、Defl%、Eva% 四列 |
| 删除 HPBase%/HPTower% | 始终显示 0.0% | 删除这两列，释放空间 |
| 新增 PosSum/NegSum | 无 | 新增正奖励分量和 / 负奖励分量和 |
| 新增 TotalReward 列 | 显示 `avg_reward` | 改为显示 `reward_mean`（batch 均值） |

#### 3.2.2 列定义调整

```python
# 当前:
action_cols_v2 = ['Noop', 'Build', 'Upgrd', 'Downg', 'Super', 'HPBase', 'HPTower']

# 改为:
action_cols_v2 = ['Noop', 'Build', 'Upgrd', 'Downg', 'LS', 'Emp', 'Defl', 'Eva']

# 计算方式:
noop_pct = _pct(ar.get('noop', 0.0), abs_sum)
build_pct = _pct(ar.get('build_tower', 0.0), abs_sum)
upgrd_pct = _pct(ar.get('upgrade_tower', 0.0), abs_sum)
downg_pct = _pct(ar.get('downgrade_tower', 0.0), abs_sum)
ls_pct = _pct(ar.get('lightning_storm', 0.0), abs_sum)
emp_pct = _pct(ar.get('emp_blaster', 0.0), abs_sum)
defl_pct = _pct(ar.get('deflector', 0.0), abs_sum)
eva_pct = _pct(ar.get('evasion', 0.0), abs_sum)

# 新增: PosSum / NegSum
pos_sum = sum(v for v in ar.values() if v > 0)
neg_sum = sum(v for v in ar.values() if v < 0)
```

#### 3.2.3 表头

```python
header = (f"{'Ep':>6}  " +
          "  ".join(f"{c:>8}" for c in action_cols_v2) +
          f"  {'PosSum':>8}  {'NegSum':>8}  {'ActSum':>8}  {'Reward':>8}  {'Batch':>6}")
```

### 3.3 format_battle_table.py — 改进默认输出

**文件**: [tools/monitor/format_battle_table.py](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/format_battle_table.py)

#### 3.3.1 对 selfplay 模式改进

当前 `status_manager.sh` 的 `get_battle_table()` 函数（第 567 行）先尝试旧版 battle 目录模式，若检测到 `selfplay_battles/` 目录则改用 `--selfplay-stats` + `--evaluations` 模式。需要确保 **两种表格都输出**：

```python
# format_battle_table.py 的 main()
def main():
    args = parser.parse_args()

    if args.selfplay_stats:
        format_selfplay_stats(args.selfplay_stats)  # 累积汇总表

    if args.evaluations:
        format_evaluations(args.evaluations)         # 逐轮评估表（新增默认输出）

    # 不再使用旧版 format_battle_table，仅在无任何参数时 fallback
```

#### 3.3.2 添加 Trend 行

在累积汇总表底添加趋势信息：

```python
# format_selfplay_stats() 末尾追加
# 从 battle_evaluations.jsonl 读取逐轮胜率，生成趋势行
if evaluations_file exists:
    with open(evaluations_file) as f:
        eval_entries = [json.loads(line) for line in f if line.strip()]
    if eval_entries:
        trends = []
        for entry in eval_entries:
            ep = entry['episode']
            results = entry.get('results', {})
            total_wins = sum(r.get('wins', 0) for r in results.values())
            total_battles = sum(r.get('total_battles', 0) for r in results.values())
            wr = total_wins / max(total_battles, 1) * 100
            trends.append(f"E{ep}:{wr:.0f}%")
        print(f"Trend{'':>15}  {' → '.join(trends)}")
```

### 3.4 scan_anomalies.sh — 新增检测项

**文件**: [tools/monitor/scan_anomalies.sh](file:///Users/raymondzeng/Documents/trae_projects/AntWar/tools/monitor/scan_anomalies.sh)

在现有 Python 检测脚本中追加以下检查逻辑：

```python
# 在 remote Python 检测脚本（scan_anomalies.sh 内联 Python 代码）中新增:

# ==== 新增检测项 ====

# 1) ratio 异常检测
if 'ratio_mean' in metrics:
    ratio_mean = metrics['ratio_mean']
    ratio_std = metrics.get('ratio_std', 0)
    if abs(ratio_mean - 1.0) > 0.2:
        anomalies.append({
            'type': 'ratio_anomaly',
            'severity': 'WARNING',
            'message': f"ratio_mean={ratio_mean:.4f} отклонен от 1.0"
        })

# 2) 价值预测发散检测
if 'value_pred_mean' in metrics:
    vpm = metrics['value_pred_mean']
    vps = metrics.get('value_pred_std', 0)
    if abs(vpm) > 2.0:
        anomalies.append({
            'type': 'value_pred_divergence',
            'severity': 'WARNING',
            'message': f"value_pred_mean={vpm:.4f} out of range [-2, 2]"
        })

# 3) 损失发散检测
if 'policy_loss' in metrics and has_history(len >= 10):
    recent_policy_losses = [m['policy_loss'] for m in recent_metrics[-10:]]
    if all(recent_policy_losses[i] < recent_policy_losses[i+1] for i in range(len(recent_policy_losses)-1)):
        anomalies.append({
            'type': 'loss_divergence',
            'severity': 'WARNING',
            'message': f"policy_loss持续上升: {recent_policy_losses[0]:.4f} → {recent_policy_losses[-1]:.4f}"
        })

# 4) 经济崩溃检测
if 'avg_our_coins' in metrics and 'avg_enemy_coins' in metrics:
    eco_ratio = metrics['avg_our_coins'] / max(metrics['avg_enemy_coins'], 1)
    if eco_ratio < 0.5:
        anomalies.append({
            'type': 'economic_collapse',
            'severity': 'CRITICAL',
            'message': f"经济崩溃: our_coins/enemy_coins={eco_ratio:.2f} < 0.5"
        })

# 5) 有效动作崩塌
if 'valid_actions_min' in metrics:
    if metrics['valid_actions_min'] < 2:
        anomalies.append({
            'type': 'valid_actions_collapse',
            'severity': 'WARNING',
            'message': f"valid_actions_min={metrics['valid_actions_min']} 接近0"
        })
```

---

## 4. 改动物件清单

### 4.1 数据生产层（ppo_v2）

| 序号 | 文件 | 方法/位置 | 改动内容 | 依赖 |
|------|------|----------|---------|------|
| D1 | ppo_trainer.py | `_process_minibatch()` | 返回值新增: value_pred_mean/std, ratio_mean/std, entropy_type/target, valid_actions_mean/min | 无 |
| D2 | ppo_trainer.py | `_ppo_update()` | 新增 TD 误差统计（advantages + values - returns） | 无 |
| D3 | ppo_trainer.py | `_aggregate_metrics()` | 返回值新增 ~10 个字段 | D1, D2 |
| D4 | selfplay.py | 新增 `_build_training_context()` | 构建完整上下文 dict | D3, D5, D6, D7, D8 |
| D5 | selfplay.py | `_collect_episode_with_swap()` | 将 `step_snapshots` 存入 `per_episode_stats_buffer` | 无 |
| D6 | selfplay.py | 新增 `_aggregate_coin_stats()` | 经济统计聚合 | D5 |
| D7 | selfplay.py | 新增 `_aggregate_reward_sources()` | 奖励来源分解（需确认 reward_detail 可用性） | env 层 |
| D8 | selfplay.py | 新增 `_aggregate_action_reward_stats()` | 动作奖励/计数扁平化 | 无 |
| D9 | selfplay.py | `_flush_batch_to_update()` | 在 `log_training_metrics()` 前调用 `_build_training_context()` | D4 |

### 4.2 展示层（tools/monitor）

| 序号 | 文件 | 位置 | 改动内容 | 依赖 |
|------|------|------|---------|------|
| F1 | format_metrics_table.py | `format_table()` | 动态列构建器重构 + 新增 GradNorm/ClipFrac/EnTowerLoss/EnGoldLoss/LastRew 列 | D3 |
| F2 | format_metrics_table.py | `format_table()` | 表底统计行增强 | 无 |
| F3 | format_action_reward_table.py | `format_table()` | Super% → 4 列拆分, 删除 HPBase/HPTower, 新增 PosSum/NegSum | D8 |
| F4 | format_battle_table.py | `main()` | 确保 selfplay_stats + evaluations 同时输出 | 无 |
| F5 | format_battle_table.py | `format_selfplay_stats()` | 新增 Trend 行 | 无 |
| F6 | scan_anomalies.sh | Python 检测脚本 | 新增 ratio/价值预测/损失发散/经济崩溃/有效动作崩塌 5 项检测 | D3 |

### 4.3 兼容性保障

| 序号 | 保障措施 | 说明 |
|------|---------|------|
| C1 | 新增字段使用 `.get(key, default)` 安全读取 | 所有 format 脚本已使用此模式 |
| C2 | 条件性表头显示 | 不存在的新增字段自动隐藏对应列 |
| C3 | 字段名不对齐时的 fallback | 如旧 JSONL 使用 `loss` 而非 `avg_loss`，format 层自动识别 |
| C4 | action_rewards 双格式兼容 | format_action_reward_table.py 同时支持 `action_rewards` dict 和 `actrw_*` 扁平字段 |

---

## 5. 实施步骤

```
Step 1: 数据层 — 扩充 PPOTrainer._aggregate_metrics()
        ppo_trainer.py: D1, D2, D3
        └→ 验证: 单元测试 + 本地运行检查 JSONL 输出

Step 2: 数据层 — 新增 SelfPlayTrainer 上下文构建
        selfplay.py: D4, D5, D6, D7, D8, D9
        └→ 验证: 运行训练 10 个 episode，检查 JSONL 包含新增字段

Step 3: 展示层 — 更新 format_metrics_table.py
        F1, F2
        └→ 验证: 用步骤2产生的 JSONL 运行 train_metrics_table

Step 4: 展示层 — 更新 format_action_reward_table.py
        F3
        └→ 验证: 用步骤2产生的 JSONL 运行 train_action_reward_table

Step 5: 展示层 — 更新 format_battle_table.py + scan_anomalies.sh
        F4, F5, F6
        └→ 验证: 运行 train_battle_table + scan_anomalies

Step 6: 端到端验证
        └→ 运行完整的 monitor_training.sh
        └→ 对照 original.md 的表格模板逐项检查
```

### 5.1 验证清单

| 验证项 | 预期结果 |
|--------|---------|
| 新旧 JSONL 兼容 | 旧 JSONL 文件不会导致 format 脚本崩溃 |
| 新增字段 | `avg_our_coins` 等字段在 JSONL 中可见 |
| 经济列显示 | OurCoin/EnCoin 列出现在 SelfPlay 技术参数表中 |
| 价值诊断列显示 | GradNorm/ClipFrac 列出现在表中 |
| Reward 表显示 | Super% 被拆分为 4 列，无 HPBase/HPTower |
| Battle 表 | 累积 + 逐轮 两张表并列输出 |
| 异常扫描 | 新增检测项正确告警 |
| 字段名对齐 | JSONL 使用统一命名（如 `grad_norm` 而非 `gradient_norm`） |
