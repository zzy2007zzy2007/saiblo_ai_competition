# 线索 14 / 17 / 18 调查报告：PPO V2 次要问题

> 调查日期：2026-06-11
> 调查范围：对手池淘汰策略、降塔惩罚、enemy 辅助标签权重

---

## 线索 14：对手池淘汰策略单一

### 问题描述

`OpponentPool._evict_worst()` 按 `mu + protection_bonus` 排序淘汰最低分对手，可能淘汰有价值的多样化对手。

### 代码确认

**文件**：`ppo_v2/src/ppo_antwar/league/opponent_pool.py`，第 48-85 行

```python
def _evict_worst(self) -> str:
    evictable = []
    for opp_id in self._opponents:
        games = self._games_played.get(opp_id, 0)
        if games < self.min_games_threshold:
            continue
        evictable.append(opp_id)

    if not evictable:
        evictable = self._opponents

    def eviction_score(opp_id):
        mu = 25.0
        if self._payoff is not None:
            rating = self._payoff._trueskill_ratings.get(opp_id, TrueSkillRating())
            mu = rating.mu
        games = self._games_played.get(opp_id, 0)
        protection = (
            max(0, self.min_games_threshold - games)
            * self._EVICTION_PROTECTION_FACTOR
        )
        return mu + protection

    victim = min(evictable, key=eviction_score)
```

### 分析

1. **淘汰唯一标准是 TrueSkill mu 分数**：`eviction_score` 仅由 `mu + protection_bonus` 构成，其中 `protection_bonus = max(0, min_games_threshold - games) * 0.5`。当 `games >= min_games_threshold`（默认 8）时，`protection_bonus = 0`，淘汰完全取决于 `mu`。

2. **缺少多样性考量**：两个 mu 相近的对手，一个可能是激进型策略、另一个是防守型策略，但淘汰时无法区分。低分但风格独特的对手会被优先淘汰，导致对手池策略趋同。

3. **pool_size=10 较小**：配置中 `opponent_pool_size: 10`，淘汰更频繁，多样性丧失的影响更显著。

4. **TrueSkill sigma 未被利用**：`TrueSkillRating` 保存了 `sigma`（不确定性），高 sigma 的对手评分尚不确定，直接按 mu 淘汰可能误删潜力对手。`is_confident` 属性（`sigma < 3.0`）也未在淘汰逻辑中使用。

5. **fallback 逻辑有风险**：当所有对手都未达到 `min_games_threshold` 时，`evictable = self._opponents`，此时 `protection_bonus` 仍按公式计算，但所有对手都受到保护，淘汰结果取决于谁的游戏数最少（protection 最大），而非谁最弱。

### 结论：**确认**

淘汰策略确实单一，仅以 mu 分数为唯一标准，缺少多样性和不确定性考量。在 pool_size=10 的约束下，可能导致对手池策略趋同。

### 修复建议

1. **引入多样性指标**：在淘汰评分中加入策略多样性因子，例如基于 payoff 矩阵的胜负模式差异，或基于检查点嵌入的余弦距离。保留与现有对手风格差异最大的低分对手。

2. **利用 sigma 信息**：对高 sigma（评分不确定）的对手给予额外保护，避免在评分尚未收敛时就淘汰。可参考 `TrueSkillRating.is_confident` 属性。

3. **引入年龄衰减**：对加入时间过长的对手（`_added_episodes` 已记录），即使 mu 不低也应考虑淘汰，防止对手池老化。

4. **优先级建议**：中等。当前淘汰策略不会导致训练崩溃，但会限制策略多样性，影响自对弈效果。可在后续迭代中改进。

---

## 线索 17：downgrade_tower_penalty=-9.0 可能过重

### 问题描述

降塔惩罚远大于建塔奖励（0.3-0.6），可能导致策略完全避免降塔操作。

### 代码确认

**文件**：`ppo_v2/src/ppo_antwar/utils/action_constants.py`，第 230-259 行

```python
REWARD_CONFIG: Dict = {
    ...
    "build_tower_tiers": [0.6, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1],  # 建塔奖励
    "upgrade_tower_l2": 3.0,                                       # 升2级奖励
    "upgrade_tower_l3": 5.0,                                       # 升3级奖励
    "downgrade_tower_penalty": -9.0,                               # 降塔惩罚
    ...
}
```

**文件**：`ppo_v2/src/ppo_antwar/env/antwar_env.py`，第 522-533 行

```python
def _compute_downgrade_tower_reward(self, _action_id, player, _type_name, target_idx):
    reward = REWARD_CONFIG["downgrade_tower_penalty"]
    if target_idx < len(TOWER_POSITIONS):
        target_pos = TOWER_POSITIONS[target_idx]
        pos_key = target_pos[0] * _POSITION_HASH_MULTIPLIER + target_pos[1]
        player_counts = self._tower_build_slot_counts[player]
        count = player_counts.get(pos_key, 0)
        if count > 0:
            player_counts[pos_key] = count - 1
    return reward
```

### 分析

1. **数量级对比**：

   | 动作 | 奖励/惩罚 |
   |------|-----------|
   | 建塔（首次） | +0.6 |
   | 建塔（后续递减） | +0.5 → +0.3 → +0.1 |
   | 升级到 L2 | +3.0 |
   | 升级到 L3 | +5.0 |
   | **降塔** | **-9.0** |

   降塔惩罚是建塔奖励的 **15-30 倍**，是升级 L3 奖励的 **1.8 倍**。

2. **降塔的实际游戏价值**：降塔操作会返还部分金币，允许玩家在关键位置重建塔或调整布局。在某些战术场景下（如卖掉低级塔重建高级塔、调整防线），降塔是合理操作。

3. **惩罚过重的后果**：
   - 策略几乎不可能学会降塔操作，因为 -9.0 的惩罚远超任何可能的收益
   - 即使降塔后重建能带来更好的局面，短期惩罚也足以阻止探索
   - 动作空间中 downgrade_tower 占 16 个 slot（action_id 81-96），这些动作基本不会被选中，相当于浪费了动作空间

4. **与 step_reward_clip=2000 的关系**：-9.0 虽大但远未触及 clip 上限，说明惩罚本身不会导致数值问题，但足以在策略梯度中产生强烈的抑制信号。

5. **降塔的 build_slot_counts 维护**：代码中降塔时会递减 `_tower_build_slot_counts`，这意味着降塔后重建可以重新获得较高的建塔奖励。但 -9.0 的惩罚远大于重建可能获得的 0.6 奖励，无法形成正循环。

### 结论：**确认**

`downgrade_tower_penalty=-9.0` 确实过重。与建塔奖励（0.3-0.6）相比，惩罚量级不合理，会导致策略完全避免降塔操作，限制了策略的战术灵活性。

### 修复建议

1. **降低惩罚量级**：建议将 `downgrade_tower_penalty` 从 -9.0 调整为 **-1.0 ~ -2.0**，使其与建塔奖励在同一量级。降塔本身已经消耗了一个动作机会（机会成本），不需要额外过重的惩罚。

2. **条件化惩罚**：根据降塔的等级给予不同惩罚。降 L3 塔的惩罚可以高于降 L1 塔，因为 L3 塔投入更多。例如：
   - 降 L1 塔：-0.5
   - 降 L2 塔：-1.5
   - 降 L3 塔：-3.0

3. **考虑净收益**：降塔返还的金币可以在 `balance_reward_weight` 中体现，因此惩罚不需要覆盖"浪费金币"的考量，只需表达"拆除已有投资"的负面信号。

4. **优先级建议**：中等偏高。过重的惩罚直接限制了策略的战术空间，且修复简单（改一个配置值即可验证效果）。

---

## 线索 18：enemy 辅助标签的损失权重与 own 相同，但预测难度不同

### 问题描述

`aux_enemy_tower_coef=0.005` 与 `aux_tower_coef=0.005` 相同，但敌方信息可能更不完整，预测难度更高。

### 代码确认

**文件**：`ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml`，第 35-38 行

```yaml
aux_tower_coef: 0.005
aux_gold_coef: 0.001
aux_enemy_tower_coef: 0.005
aux_enemy_gold_coef: 0.001
```

**文件**：`ppo_v2/src/ppo_antwar/trainer/ppo_trainer.py`，第 651-661 行

```python
aux_tower_coef = ppo_cfg.get("aux_tower_coef")
aux_gold_coef = ppo_cfg.get("aux_gold_coef")
aux_enemy_tower_coef = ppo_cfg.get("aux_enemy_tower_coef")
aux_enemy_gold_coef = ppo_cfg.get("aux_enemy_gold_coef")
...
total_loss = total_loss + aux_tower_coef * aux_losses["aux_tower_loss"]
total_loss = total_loss + aux_enemy_tower_coef * aux_losses["aux_enemy_tower_loss"]
```

**文件**：`ppo_v2/src/ppo_antwar/network/ant_war_policy_value_network.py`，第 335-370 行

```python
def compute_auxiliary_loss(self, features, aux_tower_damage, aux_gold_income, ...):
    tower_pred = self.tower_damage_head(features)  # (batch, 10)
    mid = tower_pred.size(1) // 2  # 5
    aux_tower_loss = F.mse_loss(tower_pred[:, :mid], aux_tower_damage[:, :mid])
    aux_enemy_tower_loss = F.mse_loss(tower_pred[:, mid:], aux_tower_damage[:, mid:])
```

### 分析

1. **观测信息不对称**：从 `observation.py` 的 `_encode_global` 方法可以看到，己方和敌方的观测信息存在差异：
   - 己方：精确的 HP、金币、塔等级、科技等级、武器冷却
   - 敌方：HP（可见）、塔数量和等级总和（可见但不如己方精确）、金币（**不可见**）、科技等级（**不可见**）、武器冷却（可见）

2. **预测标签的构成**：辅助任务的标签 `aux_tower_damage` 和 `aux_gold_income` 是未来 N 步的实际值（ground truth），而非观测值。因此标签本身是完整的，不存在信息缺失问题。

3. **核心问题在于特征而非标签**：虽然标签完整，但网络输入的特征中，敌方信息比己方信息更稀疏、更不精确。这意味着：
   - 从更少的敌方特征预测敌方未来伤害，本质上更困难
   - 相同权重下，enemy 辅助头的梯度噪声更大（预测误差的方差更高）
   - 高噪声梯度通过共享的编码器反传，可能干扰策略学习

4. **实际影响评估**：
   - `aux_tower_coef=0.005` 和 `aux_enemy_tower_coef=0.005` 都很小，对总损失的贡献有限
   - 但 enemy 辅助头的 MSE loss 本身可能更高（因为更难预测），乘以相同系数后，实际梯度贡献可能大于 own 辅助头
   - 在 value encoder 侧，由于 `vf_coef=0.05`，辅助损失对 value 学习的相对影响更大（见线索 05 的分析）

5. **对比 aux_base_dmg 的处理**：注意到 `aux_base_dmg_coef=0.0003` 与 `aux_enemy_base_dmg_coef=0.01` **不同**，enemy base damage 的权重是 own 的 **33 倍**。这说明设计者已经意识到 enemy/own 预测难度不同，但在 tower 和 gold 辅助头上没有应用相同的逻辑。

### 结论：**确认**

enemy 辅助标签的损失权重与 own 相同，但预测难度确实不同。观测中敌方信息更稀疏，导致 enemy 辅助头的预测误差更大、梯度噪声更高。值得注意的是，`aux_base_dmg_coef` 和 `aux_enemy_base_dmg_coef` 已经做了差异化处理，但 tower 和 gold 的系数没有，存在不一致。

### 修复建议

1. **降低 enemy 辅助头权重**：将 `aux_enemy_tower_coef` 从 0.005 降至 **0.002-0.003**，将 `aux_enemy_gold_coef` 从 0.001 降至 **0.0005**，以补偿更高的预测噪声。

2. **与 base_dmg 系数保持一致的设计逻辑**：既然 `aux_enemy_base_dmg_coef` 比 `aux_base_dmg_coef` 高（0.01 vs 0.0003），说明 enemy base damage 被认为更重要。对于 tower 和 gold，如果认为 own 预测更有价值（因为直接影响决策），则应降低 enemy 系数；如果认为 enemy 预测同样重要，则应保持但接受更高的噪声。

3. **监控 enemy 辅助头 loss**：在训练日志中分别记录 `aux_tower_loss` 和 `aux_enemy_tower_loss` 的值，如果 enemy 侧的 loss 持续显著高于 own 侧，则证实了预测难度差异，应降低其系数。

4. **优先级建议**：中等偏低。系数本身很小（0.005），对总损失的直接影响有限。但与 base_dmg 系数的设计不一致值得关注，建议在下一轮调参时统一处理。

---

## 总结

| 线索 | 结论 | 优先级 | 修复难度 |
|------|------|--------|----------|
| 14 - 对手池淘汰策略单一 | **确认** | 中等 | 中等（需设计多样性指标） |
| 17 - downgrade_tower_penalty=-9.0 过重 | **确认** | 中等偏高 | 低（改配置值即可） |
| 18 - enemy 辅助标签权重与 own 相同 | **确认** | 中等偏低 | 低（改配置值即可） |

三个问题均已确认存在。其中线索 17 的修复最为简单且可能带来明显效果，建议优先处理。线索 14 需要更深入的设计思考，但对手池多样性对自对弈质量有重要影响。线索 18 的影响相对较小，但与 base_dmg 系数的设计不一致值得关注。
