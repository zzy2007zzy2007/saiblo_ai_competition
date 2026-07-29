# 意图解码：超级武器金币不足时自动拆塔

## 问题

当前 `class_mask` 在金币不足时把超级武器类掩码掉，模型不能选该类。结果：

1. 模型为了始终保持有金币放闪电，不敢花钱建塔
2. 任何建塔尝试都会降低放闪电概率 → 胜率下降 → 淘汰
3. 进化路径被"建塔过渡期低胜率"切断，困在"只放闪电"的局部最优

## 思路

把超级武器类从"放武器"重新定义为"**想要放武器**"。解码器负责：
- 金币够 → 放武器
- 金币不够 → 自动拆塔回收金币

模型只需要学"什么时候武器好"，不需要学"怎么凑钱"。

## 兼容性

默认开启。旧 checkpoint 加载后解码器自动应用新逻辑——模型结构不变，只有 decode 行为变化（金币不足时拆塔而非 HOLD），不需要额外传参。

## 改动点

### 1. `constants.py` — 超级武器成本

| 类 | 动作 | 成本 |
|---|---|---|
| 17 | 闪电 | 90 |
| 18 | EMP | 135 |
| 19 | Deflector | 60 |
| 20 | Evasion | 60 |

基地升级（21/22）成本过高（200/250），不纳入意图解码。

### 2. `decoder.py` — class_mask 不再检查金币

`make_class_mask` 中的超级武器检查只保留冷却，移除金币检查：

```python
# 改前：检查冷却 AND 金币
def _check_super_weapon_valid(state, player, ch):
    sw = CHANNEL_TO_SUPER_WEAPON[ch]
    stats = SUPER_WEAPON_STATS[sw]
    return state.weapon_cooldowns[player, sw] == 0 and state.coins[player] >= stats.cost

# 改后：只检查冷却（金币不足时走意图解码）
def _check_super_weapon_valid(state, player, ch):
    sw = CHANNEL_TO_SUPER_WEAPON[ch]
    return state.weapon_cooldowns[player, sw] == 0
```

### 3. `decoder.py` — `decode_head` 超级武器分支

```python
if 17 <= class_id <= 20:
    sw_type = CHANNEL_TO_SUPER_WEAPON[class_id]
    op_type = SUPER_WEAPON_TO_OP_TYPE[sw_type]
    cost = SUPER_WEAPON_STATS[sw_type].cost

    if state.coins[player] < cost:
        # 金币不足：尝试拆塔凑钱
        tower = _find_tower_to_downgrade(state, player)
        if tower is not None:
            return Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
        return None  # 无塔可拆 → HOLD

    # 金币够，正常放武器
    pos_mask = position_mask[class_id]
    if not pos_mask.any():
        return None
    channel_map = action_map[class_id]
    masked_map = np.where(pos_mask, channel_map, -np.inf)
    x, y = np.unravel_index(np.argmax(masked_map), masked_map.shape)
    return Operation(op_type, int(x), int(y))
```

### 4. 拆塔选择

使用模型自己的 downgrade action_map（class 16）决定拆哪座塔，不依赖手写规则：

```python
downgrade_map = action_map[16]
pos_mask16 = position_mask[16]
if pos_mask16.any():
    masked = np.where(pos_mask16, downgrade_map, -np.inf)
    x, y = np.unravel_index(np.argmax(masked), masked.shape)
    tower = state.tower_at(int(x), int(y))
    if tower is not None and tower.player == player:
        return Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
return None  # 没有可拆的塔 → HOLD
```

这样拆塔策略完全由模型自身的 downgrade 判断决定，不需要辅助函数。

## 对现有系统的影响

| 组件 | 影响 |
|---|---|
| 训练数据 | 不影响（不改数据收集） |
| 模型 | 不影响（不改模型结构） |
| 现有 checkpoint | 加载后自动生效（只改了解码器） |
| 其他动作类 | 不受影响 |
| 对抗测试 | 对手也能用这个逻辑（改的是 decoder，双方共享） |

## 局限性

- Ant-Game 特定的修改
- 21/22（基地升级）成本太高，不适合意图解码
- 假设场上至少有一座可拆的塔
