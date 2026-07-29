# 意图解码：闪电类自动拆塔凑金币

## 问题

模型只放闪电不打其他策略，因为：
1. 任何建塔→拆塔→闪电的尝试，在不够金币放闪电的过渡态中输掉
2. 进化路径被低胜率中间态切断
3. 模型被困在"放弃建塔、只放闪电"的局部最优

## 思路

把闪电类（class 17）从"放闪电"重新定义为"**想要放闪电**"。解码器负责：
- 金币够 → 放闪电
- 金币不够 → 自动拆塔回收金币，为放闪电创造条件

这样模型只需要学"什么时候闪电好"，不需要学"怎么凑到闪电的钱"。

## 改动点

只改 `decoder.py` 的 `decode_head` 函数中超级武器分支：

```python
# 当前逻辑（class 17-20）
if 17 <= class_id <= 20:
    # 检查金币是否足够
    if state.coins[player] < 武器.cost:
        return None  # → HOLD（浪费一回合）
    ...

# 改后逻辑（只改 class 17 = Lightning）
if class_id == 17:
    cost = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost
    if state.coins[player] < cost:
        # 找一个己方 tower 拆掉凑金币
        tower = _find_best_tower_to_downgrade(state, player)
        if tower is not None:
            return Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
        return None  # 没塔可拆 → HOLD
    # 金币够，正常放闪电
    ...
```

其他超级武器（EMP、Deflector、Evasion）不改，保持原逻辑。

## 辅助函数

```python
def _find_best_tower_to_downgrade(state, player) -> Tower | None:
    """找到最值得拆的塔：优先拆低级、非关键位置的塔。"""
    best = None
    best_score = -1e9
    for tower in state.towers_of(player):
        score = -tower.level * 10 - state.slot_priority(player, tower.x, tower.y)
        if score > best_score:
            best_score = score
            best = tower
    return best
```

选择逻辑：优先拆等级低的、位置不重要的塔。避免拆关键防线。

## 对训练的影响

- 不需要重新收集数据
- 不需要改模型结构
- 改动只影响 decode，不产生新数据也不影响现有 checkpoint
- 加载旧 checkpoint 后直接生效

## 局限性

这是 Ant-Game 特定的修改，但背后的"意图解码"思想是通用的。

## 实现顺序

1. `decoder.py` 添加 `_find_best_tower_to_downgrade()`
2. 修改 `decode_head` 的闪电分支
3. 跑诊断验证：金币不足时是否自动拆塔而非 HOLD
