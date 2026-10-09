# AntWar2 — 回放字段说明

承接 `docs/replay-format-spec.md` 的通用外壳，补充 AntWar2 特有字段语义。
阅读要点承自 B 的 `replay_skill.md`。

## public_state 关键字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `round` | int | 当前回合 |
| `towers` | delta 流 | 塔增量：`type == -1` 删除该塔，否则按**全局塔 ID** 创建/更新；须检查 owner |
| `ants` | list | 蚂蚁列表（位置/HP/所属） |
| `coins` | dict[player,int] | 双方金币 |
| `bases` | dict[player,obj] | 基地：`hp` / `generation_level` / `ant_level` |
| `cooldowns` | dict | 各超级武器冷却 |
| `active_effects` | list | 生效中的状态效果 |

回放 JSON 的 `camps/speedLv/anthpLv[player]` 分别映射到
`bases[player].hp / generation_level / ant_level`（见 `decision_space.yaml` 的 `replay_mapping`）。
实时策略**没有** `state.camps`。

## action 编码

编码见 `rules.md` 的操作表（11 建塔 / 12 升级 / 13 降级 / 21–24 超级武器 / 31–32 升级 / 空=HOLD）。
每轮 `op0/op1` 为该角色**同轮被接受的有序原子操作**；空列表记为 HOLD。

## 角色可见性

公开状态对双方对称可见（无隐藏底牌）；不对称仅体现在**起始基地位置**
（P0 `(2,9)` / P1 `(16,9)`）。回放只暴露公开操作和公开效果，**不暴露意图**。

## 回放字段 → decision_space 状态映射

1. 重建 tower delta：`type == -1` 删除，否则按全局塔 ID 更新。
2. 把 `op0/op1` 解码为同轮接受的原子操作；空列表为 HOLD。
3. 将 `camps/speedLv/anthpLv` 规范化为 `base_hp/generation_level/ant_level`。
4. 记录动作前金币、基地 HP、塔、蚂蚁、冷却、坐标和合法操作类型。

## 即时效果 vs 延迟效果

区分下一公开状态的**即时效果**与若干轮后的**延迟效果**（如蚂蚁行进、状态效果衰减）。
每个因果判断须引用 `match_id:state_id` 或连续窗口；无引用的因果解释只能标为假设。

## 阅读注意（承 replay_skill）

- 不得从行为臆测隐藏目标；不得按 seed、对手身份或回放 ID 写策略。
- 被拒绝动作记录但不计入策略序列。
- Replay 只用于诊断；策略晋升须通过未修改对手的自适应实战。
