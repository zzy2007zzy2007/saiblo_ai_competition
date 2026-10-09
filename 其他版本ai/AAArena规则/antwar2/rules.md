# AntWar2 — 规则

AntWar2 是 19×19 offset-hex 网格上的双人塔防游戏。P0 与 P1 各有 50 HP 基地，
位置分别为 `(2,9)` 与 `(16,9)`。每轮双方提交**有序操作列表**；蚂蚁移动、攻击、
生成、状态效果和伤害结算由环境完成，策略不能直接指定蚂蚁移动。

> 规则权威是内容哈希固定的官方 C++ 后端；公开 Python SDK 是候选接口权威。
> 本文件已按本仓官方后端核对，并与 `decision_space.yaml` / `replay_format.md` 一致。

## 公开状态

公开 `BackendState` 包含 round、tower delta、ants、双方 coins、bases、武器冷却和
active effects。塔 ID 为全局 ID，必须检查 owner。`state.towers` 的输入是增量流：
`type == -1` 删除该塔，其余记录创建或更新；策略 SDK 维护完整状态。

实时策略读取：

- 基地 HP：`state.bases[player].hp`
- 出兵等级：`state.bases[player].generation_level`
- 蚂蚁生命等级：`state.bases[player].ant_level`
- 金币：`state.coins[player]`

回放 JSON 的 `camps/speedLv/anthpLv` 映射到上述 `bases` 字段（见 `replay_format.md`）。
实时策略没有 `state.camps`。

## 回合流程

每轮双方各提交有序操作列表，按列表顺序在"公开状态 + 同轮已接受前缀"上逐个验证；
通过后环境统一结算蚂蚁移动/攻击/生成/状态效果/伤害。

## 操作与经济

| 码 | 操作 | 参数 | 说明 |
| --- | --- | --- | --- |
| 11 | `BUILD_TOWER` | `(x,y)` | 建造 Basic 塔，无塔型参数 |
| 12 | `UPGRADE_TOWER` | `(tower_id,target_type)` | 升级 |
| 13 | `DOWNGRADE_TOWER` | `(tower_id)` | 降级或拆除 Basic 并按规则退款 |
| 21 | `USE_LIGHTNING_STORM` | `(x,y)` | 超级武器，花费 90 |
| 22 | `USE_EMP_BLASTER` | `(x,y)` | 超级武器，花费 135 |
| 23 | `USE_DEFLECTOR` | `(x,y)` | 超级武器，花费 60 |
| 24 | `USE_EMERGENCY_EVASION` | `(x,y)` | 超级武器，花费 60 |
| 31 | `UPGRADE_GENERATION_SPEED` | — | 花费 200 |
| 32 | `UPGRADE_GENERATED_ANT` | — | 花费 250 |
| — | `HOLD` | — | 空列表在测量中记为 HOLD |

经济：双方初始 50 金币，基础收入每两轮 3。下一座塔造价为 `15 * 3**(n//2)`，
现有塔数 n 为奇数时再乘 2。两级塔升级花费 60/200，基地两类升级花费 200/250。
击杀、突破和降级退款以冻结后端为准。

## 动作合法性

每个操作按列表顺序在"公开状态 + 同轮已接受前缀"上验证；地图边界、所属地块、
占用、塔所有权和升级树、同轮重复使用、EMP、金币、冷却和等级上限都参与合法性判定。
被拒绝的操作记录但不计入策略序列（见 `decision_space.yaml`）。

## 胜负与得分

- 基地降至 0 HP 时失败；同一结算中双方均归零则 **P0 获胜**。
- 第 512 轮仍未结束时，冻结后端依次比较：基地 HP → 击杀敌蚁数 → 较少超级武器
  使用量 → 较低 AI 总耗时；完全相同则 P0 获胜。
- **只有无协议错误、无基础设施故障的终局 `winner` 才是有效结果**
  （对应对战器契约的 `status == complete`）。

## 失败与判负

- 选手超时 / 输出非法动作 / 进程崩溃 → 对战器契约的 `game_error`（选手侧失败）。
- 对战器 / 后端环境故障 → `infra_error`（应重跑，不惩罚选手）。
