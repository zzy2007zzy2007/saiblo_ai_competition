# Generals / 将军棋战（第 28 届）规则说明

> 本文档从官方后端源码 `games/generals/backend/logic/gamecode_logic/`
> （`main.py` + `logic/` 目录）逐条提炼，以后端实现为准，不作发挥。

## 1. 概述

Generals 是**双人回合制即时战略对抗**（Saiblo `Game.code`）。棋盘为
`15 × 15`（`logic/constant.py`：`row = col = 15`）的格子地图，双方各控制一名
**主将（MainGenerals）**，通过扩张领地、指挥军队、升级科技/将领、释放战法与超级武器
争夺胜利。玩家编号为 `0`（先手）与 `1`（后手），对应角色 `P0` / `P1`。

## 2. 棋盘与地形

每个格子 `Cell` 具有：地形 `type`、归属 `player`（`-1` 无主 / `0` / `1`）、
`army`（驻军数）、可能的 `generals`（将领对象）。

地形（`CellType`，`logic/gamedata.py`）：

| 值 | 名称 | 说明 |
| -- | ---- | ---- |
| 0 | PLAIN 平原 | 可自由通行 |
| 1 | BOG 沼泽 | 若玩家未点出「免疫沼泽」科技，每回合己方沼泽格 `army-1`，减到 0 且无将领则变无主（`gamestate.update_round`）|
| 2 | MOUNTAIN 山脉 | 未点「攀岩」科技不可进入（军队移动/将军移动/突袭/传送均判非法）|

地形生成比例：沼泽 `bog_percent = 0.15`，山脉 `mountain_percent = 0.4`
（`logic/gamedata.py: init_cell_type`），其余为平原。`main.py: update_map` 会做地形
平滑（孤立山脉退化为平原，被平原包围不足则升为山脉）。

## 3. 初始布局（`logic/gamestate.py: init_generals`）

- 随机生成两名主将，要求曼哈顿距离 `> 18`；若二者不连通则强制打通路径。
- 生成 `subgen_num = 4` 个**中立副将（SubGenerals）**，初始驻军 `random(10,20)`。
- 生成 `farmer_num = 10` 个**中立油井 / 农民（Farmer）**（其中 3 个刷在山脉上），
  初始驻军 `random(3,5)`。
- 双方初始金币 `coin = [40, 40]`（`main.py` 覆盖 `GameState` 默认值）。
- 初始科技等级 `tech_level = [[2,0,0,0], [2,0,0,0]]`：分别是
  行动力 / 攀岩 / 免疫沼泽 / 超级武器解锁。
- 初始每回合可移动军队步数 `rest_move_step = [2, 2]`（等于行动力科技值）。

## 4. 将领类型（`logic/gamedata.py`）

| 类型码 | 类 | 说明 |
| ------ | -- | ---- |
| 1 | MainGenerals 主将 | 每方唯一核心。被消灭即败。可升级、放技能、每回合产兵 |
| 2 | SubGenerals 副将 | 中立时可花 50 金币「召唤」为己方；可升级、放技能、产兵 |
| 3 | Farmer 油井/农民 | 归己方后每回合产**金币**（等于其 produce_level）；只能升级防御/产出，不能升机动、不能放技能 |

将领属性：`produce_level`（产出）、`defense_level`（防御）、`mobility_level`（机动，
影响 `rest_move`）、`skills_cd`（5 个技能冷却）、`skill_duration`（3 个增益/减益持续）。

## 5. 回合结构（`main.py` 主循环）

一局按「玩家 0 行动 → 玩家 1 行动 →（后手结束时）`update_round` 推进回合」循环：

1. 后端把当前玩家视角的地图/操作信息发给该玩家（`send_round_info`，`listen=[player]`）。
2. 该玩家在**一次回复**里发送**一整串命令**（多行，每行一个操作），以单独一行 `8`
   结束（`finish_and_send_our_ops`）。命令按顺序即时执行（`execute_single_command`）。
3. 玩家 1（后手）操作结束后调用 `update_round`：结算产兵、产币、沼泽减兵、超级武器
   持续效果、CD 与增益递减、`round += 1`。
4. 每次执行完一步都检查 `is_game_over`；已分胜负立即终局。

`update_round`（`logic/gamestate.py`）细节：
- 主将格 `army += produce_level`；己方副将格 `army += produce_level`；
  己方农民（油井）给玩家 `coin += produce_level`。
- **每 10 回合**（`round % 10 == 0`）所有己方格 `army += 1`。
- 己方沼泽格（未免疫）`army -= 1`。
- 超级武器持续结算（见 §8），CD/duration 递减，`rest_move_step` 重置为行动力科技值。

## 6. 军队移动与战斗（`logic/movement.py`, `logic/computation.py`）

`army_move(location, direction, num)`：从己方格子沿方向移动 `num` 兵。
- 合法性：不越界、格子归己方、`rest_move_step[player] > 0`、`army > 1`、`num > 0`、
  目标非山脉（或已点攀岩）。`num` 上限为 `army - 1`。
- 目标己方：直接叠加驻军。
- 目标敌方/无主：以 `攻击力 attack`、`防御力 defence` 结算
  `vs = num*attack - target.army*defence`：`vs>0` 占领并留 `ceil(vs/attack)` 兵；
  `vs<0` 防住，守军剩 `ceil(-vs/defence)`；`vs==0` 归中立、清零。占领带将领的格子会
  夺取该将领。
- 每次军队移动消耗 `rest_move_step[player] -= 1`。

`attack`/`defence` 由 `computation.py` 结合将领等级、增益（统率/防御/减益战法）计算。

## 7. 将领移动 / 升级 / 战法

- **将军移动** `command 2`（`general_move`）：目标格无将领、BFS 最短路 ≤ 将领 `rest_move`。
- **将领升级** `command 3`（`logic/upgrade.py`）：`type=1` 产出 / `2` 防御 / `3` 机动。
  不同将领类型有不同金币阶梯（主将花费为副将的一半）。农民不能升机动。
- **战法** `command 4`（`logic/general_skills.py`，`SkillType`）：
  | 技能 | 编号(参数) | 消耗 | CD | 效果 |
  | ---- | --- | --- | --- | ---- |
  | 突袭 SURPRISE_ATTACK | 1 | 20 | 5 | 将领带兵瞬移到 2 格内目标并战斗 |
  | 击破 ROUT | 2 | 15 | 10 | 对 2 格内目标格削减最多 20 兵 |
  | 统率 COMMAND | 3 | 30 | 10 | 10 回合攻击增益 |
  | 防御 DEFENCE | 4 | 30 | 10 | 10 回合防御增益 |
  | 弱化 WEAKEN | 5 | 30 | 10 | 10 回合对敌减益 |
  （常量见 `constant.py`：`tactical_strike/breakthrough/leadership/fortification/weakening`。）

## 8. 科技与超级武器

**科技升级** `command 5`（`logic/upgrade.py: tech_update`，参数 = 科技编号 1..4）：
- 1 行动力：`army_movement_T1=80` → +1 步；`T2=150` → +2 步。
- 2 攀岩：`mountaineering=100`，可进入山脉。
- 3 免疫沼泽：`swamp_immunity=75`。
- 4 解锁超级武器：`unlock_super_weapon=250`，解锁后 `super_weapon_cd = start_cd(10)`。

**超级武器** `command 6`（`logic/super_weapons.py`，需已解锁且 CD=0，使用后 CD=`use_cd(50)`）：
- 1 核弹 NUCLEAR_BOOM（`bomb`）：中心 3×3 范围，主将格兵力减半，其余格清零、将领消灭。
- 2 攻击强化 ATTACK_ENHANCE（`strengthen`）：范围内增益，持续 5 回合。
- 3 时空传送 TRANSMISSION（`tp`）：把起点驻军（留 1）传送到目标空格；落点 2 回合眩晕。
- 4 时间停止 TIME_STOP（`timestop`）：范围内 10 回合无法操作。

**召唤副将** `command 7`（`logic/call_generals.py`）：花 50 金币，把己方领地上的一个
中立副将格变为己方副将。

## 9. 胜负判定（`main.py: is_game_over`）

按优先级：
1. 某方主将（MainGenerals）被消灭 → 对方获胜（`return 存活方`）。
2. 双方主将都在且 `round <= 500` → 继续（`return -1`）。
3. `round > 500` → 依次比较：总兵力 → 领地格数 → 金币，多者获胜；全相等则玩家 0 胜。

**非正常终局**：某方选手崩溃 / 超时 / 输出非法命令 / 主动认输（发送 `9`）→ 对方获胜，
`end_state` 标记该方为 `RE`（运行错误）/`TLE`/`OLE`/`IA`（非法操作）。

后端终局通过 `send_game_end_info(end_info, end_state)` 输出：
`end_info = {"0": s0, "1": s1}`，其中 `s = 1` 为胜、`0` 为负（分高者胜）；
`end_state = ["OK"|"RE"|"TLE"|"OLE"|"IA", ...]` 标记双方进程状态。

## 10. 隐藏信息

对局为**完全信息**回合制：后端每回合把双方当前视角的完整地图（`Cells`/`Generals`/
`Coins`/`Tech_level`/`Weapons` 等）发给对应玩家，随后转发对手上一回合的操作序列。
玩家凭本地 SDK 复算状态。没有战争迷雾式隐藏。
