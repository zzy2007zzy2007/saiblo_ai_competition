# Miracle / 神迹之战（24 届）规则

> 本文档从本 GamePack 的后端源码 `backend/`
> （`main.py` / `logic.md` / `PlayerLegality/operations.py` / `StateSystem/`）与
> 历史恢复的 `public_sdk/` 提炼，描述评测器实际驱动的真实规则。

## 1. 概述

Miracle 是**双人回合制策略对战**（Saiblo Game.code，编号 24）。两名玩家分属
阵营 0（先手）与阵营 1（后手），在六边形（立方坐标）地图上召唤生物、移动、攻击，
目标是摧毁对方的“神迹（Miracle）”或在回合上限时以得分取胜。

- 玩家数：2（角色 `P0` = camp 0，`P1` = camp 1）。
- 先后手：先手为 camp 0；地图对两侧镜像，规则对称（`roles_symmetric: true`），
  评测仍建议 side-balanced（两个 seed 互换先后手）。
- 回合上限：`MAX_ROUND = 100`。
- 每步 AI 决策时限：`AI_TIME = 3` 秒（后端 `send_init` 通知）。

## 2. 对局流程

后端 `Game.start()` 的真实流程：

1. **初始化（init）**：裁判把 `{"replay":<路径>, "player_list":[1,1]}` 发给后端。
   `player_list` 中 `1` 表示该位是正常参赛 AI（`0`=进入失败判负，`2`=观众播放器）。
2. **选卡（select_cards）**：依次让 camp 0、camp 1 发送 `init` 操作，选定
   **1 张神器 + 3 张生物**（见 §5 合法性）。
3. **主循环**：交替进行玩家回合，直到分出胜负或达到回合上限：
   - `change_round`：切换当前行动方 `listen`，`_round += 1`；
   - `get_round_ope`：反复把当前局面发给当前玩家、读取其一个操作、判合法并结算，
     直到玩家发送 `endround`（或投降 / 超时 / 异常）。
   - 每步操作后后端都会广播媒体事件并检查游戏是否结束。

一个玩家回合内可以连续发送**多个操作**（若干次 summon / move / attack / use），
每发一个操作，选手随即读取一份更新后的局面（`update_game_info`），最后以
`endround` 结束本回合。

## 3. 地图与坐标

- 采用**立方坐标** `(x, y, z)`，恒满足 `x + y + z == 0`（见 `operations.to_position`）。
- 地图随机 `map_type ∈ {0,1}` 与 `day_time ∈ {0,1}`（写入回放头部）。
- 地图元素：双方神迹（Miracle）、驻扎点（Barrack）、出兵点（Summon Pos）、
  地面/飞行障碍（Obstacle）。距离用立方距离 `cube_distance`。

## 4. 单位与神器

**生物（Creature）**：剑士 Swordsman、弓箭手 Archer、黑蝙蝠 BlackBat、牧师 Priest、
火山之龙 VolcanoDragon、海洋之龙 FrostDragon（衍生：地狱火 Inferno）。
主要属性：`id / camp / level(1..3) / type / cost / atk / hp / max_hp /
atk_range(min,max) / max_move / cool_down / pos / flying / atk_flying /
agility(迅捷) / holy_shield(圣盾) / can_move / can_atk`。

**神器（Artifact）**：圣光之耀 HolyLight、地狱之火 InfernoFlame、阳炎之盾
SalamanderShield。神器有 `Ready / 再装填` 状态与法力消耗，目标类型为 `Pos` 或 `Unit`。

关键机制（详见 `logic.md`）：
- 攻击流程：Attacking → Damage → Attacked（可反击）→ CheckDeath。
- 圣盾：受击时抵消该次伤害并破盾。
- 迅捷（agility）：可在同回合内既移动又攻击。
- 刚召唤的单位当回合不可移动/攻击（未经历 Refresh）。
- 牧师提供治疗/攻击光环；火山之龙攻击有溅射。
- 回合开始玩家恢复法力、生物与神器冷却递减、清空“新召唤”列表。

## 5. 操作与合法性

由 `PlayerLegality` 判定，非法操作会被后端记录并跳过（不结算），选手需自行避免。

- **Init（选卡）**：恰好 1 神器 + 3 生物；生物互不重复；名称均需存在。
- **Summon（召唤）**：持有该生物卡且未冷却；星级 ∈ {1,2,3}；召唤点必须是本方出兵点；
  召唤点无单位冲突；法力 ≥ 花费。
- **Move（移动）**：只能操作己方生物；终点不与其他单位冲突；存在起点到终点的路径；
  路径长度 ≤ 单位 `max_move`；本回合未行动过（除非迅捷）。
- **Attack（攻击）**：只能操作己方生物；不能攻击友军；攻击力 > 0；本回合未行动过
  （除非迅捷）；目标在攻击距离内；目标存活；攻击飞行单位需攻击者 flying 或 atk_flying。
  目标 id 为 `0` 或 `1` 时表示攻击对方**神迹**。
- **Use（使用神器）**：为本方神器；处于 Ready；法力 ≥ 花费；满足神器特殊条件
  （如 InfernoFlame 需在驻扎点/神迹附近且落点无地面单位、非深渊/己方神迹）。
- **EndRound（结束回合）** / **StartRound** / **Surrender（投降）**：无合法性检测。

## 6. 胜负判定

后端 `check_game_end`：当满足以下任一时结束——
- 达到回合上限 `_round >= 100`；或
- 任一方神迹 `hp <= 0`。

结束时调用 `end(score0, score1)`：
- **平局处理**：若 `score0 == score1`，给后手 `score1 += 1`（保证不平局）。
- **胜者**：得分高的一方（`winner = 0 if score0 > score1 else 1`）。
- **投降/对手异常/超时判负**：对方得分 +1 并立即结束。

评测器据后端终局 `end_info = {"0": score0, "1": score1}` 裁决：
`winner` = 分高者对应角色；`scores = {P0: score0, P1: score1}`；`rounds` 从回放的
`GameEnd` 事件（或最大回合号）读取。

## 7. 通信协议（评测器实现）

后端采用 **Saiblo judge 协议**（长度前缀 stdio），评测器 `evaluator/arena.py`
以 Python 实现裁判转发层，详见 `replay_format.md` 与 `decision_space.yaml`。
