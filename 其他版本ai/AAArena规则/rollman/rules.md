# Rollman / 卷王吃豆人 —— 游戏规则

> 本文由本 GamePack 的官方后端源码提炼（`backend/`），
> 以 `core/gamedata.py`（常量）、`core/GymEnvironment.py`（step/reset/render）、
> `core/pacman.py`、`core/ghost.py`、`main.py` 为准，非二次演绎。

## 1. 概述

Rollman 是 THUAI7 的**非对称双人对抗**游戏：一名玩家扮 **吃豆人 (pacman, role=0)**，
另一名扮 **幽灵方 (ghost, role=1，同时操控 3 个幽灵)**。双方在多关卡迷宫中博弈：吃豆人
尽量吃豆、用技能、闯关；幽灵方尽量抓住吃豆人、阻止其吃完豆子。

- 玩家数：**2**，角色 `rollman`(=pacman, judger role 0) / `ghost`(=ghost, judger role 1)。
- 幽灵数量：**3**（`GymEnvironment._ghost_action_space = MultiDiscrete(ones(3)*5)`）。
- 关卡：`MAX_LEVEL = 3`，棋盘尺寸随关卡缩小
  （`INITIAL_BOARD_SIZE` ≈ 39 / 30 / 21）。
- 每关最大回合：`MAX_ROUND = [_, 500, 400, 300]`（第 1/2/3 关）。
- 随机种子：`random_seed` 决定迷宫与豆子布局，保证可复现。

## 2. 操作空间（`core/gamedata.py::Direction`）

一个方向 = 一个整数：

| 值 | 方向 | 位移 (dx,dy) |
| --- | --- | --- |
| 0 | STAY 不动 | (0,0) |
| 1 | UP | (1,0) |
| 2 | LEFT | (0,-1) |
| 3 | DOWN | (-1,0) |
| 4 | RIGHT | (0,1) |

- **吃豆人**每回合提交 **1 个方向整数**（协议 action 字符串形如 `"d"`）。
- **幽灵方**每回合提交 **3 个方向整数**（每个幽灵一个，action 字符串形如 `"d1 d2 d3"`）。

## 3. 回合与协议流程（`main.py`）

1. 后端发座位号：玩家 0 收到 `"0"`、玩家 1 收到 `"1"`。
2. 后端发初始局面（`env.reset()` 的 render）给双方。
3. **角色声明**：每回合每名玩家先发 `{"role": 0|1, "action": "..."}`；后端据首个消息
   确定其 role。若两玩家角色相同 → `Same Role` 判负；后续回合角色不一致 → `Role error`。
4. 每回合：后端按玩家顺序监听两名玩家的动作，再 `env.step(pacman_action, ghost_actions)`
   推进一步，并把 `{"pacman_action", "ghosts_action"}` 结果回传给双方。
5. 关卡结束（吃完豆子或到达该关 `MAX_ROUND`）则进入下一关（新 reset 局面）；
   三关走完或吃豆人被判定终止即终局。

## 4. 棋盘元素（`core/gamedata.py::Space`）

| 值 | 元素 | 说明 |
| --- | --- | --- |
| 0 | WALL 墙 | 不可通行 |
| 1 | EMPTY 空地 | |
| 2 | REGULAR_BEAN 一分豆 | 吃到 +1 分 |
| 3 | BONUS_BEAN 两分豆 | 吃到 +2 分 |
| 4 | SPEED_BEAN 加速豆 | 获得 SPEED_UP 技能 |
| 5 | MAGNET_BEAN 磁铁豆 | 获得 MAGNET 技能 |
| 6 | SHIELD_BEAN 护盾豆 | 获得 SHIELD 技能（可叠加） |
| 7 | DOUBLE_BEAN 双倍豆 | 获得 DOUBLE_SCORE 技能 |
| 8 | FROZE_BEAN 冰冻豆 | 获得 FROZE 技能 |
| 9 | PORTAL 传送门 | 满足条件后可传送 |

## 5. 吃豆人技能（`core/gamedata.py::Skill`，`core/pacman.py`）

`DEFAULT_SKILL_TIME = [8, 8, 8, 8, 2]`，依次对应：

| 技能 | 效果 |
| --- | --- |
| DOUBLE_SCORE(0) | 吃豆得分翻倍（持续 8） |
| SPEED_UP(1) | 每回合移动 2 步（持续 8） |
| MAGNET(2) | 吸取周围 3×3 范围的豆（持续 8） |
| SHIELD(3) | 抵挡一次被抓（可叠加计数） |
| FROZE(4) | 冻结幽灵：接下来幽灵被迫 STAY（持续 2） |

传送门可用阈值 `PORTAL_AVAILABLE = [_, 60, 50]`（按关卡）。

## 6. 计分（`core/gamedata.py` 常量 + `GymEnvironment.step`）

**吃豆人得分**：
- 吃一分豆 +1、两分豆 +2（DOUBLE_SCORE 生效时翻倍）。
- 吃完全部豆子 `EAT_ALL_BEANS = +50`。
- 里程碑：吃豆数达 `PACMAN_HUGE_BONUS_THRESHOLD=100` 时 `PACMAN_HUGE_BONUS=+50`。
- 被幽灵抓住 `EATEN_BY_GHOST = -60`。

**幽灵方得分**（对每个幽灵可叠加）：
- 抓住吃豆人 `EAT_PACMAN = +50`。
- 击破护盾 `DESTORY_PACMAN_SHIELD = +10`。
- 阻止吃豆人吃完 `PREVENT_PACMAN_EAT_ALL_BEANS = +25`。
- 里程碑 `GHOST_HUGE_BONUS_THRESHOLD=5 → GHOST_HUGE_BONUS=+20`。

碰撞：吃豆人撞墙位移记 `PACMAN_HIT_OFFSET=-100` 标记（用于抓捕判定），幽灵撞墙
`GHOST_HIT_OFFSET=-200`。抓捕判定按吃豆人与幽灵路径块的欧氏距离 `<= 0.5`。

## 7. 终局与判胜（`main.py` L505-535）

- **终止**：吃豆人吃完所有豆（`Pacman ate all the beans!!!`）、到时（`time is up`）、
  或选手异常（`Role error` / `Same Role` / `player quit unexpectedly` / 判题错误）。
- **终局得分**：`end_info` 以 **judger 玩家编号** 索引，值为该玩家按其 role 的绝对得分：
  若 `players[0].role == PACMAN`：`end_info = {"0": pacmanscore, "1": ghostscore}`，否则互换。
- **判胜**（本对战器）：`winner_player = 0 if score0 >= score1 else 1`，再映射回该玩家的
  角色名（`rollman`/`ghost`）作为 `EvaluateResult.winner`。

## 8. 异常与三态

- 选手非法/超时/崩溃/角色错误 → 本对战器归 `game_error`（对应后端 `StopReason` 记录）。
- 后端进程/运行环境/依赖(numpy) 故障 → `infra_error`。
- 正常打到终止条件 → `complete`。

## 9. 角色非对称

pacman 与 ghost 的动作维度、目标、计分完全不同，故本插件 `roles_symmetric=False`，
评测时以「同一候选分别扮 rollman 与 ghost」两侧衡量。
