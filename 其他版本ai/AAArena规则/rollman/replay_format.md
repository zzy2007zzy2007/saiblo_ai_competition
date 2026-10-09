# Rollman — 回放字段说明

> 提炼自后端 `core/GymEnvironment.py::render(mode="logic")` / `get_return_dict()` 与
> `main.py`（回放写入逻辑）。

## 1. 回放文件如何产生

回放是 **JSONL**（每行一个 JSON 对象），由 `main.py` 写入 init 包传入的 `replay` 路径：

- **首行**：`env.reset()` 后 `get_return_dict()` 的初始局面（完整棋盘）。
- **每回合**：`env.step()` 后 `env.render(mode="logic")` 的 `return_dict`（逐步状态）。
- **关卡切换**：进入新关卡时再写一行新的 reset 局面。
- **末行**：带 `StopReason` 的终局状态（`env.render()` 的 dict + StopReason）。
- 后端异常时，末尾可能追加 `traceback` 文本（非 JSON，解码器会跳过）。

## 2. 初始/关卡局面（`get_return_dict`）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `level` | int | 当前关卡（1..3） |
| `round` | int | 当前回合 |
| `board_size` | int | 棋盘边长 |
| `board` | int[][] | 棋盘格子（`Space` 枚举值，见 rules.md 第 4 节） |
| `pacman_skill_status` | int[5] | 吃豆人五项技能剩余时长 `[DOUBLE, SPEED, MAGNET, SHIELD, FROZE]` |
| `pacman_coord` | [x, y] | 吃豆人坐标 |
| `ghosts_coord` | [[x,y]×3] | 三只幽灵坐标 |
| `score` | [pacman, ghosts] | 双方当前累计分 |
| `beannumber` | int | 本关初始豆子总数 |
| `portal_available` | bool | 传送门是否可用 |
| `portal_coord` | [x, y] | 传送门坐标 |

## 3. 逐步状态（`render(mode="logic")`）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `round` | int | 回合序号（对局进行的回合数取其最大值） |
| `level` | int | 当前关卡 |
| `pacman_step_block` | int[][] | 吃豆人本步路径块（含 SPEED_UP 时的两步）；撞墙点带 `PACMAN_HIT_OFFSET=-100` 偏移标记 |
| `pacman_coord` | [x, y] | 吃豆人新坐标 |
| `pacman_skills` | int[5] | 本步开始时的技能状态快照 |
| `ghosts_step_block` | int[][]×3 | 三只幽灵各自路径块；撞墙点带 `GHOST_HIT_OFFSET=-200` 标记 |
| `ghosts_coord` | [[x,y]×3] | 三只幽灵新坐标 |
| `score` | [pacman, ghosts] | 更新后双方分数 |
| `events` | int[] | 本步事件（`Event`：0 EATEN_BY_GHOST / 1 SHIELD_DESTROYED / 2 FINISH_LEVEL / 3 TIMEOUT） |
| `portal_available` | bool | 传送门是否可用 |
| `StopReason` | str/null | 非终局为 `null`；终局为终止原因文本 |

## 4. 终局裁决与 rounds

- **StopReason** 文本举例：`Pacman ate all the beans!!!`、`time is up`、`Role error`、
  `Same Role`、`player quit unexpectedly`、`Unexpected behavior of player ...`。
- **胜负**：以**终局控制消息** `send_game_end_info` 的 `end_info` 为准（arena 直接解析）——
  `end_info` 以 judger 玩家编号索引，值为该玩家按 role 的绝对得分；
  `winner_player = 0 if end_info["0"] >= end_info["1"] else 1`，再映射回其角色名。
  回放末帧的 `score = [pacman_score, ghosts_score]` 可用于交叉核对。
- **rounds**：`replay.round_count` 取回放中最大的 `round` 字段（跨关卡累计的最终回合数）。

## 5. 坐标与阵营

- 坐标 `(x, y)`；方向位移见 rules.md 第 2 节（`Direction`/`Update`）。
- `board[x][y]` 为 `Space` 枚举值；`0=WALL` 不可通行。
- `rollman` = pacman = judger role 0；`ghost` = 幽灵方（3 只）= judger role 1。角色非对称。

## 6. 解析注意

- 回放为逐行 JSON（`json.dumps(state)+"\n"`），用逐行 `json.loads` 解析；后端异常分支
  可能在文件尾部混入非 JSON 的 traceback 文本，解码器会安全跳过。
- Replay 仅用于诊断与状态重建，不得据 seed / 对手身份 / 回放 ID 编写策略。
