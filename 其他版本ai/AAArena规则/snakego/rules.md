# SnakeGo / 贪吃蛇棋 —— 游戏规则

> 本文由官方后端源码提炼（`games/snakego/backend/logic/gamecode_logic/`），
> 以 `logic/constants.py`、`logic/spawn.py`、`logic/operate.py`、`logic/result.py`、
> `logic/infrastructure.py`、`main.py` 为准，非二次演绎。

## 1. 概述

SnakeGo 是 THUAC2022 的双人对抗游戏。两名玩家（阵营 `0` / `1`）在一张矩形网格上
各操控若干条蛇，通过移动固化领地、发射穿墙、分裂繁殖，在回合上限内争夺**更大的
领地与蛇身覆盖面积**。终局按双方占据的格子数（蛇身 + 己方墙体）折算总分，分高者胜。

- 玩家数：**2**（本对战器角色标识 `P0` = 阵营 0，`P1` = 阵营 1）。
- 棋盘：`length × width` 网格，默认 `16 × 16`（`GameConfig`，`spawn.py`）。
- 回合上限：`max_round`，默认 `512`（`GameConfig`）。
- 随机种子：`random_seed` 决定道具生成，保证可复现（`spawn.py` 的 `random.seed`）。

## 2. 回合结构（`main.py::Logic.run`）

游戏以「大回合（turn）」推进，每个大回合内两名玩家依次行动（`for current in range(2)`）：

1. **回合预处理** `round_preprocess(ITEM_EXPIRE_TIME)`：处理道具的产生、被拾取、过期
   （`ITEM_EXPIRE_TIME = 16`，见 `constants.py`）。
2. **轮到某玩家**：后端发出监听（`send_normal_round_msg(game_round,[current],[],[],time)`），
   本地 AI 的思考时限字段为 `3`（判题层参数）。
3. **该玩家在本回合可连续操作多条/多次蛇**：后端进入内层 `while True`，反复
   `read_operation` → `apply` → 把操作广播给双方，直到该玩家当前回合无可动蛇
   （`next_snake == -1`）为止。**这意味着一个回合提示后可能有多次操作往返**，是本游戏
   与「一问一答」类游戏的关键差异。
4. 两名玩家都行动完毕后 `settle_round` 结算，并判断是否继续。

`next_player` 中，仅当轮回到玩家 0 时 `turn += 1`；`turn > max_round` 或双方蛇全灭即终局。

## 3. 出生与自动生长

- 初始每方一条蛇，蛇 `id` 与 `camp`（阵营）见 `infrastructure.py`。
- **自动生长期**：`turn <= auto_growth_round`（`auto_growth_round = 8`）且 `camp == id`
  时，蛇每次移动自动加长（`move` 中 `auto_grow`）。

## 4. 操作类型（`operate.py::Controller.apply_single`）

玩家每次提交**一个整数操作码 `op ∈ [1,6]`**（见协议章节）。后端映射如下：

| 操作码 | 含义 | 内部 type | 说明 |
| --- | --- | --- | --- |
| `1` | 向 **下** 移动（dx=+1,dy=0） | move, direction 0 | `dx=[1,0,-1,0], dy=[0,1,0,-1]`，direction = op-1 |
| `2` | 向 **右** 移动（dx=0,dy=+1） | move, direction 1 | |
| `3` | 向 **上** 移动（dx=-1,dy=0） | move, direction 2 | |
| `4` | 向 **左** 移动（dx=0,dy=-1） | move, direction 3 | |
| `5` | **发射（fire）** | type 2 | 消耗一个 type=2 道具，沿蛇头方向打穿一整列/行的墙 |
| `6` | **分裂（split）** | type 3 | 把当前蛇从中点拆成两条 |

### 4.1 移动 `move`

- 蛇头按方向前进一格；`length_bank>0` 时尾部保留（增长），否则尾部收缩。
- **撞边界或撞墙**（越界或 `wall_map[x][y] != -1`）：该蛇死亡（`invalid move`）。
- **禁止回头**：长度 > 2（或长度 = 2 且处于生长态）时，走向脖子格触发
  `IA`（非法动作）"DON'T TURN BACK!"。
- **自咬固化（solidify）**：蛇头进入自身身体格时，围成的闭合区域被 floodfill
  判定为内部领地并**固化为己方墙**（`set_wall(..., player, 1)`），其中若圈住敌方
  蛇则连带击杀；本蛇随后消失。这是主要的**圈地得分**手段。
- 撞到别的蛇身：本蛇死亡（`invalid move`）。
- 途经道具格自动拾取（`get_item`）。

### 4.2 发射 `fire`（op=5）

- 需要蛇身长度 > 1，且蛇持有的**首个道具为 type=2**（否则 `IA`/无道具报错）。
- 沿蛇头相对脖子的方向，把射线路径上的墙**清除**（`set_wall(walls,-1,-1)`）。

### 4.3 分裂 `split`（op=6）

- 同一阵营蛇数达到 4 条时禁止再分裂（`Split too much!!!`，`IA`）。
- 蛇长度 <= 1 不能分裂（`Split length too short!!!`，`IA`）。
- 从中点拆成两条：`head` 保留原蛇，`tail`（反向）生成新蛇；道具按类型分配
  （type=0 长度道具给尾，type=1 丢弃，其余给头）。

## 5. 道具（`spawn.py`）

- 道具按随机种子在开局批量生成，每个道具有 `x, y, type, time, param`。
- `type=0`：长度道具，拾取后 `length_bank += param`（蛇变长）。
- `type=2`：发射道具（供 op=5 使用），`param` 常与 `max_round` 相关。
- 道具在场存活 `ITEM_EXPIRE_TIME` 个 turn（`round_preprocess` 中按 `time` 判过期），
  被蛇拾取后按 `item.param` 决定其在蛇身上的保留时长。

## 6. 计分与胜负（`result.py::settle_round`）

逐格扫描棋盘：

- 有蛇的格子：该蛇阵营 `+2`。
- 空墙 `wall_map == 0`（阵营 0 的墙）：`score[0] += 2`。
- 墙 `wall_map == 1`（阵营 1 的墙）：`score[1] += 2`。
- **首拾奖励**：最先拾到道具的阵营额外 `+1`（`first_get_player`）。

**终止条件**：双方蛇全灭（`snake_num == [0,0]`）或 `turn > max_round`。

**判胜**（`main.py::run` 终局）：`winner = 0 if score[0] >= score[1] else 1`，即
**平分判阵营 0（P0）胜**。终局分数以 `end_info = {"0": score0, "1": score1}` 写入回放
与终局消息。

## 7. 非法/异常处理（`communication.py::ErrorType`）

- `IA`（非法动作，0x11）：回头、无道具发射、超量分裂等 → 判对方胜，违规方 `end_state=IA`。
- `PE`（格式错误，0x12）：操作码非 `1..6` 或无法解析 → "Invalid message format"。
- `RE`/`TLE`：选手崩溃/超时（判题层）。
- `INTERNAL_ERROR`（0x20）：后端异常。

本对战器把上述任何一方选手侧异常/超时/崩溃归为 `game_error`；后端进程或运行环境
故障归为 `infra_error`；正常打到终止条件为 `complete`。

## 8. 角色对称性

两名玩家规则完全对称，但**出生位置与行动先后**（阵营 0 先手、平分判 0 胜、首拾 +1）
带来轻微先手/位置差异，故本插件声明 `roles_symmetric=False`，评测时应做左右手（P0/P1
互换）平衡。
