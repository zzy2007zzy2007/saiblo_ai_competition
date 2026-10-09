# SnakeGo — 回放字段说明

> 提炼自后端 `logic/infrastructure.py::History.dump` 与 `logic/formatter.py::format_to`
> （自定义缩进格式，输出为 JSON 兼容文本），以及 `logic/operate.py`、`logic/result.py`。

## 1. 回放文件如何产生

终局时 `main.py::write_gameover_message` 调用 `History.dump(replayPath)`，用 `format_to`
把 `History` 对象序列化到 `--replay`（本对战器在 init 包里传入的 `replay` 路径）。
输出是缩进美化后的 JSON 文本，可被标准 `json.loads` 解析。

## 2. 顶层结构

```jsonc
{
  "game_config": { "length", "width", "max_round", "random_seed" },
  "item_list":   [ { "x", "y", "time", "type", "param" }, ... ],   // 开局道具全集
  "round_info":  [ RoundInfo, ... ],                               // 每大回合预处理信息
  "operations":  [ [ ProcessedOp, ... ], ... ],                    // 逐次操作序列（按发生顺序）
  "end_info":    { "type", "winner", "score": [s0, s1], "err" }    // 终局裁决
}
```

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `game_config` | dict | 棋盘尺寸、回合上限、随机种子（复现关键） |
| `item_list` | list | 开局生成的道具集合：`type=0` 长度道具 / `type=2` 发射道具；`time` 出现回合，`param` 参数 |
| `round_info` | list | 与大回合一一对应，长度即**进行的大回合数**（本对战器 `rounds` 取此长度） |
| `operations` | list[list] | 外层按行动次序，内层是该次 `apply` 产出的 `processed_operation`；空 `[]` 表示该玩家当回合无可动蛇 |
| `end_info` | dict | `Result` 展开：`type`（NORMAL/PLAYER_ERROR/ILLEGAL_ACTION/INVALID_FORMAT/INTERNAL_ERROR）、`winner`、`score=[s0,s1]`、`err` |

## 3. round_info 字段（`operate.py::RoundInfo`）

| 字段 | 含义 |
| --- | --- |
| `gotten_item` | 本回合被拾取的道具 `[{item, snake}]` |
| `expired_item` | 本回合过期/被删的道具 `[{item, snake}]`（`snake=-1` 为场上过期） |
| `invalid_item` | 无效道具记录 |
| `extra_length` | 长度储备变化 `[{snake, el}]`（`el = snake.length_bank`） |

## 4. operations 内元素（`processed_operation`）

每次 `Controller.apply` 产出一个 dict：

| 字段 | 含义 |
| --- | --- |
| `basic` | `Operation`：`{type, snake, direction, item_id}`，`type` 归一为 1=move/2=fire/3=split，`direction` 为移动方向 0..3 |
| `dead_snake` | 本次操作导致死亡的蛇 id 列表 |
| `should_solid` | 是否发生固化（圈地） |
| `extra_solid_area` | 因固化新增的墙格 `[{x, y}]` |
| `got_item` | 本次拾取的道具 id（-1 无） |
| `extra_length` | 操作后该蛇长度储备 |
| `new_snake_id` | 分裂产生的新蛇 id（-1 无） |

## 5. 终局裁决与 rounds

- **胜者**：以**终局控制消息** `send_game_over` 的 `end_info` 字符串 `{"0":s0,"1":s1}`
  为准（arena 直接解析），`winner = 0 if s0 >= s1 else 1`（平分判 P0）。回放文件里的
  `end_info.score` 与之一致，`end_info.winner` 同为该值。
- **rounds**：`round_info` 列表长度（`replay.round_count`）。本次真跑 rank07(P0) vs
  rank12(P1) 得 `rounds = 512`（打满默认 `max_round`）。

## 6. 坐标系与阵营

- 网格坐标 `(x, y)`，`x ∈ [0, length)`，`y ∈ [0, width)`；方向增量
  `dx=[1,0,-1,0], dy=[0,1,0,-1]` 对应操作码 1/2/3/4。
- `wall_map[x][y]`：`-1` 空、`0` 阵营0 的墙、`1` 阵营1 的墙；`snake_map[x][y]` 存蛇 id。
- 阵营 0 = `P0`，阵营 1 = `P1`。阵营 0 先手且平分判胜，故左右手（P0/P1 互换）不完全对称。

## 7. 解析注意

- 回放为**自定义格式化的 JSON 文本**（`format_to`）：字符串走 `repr(...)[1:-1]`，
  `null/true/false` 已 JSON 化，实测可被 `json.loads` 直接解析。
- Replay 仅用于诊断与状态重建，不得据 seed / 对手身份 / 回放 ID 编写策略。
