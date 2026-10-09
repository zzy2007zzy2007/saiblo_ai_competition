# Generals / 将军棋战（第 28 届）回放格式

> 从 `logic/generate_round_replay.py: get_single_round_replay`、`main.py`（写入逻辑）
> 与 `logic/gamestate.py: trans_state_to_init_json / update_round` 提炼。
> 注意：generals 的回放**不是二进制**，而是**逐行 JSON 文本**。

## 1. 文件形态

回放文件（后端 `init_info["replay"]` 指定路径）是**纯文本**，**每行一个 JSON 对象**。
后端用 `str(dict).replace("'", '"')` 写出（把 Python 字典的单引号换成双引号），
因此每行都是合法 JSON。行分为三类：

1. **初始行**（第 1 行）：整张地图的初始快照（`trans_state_to_init_json`）。
2. **增量行**（每次操作 / 每回合结束）：包含本次变化的 `Cells`、涉及的 `Generals`
   与全局 `Coins`/`Tech_level`/`Weapons` 等。
3. **终局行**：`Action` 为 `[9]` 的说明行（如 `"game end normally"`、`"give up"`、
   traceback 等），由 `main.py: write_end_info` / `quit_running` 写入。

## 2. 单行字段（`get_single_round_replay` 返回结构）

```jsonc
{
  "Round": 12,                 // 当前回合数
  "Player": 0,                 // 操作方；-1 表示系统/回合结算；9 见终局说明
  "Action": [1, 7, 11, 4, 5],  // 本步操作：首位为 command，其余为参数（见 decision_space.yaml）
  "Cells": [                   // 本步发生变化的格子（初始行为全 225 格）
    [[7, 11], 0, 5]            //   [[x, y], player, army]
  ],
  "Generals": [                // 涉及的将领（按 generals_ids 过滤；初始行为全部）
    {
      "Id": 0,
      "Player": 0,             //   归属：-1 中立 / 0 / 1
      "Type": 1,               //   1 主将 / 2 副将 / 3 油井(农民)
      "Position": [7, 11],
      "Level": [1, 1, 1],      //   [产出档, 防御档, 机动档]（后端做过换算）
      "Skill_cd": [0,0,0,0,0], //   五个战法冷却
      "Skill_rest": [0,0,0],   //   统率/防御/弱化 增益剩余回合
      "Alive": 1               //   1 存活 / 0 被消灭（如被核弹清除）
    }
  ],
  "Weapons": [                 // 生效中的超级武器
    {"Type": 1, "Player": 0, "Position": [3, 4], "Rest": 5}
  ],
  "Weapon_cds": [-1, -1],      // 双方超级武器 CD（-1 = 未解锁）
  "Tech_level": [[1,0,0,0], [1,0,0,0]], // 行动力(换算后)/攀岩/免疫沼泽/超武
  "Coins": [40, 40]            // 双方金币
}
```

## 3. 初始行额外字段（`trans_state_to_init_json`）

初始行在上述结构基础上追加 `Cell_type`：长度 `row*col = 225` 的字符串，逐格给出
地形码（`0` 平原 / `1` 沼泽 / `2` 山脉），顺序为按行展开 `board[i][j]`。
初始行的 `Cells` 覆盖全部 225 格，`Generals` 覆盖全部将领。

## 4. Action 语义速查

| Action[0] | 含义 |
| --------- | ---- |
| 1 | 军队移动/攻击 `[1, x, y, dir, num]` |
| 2 | 将领移动 `[2, id, x, y]` |
| 3 | 将领升级 `[3, id, kind]` |
| 4 | 释放战法 `[4, id, skill, (dx, dy)]` |
| 5 | 科技升级 `[5, tech]` |
| 6 | 超级武器 `[6, type, ...]` |
| 7 | 召唤副将 `[7, x, y]` |
| 8 | 回合结束标记（增量/回合快照）|
| 9 | 终局说明行（Content 为文字说明）|

## 5. 胜负与回合数的裁决

对战器（`evaluator/`）**不依赖回放文件裁决胜负**，而是以后端通过 stdio 发出的终局帧
`send_game_end_info` 的 `end_info = {"0": s0, "1": s1}`（分高者胜）为准，并从终局帧携带
的 `Round`（若有）或转发过程中观测到的最大回合数得到 `rounds`。回放文本行仅用于人工
复盘与调试，落在 gitignore 的 `data/generals-matches/` 下。

## 6. 传输帧（stdio 协议，供 evaluator/arena.py 参考）

后端 `main.py` 与评测机之间用长度前缀帧通信：
- **后端 → 评测机**：`struct.pack(">Ii", length, target)`（4B 无符号长度 + 4B 有符号
  target）+ `length` 字节 JSON body。`target = -1` 为广播/路由帧；`target ∈ {0,1}`
  为发给某玩家的回执帧。
- **评测机 → 后端**：`[4B 无符号长度][JSON body]`（`receive_from_judger` 只读长度+body）。
  - init：`{"config": {"random_seed": s}, "replay": "<path>", "player_list": [1, 1]}`
  - 每步回复：`{"player": p, "content": "<选手原始命令串>"}`
- **评测机 ↔ 选手**：选手用 `[4B 无符号长度][msg]` 发送命令串（多行，以 `8\n` 结尾）；
  评测机把后端路由帧 body 里的 `content[i]` 原样（含 `\n`）写给选手 stdin，选手用
  `readline()` 读地图、`input()` 逐行读对手操作。
