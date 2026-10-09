# AquaWar / 鱼洋陷役 —— 回放格式（replay_format.md）

> 从后端 `aqua_sdk.cpp` 的 `save_json()` / `to_json()`（`game.h/game.cpp`）与
> `end()` 提炼。回放由后端在 init 消息给定的 `replay` 路径写出。

## 1. 文件形态

- 回放是一个 **styled-JSON 数组**（人类可读缩进），文件末尾附有若干空行。
- 数组每个元素是**一回合的完整局面快照**（`Game::to_json()`），按对局推进顺序追加；
  数组最后追加一个空的 end 记录（`end()` 中的 `end_replay`）。

## 2. 每个局面记录字段（`Game::to_json`）

```jsonc
{
  "winner": -1,        // 整场当前胜者玩家 id（-1 表示未定）
  "players": [ /* 见 §3 */ ],
  "state": 12,         // 内部回合游标
  "gamestate": 4,      // 阶段：READY=0, PICK=2, ASSERT=3, ACTION=4, END=5
  "cur_turn": 0,       // 当前操作者玩家 id
  "over": false,       // 整场是否结束
  "cnt": 0,
  "score": 1,          // 比分：>0 → P0 领先/胜，<0 → P1，绝对值为小局净胜数
  "rounds": 1,         // 已结束的小局数（三局两胜，最多 3）
  "operation": [ /* 本回合被接受的操作，可选 */ ]
}
```

## 3. players 数组（每名玩家）

```jsonc
{
  "id": 0,
  "type": 1,                 // 1=AI 选手, 2=播放器
  "my_fish": [               // 该玩家本局所选/上场的鱼
    {
      "hp": 400, "atk": 100, "def": 0,
      "type": 0, "id": 3,    // id=鱼种编号(1..12)
      "state": 1,            // READY=0, FIGHT=1, DEAD=2
      "is_expose": false     // 是否已对对手暴露
    }
    // ...4 条
  ],
  "fight_fish": [ /* 上场作战阵型快照 */ ]
}
```

## 4. operation（本回合操作，可选）

- 记录被接受的选手操作及其结算 `ActionInfo`（技能 `skill`、命中 `hit`、被动
  `passive` 的目标与数值），以及断言信息。字段与 SDK `ai_client.hpp` 的
  `parseMyActionInfo` / `parseEnemyActionInfo` 一一对应。

## 5. 终局信息（`end()`）

对局结束时后端向 judger 广播：

```jsonc
{
  "state": -1,                       // 终局标志（对战器据此判定对局结束）
  "end_info": "{\"0\": 100, \"1\": 0}"  // 字符串化 JSON：各玩家天梯分增量
}
```

- `end_info` 中 `"0"`/`"1"` 的分值：胜者 100、负者 0；平局双方 0。

## 6. 对战器如何解码

- 对战器读取回放数组，取**最后一个含 `score` 字段的记录**为终局：
  - `score > 0` → 胜者 `P0`；`score < 0` → 胜者 `P1`；`score == 0` → 平局。
  - `rounds` 直接作为对局回合数（已进行的小局数）。
- 逐 `score` 变化可还原每小局胜者序列（`evaluator/replay.py` 的 `decode_replay`）。
- 契约 `EvaluateResult`：`winner`=角色胜者，`scores={"P0": score, "P1": -score}`，
  `rounds`=小局数，`replay_path` 指向该回放文件。
