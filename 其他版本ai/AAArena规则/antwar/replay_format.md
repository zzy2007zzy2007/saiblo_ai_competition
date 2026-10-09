# AntWar / 蚁国争霸 (27th) —— 回放格式（replay_format.md）

> 从后端 `output.cpp`（`save_data` / `dump_all`）与 `game.cpp`（`dump_round_state` /
> `send_end_info`）提炼。回放由后端在 init 消息给定的 `replay` 路径写出。

## 1. 文件形态

- 回放是一个 **JSON 数组**（`Output::dump_all` 把 `data` 直接序列化），每个元素是
  一回合记录 `round_msg`，按回合顺序追加。

## 2. 每回合记录字段（`round_msg`）

```jsonc
{
  "round_state": { /* 见 §3，该回合结束时的完整局面 */ },
  "op0": [ /* 玩家 0 本回合的操作数组 */ ],
  "op1": [ /* 玩家 1 本回合的操作数组 */ ],
  "seed": 20240117   // 仅第 1 回合记录，本局随机种子
}
```

## 3. round_state 字段（`Output::cur`）

```jsonc
{
  // ⚠️ 官方文档原文：「防御塔**增量**信息，即只包含回合内新建/等级状态发生变化的塔」
  //    （来源：https://docs.saiblo.net/games/antwar/replay-guide.html）
  //    所以 towers **不是**该回合场上的全部塔。要还原「当前所有塔」必须自己按
  //    回合顺序累积，并用 id 覆盖旧记录；直接把某一回合的 towers 当成全量快照，
  //    会得出「对手只有 1~2 座塔」这种完全错误的结论。
  "towers": [
    {"id": 0, "player": 0, "pos": {"x": 5, "y": 9}, "cd": 2,
     "type": 0, "attack": [/* 被攻击蚂蚁 id */]}
  ],
  "ants": [
    {"player": 0, "id": 3, "pos": {"x": 6, "y": 9}, "hp": 10,
     "move": 2, "level": 0, "age": 4, "status": 0}
    // status: Alive=0, Success=1, Fail=2, TooOld=3, Frozen=4
  ],
  "coins": [c0, c1],
  // camps = 大本营血量，开局 [50, 50]（官方文档确认）。
  // 这是本游戏唯一的连续量纲，因此评测器用**终局 camps** 作为该局分数，
  // score_margin = 自己血量 − 对手血量（见 evaluator/__init__.py::_terminal_base_hp）。
  "camps": [hp0, hp1],
  "speedLv": [lv0, lv1],
  "anthpLv": [lv0, lv1],
  "pheromone": [ /* 2 x 19 x 19 信息素矩阵 */ ],
  "winner": -1,       // 该回合胜者玩家 id（-1 未定，0/1 已分胜负）
  "message": "",
  "error": ""         // 可选，出错信息
}
```

## 4. 每回合操作（op0 / op1，`Operation` 序列化）

```jsonc
{"type": 11, "id": -1, "args": -1, "pos": {"x": 5, "y": 9}}
```

- `type` 见 `decision_space.yaml` 的 operation_types；`id`/`args`/`pos` 视操作类型
  取值（无关字段为 -1）。

## 5. 终局消息（`send_end_info`，非回放文件内容，通过 stdio 广播）

```jsonc
{
  "state": -1,
  "end_info": "{\"0\": 1, \"1\": 0}",   // 字符串化 JSON：各玩家得分（1=胜,0=负）
  "end_state": "[\"OK\", \"OK\"]"        // 两名 AI 的结束状态：OK/RE/TLE/OLE/IA
}
```

## 6. 对战器如何解码

- 对战器从终局广播的 `end_info` 判胜：`s0 > s1` → 胜者 `P0`；`s1 > s0` → `P1`；
  相等 → 平局。回合数取自终局前观察到的最大 `state`。
- 回放文件（`decode_replay`）另可从最后一个 `round_state.winner` 独立还原胜者、
  从第 1 回合 `seed` 还原本局种子（供 B 回放技能翻译使用）。
- 契约 `EvaluateResult`：`winner`=角色胜者，`scores={"P0":1,"P1":0}`（或反之/平局
  各 0.5），`rounds`=回合数，`replay_path` 指向该回放文件。
