# Miracle 回放格式 / 通信协议

> 从后端 `backend/main.py` 的 `send_media_info` / `get_media_info` /
> `logic_convert_byte` 与 `evaluator/replay.py` 提炼。评测器把回放写入
> `data/miracle-matches/<candidate>/<opponent>/<role>/seed-<n>/replay.bin`
> （`data/` 已 gitignore）。

## 1. 回放文件（replay.bin）

回放是一串**大端有符号 int32**，每 **7 个整数**为一条“媒体记录”。

### 1.1 头部（第 1 条记录）

```
[0, 0, 0, map_type, day_time, 0, 0]
```

- `map_type ∈ {0,1}`：地图类型；`day_time ∈ {0,1}`：昼夜。

### 1.2 事件记录（其后每条）

```
[round, event, payload...]         # 不足 7 个用 0 补齐；超过则跨多条 7 元组
```

`event` 到名称的映射（`get_media_info` 的 `event_names`）：

| event | 名称 | 主要 payload |
|------|------|------|
| 1 | TurnStart | camp |
| 2 | TurnEnd | camp |
| 3 | Spawn | type(+10*camp), level, posX, posY, id |
| 4 | Move | id, destX, destY |
| 5 | Attack | id1, id2 |
| 6 | Damage | targetId, sourceId, damage, damageType |
| 7 | Death | id |
| 8 | Heal | targetId, sourceId, heal |
| 9 | ActivateArtifact | camp, name(+10*camp), 目标坐标/单位 |
| 10 | **GameEnd** | winner, ... |
| 11 | GameStart | camp, artifact0, creature1..3 |
| 12 | BuffAdd | id, buffType |
| 13 | BuffRemove | id, buffType |
| 14 | Attacking | id1, id2 |
| 15 | Attacked | id1, id2 |
| 16 | Leave | id, posX, posY |
| 17 | Arrive | id, posX, posY |
| 18 | Summon | type(+10*camp), level, posX, posY |

- 生物类型编码：`creature_names.index(type) + 10 * camp`（个位为类型，十位为阵营）。
- 神器编码：`artifact_names.index(name) + 10 * camp`。
- 伤害类型：`["", "Attack", "AttackBack", "VolcanoDragonSplash", "InfernoFlameActivate"]`。

### 1.3 终局与回合数

`event == 10`（GameEnd）记录形如 `[round, 10, winner, 0, 0, 0, 0]`：
- `winner ∈ {0,1}`：胜者阵营；
- `round`：终局回合号，评测器以此作为 `rounds`。

解码器 `evaluator/replay.py::decode_replay` 返回 `ReplayReport(map_type, day_time,
records, winner, rounds)`。

## 2. 传输协议（Saiblo judge）

评测器 `evaluator/arena.py` 用 Python 实现裁判转发层，衔接 Python 后端与两名选手。

### 2.1 后端 → 裁判（后端 stdout）

帧格式：`[4B big-endian 长度][4B big-endian goal][JSON body]`（本游戏 goal 恒为 -1）。
body 的三种形态：

- **计时通知**：`{"state":0, "time":<秒>, "length":<字节上限>}` —— 裁判忽略、读下一帧。
- **下发局面**：
  ```json
  {"state":s, "listen":[p], "player":[p], "content":["<内层长度前缀的 JSON>"]}
  ```
  裁判把 `content[0]` 的字节**原样**写入玩家 `p` 的 stdin（其内已含选手侧长度前缀：
  前 6 位十进制表示长度），随后从玩家 `p` 读取一帧回复。
- **终局**：`{"state":-1, "end_info":"<JSON {\"0\":s0,\"1\":s1}>"}` —— 裁判据此裁决胜负。

### 2.2 裁判 → 后端（后端 stdin）

帧格式：`[4B big-endian 长度][JSON body]`，body 为：
```json
{"player": p, "content": "<选手提交的操作 JSON 字符串>"}
```
（异常上报可用 `player=-1`、`content` 内含 `{"error":0/1,"player":p,"state":s}`。）

### 2.3 选手 ↔ 裁判

- 裁判 → 选手：直接写后端给出的 `content` 字节（选手 `read_opt` 读前 6 字节十进制
  长度再读该长度的 JSON）。
- 选手 → 裁判：`[4B big-endian 长度][操作 JSON]`（选手 `send_opt`）。

## 3. 传输事件日志（transport-events.jsonl）

评测器另写一份便于排障的 JSONL（每行一个事件：`send_init` / `game_packet`
或 `timing` / `end`），以及三方 stderr 日志 `*.backend/p0/p1.stderr.log`。
均落在与 `replay.bin` 同级的对局目录内（`data/`，gitignore）。
