# LostSpace 回放格式 / 通信协议

> 从后端 `src/replay.py`（`Replay`）与 `src/communicate.py` 提炼。评测器把回放写入
> `data/lostspace-matches/<match_id>/seed-<n>/replay.json`（`data/` 已 gitignore），
> 并提供 `evaluator/replay.py::decode_replay` 解码。

## 1. 回放文件（replay.json）

回放是一个 **JSON 数组** `lines`：

```
[ birth_positions, round_1, round_2, ..., round_k, score_dic ]
```

### 1.1 首元素：出生点

`birth_positions`：长度 4 的列表，第 i 项为玩家 i 的出生点，坐标经
`__get_pos__` 变换为 `[x-3, y-3, 1]`（回放坐标系相对地图中心）。

### 1.2 中间元素：每个大回合

每个 `round_k` 是一个列表，元素为该大回合内依次进行的**小回合**（每名存活玩家一个）。
每个小回合又是该回合内产生的**动作/事件记录**列表，记录形如：

```json
{"type": "<事件>", "playerid": <玩家>, ...额外字段}
```

主要事件类型（`Replay` 的方法）：

| type | 额外字段 | 含义 |
|------|---------|------|
| move | pos | 移动到（回放坐标） |
| flink | pos | 传送 |
| attack | attack=[起点, 终点] | 攻击 |
| detect | tar_pos | 探查 |
| place_trap | pos, trap_type | 布置陷阱 |
| kit | hp | 使用医疗包后血量 |
| cure | hp | 治疗 |
| hp_update | hp | 血量变化 |
| died | box? | 死亡（可能掉落宝箱位置） |
| regenerate | pos | 复活 |
| keymachine | pos | 开启钥匙机 |
| getkey | keyid | 获得密钥 |
| escaped | — | 逃生成功 |
| ai_error | error_log | 选手异常 |
| escape_capsule | to_escape | 与逃生舱交互 |
| inspect | pos, interprops | 检视交互物 |
| map_update | args | 地图变化（box_disappear / door / trap_trigger / trap_destroy） |

### 1.3 末元素：名次分

`score_dic`：`{"0":s0, "1":s1, "2":s2, "3":s3}`，`4` = 第一名，递减至 `1` = 第四名。

### 1.4 回合数

评测器 `_rounds_from_replay` 统计中间为 `list` 的元素个数即大回合数。
`decode_replay` 返回 `ReplayReport(birth_positions, rounds, turns, score_dic,
winner_player)`。

## 2. 传输协议（Saiblo judge，四人异步双向泵）

后端在**单个玩家回合内多次 `recieve_data`**，且广播/见证消息为**单向通知**（选手
收到但不必回复）。因此评测器 `evaluator/arena.py` 采用**纯异步双向泵**（与真实
Saiblo judge 一致），而非“发一条等一条”。

### 2.1 后端 → 裁判（后端 stdout）

帧格式：`[4B big-endian 长度][4B big-endian goal][JSON body]`（goal 恒为 -1）。
body 形态（`__send__` / `send_init` / `send_end_info`）：

- **计时通知**：`{"state":0, "time":3, "length":2048}` —— 裁判忽略。
- **下发/广播**：
  ```json
  {"state":s, "listen":[curplayer], "player":[p0,p1,...], "content":["<4 位长度前缀 JSON>", ...]}
  ```
  裁判把每个 `content[i]` 的字节**原样**写给玩家 `player[i]` 的 stdin，**不阻塞
  等待**。`listen` 指示“当前回合玩家”，不是“必须回复者”。
- **终局**：`{"state":-1, "end_info":"<JSON score_dic>"}`。

### 2.2 裁判 → 后端（后端 stdin）

帧格式：`[4B big-endian 长度][JSON body]`，body 为：
```json
{"player": p, "content": "<选手提交的操作 JSON 字符串>"}
```
后端 `recieve_data` 会对 `content` 再做一次 `json.loads`。选手异常上报可用
`player=-1`、`content` 内含 `{"error":..,"player":p,"error_log":..}`。

### 2.3 选手 ↔ 裁判

- 裁判 → 选手：写后端给出的 `content` 字节（选手 `receive_data` 读 4 字节十进制
  长度再读该长度的 JSON）。
- 选手 → 裁判：`[4B big-endian 长度][操作 JSON]`（选手 `send_opt`）。裁判据此包成
  `{"player":p,"content":...}` 写给后端。

## 3. 传输事件日志

评测器另写 `transport-events.jsonl`（send_init / timing / player_eof / end）与
四方 + 后端的 stderr 日志 `*.backend/p0/p1/p2/p3.stderr.log`，均落在与
`replay.json` 同级的对局目录（`data/`，gitignore）。
