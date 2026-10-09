# LostSpace / 失落深空（25 届）规则

> 本文档从本 GamePack 的后端源码 `backend/`
> （`src/GameController.py` / `src/communicate.py` / `src/config.py` / `src/replay.py`
> / `src/player.py`）、历史样例 `public_sdk/legacy_main.py` 与同目录的兼容 SDK
> 提炼，描述评测器实际驱动的真实规则。

## 1. 概述

LostSpace 是**四人对抗生存对战**（Saiblo Game.code，编号 25）。四名玩家降落在
一张 `7×7×3`（长×宽×层高）的立体星图上，通过移动、搜集密钥、开启逃生舱逃离，
或战斗淘汰对手。目标是**尽早逃生（名次靠前）**。

- 玩家数：**4**（角色 `P0..P3`，对应后端玩家 id `0..3`）。后端 `game_init`
  硬性断言 `player_num == 4`。
- 出生点：`BirthPos = [(0,0),(6,0),(6,6),(0,6)]`（四角）。
- 大回合上限：`RoundMax = 100`。
- 每步 AI 时限：`send_init(3, 2048)` → 3 秒 / 消息上限 2048 字节。

## 2. 对局流程

1. **初始化**：裁判发 `{"player_num":4, "player_list":[1,1,1,1], "replay":<路径>}`
   （`1`=正常 AI，`0`=进入失败判负，`2`=观众播放器）。后端回 `send_init` 计时通知，
   并 `send_init_mes` 把每个玩家的编号与出生点广播给四人（无人需回复）。
2. **大回合循环**（`big_round`）：每个大回合内四名玩家**依次**行动（`turn`）：
   - 回合开始重置行动力 `sp = SpMax(=1)`，处理复活/获钥/逃生判定；
   - 广播 `roundbegin`（`listen=[当前玩家]`，附带该玩家详细状态与他人简要状态）；
   - 进入 `inround`：反复接收当前玩家操作、结算、`respond`，直到该玩家发 `finish`
     或死亡/进入逃生等待。
   - 每大回合末检查缩圈（`ShrinkTime = [40,50,60,70,80]`），到 80 回合后进入加时赛。
3. **终局**：当四人全部“逃生成功 + 淘汰”结算完毕，或到达回合上限时 `gameover`，
   计算名次并 `send_end_info(score_dic)`。

## 3. 地图与坐标

- 尺寸 `7×7×3`，坐标 `(x, y, z)`，`z` 为层（0 底层 / 2 顶层）。
- 地图由后端用 **ANTLR** 解析 `src/mapconf2.map` 生成（节点/边/门/物资点/钥匙机等）。
- 关键地点：逃生舱 `EscapeCapsulePos=(3,3,0)`、电梯 `ELEVATOR=[(3,0),(5,3),(3,6),(1,3)]`、
  四角钥匙机、物资点（每 `RefreshInterval=8` 回合刷新）。
- 视野：玩家仅能看到可视范围内的他人行动（`witness` / `see` 消息）。
- 缩圈：随回合推进逐层封禁区域，加时赛阶段死亡不再复活。

## 4. 资源与数值

- 生命 `HpMax = 200`；攻击伤害 `AttackHarm = 70`；地雷伤害 `LandMineHarm = 120`；
  医疗包回复 `KitRecovery = 100`。
- 道具（`ToolList`）：`LandMine`（地雷，最多 3）、`Sticky`（黏液陷阱，最多 3）、
  `Transport`（传送，最多 2）、`Kit`（医疗包，最多 2）。陷阱类 = `LandMine`+`Sticky`。
- 交互物（`PropList`）：`Materials`（物资，取道具）、`EscapeCapsule`（逃生舱）、
  `KeyMachine`（钥匙机，取密钥）、`Box`（宝箱，开出密钥）。
- 每回合行动力 `sp = 1`；攻击/使用道具等消耗 `sp`（`AttackCost=1`，`PropSpCost=[1,1,1,1]`）。

## 5. 操作与合法性（`solve`）

当前玩家在自己回合内可连续发送多个操作，每个由 `{"type":..., "action":[...]}`
描述；后端逐个 `respond`。以 `{"type":"finish"}` 结束本回合。

- **move** `["move", [x,y,z]]`：移动到相邻可达格；返回是否成功、新血量、状态、视野。
- **attack** `["attack", [x,y,z], target_id]`：攻击指定坐标上的目标玩家
  （`0<=target<=3` 且 `!= 自己`）。
- **interact** `["interact", Prop, ...]`：与物资/逃生舱/钥匙机/宝箱交互
  （`Box` 开箱取钥匙；`Materials` 需指定 `ToolList` 中的道具类型）。
- **trap** `["trap", TrapType]`：布置陷阱（`LandMine` / `Sticky`）。
- **tool** `["tool", Tool, ...]`：使用道具（`Kit` 治疗；`Transport` 传送到坐标）。
- **detect** `["detect", [x,y,z]]`：探查某格是否有陷阱（有冷却）。
- **finish**：结束当前回合。

非法/格式错误：后端回 `format error` 或 `action.success=false`，该操作不生效；
逃生等待状态下只允许与逃生舱交互。

## 6. 胜负判定（名次分）

`gameover` 计算玩家名次 `player_order`（逃生成功者按逃生先后在前；未结算者按
积分降序，后手优先；失败者在后），并给出名次分：

```
score_dic[str(player_id)] = 4 - 名次索引     # 第一名 4 分，第二名 3 分，……第四名 1 分
```

评测器据终局 `end_info = score_dic` 裁决：
- `scores = {P0:.., P1:.., P2:.., P3:..}`（名次分，4=第一名）；
- `winner` = 名次分最高者对应的角色；
- `rounds` = 回放中大回合数。

## 7. 通信协议与后端依赖

- 后端采用 **Saiblo judge 协议**（长度前缀 stdio）。评测器 `evaluator/arena.py`
  以 Python 实现**四人异步双向泵**裁判层（后端在单回合内多次读取，广播消息为单向
  通知），详见 `replay_format.md` 与 `decision_space.yaml`。
- **后端依赖（评测硬约束）**：地图由 **ANTLR** 解析生成，需要
  `antlr4-python3-runtime==4.9.1`（须与后端自带的 ANTLR 4.9.1 生成代码版本一致，
  否则 ANTLR 会把版本告警打印到 stdout 污染协议流）；且生成代码使用
  `from typing.io import TextIO`，在 Python 3.12+ 已移除。评测器
  `evaluator/runtime.py` 在隔离目录 `data/lostspace-build/` 内安装该运行时并注入
  `typing.io` 兼容 shim（不污染全局解释器）；**引导失败时评测器诚实返回
  `infra_error`**。此外 C++ 选手 SDK 会在运行目录读取 `mapconf2.map`，评测器会把
  后端同一份地图文件布置到每个选手的运行目录。
