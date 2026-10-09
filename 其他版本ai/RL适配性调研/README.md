# RL 适配性调研：子 agent 原始报告（**未经我审**）

> 建立时间：2026-10-10
> 这里放的是做「AAArena 12 个游戏机制层面 RL 适配性评估」时，**子 agent 自己写出来的原始详报**。
> **结论请以 [`../../docs/12游戏RL适配性评估.md`](../../docs/12游戏RL适配性评估.md) 为准** —— 那份是我逐条读代码裁决后的版本。本目录只是证据留存。

## 文件清单

| 文件 | 来源 | 内容 |
|---|---|---|
| `四游戏机制RL适配性分析.md`（44 KB / 285 行） | 第一轮 D 组（DeepSeek 路由） | dorado / snakego / aquawar / lostspace |
| `legacy_ai9三游戏RL机制评估.md`（20 KB / 117 行） | 第二轮 C 组（Kimi-K3） | pacman / monecraft / lota |
| `map_calc.py`（2.8 KB） | 第二轮 A 组（Kimi-K3） | antwar2 地图几何验算脚本（271 格 / 99 障碍 / 172 可走 / 双方塔区各 33 格） |

> 注：当时我要求子 agent **只读**，但仍有两组写了新文件——这两个 `.md` 和这个 `.py` 就是它们留下的。内容我有选择地核实过（见下）。

## 读之前必须知道的几点

### 1. 报告里的「适配度 X/5」是**被打分者自己的口径**，不要直接采信

两轮对同一游戏给出过不同分数（Pacman 2 vs 4、LostSpace 4 vs 2、Generals 3 vs 2、Rollman 5 vs 4、LOTA 3 vs 2、AntWar2 3 vs 2）。
**除 AntWar2 那处外都是"准则/权重不同"，不是事实分歧。** 最终版已把主表换成「游戏 × 方法族」矩阵。

所以本目录两份报告里的分数**只代表各自作者的口径**：

- `四游戏机制RL适配性分析.md`：Dorado **2/5**、SnakeGo **4/5**、AquaWar **3/5**、LostSpace **4/5**
- `legacy_ai9三游戏RL机制评估.md`：Pacman **4/5**、MoneCraft **3/5**、LOTA **2/5**

### 2. 三处**错误**在裁决/复查中被纠正（本目录报告可能仍带着旧说法）

| 争点 | 错的那一方 | 正确结论 | 裁决依据 |
|---|---|---|---|
| AntWar2 的观测是否含**信息素** | 第一轮 **A 组**（不在本目录，但结论被其他报告引用过） | **AI 模式不下发信息素**（只发 `id + seed`）；人类模式才发 | `comm_judger.cpp:161-187` |
| LOTA 的 `Move` 是**单坐标**还是**路径** | 第一轮 C 组（不在本目录） | **是 4 连通变长路径**，总长 ≤ `sqrt(Speed)` | `Unit::checkMoveCmd()` |
| 那组"**500 局**"数据的归属与游戏 | **我**（主文档里，转述自第一轮 A 组） | 是 **omegafantasy 的**数据，且统计的是他 **AntWar(27届)** 的树搜索 AI —— **不是你的、也不是 AntWar2** | `antwar_behavior_analysis.md:3-5`（明写 AI version = `past_AIs/ANTWar-AI/main.cpp`） |

→ 本目录的 `legacy_ai9三游戏RL机制评估.md` 是**第二轮**报告，前两条它**都是对的**。

### 3. 已被作者自己标注作废的内容

`四游戏机制RL适配性分析.md` 里有一条 `⚠️ 已修正错误（留档备核）`：作者第一遍把 `mapconf2.map` 按 `x y z` 解析，得出"7 层"；正确轴序是 `z x y`，实为 **3 层 / 110 节点 / 148 边**。**以修正后的为准**（这条同类错误我自己也犯过一次——把 Rollman 第三张棋盘算成 23，实为 22）。

### 4. 我核实过的事实（可以当依据用）

- `legacy_ai9三游戏RL机制评估.md`：Pacman 的 `Move` 结构（`sdk/AI.h:96-102`）+ `step∈{1,2}`（`GameInfo.cc:33`）、MoneCraft 的 `Action` 结构 + `BLINK_LIMIT=2.9`（我算术验过半径内 25 格含自身）、LOTA 的 `Map_size=151` + 25 种操作 + `0.1 CPU s/回合`（`ActMakerWeb.cpp:13-16`）、LOTA 的 `Move=路径`（`Unit::checkMoveCmd()`）
- `四游戏机制RL适配性分析.md`：SnakeGo 非法即 `PlayerError(IA)`（`operate.py:248`）、AquaWar 用 `gettimeofday().tv_usec` 播种（`fish.cpp:17-23`）、LostSpace 名次分 `4 - i`（`GameController.py:390`）+ `can_see` 只判同层（`map.py:302-309`）

### 5. 未核实项

- `四游戏机制RL适配性分析.md` 里 LOTA/Dorado 的**可控单位数**（第二轮 A 组在文字里说"英雄+2 塔+1 泉"却按 4 个单位算空间，自身不一致）——**待核**
- 两份报告里凡**我没有亲自读代码确认**的数字，都只能算"子 agent 声称"

## 相关文件

- 审校后的主结论：[`../../docs/12游戏RL适配性评估.md`](../../docs/12游戏RL适配性评估.md)
- 后端源码（只读副本，重解方式见主文档 §8）：`C:\Users\曾子岩\AppData\Local\Temp\aaarena_backends\<game>\games\<game>\backend\`
- 规则文档（已解包）：[`../AAArena规则/`](../AAArena规则/README.md)
