# 协议 AI 接入规范（stdio 协议）

> 2026-09-22。目的：**任何语言、任何算法设计**的 AI，只要能讲这套 stdio 协议，就能用
> `code/test_match/match_via_sdk.py --ai0=<你的程序>` 接进来对局，**不必迁就我们的技术路径**
> （不需要是 Python、不需要用我们的网络/搜索/状态对象）。
>
> 最小可运行示例：`code/test_match/example_ai.py`（约 60 行，直接照抄改）。
> 一致性体检：`code/test_match/conformance.py`（"这个 AI 到底在下棋吗"，与实现无关）。

## 0. 一句话

你的程序是一个**独立进程**。裁判通过 **stdin** 把局面和对手操作写给你，你通过 **stdout**
把本回合的操作回给裁判。就这些。

## 1. 传输层的两个方向（**不一样，最容易错**）

| 方向 | 格式 |
|---|---|
| 裁判 → 你（stdin） | **裸 UTF-8 文本，没有长度前缀**。以 `\n` 分行；你需要按 §3 的结构自己消费。 |
| 你 → 裁判（stdout） | **4 字节大端长度前缀 + UTF-8 payload**：`struct.pack(">I", len(data)) + data` |

⚠️ **stdout 是二进制协议通道**：任何漏出去的 `print`/日志都会把后续包的长度前缀冲掉，整局腐坏。
请把所有调试输出写 **stderr**（裁判会收进日志文件）。Python 里建议开头就
`sys.stdout = sys.stderr`，把真正的 stdout 单独存下来发协议。

## 2. 时序

```
裁判: "<player> <seed>\n"                    ← init，只有这一行
  ┌─ 第 0 回合 ────────────────────────────────────────────────
  │ 若你是 player 0:  先写第 0 回合操作（此时你还没收到任何局面！见 §4）
  │ 若你是 player 1:  先读「player 0 的第 0 回合操作」，再写自己的
  │ 之后你收到「第 1 回合的局面文本」
  ├─ 第 N 回合（N ≥ 1）───────────────────────────────────────
  │ 你先读到的是 <上一轮> 发来的局面文本（= 第 N 回合开始时的局面）
  │ player 0: 写第 N 回合操作 → 读 player 1 的第 N 回合操作 → 读第 N+1 回合局面
  │ player 1: 读 player 0 的第 N 回合操作 → 写第 N 回合操作 → 读第 N+1 回合局面
  └──────────────────────────────────────────────────────────
对局结束时裁判不再读你的操作 ⇒ 你下一次读会拿到 **EOF**，正常退出即可。
```

要点：
- **同一个回合里 player 0 先动**，所以 player 1 在决定前能看到 player 0 本回合的操作并据此调整。
- 局面文本是**推进之后**的局面，即"你接下来那一回合的开局状态"。
- 你先手（player 0）时，**第 0 回合是在没有任何局面的情况下出招的**（官方协议就这样：
  init 只给 `<player> <seed>`）。想在第 0 回合就行动，只能用 seed 自己重建初始局面。
  冠军 AI 就是这么做的；我们的判断是"第 0 回合空过"完全正常。

## 3. 局面文本结构（逐行，严格按顺序）

```
<round_index>                       # 回合号
<tower_count>                       # 塔数
  <tower_id> <player> <x> <y> <tower_type> <cooldown_clock> <hp>      × tower_count
<ant_count>                         # 蚂蚁数
  <ant_id> <player> <x> <y> <hp> <level> <age> <status> <behavior> <kind>   × ant_count
<coins_p0> <coins_p1>
<base0_hp> <base1_hp> <base0_gen_level> <base1_gen_level> <base0_ant_level> <base1_ant_level>
<cooldown_row_count>                # 恒为 2，每行一个玩家，按武器 ID 1..4
  <cd1> <cd2> <cd3> <cd4>           × 2 行
<effect_count>
  <weapon_type> <player> <x> <y> <remaining_turns>    × effect_count
```

⚠️ 效果（effects）按 **player0 → player1、再按武器类型** 排序输出。这是项目踩过的坑：
Python 侧的列表是插入顺序，必须重排后再输出，否则 AI 内部模拟会和裁判分歧。

## 4. 操作文本结构

```
<count>
<op_type> [arg0] [arg1]
...                                 × count 行
```
每行是空格分隔的整数；`arg0/arg1` 视操作类型而定。**空操作 = 只有一行 `0`。**

## 5. 判词（谁赢）

1. 有基地 HP ≤ 0 ⇒ 该玩家输（**双方都 ≤0 时判 player 0 胜**）。
2. 打满 512 回合 ⇒ 引擎按**5 级级联**判：① 基地 HP 高者胜 → ② `opponent_killed_ant` 多者胜
   → ③ 超武使用次数**少**者胜 → ④ `AI_total_time` **少**者胜 → ⑤ 全相同则**判 player 0 胜**。
   **官方规则从不判平局。**
3. ⚠️ 别把"平局"当真：我们的 bridge 在**对局被提前截断**（`--rounds=N`）时会退化成"只比 HP"
   并可能打出 `draw`——那不是官方判词。每局的 `RESULT` 行里有 `verdict=engine|hp_tiebreak`
   标出用的是哪条路，**读结论前先看它**。

## 6. ⚠️ 裁判有两种，同一个 AI 可能只对其中一个正确

| 裁判 | 怎么起 | 第 1 回合蚂蚁 `age` |
|---|---|---|
| **我们的 `code/cpp_engine`**（默认，`verdict=engine`） | `match_via_sdk.py`（本文件描述的协议） | **1** |
| 官方 `Ant-Game/game/output/main.exe` | `run_cpp_ai_match.py`（二进制 judger 中继） | **0** |

这是项目**花了很多时间才定位到**的坑：冠军/亚军的内嵌引擎按 `age=1` 写，所以喂给它们
`main.exe`（`age=0`）时它们会**检测到分歧并全程静默弃权**——而"跑完一盘、有赢家"完全看不出来。
因此：**你必须知道你的 AI 是按哪一版引擎写的**，并选对应的裁判。你的 AI 内部若有自己的
模拟器，它和裁判的状态一旦分歧，你很可能也会静默失效——所以我们的壳默认每回合对拍（见 §7）。

## 7. 接进来的步骤（建议顺序）

1. **从 `example_ai.py` 抄**：它已经把 §1–§4 的收发写对了，你只需要在 `decide()` 里填你的算法。
2. **先跑一致性体检**：`python code/test_match/conformance.py --ai0=<你的程序>`
   ——它会检查"进程没提前退出 / 没有非法操作 / 双方真的出招了（不是僵局）/ 判词来源已知"。
   ⚠️ 它只能对**硬失败**下结论；"短跑里没出招"**不能**判定 AI 弃权
   （历史教训：用 8 回合上限得出过"两个 AI 全弃权"的错误结论，实际是前期攒钱的正常策略）。
3. 再跑正式对局：`python code/test_match/match_via_sdk.py --ai0=<你的程序> --ai1=<对手> <seed...>`
4. 如果你自己维护影子状态（推荐，用于搜索）：**每回合用收到的局面文本和你的状态对拍**，
   分歧要计数并打印（`az_bridge_ai.py` 就是这么做的，见 §8）。若你的影子状态分歧了却继续出招，
   你会发出一串非法操作，`RESULT` 里的 `ill` 会暴露它。

## 8. 参考实现

| 文件 | 说明 |
|---|---|
| `code/test_match/example_ai.py` | 最小示例（总是空过，带"怎么出招"的注释） |
| `code/test_match/rv4_pkg/` | rule_v4 的壳：官方 SDK 的 `ProtocolSession` + 规则决策。**known-good control，保持冻结** |
| `code/test_match/az_bridge_ai.py` | 我们自己的 AI：影子状态 = 裁判本身（`GameStateFacade`），每回合对拍 |
