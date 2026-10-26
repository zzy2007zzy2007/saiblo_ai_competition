# 裁判↔AI 协议：确切报文格式 + 亚军出招查询器

> 2026-10-26。**结论全部来自源码行号或真实字节 dump，没有推测**。
> 交付物：
> - 查询器 `code/test_match/runnerup_oracle.py`（`RunnerUpOracle.ask(state) -> list[Operation]`）
> - 验证 `code/test_match/probe_oracle_verify.py`（seed=10 30 回合，**30/30 = 100%**）
> - 协议保真自对弈 `code/test_match/probe_oracle_selfplay.py`（90 回合，**PASS**）
> - 真实桥接字节 dump `code/test_match/probe_wire_trace.py`（40 回合，**PASS**）

## 0. 一句话

- **裁判→AI：裸 UTF-8 文本，`\n` 分行，无长度前缀**；**AI→裁判：4 字节大端长度前缀 + UTF-8 payload**。
- 一回合裁判发**两条**消息，顺序固定：**① 对手本回合的操作块 → ② 下一回合的局面文本**；
  AI 在两条之间写出自己的操作（一个长度前缀包）。
- **第 0 回合只有第 ① 条**。P1 的 stdin 上第一条（init 之后）就是对手的 `"0\n"`——
  **没有任何局面文本**，它要靠 `init` 里的 seed 自己重建初始局面
  （`docs/protocol_ai_spec.md:32-33`、`:45-47`）。
- `state_to_text()` 与官方格式**逐字段完全一致**（顺序/字段数/必需字段/结尾换行全对得上），
  40 回合真实 dump 里 40/40 条局面文本被「照抄 `sdk.hpp` 的严格解析器」全量吃光。
- 旧探针 round 1 的 `invalid ant row` **不是格式问题**，是**第 0 回合多发了一条局面文本**
  ⇒ 整条流错位一条消息（§5 有逐行推演）。

---

## 1. 传输层（两个方向不一样）

| 方向 | 格式 | 依据 |
|---|---|---|
| 裁判→AI | 裸 UTF-8，`\n` 分行，**无前缀** | `docs/protocol_ai_spec.md:19`；`match_via_sdk.py:197-199`（`send_line` 直接 `write(text.encode())`） |
| AI→裁判 | `struct.pack(">I", len) + payload` | `docs/protocol_ai_spec.md:20`；`match_via_sdk.py:186-189`（`read_packet`）；亚军侧 `sdk.hpp:1216-1239`（`send_operations`） |

真实回包字节（`probe_dump_reply_bytes.py` 打的两个亚军进程）：

```
空回包   prefix=00 00 00 02  size=2   payload=30 0a                          text=b'0\n'
1 个操作 prefix=00 00 00 09  size=9   payload=31 0a 31 31 20 38 20 37 0a     text=b'1\n11 8 7\n'
2 个操作 prefix=00 00 00 0e  size=14  payload=32 0a 31 33 20 30 0a 31 31 20 38 20 32 0a
                                      text=b'2\n13 0\n11 8 2\n'
```

⚠️ 亚军在 `main()` 里就 `_setmode(_fileno(stdin|stdout), _O_BINARY)`
（`ai_cpp_lure_v4.cpp:157-162`），所以**不会有 `\r\n` 翻译**。
`match_via_sdk.ProcIO(strip_cr=True)` 是历史遗留参数、现在是 no-op
（`match_via_sdk.py:134-137`：读取层绝不删字节，删了会破坏二进制前缀里的 `0x0D`）。
裁判侧 `recv_line` 也会主动剥掉行尾 `\r`（`sdk.hpp:1250-1254`），双保险。

---

## 2. 时序：线上字节序列（真值）

### 2.1 对 P1（亚军执后手）——这是查询器要复刻的那条

`probe_wire_trace.py` 在**真实桥接**（`match_via_sdk.run_match`，只替换 `send_line` 记录，
不改一行业务代码）里截到的流：

```
[00] 裁判->AI    5B  init           '1 10\n'
[01] 裁判->AI    2B  对手操作块       '0\n'            ← init 之后**第一条**，不是局面！
[02] AI->裁判    2B  回包           '0\n'
[03] 裁判->AI   89B  局面文本        round=1 塔=0 蚁=2 金币=[50,50] 基地=[50,50] 冷却行=2 效果=0
[04] 裁判->AI    2B  对手操作块       '0\n'            ← 第 1 回合
[05] AI->裁判    2B  回包           '0\n'
[06] 裁判->AI   89B  局面文本        round=2 ...
[07] 裁判->AI    2B  对手操作块       '0\n'
...
[18] 裁判->AI  133B  局面文本        round=6 ...
```

对应 `ai_cpp_lure_v4.cpp:179-188` 的主循环（P1 分支）：

```cpp
if (!receive_and_apply_opponent(runtime, io)) break;   // 读 ①
perform_self_turn(runtime, io);                        // 写（长度前缀包）
finalize_local_round(runtime);                         // simulator.advance_round()
if (!receive_and_sync_round(runtime, io)) break;       // 读 ②
```

同一个循环在 Python 壳里有逐行镜像：`其他版本ai/.../ai_cpp_protocol.py:119-171`、
以及我们的 known-good control `code/test_match/rv4_pkg/main.py:38-51`。

### 2.2 对 P0（亚军执先手）

主循环（`ai_cpp_lure_v4.cpp:170-178`）是 `self → opponent → finalize → sync`，
但**读**顺序完全一样：`对手操作块 → 局面文本`。区别只在**裁判什么时候读它的回复**：

| | 裁判写 | 裁判读 |
|---|---|---|
| P0 | `对手(N)`, `局面(N+1)` | 先读回复，**再**写 `对手(N)` |
| P1 | `对手(N)`, `局面(N+1)` | **先**写 `对手(N)`，再读回复 |

⇒ **对两个玩家，裁判→AI 的字节流形状完全相同**：
`init` → （第 0 回合）`对手(0)` → （每回合）`局面(N)`、`对手(N)` …
P0 只是「回复先到」而已（管道的缓冲让这两件事可以并发，不存在死锁）。

### 2.3 一回合几条消息

- **第 0 回合：1 条**（对手操作块）。此时 AI 手里没有任何局面。
- **第 N≥1 回合：2 条**（对手操作块 + 下一回合局面文本）。
- 于是「局面文本」的 `round_index` 是**它接下来要打的那一回合**，与桥接
  `advance_round()` 之后再 `state_to_text()` 一致（`match_via_sdk.py:474`、`:483-491`）。

---

## 3. 每条消息的确切行格式

### 3.1 init（唯一带 seed 的消息）

```
<player> <seed>
```
`recv_init` 用 `sscanf("%d %d")`，不匹配就抛 `invalid init line`（`sdk.hpp:1050-1057`）。
桥接发的就是 `f"{player} {seed}\n"`（`match_via_sdk.py:294`）。
真实字节：`'1 10\n'`。

### 3.2 对手操作块（不是 `"0"`，也不是 `"0 <tokens>"`）

**是 `ops_to_text()` 的输出，即「操作数一行 + 每操作一行」的块**：

```
<count>\n
<op_type> [arg0] [arg1]\n      × count
```

- 空操作 = **`"0\n"`**（两字节；`ops_to_text([])` 恰好就是它，`match_via_sdk.py:206-209`）。
- 解析：第一行 `std::stoi` ⇒ count，然后读 count 行、每行 `sscanf("%d %d %d")`，
  1/2/3 个整数分别构造 1/2/3 参数操作，`matched <= 0` 抛 `invalid operation line`
  （`sdk.hpp:1059-1085`）。
- 真实非空字节（桥接把一方的回包 payload 原样转发给另一方，所以这两种块在线上都出现过；
  例子分别来自 rv4 和亚军自己）：
  `'1\n21 11 7\n'`（1 个操作：`21`=USE_LIGHTNING_STORM @(11,7)）、
  `'2\n13 0\n11 8 2\n'`（2 个操作：`13`=DOWNGRADE_TOWER #0、`11`=BUILD_TOWER @(8,2)）。
- 每个操作的 token 个数由 op_type 决定（`model.py:46-59` 与 `sdk.hpp:360-370` **逐条相同**）：

| op_type | 含义 | token |
|---|---|---|
| 11 | BUILD_TOWER（**只建 BASIC 塔**，arg0=x, arg1=y；`sdk.hpp:920-922`） | `11 x y` |
| 12 | UPGRADE_TOWER（arg0=tower_id, arg1=目标塔型） | `12 id type` |
| 13 | DOWNGRADE_TOWER（arg0=tower_id） | `13 id` |
| 21/22/23/24 | 四种超级武器（arg0=x, arg1=y） | `2x x y` |
| 31/32 | 升级基地产速 / 升级产出蚂蚁 | `3x` |

⚠️ 裁判/桥接**不会**替 AI 检查合法性；非法操作由 AI 自己筛
（`ai_cpp_lure_v4.cpp:122-128` 逐条 `can_apply_operation` 后才发）。

### 3.3 局面文本（9 段，严格按顺序）

真实样例（`match_results/wire_trace_seed10.jsonl`，round=40，3 塔 11 蚁）：

```
40                                  ← ① round_index
3                                   ← ② tower_count
0 0 6 17 0 0 10                     ←   塔行: id player x y type cooldown hp
1 1 14 9 2 0 15
2 1 12 9 0 2 8
11                                  ← ③ ant_count
6 0 8 15 20 0 26 0 0 0              ←   蚁行: id player x y hp level age status behavior kind
8 0 12 8 15 0 22 0 0 0
...
15 1 16 12 30 0 8 0 0 1             ←   最后一位 kind=1 表示战斗蚁
41 35                               ← ④ coins[0] coins[1]
47 47 0 0 0 0                       ← ⑤ base0_hp base1_hp gen_lv0 gen_lv1 ant_lv0 ant_lv1
2                                   ← ⑥ 冷却行数（**恒为 2**）
16 0 0 0                            ←   玩家 0 的 cd[1..4]：闪电/EMP/偏转器/紧急回避
0 0 0 0                             ←   玩家 1 的 cd[1..4]
0                                   ← ⑦ 效果数
```

带超级武器效果的样例（同文件，round=22，末两行）：

```
1                                   ← 效果数
1 0 11 6 14                         ← weapon_type player x y remaining_turns
                                      （type=1 LIGHTNING_STORM，玩家 0，(11,6)，还剩 14 回合）
```

逐段字段表（**必需字段数**是硬约束，少一个就整局废）：

| 段 | 行 | 字段（顺序即字节顺序） | 解析器与硬约束 |
|---|---|---|---|
| ① | 1 | `round_index` | `sdk.hpp:1093` `stoi` |
| ② | 1 | `tower_count` | `sdk.hpp:1095` |
| ② | ×N | `id player x y type cooldown hp` | `sdk.hpp:1106-1109`：`sscanf("%d %d %d %d %d %d %d")`，**<6 个就抛 `invalid tower row`**（hp 可省，省了按塔型满血补） |
| ③ | 1 | `ant_count` | `sdk.hpp:1121` |
| ③ | ×N | `id player x y hp level age status behavior kind` | `sdk.hpp:1135-1139`：10 个 `%d`，**<8 个就抛 `invalid ant row`**（behavior/kind 可省，默认 0/Worker） |
| ④ | 1 | `coins0 coins1` | `sdk.hpp:1157`：**必须正好 2 个**，否则 `invalid coin row` |
| ⑤ | 1 | `base0_hp base1_hp gen_lv0 gen_lv1 ant_lv0 ant_lv1` | `sdk.hpp:1170-1180`：**≥2 必需**；≥4 才读 speed_lv，≥6 才读 anthp_lv |
| ⑥ | 1+2 | 行数（恒 2），每行 `cd1 cd2 cd3 cd4` | `sdk.hpp:1182-1199`：行内 **4 或 5 个**都收；**4 个时它自己补前导 0**（`{0,v0,v1,v2,v3}`）⇒ 下标 1..4 = 武器 1..4 |
| ⑦ | 1+E | `weapon_type player x y remaining_turns` | `sdk.hpp:1200-1212`：**必须正好 5 个**，否则 `invalid effect row` |

### 3.4 AI→裁判的回包

`send_operations`（`sdk.hpp:1216-1239`）：

```cpp
payload = to_string(operations.size()) + "\n";
for (op : operations) { tokens 用空格连起来 + "\n"; }
out.write(4 字节大端长度); out.write(payload); out.flush();
```

⇒ **payload 与「对手操作块」是同一个格式**（这也是桥接能把 p0 的 payload 直接转发给 p1 的原因，
`match_via_sdk.py:429-437`）。内容 = 它这一回合要做的操作；`"0\n"` = 什么都不做。

---

## 4. `state_to_text()` 与官方格式：逐段核对

`state_to_text` 在 `code/test_match/match_via_sdk.py:212-256`。
官方二进制裁判输出同一份文本在 `Ant-Game/game/src/comm_judger.cpp:242-313`（`cur_json_to_oj`）。

| 段 | `state_to_text` | 官方 `cur_json_to_oj` | 是否一致 |
|---|---|---|---|
| ①回合号 | `:215` | `:245` | ✅ |
| ②塔数+塔行 7 字段 | `:217-220` | `:246-257` | ✅ 字段与顺序相同 |
| ③蚁数+蚁行 10 字段 | `:222-228` | `:259-274` | ✅ 相同（官方 behavior/kind 用 `value(...,0)` 兜底） |
| ④金币 2 字段 | `:230` | `:275-277` | ✅ |
| ⑤基地 6 字段 | `:232-236` | `:279-287` | ✅ |
| ⑥冷却 行数+2 行×4 | `:239-247` | `:289-300` | ✅（官方按 `ItemType::Count`=4 个输出） |
| ⑦效果 行数+5 字段 | `:249-254` | `:302-312` | ✅ 含**排序与过滤**，见下 |
| 结尾 | `+ "\n"`（`:256`） | 每行都 `+= "\n"` | ✅ |

**效果段为什么算一致**（这是最容易错的一段）：

- 官方 `Output::add_active_effects`（`output.cpp:104-124`）：外层 `player=0→1`，
  内层 `index=0..ItemType::Count`（=武器类型序），**`duration <= 0` 的跳过**，`type = index+1`。
- 我们的引擎 `Game::active_effects()`（`code/cpp_engine/m1_support.cpp:142-151`）：
  **同一个双重循环、同一个 `duration > 0` 过滤**，`t` 从 0 开始。
- facade 把 `t` 转成 `SuperWeaponType(t+1)`（`game_state_facade.py:272`），
  `state_to_text` 再按 `(player, weapon_type)` 排序输出 `int(weapon_type)`（`:253-254`）
  ⇒ **输出行与官方逐字节等价**（已由 40 回合真实 dump 中 14 条含效果的局面文本验证）。

### 结论：**格式完全一致**，只有两处「语义」差异（不影响解析）

1. **新蚂蚁的 `age` 起点**：我们的 `cpp_engine` 是 `age=1`，官方 `main.exe` 是 `age=0`
   （`docs/protocol_ai_spec.md:89-96`）。真实字节验证：round=1 的新蚁是
   `0 0 2 9 20 0 1 0 1 0`（第 7 位 `age=1`）。亚军的内嵌引擎按 `age=1` 写，
   所以**必须用我们的 `cpp_engine` 当裁判**，不能喂 `main.exe`。
2. **塔/蚁列表顺序**：官方按引擎容器顺序输出，我们按 facade 顺序。
   亚军自己 `to_public_round_state` 会按 id 重排（`sdk.hpp:1011-1012`），所以无影响。

---

## 5. round 1 报 `invalid ant row` 的根因（真实字节推演）

### 5.1 现象（原始报错）

```
$ python code/test_match/probe_runnerup_oracle.py 10 4
  [round 0] ops=[]（共 0 个）
  [round 1] ✗ 通信失败 EOFError: runnerup exited (code 0)
  结果：1/1 回合成功拿到并应用它的操作
[stderr] [cpp_sdk] sync_round failed: invalid ant row
```

### 5.2 根因：第 0 回合**多发了一条局面文本**

旧探针每回合发 `state_to_text(st)` + `"0"`。但 P1 的**第一条**读的是「对手操作块」
（`ai_cpp_lure_v4.cpp:180`），于是：

```
state(r0) 的 9 行（seed=10，真实）：
  0 / 0 / 0 / '50 50' / '50 50 0 0 0 0' / 2 / '0 0 0 0' / '0 0 0 0' / 0
```

`probe_repro_old_probe_bug.py` 按 `sdk.hpp` 的读序重放旧探针写出的字节，逐步打印：

```
① receive_opponent
   第 1行 读到 '0'      ⇒ 对手操作数 = 0（把「回合号」当成了操作数！）
② perform_self_turn：它先写自己的操作 ⇒ 探针**此时收到 '0\n'，以为 round 0 成功了**
③ sync_round
   第 2行 '0'          → round_index = 0
   第 3行 '0'          → tower_count = 0
   第 4行 '50 50'      → ant_count = 50   ⚠️ 这是金币行！C++ 的 std::stoi 只吃前缀 ⇒ 50
   第 5行 '50 50 0 0 0 0' → 当成第 1 个蚂蚁行，只有 6 个整数 < 8
   ✗ sdk.hpp:1137 抛 invalid ant row
```

⇒ 三个后果，全都对得上观察：

1. 探针以为 round 0 成功（确实收到了 `'0\n'`），**但亚军在同一回合的 `sync_round` 里就死了**；
2. 所以「round 1 失败」其实是**在 round 0 就注定的**，round 1 只是撞上进程已退出
   （表现为 `EOFError` 或 Windows 上写死管道的 `OSError: [Errno 22] Invalid argument`）；
3. 报错是 `invalid ant row` 而不是 `invalid coin row`/`invalid camp row`，
   纯粹是因为 seed=10 的初始金币是 `50 50` —— `stoi("50 50") = 50` 让它去等 50 个蚂蚁行。
   **同一份代码换个 seed/局面会报别的错，所以别按错误文本去猜格式。**

### 5.3 为什么不是 `state_to_text` 的格式问题

- 只改**顺序**（第 0 回合不发局面）就 30/30、40/40、90/90 全通（§7），
  `state_to_text` 一个字节都没改；
- 真实桥接的 40 条局面文本全部被严格解析器全量吃光（§7.3）。

---

## 6. 查询器 `RunnerUpOracle`

`code/test_match/runnerup_oracle.py`。

```python
from test_match.runnerup_oracle import RunnerUpOracle

oracle = RunnerUpOracle(seed=10, player=1)        # 起进程 + 发 init，常驻
for rnd in range(N):                             # 必须从 round 0 起、连续 +1
    ops = oracle.ask(st)                         # -> list[SDK.backend.model.Operation]
    oracle.set_opponent_ops(rnd, opp_ops)        # 可选：告诉它对手这回合走了什么
    st.apply_operation_list(1, ops)
    st.advance_round()
oracle.close()                                   # 或 with 语句
```

- **`ask(state)`**：传入 `GameStateFacade`，内部按 §2 的顺序发消息、读 4 字节前缀包、
  用桥接同一个 `parse_ops_text` 解析成 `Operation` 列表。
- **`player=0/1` 都支持**：两边调用方式一样（P0 的对手操作块语义是「上一回合」，见 3.2/§2.2）。
- **连续性是硬要求**：`state.round_index` 必须严格 = 上一回合 +1，跳回合/回退会抛
  `OracleDesync`（不能恢复，只能重建会话）。
- **超时保护**：`timeout` 默认 5.0s。超时抛 `OracleTimeout`，进程退出抛 `OracleDead`，
  任何失败都会把会话标记作废（因为读端已错位）⇒ **调用方永远不会卡死**。
  默认 5s 而不是 2s 的原因：它自己的搜索耗时随局面增长，见 §7.2（第 90 回合单侧约 2.0s）。
- **失败一律抛 `OracleError` 子类**（`OracleTimeout` / `OracleDesync` / `OracleDead`）；
  想拿 `None` 就用 `try_ask()`。
- 进程启动**照抄桥接的 `start_ai`**（`match_via_sdk.py:259-295`）：同一套 env、`cwd=exe.parent`、
  stderr 落文件、init 行格式一致 ⇒ 不会重踩 DLL 缺失（`0xC0000139`）和 `invalid init line`。

⚠️ **合法性要自己查**：facade 的 `apply_operation_list` 走冷路径，
**非法操作会被静默跳过**（`code/cpp_engine/binding.cpp:46-59`
返回 `(True, accepted_count)`，而 `game_state_facade.py:167-170` 把返回值丢了）。
所以判断「它这步合不合法」必须逐条 `can_apply_operation`，像 `probe_oracle_verify.py` 那样。

---

## 7. 验证结果

### 7.1 `probe_oracle_verify.py`：seed=10 连喂 30 回合

判据「≥25/30 成功」（成功 = 拿到回包 **且** 回包操作全部通过 `can_apply_operation`）：

```
[verify] seed=10 rounds=30 查询方=player 1（亚军）；对手=空过
[verify] 局面：ghost 起点 coins=[50, 50] hp=[50, 50]
  r00 ✓ ops=0 合法=0 (空过)   94.0ms coins=[50, 50] hp=[50, 50]
  r01 ✓ ops=0 合法=0 (空过)  140.0ms coins=[50, 50] hp=[50, 50]
  r02 ✓ ops=0 合法=0 (空过)  157.0ms coins=[53, 53] hp=[50, 50]
  ...
  r29 ✓ ops=0 合法=0 (空过)  688.0ms coins=[102, 112] hp=[48, 49]
==============================================================================
[verify] 成功 30/30 = 100.0%   判据 ≥25/30 ⇒ PASS
[verify] 每回合耗时 ms: min=94.0 median=499.5 max=704.0
[verify] 总墙钟 14.8s   oracle.stats={'asks': 30, 'ok': 30, 'timeouts': 0, ...}
[verify] 亚军 stderr: match_results\oracle_logs\oracle_seed10_p1_55184.stderr.log
==============================================================================
```

30 回合它**全部空过**——这是真实的策略行为，不是会话卡住。两个独立证据：

1. 它的 stderr 每回合都写 `{"kind":"decision","round":N,"plans_total":74,...,"action_count":0}`，
   `round` 与 `serial` 跟我们的回合号严格同步（证明它一回合都没漏）；
2. 拿**真实对局**（`match_results/ai1_runnerup_seed8.stderr.log`）对比：同样 `plans_total=74`、
   同样 `action_count=0` 一直到 `round=17`，第 18 回合才第一次出招
   ⇒ 我们的查询器复现了它的真实决策过程。亚军 v4 的核心是「勾引对手蚂蚁」，
   空局面下它的 `hold` 先验（+80）压过一切。

### 7.2 `probe_oracle_selfplay.py`：两个亚军通过查询器互相对打 90 回合

这才是协议层的硬检验（消息顺序/字段错一处都不可能跑完 90 回合）：

```
[selfplay] 跑完 round=90 terminal=False winner=None
[selfplay] 出招回合数 p0=15 p1=10  非法操作 p0=0 p1=0
[selfplay] coins=[245, 217] base_hp=[39, 39] ants=11 towers=3
[selfplay] oracle 统计 p0={'asks': 90, 'ok': 90, 'timeouts': 0, ...}
                        p1={'asks': 90, 'ok': 90, 'timeouts': 0, ...}
[selfplay] 判定：PASS（每回合都收到回包=True 局面演化=True 有出招=True 零非法=True）
```

- 90/90 每回合都拿到回包、**零超时、零非法操作**；双方各自造塔/升级/降级（`BUILD_TOWER`、
  `UPGRADE_TOWER`、`DOWNGRADE_TOWER`），基地 HP 从 50:50 打到 39:39。
- **P0 侧也验证了**（p0 出招 15 回合）。
- 「它真的在读我们发的局面」的旁证：P1 的 `plans_total` 在 r0–r20 恒为 **74**（空棋盘），
  棋盘上出现第一座塔后立刻变成 **165**（r21），之后随局面涨到 135 / 228 / 319，
  r89 时 **327**。如果它没读到我们的局面，这个候选数会一直停在 74
  （日志：`match_results/oracle_logs/oracle_seed10_p1_58432.stderr.log`）。
- 单次查询耗时：早期 ~0.1s，r90 时单侧 ~2.0s（所以超时默认给 5s）。

### 7.3 `probe_wire_trace.py`：真实桥接 40 回合，严格解析器复验

用一份**逐行照抄 `sdk.hpp:1087-1214`** 的解析器（必需字段数、`stoi` 语义、结尾必须无剩余行）
复验桥接实际发给亚军的每一条局面文本：

```
[trace] 局面文本严格解析失败 0 条 / 共 40 条
[trace] 结论：PASS
```

覆盖到的形态：塔最多 3 座、蚂蚁最多 11 只（含 `kind=1` 战斗蚁）、金币/基地等级齐全、
冷却行 2 行、**含超级武器效果的回合**（14 条）。

> 顺手记一个自己踩的坑：`probe_wire_trace.py` 最初用「首行是数字且行数=首行+1 就是操作块」
> 来分类消息，结果一条 `round=14`、恰好 15 行的局面文本被误判成「14 个操作」，
> 计数器一旦漂移，后面每条都报「round_index 期望值不对」⇒ 一堆假失败。
> 真实桥接的流是严格交替的，必须用**位置状态机**而不是内容嗅探。

---

## 8. 能不能直接接进 `az_bridge_ai.py` 做候选注入？

**协议层可以，但不建议每回合都用。** 具体：

✅ **可行的部分**
- `az_bridge_ai` 每回合被调用一次、手里就是一个 JSON 局面 + `GameStateFacade` 形态的状态，
  与 `ask(state)` 的形状完全一致；把 oracle 当常驻成员、每回合 `ask` 一次即可。
- 它返回的就是 `Operation` 列表，可以直接和我们的候选并集、也可以只用来给候选打分/排序。
- 注意侧别：`az_bridge_ai` 执先手时要用 `player=0`（P0 的顺序也已验证）。

⚠️ **三个硬约束**
1. **必须从 round 0 起每回合都喂**。会话是有状态的（跳一回合就永久错位），
   所以「第 40 回合临时问一下」做不到——要么从头跟到尾，要么重建（重建 = 从 r0 重放）。
   进程崩溃/超时的会话**只能丢弃重建**。
2. **成本**：单次查询 = 亚军自己的 8×125 rollout 搜索，实测 0.1s（早期）→0.7s（r30）→2.0s（r90）。
   这比我们的搜索预算贵一个量级，**不可能每回合注入**。
3. **它的前瞻里对手是「空过」**（我们发 `"0\n"`）。局面本身每回合都会被真局面覆盖所以是对的，
   但它以为对手这回合不动 ⇒ 它的出招是「对着一个不会反抗的对手」算出来的。
   做候选注入时这是个系统性偏差（对它有利的保守动作会被高估）。

✅ **推荐用法（按性价比排序）**
1. **离线批量标签/对照**（最合适）：给已存的对局轨迹逐回合重放，采它的出招做
   「亚军会怎么走」的监督信号或 baseline 对照，成本可控、不占对局时间；
2. 在线只做**抽样回合的第二意见**（如每 8 回合真正采信一次，但仍需每回合喂），
   用于发现我们搜索的明显失误；
3. 若要参与搜索，只用它做**候选补充**（不信任它的分数），并限制在关键回合。

---

## 9. 复现命令

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
cd /d/2026智能体大赛_新2_distill
export PYTHONIOENCODING=utf-8

$P code/test_match/probe_oracle_verify.py 10 30      # 30/30（主判据）
$P code/test_match/probe_oracle_selfplay.py 10 90    # 协议保真自对弈（90 回合）
$P code/test_match/probe_wire_trace.py 10 40         # 真实桥接字节 dump + 严格复验
$P code/test_match/runnerup_oracle.py 10 3           # 查询器自测
$P code/test_match/probe_repro_old_probe_bug.py      # 旧探针 bug 的逐行推演（§5）
$P code/test_match/probe_dump_reply_bytes.py 25      # 回包原始字节（含前缀）

# 旧探针（保留作反例，别用）：
$P code/test_match/probe_runnerup_oracle.py 10 4     # → [cpp_sdk] sync_round failed: invalid ant row
```

产物：
- `match_results/oracle_verify_seed10.log`（30 回合验证输出）
- `match_results/selfplay_90.log`（90 回合自对弈）
- `match_results/wire_trace_seed10.jsonl` + `wire_trace_40.log`（每一条消息的原文）
- `match_results/oracle_logs/oracle_seed*_p*.stderr.log`（亚军自己的 `{"kind":"decision",...}`）
