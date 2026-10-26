# 引擎一致性排查报告（GameStateFacade / cpp_engine vs 官方 C++ 判题器）

- 时间：2026-10-26
- 范围：只读排查 + 小规模验证（16 核机器上当前还有 10 局 ladder 在跑，全程未 kill 任何既有进程，探针只占 1 个进程 / 1 个 GPU 之外的 CPU 核）
- 触发问题：用 `GameStateFacade` 从 ladder 日志的出招序列**重放**对局，终局 `base_hp` 与日志 `RESULT` 行不一致（例：重放得 `(4,0)`，记录是 `(0,28)`）。原假设"Python 引擎（GameStateFacade / cpp_engine 绑定）与官方 C++ 引擎行为不一致"。
- 结论：**该假设不成立**。不一致来自**回放脚本自身**（出招解析 + 出招施加方式），不是引擎。已在同一台机器上把官方 C++ 判题器从源码编出来，和我们的 pyd 逐回合对拍，两者一致到每一回合的 `base_hp` / `coins`。

---

## 0. 结论摘要

1. **我们根本没用"Python 引擎"。** `GameStateFacade` 是 `native_game`（pybind11）的薄封装，底层就是 `code/cpp_engine/src/game.cpp`；而 `code/cpp_engine/src/*.cpp` 与官方 `Ant-Game/game/src/*.cpp` **逐字节相同**（只有 `comm_judger.cpp` 改了 5 处 `std::cout`→`fwrite` 的 Windows 二进制安全补丁，`main.cpp` 加了 `_setmode`）。规则实现只有一份。
2. **`(4,0) vs (0,28)` 已 100% 复现并定位。** 直接跑 `code/test_match/recall_on_runnerup.py` 就打印 `重放 base_hp=(4, 0) 记录=(0, 28)`。原因：该脚本用 `ast.literal_eval` 解析日志里 `-> [Operation(op_type=<OperationType...>)…]` 这种 **Python repr**，必然抛异常被 `except: continue` 吞掉 ⇒ **这条出招被静默丢弃**。seed 10 那局 228 条真实出招被丢掉 228 条（100%），140/440 个回合连 `advance_round()` 都没被调用。
3. **官方判题器从源码编出来后，与我们的 pyd 逐回合完全一致。** 4 条真实对局（440 / 463 / 512 / 377 回合，共 1792 回合）三条链路（facade、官方判题器 `-O0`、官方判题器 `-O2`）每一回合的 `base_hp` 与 `coins` 全部相同，终局 `base_hp` 与日志 `RESULT` 全部相同。
4. **批量验证：48 条 ladder 日志、facade 重放 100% 忠实。** 终局 `base_hp` 与 RESULT 一致 48/48；**且每回合的完整塔快照**（日志里 `[round N] towers=[...]`）逐回合一致 48/48。
5. **两个"真实存在但这次没咬人"的隐患**，值得写进规范：
   - **出招必须"一回合整列表一次 apply"**。cold 路径的 `used_tower`（同塔一回合只能操作一次）/`camp_upgraded_flag`（一回合只能升一次基地）是**单次调用内的局部变量**；逐条 apply 会绕过它们，局面立刻分叉。已给出 3 方最小对比实验（官方判题器 `/` 整列表 `/` 逐条：塔类型 `1` / `1` / `0`）。
   - **官方判题器不能用含中文的路径写 replay**。MinGW 窄字符 `std::ofstream` 在 `D:\2026智能体大赛_新2_distill\...` 下**静默写不出文件**（返回码 0、replay 不存在），换 `C:\Temp\...` 就正常。这会让"官方判题"链路在本机悄悄产不出结果。
6. **编译选项不一致（应消掉的隐患）**：`code/cpp_engine/build.sh` 注释写"`-O0` matches the official Makefile flags"，但实际命令行是 **`-O2`**；官方 `Ant-Game/game/Makefile` 的 `CXXFLAGS` 里**没有 `-O`**（= `-O0`）。本次 1792 回合里 `-O0` 与 `-O2` 没有任何差异，所以它**不是**这次的病根，但 Enhanced 移动/寻路大量用 `double` 打分并排序，属于"换编译器/换优化级别可能翻边"的高风险区，建议对齐。
7. **一个和平台判题有关、比引擎一致性更值得警惕的差异**：官方 `main.exe` 的 `cold_handle_rule_illegal` **默认 false**（`game.cpp:134-140`），即"出一次非法操作 = 立刻判负"；而我们的 `match_via_sdk.py:405` 传的是 `cold_handle_rule_illegal=True`（非法操作被静默跳过）。也就是说**本地对局里非法出招没有惩罚，平台上会直接输**。这 48 条日志里 0 次非法出招，暂时安全，但这是"本地测不出来"的失分点。

**是否该改用 C++ 引擎做判题？**
> 不是"改用"，而是**"现在用的本来就是 C++ 引擎"**。真正该改的是两件事：(a) 需要**权威判题**时用官方 `main.exe`（官方 Makefile 原样编译，`-O0`）跑一遍，作为最终裁决；(b) 把 `native_game.pyd` 的编译选项对齐到 `-O0`，让它继续当"快 20 倍的自对弈/回放引擎"。日常训练/自对弈继续用 facade 完全没问题——它已经被证明等价。

---

## 1. 问题 1：我们的桥接和 AI 到底用哪个引擎？

**答案：pybind11 调 C++，纯 Python 规则实现只有 `Ant-Game/SDK/backend/engine.py` 那一份（我们没在判题链路上用它）。**

### 调用链（文件 + 行号）

| 层 | 位置 | 内容 |
|---|---|---|
| 桥接入口 | `code/test_match/match_via_sdk.py:25` | `from my_ai.az_intent.game_state_facade import GameStateFacade as GameState` |
| 建局 | `code/test_match/match_via_sdk.py:405` | `state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)` |
| 结算 | `code/test_match/match_via_sdk.py:464-474` | `state.apply_operation_list(p, ops)` ×2 → `state.advance_round()` |
| 自证 | `code/test_match/match_via_sdk.py:632` | `print("[bridge] judge=cpp_engine (in-process GameStateFacade)")` |
| facade 建局 | `code/my_ai/az_intent/game_state_facade.py:155-161` | `import native_game; g = native_game.NativeGame(); g.init_game(seed=seed, movement_policy="enhanced", cold=cold_handle_rule_illegal)` |
| facade 出招 | `code/my_ai/az_intent/game_state_facade.py:167-170` | `self._g.apply_operation_list(player, [(int(op.op_type), op.arg0, op.arg1) …])` |
| facade 推进 | `code/my_ai/az_intent/game_state_facade.py:172-174` | `self._g.advance_round()` |
| pybind 模块 | `code/cpp_engine/binding.cpp:38` | `PYBIND11_MODULE(native_game, m)` |
| pybind 绑定 | `code/cpp_engine/binding.cpp:41,46-59,60,102` | `py::class_<Game>("NativeGame")`；`apply_operation_list`→`Game::apply_operation_list_cold`；`advance_round`→`&Game::next_round`；`clone`→`Game::deep_clone` |
| 嵌入接口实现 | `code/cpp_engine/m1_support.cpp:18-69` | `Game::init_game(seed, policy, cold)`（手写复刻 `Game::init()` 的可判题无关部分） |
| cold 语义复刻 | `code/cpp_engine/m1_support.cpp:171-207` | `Game::apply_operation_list_cold`（逐行复刻 `game.cpp:2627-2681`） |
| 规则本体 | `code/cpp_engine/src/game.cpp` | 与 `Ant-Game/game/src/game.cpp` **完全相同**（107300 字节，diff 0 行） |

### 关键证据：源码就是官方的

```powershell
PS> # 逐文件 Compare-Object（official vs ours）
aco.cpp diffLines=0        comm_judger.cpp diffLines=13      main.cpp diffLines=10
ant.cpp diffLines=0        game.cpp        diffLines=0
building.cpp diffLines=0   item.cpp        diffLines=0       map.cpp diffLines=0
coin.cpp diffLines=0       operation.cpp   diffLines=0       output.cpp diffLines=0
game.hpp diffLines=57（全部是"只在我们这边"的新增声明，无一行修改/删除）
```

`comm_judger.cpp` 的 13 行差异全在这里：

```
=>  fwrite(&tmp, 1, 1, stdout);   // binary-safe (cout would translate 0x0A on Windows)
=>  fwrite(dumped.data(), 1, dumped.size(), stdout);
<=  std::cout << tmp;  std::cout << info;  std::cout.flush();
```

`main.cpp` 的 10 行差异只有：

```cpp
=>  #ifdef _WIN32
=>      _setmode(_fileno(stdin), _O_BINARY);
=>      _setmode(_fileno(stdout), _O_BINARY);
```

即"把协议流切成二进制模式"，**不碰规则**。

### 但编译选项不一致（隐患）

```bash
# code/cpp_engine/build.sh:26-29
# -O0 matches the official Makefile flags (g++ -g, no -O): avoids FP
# optimization differences in the Enhanced ant-movement pathfinding that would
# diverge behavior from the official binary.
"$GXX" -std=c++17 -O2 -fPIC -shared \      # <-- 注释说 -O0，实际是 -O2
```

```make
# Ant-Game/game/Makefile:10 （官方，无 -O ⇒ -O0）
CXXFLAGS := -std=c++17 -Wall -Wextra -g -MMD -MP
```

> 附带发现：`build.sh:8` 里的 `REPO="D:/2026智能体大赛_新2"` 指向的是**另一个目录**（不是 `_distill`）。当前 `code/cpp_engine/src` 与官方 diff 为 0，所以不影响正确性，但这条命令已经不能在本工作目录复现构建了。

**回答小结**：桥接和 AI 用的都是 C++（`code/cpp_engine` = 官方源码 + 二进制 IO 补丁 + 嵌入接口），`advance_round` / `apply_operation_list` 都是 pybind11 直调 C++，**没有任何 Python 重写的规则**在链路上。

---

## 2. 问题 2：官方仓库有没有权威的 JSON 回放格式 / 工具？

**答案：有权威的 replay schema 和"跑一局并生成 replay"的工具，但\*\*没有\*\*"给 replay + 引擎 ⇒ 复现终局"的官方 judge/replay 工具。**

### 2.1 replay schema（权威文档）

`Ant-Game/README.md:477-529`，要点：

- 顶层是 JSON **数组**，下标 = 回合号（从 0 起），每个元素是该回合双方操作执行完 + 结算完的记录。
- 单回合结构：`{"op0":[…], "op1":[…], "round_state":{…}, "seed":7}`（`seed` 只在第 0 回合出现）。
- `op0/op1` 字段格式 = 操作协议；未使用字段通常记 `-1`。
- `round_state` 含 `coins / camps / speedLv / anthpLv / weaponCooldowns / activeEffects / ants / towers / winner / message / pheromone`。
- **`round_state.towers` 不是全量快照，是"本回合发生变化的塔"的增量**；`type = -1` 表示塔被真正拆除；对非 Basic 塔执行 `13 id` 是**降级**而不是移除，所以会写入降级后的真实类型。要还原全量塔状态必须自己维护 `tower_id -> tower` 映射并逐回合应用增量（我的探针就是这么做的）。
- `round_state.ants` 每条含 `player/id/pos/hp/level/age/status/behavior/kind/move`；`move = -1` 表示本回合没记录到移动方向（例如刚生成）。

### 2.2 生成 replay 的官方命令

```bash
cd Ant-Game/game && make           # Makefile:10，CXXFLAGS 无 -O ⇒ -O0；产物 game/output/main(.exe)
python Ant-Game/tools/run_local_match.py --ai0 greedy --ai1 greedy --seed 7 --keep-dir <dir>
#   -> <dir>/replay.json
```

- `tools/run_local_match.py:19` `DEFAULT_GAME_BIN = game/output/main`
- `:24-26` 若用默认 bin 就先 `make`
- `:187-192` 初始化包 `{"player_list":[1,1],"player_num":2,"config":{"random_seed":seed},"replay":<path>}`
- `:221-224` 收到 `end_state` 收尾
- **注意**：`game/output/` 被 `Ant-Game/.gitignore` 忽略，仓库里**没有**预编译的 `main.exe`（我确认过 `Test-Path Ant-Game/game/output` = `False`，全仓也没有 `main*.exe`）。

### 2.3 判定流程（judger 协议，`main.exe` 的真实行为）

`Ant-Game/game/src/main.cpp:39-73` + `game.cpp`：

```
while (!is_ended) {
    listen(0);  round_read_from_judger(0);   // 读 p0 的一包
    listen(1);  round_read_from_judger(1);   // 读 p1 的一包
    if (!next_round()) break;                // 结算；false = 结束
    dump_round_state();                      // 追加一条 replay 记录
}
send_end_info();                             // -> dump_last_round + dump_result(record_file)
```

- 每个包 = `4 字节大端长度 + JSON`（`game.cpp:2811-2840`）；EOF 时 `exit(0)`，**不会**落盘 replay。
- `config.movement_policy` 缺省 = `Enhanced`（`game.cpp:125-132`）；`config.cold_handle_rule_illegal` 缺省 = **false**（`game.cpp:134-140`）。
- **会导致对局结束的那一回合不进 `dump_round_state()`**（`next_round()` 返回 false 直接 `break`）；但 `send_end_info()` 里的 `dump_last_round()`（`game.cpp:2505-2535`）会再 `save_data()` 追加一条"最终局面"记录（同样带变化塔增量 + camps/coins/winner）。所以：
  - 自然结束（把基地打爆 / 打满 512 回合）：`len(replay) == 回合数`（439 条 dump_round_state + 1 条 dump_last_round，实测 seed 10 = 440 条）；
  - 用错误包 `{"player":-1,…}` 强制收尾（对局没自然结束）：`len(replay) == 已喂回合数 + 1`（实测喂 42 回合 → 43 条）。
- `op0/op1` 记的是**被判题器接受的** op 列表（cold 路径 `op[player] = accepted_ops`，`game.cpp:2678`），被跳过的非法操作不在里面 —— 这一点很好用：可以拿 replay 的 `op0` 反推"引擎到底接受了哪几条"（§4.3 的实验就是靠它）。
- 官方**没有**独立 judge 程序去读 replay 复现；`replay.json` 只是"记录"。要做"replay ⇒ 复现"，只能把 `replay[i].op0/op1` 重新按协议喂给 `main.exe`（这正是我下面探针做的事）。

### 2.4 官方其实还提供了 C++ 引擎的 Python 绑定

`Ant-Game/SDK/native_antwar.cpp` + `Ant-Game/SDK/native_adapter.py` + `Ant-Game/tools/setup_native.py`：

```python
# tools/setup_native.py
Pybind11Extension("SDK.native_antwar",
                  [REPO_ROOT/"SDK"/"native_antwar.cpp", *GAME_SOURCES],  # 除 main.cpp 外全部 game/src
                  include_dirs=[REPO_ROOT, REPO_ROOT/"game"/"include"], cxx_std=17)
```

`SDK/native_adapter.py` 暴露 `NativeGameStateAdapter.initial(seed, movement_policy, cold_handle_rule_illegal)`、`apply_operation_list`、`advance_round`、`resolve_turn`、`sync_public_round_state`。
→ **如果只是想要"官方背书的 C++ 引擎 Python 接口"，官方自己就有一份**。但它是"native + Python 影子状态"双写设计（每次操作都同步一份 `GameState`），比我们的 facade 慢，且 `setup_native.py` 用 setuptools 默认 MSVC 选项编译，同样不是 `-O0`。

---

## 3. 问题 3：两个引擎是否真的不一致？——最小验证

### 3.1 验证设计

1. **把官方判题器从源码编出来**（不动既有二进制，产物只放 `code/cpp_engine/build_probe/`）：
   ```bash
   # 官方 flags（-g，无 -O == -O0）
   g++ -std=c++17 -g  -I Ant-Game/game/include Ant-Game/game/src/*.cpp \
       -o code/cpp_engine/build_probe/judge_O0.exe
   # 我们的 pyd 的 flags（-O2），做对照
   g++ -std=c++17 -O2 -I Ant-Game/game/include Ant-Game/game/src/*.cpp \
       -o code/cpp_engine/build_probe/judge_O2.exe
   ```
   两个坑（都已解决，命令见 §3.2）：MinGW 需要把 `libstdc++-6.dll / libgcc_s_seh-1.dll / libwinpthread-1.dll` 拷到 exe 旁边，否则 `0xC0000139 ENTRYPOINT_NOT_FOUND`；`TMP` 必须设成**纯 ASCII** 路径（默认 `C:\Users\<中文名>\AppData\Local\Temp` 会让 `cc1plus` 写不出 `.o`，报 `Permission denied`）。
2. **从日志取真实出招序列**：解析 `[round N] pX(name) 原文='…' -> […]` 里的 `原文`（对 `'…'` 做 `ast.literal_eval` 拿到真正的多行文本），再用与 `match_via_sdk.py:560 parse_ops_text` **逐字相同**的解析器转成 op；重建成协议文本喂给官方判题器。
3. **三方对拍**：`facade`（in-process pyd）/ `judge_O0.exe` / `judge_O2.exe`，逐回合比 `base_hp`、`coins`，终局比 `RESULT`。

### 3.2 命令

```powershell
# 0) 编译（一次性）
New-Item -ItemType Directory -Force C:\Temp\probe_build | Out-Null
bash -lc 'cd /d/2026智能体大赛_新2_distill && export TMP="C:\Temp\probe_build" TEMP="C:\Temp\probe_build" TMPDIR="C:\Temp\probe_build" && \
  /c/mingw64/bin/g++.exe -std=c++17 -g  -I Ant-Game/game/include Ant-Game/game/src/*.cpp -o code/cpp_engine/build_probe/judge_O0.exe'
bash -lc 'cd /d/2026智能体大赛_新2_distill && export TMP="C:\Temp\probe_build" TEMP="C:\Temp\probe_build" TMPDIR="C:\Temp\probe_build" && \
  /c/mingw64/bin/g++.exe -std=c++17 -O2 -I Ant-Game/game/include Ant-Game/game/src/*.cpp -o code/cpp_engine/build_probe/judge_O2.exe'
Copy-Item C:\mingw64\bin\libstdc++-6.dll,C:\mingw64\bin\libgcc_s_seh-1.dll,C:\mingw64\bin\libwinpthread-1.dll code\cpp_engine\build_probe\ -Force

# 1) 三方对拍一条日志
$env:PYTHONIOENCODING="utf-8"
D:\anaconda3\envs\pytorch-gpu\python.exe code/test_match/probe_replay_engines.py `
  "match_results/ladder_logs/VS_RUNNERUP_cells4/10.log" `
  --judge code/cpp_engine/build_probe/judge_O0.exe `
  --judge code/cpp_engine/build_probe/judge_O2.exe
```

### 3.3 输出（seed 10，440 回合，就是"重放得 (4,0) / 记录 (0,28)"那一局）

```
[log] seed=10  RESULT={'rounds': 440, 'hp': (0, 28)}
[facade] native_game.pyd  440 rounds in 0.13s
[facade] final hp=(0,28) coins=(81, 370) terminal=True winner=1
[facade vs log] 每回合塔快照完全一致 ✅
[facade vs RESULT] hp: facade=(0,28) log=(0, 28)  -> 一致 ✅

[judge judge_O0.exe] 440 rounds in 2.62s  rc=0
[judge judge_O0.exe] final hp=(0,28) coins=(81, 370) winner=1
[judge judge_O0.exe vs RESULT] hp=(0,28) log=(0, 28)  -> 一致 ✅

[judge judge_O2.exe] 440 rounds in 1.05s  rc=0
[judge judge_O2.exe] final hp=(0,28) coins=(81, 370) winner=1
[judge judge_O2.exe vs RESULT] hp=(0,28) log=(0, 28)  -> 一致 ✅

[facade vs judge_O0.exe] 每回合 base_hp 完全一致 ✅
[facade vs judge_O0.exe] 每回合 coins 完全一致 ✅
[facade vs judge_O2.exe] 每回合 base_hp 完全一致 ✅
[facade vs judge_O2.exe] 每回合 coins 完全一致 ✅
[judge_O0.exe vs judge_O2.exe] 每回合 base_hp 完全一致（编译选项不影响）✅
```

另外 3 条不同目录、不同对手/长度的日志（`VS_RUNNERUP_cells4/7.log` 463 回合、`RU_SELFPLAY/865.log` 512 回合、`HZ_h128/10.log` 377 回合）结果一致：**facade == judge_O0 == judge_O2，逐回合 base_hp/coins 全等，终局都等于 RESULT**。合计 1792 回合无一处差异。

### 3.4 批量：48 条日志的忠实性

```powershell
D:\anaconda3\envs\pytorch-gpu\python.exe code/test_match/probe_replay_batch.py `
  "match_results/ladder_logs/VS_RUNNERUP_cells4" --limit 32     # == cold=1: OK=32 BAD=0 SKIP=0 / 32 logs
D:\anaconda3\envs\pytorch-gpu\python.exe code/test_match/probe_replay_batch.py `
  "match_results/ladder_logs/RU_SELFPLAY" --limit 8             # == OK=8 BAD=0 SKIP=0 / 8
D:\anaconda3\envs\pytorch-gpu\python.exe code/test_match/probe_replay_batch.py `
  "match_results/ladder_logs/HZ_h128" --limit 8                 # == OK=8 BAD=0 SKIP=0 / 8
```

判定标准不是只看终局 hp：每一项都同时检查**日志里每回合的完整塔快照**（`[round N] towers=[(id,player,x,y,type,hp)…]`）是否与重放逐回合相同 —— 48/48 `towers=ok`。

---

## 4. 问题 4：如果不一致，差在哪？——定位到具体机制

**先给出结论：这次的不一致与引擎无关，是回放脚本的两个 bug + 一个环境坑。**

### 4.1 直接复现用户看到的那行输出

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:CUDA_VISIBLE_DEVICES=""
D:\anaconda3\envs\pytorch-gpu\python.exe code/test_match/recall_on_runnerup.py `
  "match_results/ladder_logs/VS_RUNNERUP_cells4/10.log"
```

```
  10.log: seed=10 重放 base_hp=(4, 0) 记录=(0, 28) ❌不一致  亚军决策点 0
  忠实重放 0/1 局
```

`亚军决策点 0` 是关键线索：它根本没能从日志里解析出任何一条出招。

### 4.2 机制 A（病根）：日志里出招是 Python `repr`，用 `ast.literal_eval` 解析必然失败

日志里的行（`match_via_sdk.py:448` 打印的）：

```
  [round 25] p0(az_bridge_ai) 原文='1\n11 8 16' -> [Operation(op_type=<OperationType.BUILD_TOWER: 11>, arg0=8, arg1=16)]
```

而 `recall_on_runnerup.py:34-48` 是：

```python
LINE = re.compile(r"\[round (\d+)\] (p[01])\((\w+)\) 原文='[^']*' -> (\[.*\])")
...
        try:
            lst = ast.literal_eval(ops)      # ops = "[Operation(op_type=<OperationType...>, arg0=8, arg1=16)]"
        except Exception:
            continue                         # ← 静默丢弃这条出招
        per.setdefault(int(rnd), {})[side] = lst
```

`Operation(...)` 不是 Python 字面量 ⇒ 只要一侧**真的有出招**，这一行必定进 `except: continue`。后果不只是丢 op：如果某回合双方都有出招，`per[rnd]` 里两边都没有条目，`for rnd in sorted(per)` 直接**跳过整个回合**（连 `advance_round()` 都不调用）。

量化（`code/test_match/probe_replay_bug_demo.py`，同一份 10.log）：

```
== bug-1  日志解析 ==
  日志里打印出招的行数            : 880
  ast.literal_eval 失败（被丢弃） : 158
  recall 解析后还剩的回合数       : 422
  被保留的回合里'某侧没有出招'的次数: 844
  真实出招总数                    : 228
  recall 解析后剩下的出招总数     : 0  (丢掉 228 条 = 100.0%)
  整回合被跳过（该回合出招一条都没剩下）: 140 回合
```

**228 条出招 0 条被用上**，所以那次"重放"实际上是一局"双方全程不出招"的模拟，终局 `(4,0)` 与真实 `(0,28)` 毫无关系。这就是 `(4,0) vs (0,28)` 的全部来源。

隔离实验（`probe_replay_bug_demo.py`，**同一个 pyd、同一个 seed**，只改解析/施加方式）：

```
V0  recall 原样：错解析 + 逐条 apply     base_hp=(4, 0)  ❌ 与 RESULT 不一致
V1  只修解析    ：对解析 + 逐条 apply    base_hp=(0, 28) ✅ 与 RESULT 一致
V2  只修 apply  ：错解析 + 整列表 apply  base_hp=(4, 0)  ❌ 与 RESULT 不一致
V3  两处都修    ：对解析 + 整列表 apply  base_hp=(0, 28) ✅ 与 RESULT 一致
```

⇒ 决定 hp 的是**解析**（V0/V2 都错、V1/V3 都对），与"哪个引擎"无关。

### 4.3 机制 B（潜在坑）：出招逐条 apply 会绕过"一回合一次"的规则

官方 cold 路径（`game.cpp:2629-2677`，我们在 `m1_support.cpp:171-207` 逐行复刻）：

```cpp
std::vector<int> used_tower;          // ← 每次 apply_operation 调用内的局部变量
bool camp_upgraded_flag = false;
for (const auto &operation : op_list) {
    if ((type == TowerUpgrade || type == TowerDestroy) &&
        std::find(used_tower.begin(), used_tower.end(), operation.get_id()) != used_tower.end())
        { … continue; }               // 同一座塔一回合只能操作一次
    if (is_base_upgrade_operation(type) && camp_upgraded_flag)
        { … continue; }               // 一回合只能升一次基地
    …
}
```

`used_tower` / `camp_upgraded_flag` 是**调用内**的局部变量 ⇒ 把一回合的 `[op_a, op_b]` 拆成两次调用，"同塔已用过"/"本回合已升基地"这两条规则就被重置了。

最小 3 方实验（`code/test_match/probe_oplist_batching.py`，seed 10，先空过 40 回合攒钱，在 `(4,2)` 建 0 号塔，然后 p0 出一回合 `2\n12 0 1\n13 0`，即"把 0 号塔升成 Heavy(1) 后再降级同一座塔"）：

```
facade 整列表一次调用（正确用法）: 测试回合 apply 后=[(0, 1, 15)] 金币=[55, 150]
facade 拆成两次调用（错误用法）  : 测试回合 apply 后=[(0, 0, 10)] 金币=[109, 150]
  -> 两种调用方式**不一致**

官方判题器 judge_O0.exe: replay 43 条
  round  42 op0(已接受的 op)=[{'args': 1, 'id': 0, 'pos': {'x': -1, 'y': -1}, 'type': 12}] 塔状态(id->type)={0: 1}
官方判题器 最终塔(id->type)    : [(0, 1)]
facade 整列表 最终塔(id->type) : [(0, 1)]   ✅ 与官方一致
facade 拆两次 最终塔(id->type) : [(0, 0)]   ❌ 与官方不一致
```

官方 replay 的 `op0` 里只剩 `12 0 1` 一条 —— `13 0` 被 "Tower has been used" 吃了，塔留在类型 1；而逐条 apply 时 `13 0` 生效，塔降回类型 0、还多拿一笔降级返还（金币 109 vs 55）。**从这一回合起局面就分叉了。**

补充说明（避免过度归因）：在 48 条日志的 **17513 次"有出招的(回合,侧)"** 里，37.4% 一回合出招 >1 条，但**没有任何一次**出现"同一座塔一回合被操作两次"或"31+32 同回合两个基地升级"。所以机制 B 在这批日志里**没有真的咬到**，属于必须写进规范但这次不是病根的坑。

### 4.4 机制 C（环境坑）：官方判题器在中文路径下静默写不出 replay

```powershell
path= C:\Temp\probe_build\p_ascii.json
   rc 0 exists True 70057 stderr b''
path= D:\2026智能体大赛_新2_distill\code\cpp_engine\build_probe\replay\p_nonascii.json
   rc 0 exists False  stderr b''
```

同样 3 个 pass 回合、同样的输入，只有路径不同。MinGW 窄字符 `std::ofstream` 在 GBK 代码页下解析这个 UTF-8 中文路径失败，`fout` 打开失败而代码**没有检查**（`output.cpp:221-227 dump_all`），于是 `rc=0`、`stderr` 空、replay 不存在。
→ 本机做"官方判题"时必须把 replay 落到 `C:\Temp\...` 这类纯 ASCII 路径，否则会以为"官方判题跑通了"其实什么都没产出。

### 4.5 顺带排除的假设

| 假设 | 结论 | 依据 |
|---|---|---|
| "Python 引擎重写了规则" | 否 | `game.cpp` 与官方 diff 0 行；facade 直调 pybind |
| "编译器 `-O2` 导致浮点分叉" | 本次未发生 | `judge_O0` vs `judge_O2` 在 1792 回合里逐回合 base_hp/coins 全等 |
| "`movement_policy` 用错了" | 否 | 官方缺省就是 `Enhanced`（`game.cpp:125-132`），我们传的也是 `enhanced` |
| "seed 对不上" | 否 | 探针用 `对战 seed=N`，与 `RESULT seed=N` 一致 |
| "`cold_handle_rule_illegal` 造成差异" | 不是本次病根，但是**平台风险** | 见 §5.0；48 条日志 0 次非法操作 |

---

## 5. 建议：可靠回放 / 判题的落地方案

### 5.0 先说一条比引擎一致性更该修的

`match_via_sdk.py:405` 用 `cold_handle_rule_illegal=True`（非法操作静默跳过），而**官方判题器缺省 false**（`game.cpp:134-140`），非 cold 路径一旦 `apply_operation` 失败就 `set_AI_state_IO(player)` ⇒ `is_end=1, winner=1-player`（`game.cpp:2618-2626, 2804-2809`）。**本地对局里出非法招没有惩罚，平台上直接判负。**
建议：在自对弈/评测里加一个 `cold_handle_rule_illegal=False` 的"影子校验"（只有第一次非法操作时才需要跑，成本极低），或者至少在 `match_via_sdk.py` 里把 `il0/il1` 非空但 `cold=True` 的对局标记为"平台上会输"，统计比例。本次抽查的 48 条日志里非法操作 0 次（`Select-String 'ILLEGAL'` 命中 0 文件），暂时是安全的。

### 5.1 三种用途，分三种引擎用法

| 用途 | 用什么 | 命令 | 期望输出 |
|---|---|---|---|
| 日常自对弈 / 训练 / 快速批量评测 | **`GameStateFacade`（现在的 pyd）** | `python code/test_match/match_via_sdk.py --trace-ops --ai0=… --ai1=… 7` | 已是权威等价（48/48），440 回合约 **0.13–0.25 s**，比官方判题器快 ~10–20× |
| **权威判题**（争议局 / 提交前复核） | **官方 `main.exe`（`-O0` 原样编译）** | 见 §5.2 | 逐回合 `base_hp`/`coins` 与日志 RESULT 一致 |
| 只要"官方背书的 Python 接口" | `Ant-Game/SDK/native_adapter.py` + `SDK.native_antwar` | `cd Ant-Game && python tools/setup_native.py build_ext --inplace` | `NativeGameStateAdapter.initial(seed=…)` 可用（但慢、且 MSVC 默认优化非 `-O0`） |

### 5.2 权威判题的标准动作（推荐直接落地）

```powershell
# (1) 编官方判题器（一次性；产物不进 git）
New-Item -ItemType Directory -Force C:\Temp\probe_build,code\cpp_engine\build_probe | Out-Null
bash -lc 'cd /d/2026智能体大赛_新2_distill && export TMP="C:\Temp\probe_build" TEMP="C:\Temp\probe_build" TMPDIR="C:\Temp\probe_build" && \
  /c/mingw64/bin/g++.exe -std=c++17 -g -I Ant-Game/game/include Ant-Game/game/src/*.cpp \
  -o code/cpp_engine/build_probe/judge_O0.exe'
Copy-Item C:\mingw64\bin\libstdc++-6.dll,C:\mingw64\bin\libgcc_s_seh-1.dll,C:\mingw64\bin\libwinpthread-1.dll code\cpp_engine\build_probe\ -Force

# (2) 复核任意一条 ladder 日志（三方对拍；replay 落 ASCII 路径）
$env:PYTHONIOENCODING="utf-8"
D:\anaconda3\envs\pytorch-gpu\python.exe code/test_match/probe_replay_engines.py `
  "match_results/ladder_logs/<dir>/<seed>.log" --judge code/cpp_engine/build_probe/judge_O0.exe
```

期望：

```
[facade vs RESULT] hp: facade=(0,28) log=(0, 28)  -> 一致 ✅
[judge judge_O0.exe vs RESULT] hp=(0,28) log=(0, 28)  -> 一致 ✅
[facade vs judge_O0.exe] 每回合 base_hp 完全一致 ✅
[facade vs judge_O0.exe] 每回合 coins 完全一致 ✅
```

### 5.3 回放/重放的三条硬规则（写进代码规范）

1. **出招从 `原文` 解析，绝不要 `ast.literal_eval` 那个 `Operation(...)` repr。** 想用 repr 就必须写专门的 regex 提取 `op_type`/`arg0`/`arg1` 数字。现成实现：`code/test_match/probe_replay_engines.py:parse_log / parse_ops_text`。
2. **一回合的出招必须一次 `apply_operation_list(player, ops)` 全量传进去。** 绝不要 `for op in ops: apply(p, [op])`。理由见 §4.3（`used_tower` / `camp_upgraded_flag` 是调用内的局部变量）。
3. **官方判题器相关路径必须纯 ASCII**（exe 所在目录、replay 输出目录、`TMP`）。理由见 §4.4。

### 5.4 建议顺手修的两处（本次只读，未改）

- `code/cpp_engine/build.sh:29`：`-O2` → `-g`（对齐官方 `-O0`），并把 `REPO` 从 `D:/2026智能体大赛_新2` 改成 `_distill`。改完重编 pyd 后，用 §5.2 的三方对拍回归一遍（预期全绿）。**注意：重编 pyd 会改变正在跑的实验的数值行为（哪怕理论上等价），建议等当前批跑完再动。**
- `code/test_match/recall_on_runnerup.py:34-48,126-136`：换成 `probe_replay_engines.py` 的解析 + 整列表 apply，即可让它自己的"忠实重放"自检通过。

---

## 6. 证据索引（本次新增的探针与命令）

| 文件 | 作用 | 运行 |
|---|---|---|
| `code/test_match/probe_replay_engines.py` | 单条日志三方对拍（facade / judge_O0 / judge_O2），逐回合 base_hp+coins，并核每回合塔快照 | `python code/test_match/probe_replay_engines.py <log> --judge code/cpp_engine/build_probe/judge_O0.exe` |
| `code/test_match/probe_replay_batch.py` | 批量日志忠实性统计（终局 hp + 每回合塔快照） | `python code/test_match/probe_replay_batch.py <dir> --limit 32` |
| `code/test_match/probe_replay_bug_demo.py` | 复现 `(4,0) vs (0,28)`，4 种解析/施加组合隔离归因 | `python code/test_match/probe_replay_bug_demo.py <log>` |
| `code/test_match/probe_oplist_batching.py` | "整列表 apply vs 逐条 apply"最小 3 方实验（官方判题器参与） | `python code/test_match/probe_oplist_batching.py` |
| `code/cpp_engine/build_probe/` | 临时编译产物（`judge_O0.exe` / `judge_O2.exe` / MinGW DLL / replay），**已加入 .gitignore** | 见 §5.2 |

所有命令都在 `PYTHONIOENCODING=utf-8` + `D:\anaconda3\envs\pytorch-gpu\python.exe` 下执行；全程只用了 1 个编译进程 / 1 个 python 进程，未触碰正在跑的 10 局 ladder。

---

## 7. 未完成事项

1. **没验证 MSVC 构建**。比赛平台大概率用 MSVC；本次只对比了 MinGW `-O0` / `-O2`。`double` 打分 + 排序的代码在 MSVC 下可能有不同的 FP 收缩（`/fp:precise` vs `-ffp-contract`），**这是目前唯一还没排除的引擎分叉风险**。若要彻底排除，需要拿平台使用的编译器和选项再跑一次 §3.3。
2. **没跑通 `Ant-Game/tools/run_local_match.py` 端到端**（它需要先把 AI 用 `AI/package_ai.sh` 打包成 `main.py` 目录，属于"改我们的 AI 打包方式"，超出本次只读范围）。我只验证了它依赖的 `main.exe` 协议层能被驱动。
3. **`code/cpp_engine/native_game_judge.pyd` 来源未知**：全仓无任何引用，`git log --all -S native_game_judge` 也无记录（`.pyd` 本来就被 `.gitignore` 忽略），时间戳 2026-08-10。看起来是历史上某次试验留下的，**未使用、也未验证**，不影响当前链路。
4. **`native_game.pyd` 是否为当前源码所编未做逐位校验**（它的 mtime 比源码早，但 git checkout 会重置 mtime，不可信）。间接证据是：它在 48 条日志 / 1792 回合上与现编的官方判题器完全一致，所以**行为上是当前源码**。
5. **只覆盖了 48 条日志**（约 2 万回合），没有跑完全部 `match_results/ladder_logs` 下所有目录。若要更彻底，可对全部日志跑一遍 `probe_replay_batch.py`（每条约 0.2 s，成本很低）。
