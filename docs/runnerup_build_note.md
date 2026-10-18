# 亚军（saiblo-30th-AI/Game1, cpp_lure_v4）如何在本机正确编译

> 2026-10。为做"把亚军价值函数换成线性/零值再对战"的消融，需要重建亚军 exe。
> 这里记录踩过的坑，避免以后再花时间。

## 1. 依赖

- 官方引擎源码：`Ant-Game/game/src/*.cpp` + `Ant-Game/game/include`（本仓库子模块里有）。
- SDK：`其他版本ai/saiblo-30th-AI/Game1/antgame_cpp_sdk`。
- **预编译静态库**：主工作区 `D:/2026智能体大赛_新2/其他版本ai/saiblo-30th-AI/Game1/antgame_cpp_sdk/build/libantgame_cpp_sdk.a`
  （本 worktree 没有 build 产物；复制过来即可，约 1.5 MB）。

## 2. 关键坑

1. **`Game1/Ant-Game` 不存在**：亚军的 Makefile 用 `../../Ant-Game/game/include`。
   需要建 junction：`mklink /J 其他版本ai\saiblo-30th-AI\Game1\Ant-Game D:\...\Ant-Game`。
2. **MinGW ld 不认中文路径**：库文件 / `-o` 输出路径都要放**纯 ASCII 目录**（如 `D:\dsh_build_tmp\`），
   否则报 `cannot open output file` / `cannot find ...libantgame_cpp_sdk.a`。
3. **临时目录**：`TMP`/`TEMP` 也要设成 ASCII 路径，否则汇编器报 `can't create ...Temp\cc*.o`。
4. **⚠️ 最重要：Windows 文本模式会破坏协议**。
   亚军的协议是「4 字节大端长度 + 原始字节」；`std::cout` 在 Windows 默认**文本模式**会把 payload 里的
   `\n` 翻成 `\r\n`，导致**长度与实际不符**，桥接端表现为 `round 1` 直接 `TimeoutError`。
   **修复**：在 `ai_cpp_lure_v4.cpp` 的 `main()` 里加
   ```cpp
   #ifdef _WIN32
   #include <fcntl.h>
   #include <io.h>
   ...
   _setmode(_fileno(stdin), _O_BINARY);
   _setmode(_fileno(stdout), _O_BINARY);
   #endif
   ```
   原版 exe 应该是 Linux/交叉编译的，所以没有这个转换。加完即可正常跑（实测到 round 55+）。

## 3. 编译命令（已验证）

```powershell
$env:TMP = "D:\dsh_build_tmp"; $env:TEMP = $env:TMP
$env:PATH = "C:\mingw64\bin;" + $env:PATH
g++ -std=c++17 -O3 -DNDEBUG -w -pipe `
    -I源/include -Iantgame_cpp_sdk/include -IAnt-Game/game/include `
    源/ai_cpp_lure_v4.cpp D:\dsh_build_tmp\libantgame_cpp_sdk.a -o D:\dsh_build_tmp\ai.exe
```

## 4. 消融开关（已加在 `lure_strategy_v4_terminal_eval.hpp`）

- `-DLURE_LINEAR_VALUE`：`total_score = base_hp*200 + (coins + 塔回收)*10`（去掉 stepped decay / threat / c1 / future）
- `-DLURE_ZERO_VALUE`：`total_score = 0`（纯 heuristic 打分）

## 5. 已得到的结论

- **离线消融**（`analyze_value_ablation.py`，用 seed900 的 plans 日志）：把价值项去掉后，
  **41.8% 的决策 argmax 会变** ⇒ 价值**不是**可有可无的弱 tie-breaker，之前的"先验更重要、value 可收尾"**结论过强**。
- **直接对战消融（各 32 局，seed 700–731，jobs=8，带 `MVS_RAW_STATE_DIR`）**：
  - **zero 版（价值=0）**：原版 **27 胜 5 负**（变体胜率 15.6%，−3.9σ，**极显著更弱**）。
    zero 版经常被直接推平（base_hp=0，回合数 306/315/324…）。⇒ **价值是亚军的命脉，去掉就崩**。
  - **linear 版**（`base_hp*200 + 等价金币*10`，去掉 threat/stepped decay/c1/future）：
    **32 局 13:19** + **64 局 26:38** ⇒ **合计 96 局 39:57（linear 胜率 40.6%，z≈−1.84，双侧 p≈0.066）**。
    ⇒ **略差约 9pp（边缘显著）**：亚军的 threat/stepped-decay 精细项是**真实但不大**的贡献。
    这与"价值近似线性于 stats"（线性 R²=0.97）吻合但不完全等价。
- 消融局本身也是**有效对局数据**（价值标签由 `evaluate_terminal` 从状态算出，与谁执子无关），
  已开 dump（`abl_zero_raw` / `abl_linear_raw`），可并入价值训练集。
- **注意**：单局不能下结论（亚军自对弈本来就近 50% 波动），至少 32 局才统计可用。

