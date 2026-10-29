# 我们 vs 冠军：**首次基线**（E1），以及 E2 / E4 / E4-rand 的预注册设计（2026-10-28）

> 一句话：**我们（不注入）对冠军 = 0/16（0.0%）**，16 局全部 `verdict=engine`、`illegal=0`；
> 冠军当对手的三个工程坑（DLL/PATH ⇒ 0xC0000139 秒退、stderr ~0.26MB/回合、单局墙钟）都已量化并有修法。
> E2（注入冠军动作）/ E4（候选集只剩「冠军动作+亚军动作」）/ E4-rand（随机挑一个）的开关与脚本已就绪，
> **默认关闭且逐位无影响**（两层证据见 §5.4）。

---

## 0. 口径声明（读结论前必看）

### 0.1 "我们 vs 冠军"在这套配置下**是第一次**

- 历史只有 **`rule_v4` vs 冠军 = 0/16**（`docs/experiment_log.md` 的 A2a 批，2026-09-22，用时 1417s，
  每局 311–437 回合被爆基地）；
- 2026-09-23 的 `B1_us_vs_champion` 是**老配置**（`AZAI_MODE=joint / POSPIN=argmax / MCTS iters=256`，
  那时还没有 `AZAI_SEARCH=flat`、没有 K=48 / FLAT_ROUNDS=49152），读数 0/16、用时 2741s，
  日志目录**已不在仓库里**（只剩 `experiment_log` 的文字），**不能与本批配对**；
- 本批 = 现在现役的 **BASE**（K=48 / FLAT_ROLLOUTS=64 / FLAT_ROUNDS=49152 / random+decay(1.0,3.0) /
  horizon16 / share_prefix=1 / FLAT_SEED=53），与"注入 A/B 打亚军"那批**逐字相同**（脚本里有断言）。

### 0.2 与"我们 vs 亚军"的读数**不能直接相减**

对手不同（冠军 vs 亚军），而且那批的 A 臂开了注入。**只做并排展示**（§4.3），不做差。

### 0.3 冠军**不是确定性对手**（历史证据），所以本批只做"单批内"的比较

`experiment_log` 2026-09-22 那批（冠军 vs 亚军，16 局）记录了**同 seed 同配置胜负翻转**（seed 9 / 11），
最可能的原因是冠军的决策受机器负载影响（当时的怀疑：MCTS 有墙钟成分；**本批没有去证实**）。
⇒ ① 不要用"同 seed 重跑应逐位相同"来校验冠军这一侧的 harness；
   ② 16 局的分辨率只够分辨**大效应**（Wilson 95%CI 在 0/16 时上界 ≈ 19.4%）。

---

## 1. 三个工程坑（本次最可复用的产出）

### 1.1 冠军 exe 缺 DLL ⇒ `0xC0000139` **秒退**，在 bridge 里伪装成"协议错/超时"

- **症状**：每一局都是 `verdict=aborted rounds=0`，`base_hp=50,50`；bridge 侧的异常是
  `OSError: [Errno 22] Invalid argument`（写一个已经死掉的 stdin）或 `EOFError`；
  冠军侧 stderr **为空**，进程 `poll = 3221225785`。
- **根因**：`3221225785 = 0xC0000139 = STATUS_ENTRYPOINT_NOT_FOUND`。
  `magica_v3_O0.exe` 是 **mingw 动态链接**的，而它的目录里**一个 DLL 都没有**（只有 exe + main.cpp + readme.md）
  ⇒ Windows 加载器沿 **PATH** 找 `libstdc++-6.dll` / `libgcc_s_seh-1.dll` / `libwinpthread-1.dll`；
  而本机 PATH 里 **anaconda 的 mingw 目录**（`D:/anaconda3/Library/mingw-w64/bin`）排在 `C:/mingw64/bin`
  **前面**，那份 libstdc++ 与它的导入表**不匹配** ⇒ 进程还没进 main 就退出。
- **实证**（三种 PATH，同一 exe、同一 `printf '0 7\n'` 输入）：

  | PATH | 结果 |
  |---|---|
  | 干净 PATH（只 `C:/mingw64/bin` + 系统） | `rc=0`，`[Magica] hello ... ReadInit 成功`，进入主循环 ✅ |
  | 亚军 build 目录（那三份 DLL）放最前 | `rc=0`，同上 ✅ |
  | 当前 shell 的 PATH（anaconda mingw 在前） | `rc=127`（等价 0xC0000139），无输出 ❌ |

- **修法（两处，都在仓库里）**：
  1. `code/test_match/match_via_sdk.py:magica_env()`：**只给 magica 这一个子进程**把已知可用的
     DLL 目录前置到 `PATH`（顺序：`MVS_MAGICA_DLL_DIR` → `C:/mingw64/bin` → 亚军 build 目录），
     并在 `start_ai()` 里按文件名 `magica` 自动套用 ⇒ **真实对局、注入会话、探针**三条路径一起覆盖。
     ⚠️ 故意**不改我们自己进程的 PATH**（我们跑 torch/pybind，改全局会带偏 `cpp_engine` 的 DLL 搜索）。
  2. （并行的兜底）把 `libstdc++-6.dll` / `libgcc_s_seh-1.dll` / `libwinpthread-1.dll` 拷进
     `其他版本ai/ant-war2-magica-v3/`：exe 目录的优先级高于 PATH ⇒ 谁先起来都不会再撞。
- **为什么这条值得写进报告**：它**不是**"冠军很慢"，但看起来**特别像**（都是"开局 0 回合就结束"）。
  下次换机器/换构建，先查 `3221225785`。

### 1.2 冠军的 stderr **爆量**：~0.26 MB/回合（一局 100 MB+），而且 O0 版**不认** `LOG_LEVEL`

- 实测（同一 BASE 配置、正常对局）：

  | 回合数 | 冠军 stderr |
  |---|---|
  | 3 | 10 KB / 107 行 |
  | 24 | 5.5 MB / 31,024 行 |
  | 60 | **15.7 MB / 94,496 行** |

  增长近似**线性于回合数**（越到后期越吵：r13 单回合 2046 行）⇒ 一局 400–500 回合 ≈ **100–130 MB**。
- 它吵在**自己的 MCTS 调试行**（`[MCTS:Sim]` / `[MCTS:Expand:...]` / `=== expand end ===` / `[Planner:*]`）。
- **`ANTWAR_MAGICA_LOG_LEVEL` 在 O0 版里不起作用**：设 `-1 / 0 / 2 / 不设` 四种情况，3 回合都是
  **10,096 B / 107 行**（逐字节相同）。用二进制里的字符串表核对，这个 exe 只认这些环境变量：
  `ANTWAR_MAGICA_IO_TRACE` / `ANTWAR_MAGICA_RUNTIME_ARTIFACT_DIR` / `ANTWAR_POLICY_RECOVER_ROUNDS` /
  `ANT_MAGICA_MCTS_INFO_OUTPUT_PATH` / `ANT_MAGICA_TRACE_TIMING` / `MAGICA_ROLLOUT_HORIZON`
  —— **没有** log level 那一项（当前源码里有，O0 二进制是 2026-08-10 的更早版本）。
- **处置**：诊断只需要"它有没有自己报 desync / 有没有秒退"⇒ 归档时**冠军那份 gzip**（约 10:1），
  我们自己的 stderr 原样存（老工具直接读）。驱动里 `archive_stderr()` 按名字分流。
- ⚠️ 顺带一个**读数影响**：这 100MB+ 的写盘是**冠军进程自己**的开销（每局 ~40 分钟 ⇒ ~50 KB/s），
  量级小到可以忽略，但**不要**把它算进"我们的决策耗时"。

### 1.3 协议与超时（易踩的三条）

1. **magica 只接受 `\n`**（CR 会把它顶掉）：`match_via_sdk.run_match` 里已有 `is_magica` 专门分支，
   我们发过去的文本一律 `normalize_lines()` 归一化；**不要**自己起进程喂它（会绕过这个分支）。
2. **单局墙钟**：冠军是 O0 编译 + 最重的 MCTS ⇒ `_tmp_ladder.py` 新增 `LADDER_TIMEOUT`（默认仍是
   7200s = 改动前；本批 10800s），单次读包（卡死检测）`MVS_TIMEOUT_SECONDS=1800s`（默认 300）。
   这两个**只是防"慢局被掐断"**，不是每步预算；我们的搜索是**预算制**（没有 wall-clock 提前停止）。
3. **冠军当 oracle 时更慢**（E2/E4 用）：r0–r29 连续喂 30 回合的实测延迟见 §5.2。

---

## 2. E1 的设计（**跑之前登记**）

| 项 | 值 |
|---|---|
| 被试 | 我们（`code/test_match/az_bridge_ai.py`，**不注入**） |
| 对手 | 冠军 `其他版本ai/ant-war2-magica-v3/magica_v3_O0.exe` |
| 配置 | BASE（与 `_inject_vs_runnerup.sh` / `_oracle_inject_ab.sh` **逐字相同**，脚本里有断言） |
| 规模 | 16 局 = seeds **7..14** × 2 座位（`7` = 我们执 P0；`7r` = 镜像，我们执 P1）|
| 驱动 | `code/test_match/_us_vs_external.sh`（`EXP=E1`）→ `_tmp_ladder.py` → `match_via_sdk.py` |
| 并发 | `--jobs 4`（机器上另有升级诊断臂 8 局在跑，不抢满）|
| 超时 | 单局 `LADDER_TIMEOUT=10800s`；单次读包 `MVS_TIMEOUT_SECONDS=1800s` |
| 每局断言 | ① `verdict=engine`；② 裁判侧 `illegal=0` 且我方 `illegal_total=0`；③ 冠军侧 stderr 非空（证明它没秒退）|
| 主读数 | 胜率（Wilson 95%CI）、血量差（t 95%CI）、按座位拆分、每局墙钟 |
| 日志 | `match_results/ladder_logs/US_VS_CHAMPION_BASE/<token>.log`（`--trace-ops` 全量）+ `_run.log` |
| stderr | `match_results/us_vs_external/US_VS_CHAMPION_BASE/stderr/`（我方原样 + 冠军 `.gz`）|
| 命令 | `bash code/run_logged.sh us_vs_champion_baseline bash code/test_match/_us_vs_external.sh` |

**为什么这批的 seed 与亚军那批相同**：为了以后能**配对**看"同一张图上，换对手之后我们的相对表现"。
⚠️ 但配对只在"同一批内"成立（§0.3：冠军侧不可复现），**不要**把 E1 的 Δhp 与亚军批的 Δhp 逐局相减。

---

## 3. E1 结果（跑完回填）

（待填）

---

## 4. 判读

（待填）

---

## 5. E2 / E4 / E4-rand：脚本与开关（**默认关闭**，附零影响证据）

### 5.1 开关一览（全部默认关，关闭时行为逐位不变）

| 环境变量 | 作用 | 引入 | 默认 |
|---|---|---|---|
| `AZAI_ORACLE_INJECT=<exe>` | 每回合问一个外部 AI，把它的动作当**额外候选**注入搜索 | 2026-10-26 | 空 = 关 |
| `AZAI_ORACLE_TIMEOUT=<s>` | 上面那条会话的单次读包超时 | 2026-10-26 | 10s |
| **`AZAI_ORACLE_INJECT_B=<exe>`** | **第二个**外部会话（E4 用：A=冠军、B=亚军）；两条动作一起进候选 | **2026-10-28（本次）** | 空 = 关 |
| **`AZAI_ORACLE_TIMEOUT_B=<s>`** | B 会话的超时（默认取 `AZAI_ORACLE_TIMEOUT`） | **本次** | 20s |
| **`AZAI_ONLY_EXTRA=1`** | **候选集只保留额外候选**（网络那 K 条被类轴 pinned 的不进候选） | **本次** | 0 = 关 |
| **`AZAI_ONLY_EXTRA_RANDOM=1`** | 同一候选集，但**不看评估器分数**随机挑一个（固定种子可复现；隐含上一条） | **本次** | 0 = 关 |
| **`AZAI_ONLY_EXTRA_SEED=<int>`** | 随机挑的随机流种子（默认 = `AZAI_FLAT_SEED`） | **本次** | 53 |

实现位置：
- `code/my_ai/az_intent/flat_rollout_greedy.py`：`self.only_extra = False`（默认）+ `set_only_extra()`；
  `search()` 里只有**一个** `if _only_extra:` 分支改变候选集，预算公式/rollout/叶值网/选择规则不动。
- `code/test_match/az_bridge_ai.py`：`_oracle_b_init/_oracle_b_ask/_oracle_b_step/_oracle_b_set_opp/
  _oracle_b_log_round`（B 会话）+ `_only_extra_enable/_only_extra_rng/_only_extra_pick`；
  `decide()` 里只多一个 `if _ORACLE_B["off"]`（默认恒 True ⇒ 走**原来那条** `_oracle_step()`）。

### 5.2 三个实验臂（脚本已就绪，命令可复制）

```bash
# E2：我们 + 注入**冠军**动作 vs 冠军（唯一变量 = AZAI_ORACLE_INJECT，timeout=20s）
bash code/run_logged.sh inject_champion_vs_champion bash code/test_match/_e2_inject_vs_champion.sh

# E4：候选集 = {冠军动作, 亚军动作}，评估器裁决 —— 对手分别是冠军 / 亚军（各 16 局）
bash code/run_logged.sh only_extra_vs_champion bash code/test_match/_e4_only_extra.sh
OPP=runnerup bash code/run_logged.sh only_extra_vs_runnerup bash code/test_match/_e4_only_extra.sh

# E4-rand：同一候选集，随机挑一个（对照臂）
bash code/run_logged.sh only_extra_rand_vs_champion bash code/test_match/_e4r_only_extra_rand.sh
OPP=runnerup bash code/run_logged.sh only_extra_rand_vs_runnerup bash code/test_match/_e4r_only_extra_rand.sh
```

三条**硬前置断言**（驱动里，任一失败立刻退出）：`AZAI_ORACLE_INJECT` 必须是 **`D:/` 形态**的
Windows 绝对路径（`/d/...` 会让 `Path.is_absolute()` 判 False ⇒ 注入**静默失败**）；
`AZAI_ONLY_EXTRA=1` / `AZAI_ORACLE_INJECT_B` 必须真的等于脚本里那一串；E1 必须一个注入开关都没有。
`AZAI_ORACLE_TIMEOUT=20` 的依据：冠军当 oracle 的实测单次延迟（`code/test_match/_champ_oracle_latency.py`，
从 r0 连续喂 30 回合，seed 7 / player 0）——p50 **211 ms**、mean 1167 ms、**max 3313 ms**（r9 起它开始出招，
r7–r20 集中在 2–3.3 s；空过回合只要 ~70–170 ms）。⇒ 20 s ≈ 实测最大值的 **6 倍**。
⚠️ 这是**早期回合**的量级：冠军的搜索随局面复杂度增长（它的 stderr 量就是证据，§1.2），
所以 20 s 既不该再压小、也不该当成"它一定不会超时"的保证；超时的局在
`[oracle] 注入统计` 里表现为 `timeout>0` 且**该局之后不再注入**（驱动与读数脚本都会显式报这一项）。

| r0–6 | r7–10 | r11–13 | r14–20 | r21–29 |
|---|---|---|---|---|
| 62–125 ms（全空过）| 3.1–3.3 s（r9 起出招）| 78–140 ms | 0.95–2.4 s | 94 ms–2.3 s |


### 5.3 预注册判据（跑之前定好，避免事后挑）

| 实验 | 主读数 | 与谁比 | 判据 |
|---|---|---|---|
| **E2** 注入冠军 vs 冠军 | 胜率 + 血量差 | **E1**（同 seeds/座位，配对） | 配对 McNemar 精确 p < 0.05 **且** 胜率差 ≥ +12.5pp（2/16）才算"注入有效" |
| **E4** 只留两条外部动作 vs 冠军 | 同上 | E1（配对） | 同上（这是"候选/先验上限"的直接读数）|
| **E4 vs E4-rand** | 同上 | 彼此（配对） | 差值 = "评估器从 2 个候选里挑对"的贡献；显著 ⇒ 评估侧仍有增益空间 |
| E4 / E4-rand 的**对手换亚军** | 同上 | E1 的亚军对应批（`VS_RUNNERUP_cells4`） | 只做**并排**（§0.2 的口径限制）|

⚠️ 三条口径提醒：
1. **E4 的 `hit_sel` 不能当命中率读**：候选集只有 1–2 个 ⇒ "额外候选被选中"按构造恒为 1；
   要看 `[onlyextra]` 的 `ncand` 直方图与 `pick` 直方图。
2. E4/E4-rand 的**成本**比 E1 高：每回合多两个外部查询（冠军 O0 + 亚军）⇒ 单局墙钟会明显上涨，
   但**决策质量**不受墙钟影响（预算制）。
3. E4-rand 的随机流是 `sha256("onlyextra|<seed>|<dec_n>")`，与 rollout 的随机流**完全独立**
   ⇒ 同 (seed, 决策序号) 恒得同一选择，**可复现**。

### 5.4 零影响证据（开关关 ⇒ 逐位不变）

**① 搜索器层（逐位，`match_results/only_extra/zero_impact_probe.log`）**
`code/test_match/_only_extra_zero_impact.py` 在**真实局面**上（从 `VS_RU_dump/8.log` 的
`--trace-ops` 重放出来的 r20/r80 × 两个座位）把"**改动前**的 `flat_rollout_greedy.py` 副本"
（`_flat_rollout_greedy_preonlyextra.py`，md5 `60C895D34D93A9AE6B244FDC4BBDC944`，脚本里核对）
与**现行版本**放在**同一个进程**里跑同一个 `search()`，对
`last_scores`(float64 原始字节)/`visit_policy`/`chosen_bundle`/候选数/rollout 数/回合数打哈希：

- 8 次比较（4 个局面 × 默认路径/注入路径）**全部逐字段相同**（日志里没有任何 `!! 不一致` 行）；
- **正向对照**（证明开关不是死代码）：`only_extra=ON` 时候选数从 **48 条**塌成 **2 条**
  （= 额外候选去重后的条数），且 `chosen_bundle ∈ 额外候选`（2/2）。
- ⚠️ 这个探针**不动工作区**（不换文件）⇒ 不会污染当时正在跑的 E1。

**② 对局层（`RESULT` 逐字段，`_only_extra_zero_impact.sh` 第二段）**

（结果见 §5.4.1，跑完回填）

---

## 6. 双 oracle 逐回合记录（`[xor]`）：白拿的一批研究数据

E4 / E4-rand 里**两个外部会话同时在跑**（冠军 + 亚军）⇒ 每回合都能免费拿到"两个强 AI 在同一局面上
各选了什么"。这组记录**默认零日志**（两个会话都没开时 `_xor_log_round()` 第一行就 return），
只在 `AZAI_ORACLE_INJECT` / `AZAI_ORACLE_INJECT_B` 至少一个开着时才写，**绝不进 stdout**
（stdout 是二进制协议通道）。

### 6.1 格式（我们自己的 stderr，每回合一行 + 收尾/每 25 决策一行汇总）

```
[xor] round=R champ=<动作或()> ru=<动作或()> agree=<0/1|NA> dup_c=<0/1|NA> dup_r=<0/1|NA> chosen=<我们最终动作> hit_c=<0/1|NA> hit_r=<0/1|NA>
[xor-sum] rounds=N agree=<n>(xx.x%) both_hold=<n>(xx.x%) disagree=<n> champ_new=<n> ru_new=<n> chosen_eq_champ=<n> chosen_eq_ru=<n> chosen_eq_neither=<n> eq_champ_when_disagree=<n>/<n> champ_hold=<n>(xx.x%) ru_hold=<n>(xx.x%) champ_ops=<n> ru_ops=<n>
```

- `champ` / `ru`：冠军 / 亚军在该局面给出的动作（`()` = HOLD；多 op 束是 `((t,a0,a1),...)`）；
  E2 只有一个会话 ⇒ 那一行是 `ru=NA agree=NA dup_r=NA hit_r=NA`；
- `agree`：两者**完全相同**；
- `dup_c` / `dup_r`：该动作**已在我们自己的网络候选集里**（`1` = 我们本来就有，`0` = 真·增量）。
  口径 = `mcts.last_root.bundles`（`flat_rollout_greedy` 的**网络候选**，不含额外候选）；
- `chosen`：我们**最终真的发出去**的动作（过完闸门/执行层过滤之后）；
- `hit_c` / `hit_r`：`chosen` 分别等于冠军 / 亚军那一手；
- ⚠️ `_xor_log_round()` 在 `_oracle_log_round()` / `_oracle_b_log_round()` **之前**调用，
  且只**读**它们的 `round_log` ⇒ A/B 两条既有路径的日志与行为一个字节都不变。

### 6.2 这组数据能回答什么（E4 跑完后按这四条读）

1. **冠亚军一致率**（`agree`）：高（>70%）⇒ 好动作集中、最优解唯一，动作敏感度可能不低；
   低（<40%）⇒ 存在**大量近似等价的好动作**（支持"近最优高原"假设，也解释了"改一手不改变胜负"）；
2. **两者都 HOLD 的比例**（`both_hold`）：亚军 77.9% 是 HOLD（`docs/inject_vs_runnerup_20261028.md`），
   冠军也常在攒钱 ⇒ 预期很高。这直接解释"E4 与 E4-rand 的差别可能很小"（很多回合两个候选是同一个 `()`，
   拿掉评估器也不损失什么）；
3. **我们更接近谁**：`chosen_eq_champ` vs `chosen_eq_ru`，以及**两者不一致**的那些回合里我们选谁更多
   （`eq_champ_when_disagree / disagree`）——这是"我们的先验/评估器更像冠军还是更像亚军"的**行为学读数**；
4. **新信息占比**：`champ_new` / `ru_new`（有多少候选是我们网络根本不会给的），
   与 `hit_c/hit_r` 交叉 ⇒ "借来的招里有多少被我们采纳"（对应亚军那批的 45.7% 一致率）。

### 6.3 解析

```bash
# E4（对手=冠军）——同时把该臂的我方 stderr 复制到 match_results/xoracle/<TAG>/
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_xoracle.py \
    --tag ONLYEXTRA_VS_CHAMPION \
    --stderr match_results/us_vs_external/ONLYEXTRA_VS_CHAMPION/stderr
# E2（单会话；ru 全 NA）
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_xoracle.py \
    --tag US_VS_CHAMPION_INJ \
    --stderr match_results/us_vs_external/US_VS_CHAMPION_INJ/stderr
```

冒烟验证（3 回合 × 2 座位，`SMOKE_XOR2`）：`[xor]` 行正常产出、解析器能读、汇总行/JSON/独立归档目录都生成；
读数为 `agree=6/6 both_hold=6/6`（三个回合双方都在 HOLD —— **这正是 §6.2 第 2 条要看的现象**）。

---

## 7. 复现清单（全部落盘）

```bash
# 0) 冒烟（先把"冠军真的会出招"钉死，别拿几小时去赌）
ROUNDS=24 JOBS=2 SEEDS="7 7r" TAG=US_VS_CHAMPION_SMOKE bash code/test_match/_us_vs_external.sh

# 1) E1（16 局；本批实测 35–44 min/局）
bash code/run_logged.sh us_vs_champion_baseline bash code/test_match/_us_vs_external.sh
#    续跑：原样再敲（已 verdict=engine 的局会跳过）

# 2) 读数（逐局表 + 胜率/血量差/按座位/墙钟/verdict·illegal 核查）
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/_batch_table.py \
    --tag US_VS_CHAMPION_BASE --stderr match_results/us_vs_external/US_VS_CHAMPION_BASE/stderr \
    --opp-label champion --seeds 7 8 9 10 11 12 13 14
#    统计口径（Wilson CI / t 区间 / 行为画像）用同一批日志：
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_inject_vs_runnerup.py \
    --arm E1=match_results/ladder_logs/US_VS_CHAMPION_BASE \
    --stderr E1=match_results/us_vs_external/US_VS_CHAMPION_BASE/stderr \
    --json match_results/us_vs_external/US_VS_CHAMPION_BASE/summary.json

# 3) 零影响（开关默认关的证明：探针 + before/after self-play 的 RESULT 比较）
PROBE_ONLY=1 bash code/test_match/_only_extra_zero_impact.sh      # 轻，随时可跑
bash code/test_match/_only_extra_zero_impact.sh                    # 含 4 局 self-play，挑机器空的时候
```

## 8. 没做 / 没覆盖（诚实清单）

1. **没有**跑 E2/E4/E4-rand（任务书要求先只跑 E1）；脚本与开关已就绪（§5）。
2. **没有**解释冠军为什么会输/赢我们（黑盒，且它不可复现，§0.3）。
3. **没有**做"亚军当对手 + 相同 seeds"的同批对照（那批是 2026-10-28 的 `INJECT_VS_RU_B`，
   配置逐字相同 ⇒ 可以并排，但**不是**同一批）。
4. **没有**测冠军当对手时的**每步延迟**分位（只有单局墙钟与"当 oracle 时的延迟"，§5.2）。
5. `ANTWAR_MAGICA_LOG_LEVEL` 在 O0 版里无效这件事只做了 3 回合的对照（够定性，不够定量）。
6. `[xor]` 那组数据**只有冒烟**（3 回合 × 2 座位），E4 的正式读数还没有（实验未跑）。

