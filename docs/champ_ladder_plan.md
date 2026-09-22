# 计划：把"我们 ↔ rule_v4 ↔ 冠军/亚军"接到同一把尺子上（含环境探针）

> 日期：2026-09-22（用户提出）。
> 状态：**计划阶段，待过目；不要先动 `match_via_sdk.py`。**
> 动机：用户 2026-09-22 想试"用 dsh goal 跑几周，让 AI 自己做 ML/RL 做出能打败冠军和亚军的 AI"。
> 讨论后的结论是：**在阶梯接上之前，"打败冠军"不是一个可优化的目标（距离未知）**，所以先补度量。

## 0. 一句话

我们现在有两条互不相通的读数链：
（我们 vs rule_v4）在 **Python 引擎的 eval 协议**下；（rule_v4 vs 冠军/亚军）在 **bridge（我们的 cpp_engine）**下。
**我们 vs 冠军/亚军这一格是空的**，而它是目标本身。本计划就是把它接上，并顺带把环境健康度做成可判定的。

## 1. 已核实的事实（2026-09-22，不是记忆）

| 项 | 状态 | 来源 |
|---|---|---|
| 我们 raw vs rule_v4 | **0%** | 本轮会话核对 |
| 我们 + 256 搜索 vs rule_v4 | **30–47%**（最好 46.9% = 32 局 parity；56.2% 是幸运 16 局）| `memory/feedback_az_strong_baseline_parity_ceiling.md` |
| rule_v4 vs 冠军 / 亚军 | **0%**（用户实测）| 用户 2026-09-22 |
| 冠军 vs 亚军 | ~50/50（9 局 512 回合 **4:5**，全部打满回合上限）| `memory/project_champion_runnerup_cpp_match.md` |
| **我们 vs 冠军/亚军** | **完全没测过** | — |

**两个 runner 各缺一半（这是本计划要修的第一件事）**：

| runner | 裁判 | `.py` AI 支持 |
|---|---|---|
| `code/test_match/match_via_sdk.py` | ✅ `GameStateFacade`（= `code/cpp_engine`，**两个 C++ AI 在它下面打得正常**）| ❌ `start_ai` 直接 `Popen([exe])` |
| `code/test_match/run_cpp_ai_match.py` | ❌ `GAME_BIN = Ant-Game/game/output/main`（**正是当年让两个 AI 全弃权的那份**）| ✅ `proc.suffix == ".py"` → `sys.executable` + `Ant-Game` 进 PYTHONPATH（`:65-70`）|

⇒ 阶梯必须建在 `match_via_sdk.py` 上，并**把 `.py` 支持从 `run_cpp_ai_match.py:65-70` 搬过去**（照抄，不重写）。

`code/test_match/rv4_pkg/`（known-good control：官方 `main.py` + `protocol.py` 的 `ProtocolSession` + rule_v4 的
`ai.py`/`common.py`/`position_utils.py`/`score_utils.py`/`operation_utils.py`）**仍在**，可直接复用。

## 2. Phase A —— 环境探针（先做，且在接我们的 AI 之前）

### A1. `rule_v4 vs rule_v4`（镜像）——**唯一有可推导零假设的对照**

**预注册签名**：同一份 AI、同 seed、交换先手 ⇒ 确定性下**每一对必然恰好一胜一负** ⇒
**配对分 = 1.0，方差 = 0**。

⚠️ 这条签名来自一次真实翻车（`memory/feedback_known_good_control_for_harness.md` 第四条）：
同一份代码的镜像 A/B 报出"8 局 4 胜 4 平 **0 负**"，被当成"镜像不可能输"接受了——
实际是判定代码把 `== opp_player` 写成了 `!=`，**把对手的胜利记成了平局**（`eval.py` 里是对的，那份新脚本是重打的）。

⇒ **纪律**：① 读任何胜负前先审判定代码；② 判定逻辑**从 `eval.py` 抄**，不要重打；
③ 若 A1 不是 1.0/方差 0，**先怀疑自己的判定与中继，不要往下走**。

### A2. `rule_v4 vs 冠军` / `rule_v4 vs 亚军`（镜像）——**用户提出的环境探针**

**预期 ~0%**（一个跟我们打平的 13 名规则 AI，被前两名打到 0%）。

⇒ **若明显非 0，那就是环境有问题的信号**，不是"rule_v4 变强了"。这个方向的逻辑与 08-10 那次
"用 rule_v4 当 known-good 证明 relay 正确"完全对称（那次是"它动了 ⇒ 环境对"，这次是"它赢了 ⇒ 环境可疑"）。

### A3. 胜负口径对齐（否则 0% 可能是口径的产物）

`match_via_sdk.py` 的胜负 = `state.winner`（若为 0/1）**否则 base_hp tie-break**；
而官方 judge 有自己的 score map（`end_info {"0":0,"1":1}`，赢家=1）。
**必须把两者的口径写下来、并在同一批局上对照**——尤其是 512 回合打满、由 tie-break 定胜负的那些局
（冠军 vs 亚军 9 局**全部**是这种）。**若 tie-break 与官方 score map 不一致，"0%"和"4:5"都要重读。**

## 3. Phase B —— 把我们自己的 AI 接进去

1. `match_via_sdk.py` 加 `.py` AI 支持（抄 `run_cpp_ai_match.py:65-70`）。
2. **把我们当前最强的配置包成协议 AI**。⚠️ 这一格的工作量**尚未估**：我们的 AI 是 in-process 的，
   而 `rv4_pkg` 的壳是"官方 SDK 的 `ProtocolSession` + rule_v4 的决策函数"——
   我们的 AI 需要同样一个"收局面文本 → 出操作文本"的壳，且**内部要跑那套 256 迭代搜索**。
   先估再写（这是本计划里唯一可能"比预想大"的一步）。
3. 同裁判、同镜像 seed 测四条线：**我们 vs rule_v4、我们 vs 亚军、我们 vs 冠军**（外加 A1/A2 做对照）。

## 4. 判据与读法

- **主读数**：镜像**配对**胜率（每个 seed 两局、交换先手），报 SE；不报单臂绝对值（本项目历史：单臂排名不可复现）。
- **附带读数**：每局回合数、base_hp、coins、双方出招构成（`--trace-ops`），以及第一座塔的回合。
- **读任何"我们赢了 X"之前，先看 A1/A2 的环境签名**：环境不可信 ⇒ 所有结论作废。
- ⚠️ 我方每步搜索约 2.4 s × ~512 步 ≈ **20 min/局**；并行度按 §7 的拐点走（6–8），不要贪 16。

## 5. 风险

| 风险 | 说明 |
|---|---|
| **判定代码重打** | 已知翻车（`!=` 把败记成平）。只从 `eval.py` 抄，且 A1 先验签名 |
| **tie-break 与官方 score map 不一致** | 冠军/亚军 9 局**全部**靠 tie-break 定胜负 ⇒ 口径错了整张表都错（§A3）|
| **我们的 AI 在 bridge 下"看起来不动"** | 08-10 已踩过（两个 C++ AI 因 desync 全弃权，其实是判官问题）⇒ 先 A1/A2 排除环境，再解释 |
| **协议壳工作量未估** | §3.2 是唯一可能超预期的一步 |
| **成本** | 我方带搜索 20 min/局 × 4 条线 × 镜像 ⇒ 按 6–8 路并行，整条阶梯约数小时 |

## 6. 与本文档之外的关系

- 这是"**让 AI 自己做 ML/RL 打赢冠军**"这个大目标的前置度量工作，**不是**那个大目标本身。
  大目标的判断（乐观/悲观、goal 模式怎么用）见对话记录与 §7。
- `memory/project_champion_runnerup_cpp_match.md`（08-10）给了"哪个裁判可用"；
  本文档要回答的是"**在这个可用的裁判下，四方各自的真实距离**"。

## 7. 实施顺序

1. **【已做】16 局镜像（seed 7–14 × 双方各先手）** —— 修好判定口径后它本身就是 A2 的一半（冠军/亚军互打）。
2. `match_via_sdk.py`：加 `.py` AI 支持 + **加一行 ASCII 机器可读的 `RESULT` 行**
   （`seed/first/winner/rounds/terminal/state_winner/base_hp/coins`），让脚本别再自己推胜负。
3. **A1** `rule_v4 vs rule_v4` 镜像（8 seed × 2）⇒ 校验签名 = 1.0 / 方差 0。
4. **A2** `rule_v4 vs 冠军` + `rule_v4 vs 亚军` 镜像 ⇒ 校验 ~0%。
5. **A3** 口径对照（tie-break vs 官方 score map）。
6. **B** 包装我们的 AI → 跑"我们 vs rule_v4 / 亚军 / 冠军"。
7. 记档：把阶梯表写进 `docs/experiment_log.md`，并把"环境是否健康"单独写成一条结论。

## 8. 预注册：具体命令与预期读数（2026-09-22 写死，跑之前）

运行器 = `_tmp_ladder.py`（并行跑 bridge、只读 `RESULT` 行、报镜像配对）。**每个 seed 跑两局**
（`N` = AI0 先手、`Nr` = AI1 先手）。并行度按实测拐点取 **8**（不要 16）。

| 步 | 命令（`P` = `D:/anaconda3/envs/pytorch-gpu/python.exe`） | **预期读数（跑之前写死）** |
|---|---|---|
| **A1** | `bash code/run_logged.sh ladder_A1_rv4_self $P -u _tmp_ladder.py --tag=A1_rv4_self --jobs=8 --ai0=code/test_match/rv4_pkg/main.py --ai1=code/test_match/rv4_pkg/main.py 7 7r 8 8r 9 9r 10 10r` | **每一对 AI0 得分 = 1.0，方差 = 0.0000**；`verdict=engine`；`ill=0`；**僵局检查 = 无**（rule_v4 会出招）；两局逐字段（rounds/base_hp/coins）**完全相同** |
| **A2a** | `bash code/run_logged.sh ladder_A2_rv4_vs_champion $P -u _tmp_ladder.py --tag=A2a --jobs=8 --ai0=code/test_match/rv4_pkg/main.py --ai1=其他版本ai/ant-war2-magica-v3/magica_v3_O0.exe 7 7r 8 8r 9 9r 10 10r` | rule_v4 胜率 **≈ 0%**（配对得分 ≈ 0.0）。**非 0 就是环境可疑的信号**，不是"rule_v4 变强" |
| **A2b** | 同上，`--ai1=其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe` | 同上 |
| **B1** | `AZAI_CKPT=<选定> AZAI_ITERS=256 AZAI_DEPTH=4 ... $P -u _tmp_ladder.py --tag=B1_us_vs_rv4 --jobs=6 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py 7 7r ... 14 14r` | 我们 vs rule_v4（**这一格现在完全没测过**）|
| **B2/B3** | 同 B1，`--ai1` 换成冠军 / 亚军 | 我们 vs 冠军 / 亚军（目标本身）|

**读 A1 的一个额外好处**：`rule_v4` 是纯 Python 确定性代码（没有墙钟预算）⇒ A1 把"**harness 的确定性**"
与"**AI 侧的时间依赖**"分开了。即使冠军那种时间依赖的不确定性存在，A1 仍应给出干净的 1.0。

⚠️ **跑之前必须确认机器空闲**：实测 `DET_seed11_run1` 就是因为我在它旁边跑集成冒烟抢了 CPU，
第 84 回合报 `timed out reading ai_cpp_lure_v4` 整局作废（当时门限还是 120s，现已改 300s）。
**同一个 seed 的对照必须同批、同并行度**——`mirror8x2` 那次已经证明 6 路与 16 路跑出的胜负会翻转。

### 8.1 ⚠️ 已证实：**冠军/亚军的对局不可复现**（2026-09-22，四跑）

`DET_seed11_run1` / `DET2_seed11_run1` / `DET2_seed11_run2` / `DET3_seed11_hexdump`（全部 1 路、同 seed 11）：
前三次都停在第 84 回合且读数逐字段相同，第四次却**打满 512 回合**，而且**第 84 回合的局面已经完全不同**
（5 塔/11 蚁 vs 3 塔/9 蚁，位置与 ID 全不一样）。

⇒ **同一 seed 会跑出不同的棋局**（分歧发生在第 84 回合之前）。bridge 是确定性 Python，局面文本由同一份
`state_to_text` 生成 ⇒ 不可复现只能来自**两个黑盒 AI 自身**（最可能：墙钟预算，或进程相关的遍历顺序）。
**我们改不了别人的二进制。** 详见 `docs/experiment_log.md` 该条 result。

**三条后果**：
1. **逐 seed 跨批对照无效**：`mirror8x2` 与 08-10 那份名单之间 seed 9/11 的"翻转"根因在此，**不是并行度**。
   只有**同批内的聚合胜率**（足够多局）才有意义；单局/单 seed 的胜负不能当证据。
2. **§A1 的镜像签名只对确定性 AI 成立**：它不能用在冠军/亚军上，**但能用 rule_v4**（纯 Python）——
   这正是 A1 被设计成"rule_v4 打自己"的价值：它测的是 **harness**，不是那两个黑盒。
3. **按"偶发中止"设计**：1 路下 seed 11 有 3/4 次卡在第 84 回合（亚军那边 1.8 秒就算完了，是收发的锅）。
   异常局现在标 `INVALID` 且不计分（commit `57724c0`）；必要时对 `INVALID` 的局重试一次再判。

### 8.2 待用户拍板（B 的前置）

壳从环境变量读搜索配置（默认 `iters=256 depth=4 mode=joint pospin=argmax t_class=0.5
t_pos=1.0 temp=1e-6`）。**B 要报强度，"我们"必须是项目的"最强已知"那一套**，而记忆里对
rule_v4 最好的读数是 `r69p+v50v` / `az_mix_gen0120p_v50v` 那一族（46.9%），
与近期的 `posnet_A_k5_m32` + `valnet_A2` **不是同一条线** ⇒ 用哪套当"我们"需用户指定。
