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
