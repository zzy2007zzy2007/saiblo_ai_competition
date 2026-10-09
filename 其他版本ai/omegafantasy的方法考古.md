# omegafantasy 的方法考古：他各届 AI 用的是什么算法

> 整理时间：2026-10-10
> 方法：**4 个子 agent 各自精读 + 我逐条抽查核实**（抽查结果见 §8）。所有结论都有 `文件:行号`。
> 对象：GitHub `omegafantasy` 的 9 个比赛 AI 仓库（本目录下 `Miracle-AI/` … `saiblo-30th-AI/`，清单见 [`omegafantasy的历届AI.md`](omegafantasy的历届AI.md)）。

---

## 0. 一句话结论

**全程零机器学习。** 9 个仓库里：

- **0 个权重文件**（`.pt/.pth/.onnx/.ckpt/.npz/.h5/.pkl/.safetensors` 一个都没有）
- **0 处 ML 框架导入**（`torch|tensorflow|jax|sklearn|xgboost|lightgbm|stable_baselines|nn.Module` 全仓库 0 命中）
- **0 处遗传/进化**（`genetic`/`evolution`/`CEM`/`GA` 真实命中全为 0，见 §8 第 1 条）
- **0 个训练脚本**；他租的是**高性能 CPU 服务器**（README 自述），不是 GPU

他的方法一直是：**手写规则 / 手写估值 + 手调常数 + 极浅的显式推演**，差别只在"有没有搜索骨架、搜索长什么样"。

而 2026 年真正新增的东西**不是算法范式**，是两件工程件：**大规模自动化实验设施（autolab）** + **LLM(Codex) 驱动的自动迭代闭环**。

---

## 1. 逐游戏方法表

| 游戏 | 届次 | 他自研的代码量 | 主算法 | 搜索形态 | 时间预算 |
|---|---|---|---|---|---|
| **Miracle** | 24 | `ai2.cpp` 1,620 行 | 优先级阶梯 + 加法评分 + **硬编码开局库** | 深度 1（枚举可达格） | 无（不搜索） |
| **AquaWar** | 25 | 3,897 行 | if-else + **身份约束传播（真信念推断）** | **无** | 无 |
| **LostSpace** | 25 | 2,229 行 | if-else 瀑布 + **字典序**多指标 + 跨回合信念状态 | 深度 1（sp=1 ⇒ 就是全部动作） | 无 |
| **SnakeGo** | 26 | `main.cpp` 2,743 行（另有官方 `adk.hpp` 905 行） | **对抗 MCTS / UCT**（树内显式换手） | 树，30 万节点 | **0.95 s** |
| **AntWar** | 27 | **`main.cpp` 1,671 行**（`include/` 8 个头是官方模板） | **单边 best-first 树搜索 + 规则状态机优先**（**不是 MCTS**） | 树，2 万节点，**无 UCB / 无随机 rollout / 无回传** | **0.15 s** |
| **Generals** | 28 | `main.cpp` 4,646 行（我实测） | 启发式流水线 + **112 分支宏动作 lookahead** + 单方 DFS + **背包 DP 威胁模型** + 模拟-验证闭环 | 浅层宏动作枚举 | `MAX_T1..T4` = 0.15/0.4/0.6/0.85 s |
| **RollMan** | 29 | `ai.h` 为主 | **MCTS + UCB1（`ARG=1.414`）** + 手写估值；另用 pac 视角 MCTS **预测对手** | 树，10,000 节点 | `TIME=0.1s` / `MAX_TIME=0.9s` 双闸 |
| **AntWar2** | 30 (Game1) | `lure_strategy_v4` 19 个模块头 | **模板候选 + 扁平 UCB1 bandit + 蒙特卡洛 rollout(4/8 回合) + 手写终点评估**，**无树** | 只在根节点做 bandit | `action_target_time_ms=4500` / `budget=6000` |
| **SnowCup** | 冰雪杯 | **1,288 行**（"23,100 行"里 25,855 是 `json.hpp`） | 纯启发式 + **人工势场（APF）** + 角色状态机 | 无 | 无（实时制，0.1 s 回调） |
| **DeepClue** | 30 (Game2) | ~1,144 行硬编码剧本 | **穷举式硬编码剧本状态机**（单人推理游戏，非对弈） | 无 | 无 |

---

## 2. 方法族：他用的是**四种不同的东西**，不是一条路线走到底

**族 A：对抗 MCTS + 手写估值**（SnakeGo 26届、RollMan 29届）
- 两届的 `Node` 结构**逐字段同构**（`count/win/origin/p/lc/rc` + `node[MAX_NODE_COUNT]`），`value_judge` / `randomplay` / `newnode` / `mcts` 一一对应
- SnakeGo：`main.cpp:158-161` Node、`:367-388 utcbest`（`ARG=1.0`）、树内**显式换手** `:1937-1999` → 真对抗（含对手建模）；rollout **不是均匀随机**，而是**手写启发式策略**（自杀规避 `:1567-1624`、气/贴边判定 `:853`、敌位 `:1086`、射线价值 `:1497`、道具势场 `:1667-1729`）；回传用 sigmoid 软胜率 `:1809-1810`；两段搜索（全局 + 局部 11×11）`:2535/:2617`
- RollMan：`ai.h:1712` UCB = `child.win + ARG*sqrtf(logf(parent.count)/child.count)`；**另有一套 pac 视角的 MCTS 专门用来预测对手走哪一步**，再按 `PREDICT_THRES=0.75` 在两条启发式之间切换
- 估值都是**手写线性特征**：RollMan 是分数差/护盾/存活回合/距离/BFS 可达性；SnakeGo 是围棋味的地盘、长度差、气、射线、贴边罚

**族 B：单边 best-first + 规则状态机优先**（AntWar 27届）
- **不是 MCTS**：`select_expand`（`:865-893`）按"最少扩展优先 + 未扩展 +1000 + danger/safe 调整"选节点，**整个文件没有 `sqrt`/`log`/`rand`** → 无 UCB、无随机 rollout、无回传
- 扩展用 **8 种战术模板** `series_action(i=0..7)`（`:789-806`）+ 一个"空操作"子节点
- 估值 = **单边 60 轮确定性前推**（`:629-656`），**只 apply 自己的操作、不建模对手**；root 甚至把敌基地 HP 设成 1000（`:1590`）→ **它搜的是"我能撑多久"，不是"我怎么赢"**
- **规则优先于搜索**：`advanced_ai`（`:1509-1660`）先按规则试 EMP/攻击/风暴，**一有动作就直接 return 跳过搜索**；搜索只做兜底
- 预算 0.15 s / 2 万节点 —— 比 SnakeGo 的 0.95 s / 30 万节点小一个量级

**族 C：模板候选 + 扁平 bandit + MC rollout**（AntWar2 30届）—— 他最新的形态
- 流程：`generate_root_plans` → `evaluate_root_plans` → 取最优 → 追加进攻性 EMP/回避后处理（`lure_strategy_v4_decision.hpp:30/34/45`）
- **不打树**，只在根节点的候选计划上跑 **UCB1 bandit**：`mean + exploration*sqrt(log(max(2,group_samples+1))/arm.samples)`（`..._evaluation.hpp:310-315`），而且**跑两组独立的 UCB**（先闪电组、再普通动作组）
- rollout horizon 4 / 8 回合（`..._params.hpp:23-26`），终点评估是**精细手写加权**：`base_hp*200 + 阶梯金币分 + c1_terminal_bonus - worker_threat - combat_threat`
- **权重表就是参数头文件本身**（`lure_strategy_v4_params.hpp`），注释里直接写公式
- v4 还会**实测测速再外推预算**（先探 5 个样本算 `us_per_sample`，再估算还能跑多少，硬停）

**族 D：纯启发式 / 无搜索**（Miracle、AquaWar、LostSpace、SnowCup、DeepClue）
- 但**不是"随便写写"**：AquaWar 做了**真正的信念约束传播**（`possible/truepossible` 候选集 + 交集/差集，靠"12 种鱼每队各一次"这个约束收缩，并用**对手动作反推身份**：AOE ⇒ 只可能是 1/3/11；暴击 120 且非最低血 ⇒ 排除 9/10）；LostSpace 做了跨回合信念状态（只在**同层可见**时才把敌人入表）+ **敌人足迹衰减缓存**去猜 `detect` 落点
- **都没有概率建模**：AquaWar 的断言靠优先级 if-else 表 + 线性赌注代价，没有贝叶斯/期望值

---

## 3. 元方法层：他用 **Codex（LLM agent）跑自动迭代闭环**——这是重点

`saiblo-30th-AI/scripts/` 里有一整套：

```
codex_iterate_loop.sh          每 INTERVAL_SEC(默认 1800s) 用 flock 跑一次迭代
codex_iterate_once.sh          （本地线）
codex_anchor_iterate_once.sh   （第二线）
codex_saiblo_iterate_once.sh   （在线线，喂 CODEX_BIN=/usr/local/bin/codex）
codex_quota_guard_wrapper.sh   （配额保护）
```

配套 7 个固定输入文档（`docs/`）：`codex_objective_fixed.md`（**硬规则：先看当前代码真值再做判断；不引用历史版本/历史结果；发现文档与实现冲突先记冲突**）、`codex_iteration_prompt.md`（**默认优先做最小可验证改动**）、`codex_automation_runbook.md`（本地侧固定查 `constants.py`/`engine.py`/`test_engine.py`/`game/src/*`；在线侧"先只读→上传→等编译→非 ladder 验证→明确要求才发 ladder"；**每轮记录 `entity_id`/`code_id`/`version`/`compile_status`/`match_id`**）、`game1_code_truth.md`、`game1_known_discrepancies.md` 等。

README 那句"**基本上是定了策略之后让 AI 自己迭代**"，指的就是这个。

**配套基础设施 `autolab`**："管理 Game1 AI 版本、本地并行评测、累计 Elo 与 replay 分析"；默认 `adaptive`、最多 **24 并发**；另有 `elo_web/` 看板、`saiblo_game1_elo.py`（抓 Saiblo 对局按 `code_id` 算 Elo）、`ProcessPoolExecutor` 并行自战、`Game2/automation/run_cycle.py`（上传→编译→跑批→下载回放→分析闭环）。

> ⚠️ 注意措辞：「**让 AI 自己迭代**」里的 "AI" 是 **LLM（Codex）在改代码**，**不是机器学习**。证据：`Game2/tools/make_n623_candidates.py` 生成候选的方式是**字符串锚点替换拼手写文本块**——`population`/`fitness`/`cross`/`mutat`/`random` **各 0 命中**，没有种群、没有适应度、没有交叉变异算子。

---

## 4. 演进结论（2022 → 2026）：**算法范式没变，变的是工程规模**

**依据（都是代码内痕迹，因为仓库历史被压成 1~2 个 commit）：**

1. **他的 MCTS 模板至少复用三年**：SnakeGo（2022）的 `Node`/`value_judge`/`randomplay`/`newnode`/`mcts` 与 RollMan（2025）**逐字段同构**。
2. **跨届延续**：`saiblo-30th-AI/past_AIs/ANTWar-AI/main.cpp` 与他第 27 届的 `ANTWar-AI` 是**同一算法**（只多 40 行日志 + `#ifdef _WIN32 _setmode` 修复），`include/` **逐字节相同** ⇒ "规则 + 浅搜索"路线保留两届未推倒。
3. **搜索形态是随游戏变的，不是"从搜索退回启发式"**：
   - SnakeGo（16×16、完全信息、交替、6 个动作、围棋味）→ **UCT + 长程 rollout + 对手建模**行得通
   - AntWar / AntWar2（单回合动作是**组合操作序列**、对手同时提交、动作不可枚举）→ rollout 到终局没意义，于是改成**确定性前推 + 静态估值 + 规则状态机优先**（AntWar 的树/`evaluate`/`expand`/`max_val` 都还在，只换了选择策略和 rollout）
   - AntWar2（512 回合、终局 1 HP 净差）→ 再退一步，**树都不要了**，只在人工设计的根候选上做 bandit + 浅 rollout
4. **2026 真正新增的两样**：(a) 大规模自动化实验设施（autolab + 并行自战 + Elo 爬取看板）；(b) **LLM 驱动的自动迭代闭环**。**都不是算法范式变化。**

---

## 5. 考古花絮（都是能直接看到的历史化石）

| 发现 | 位置 | 说明 |
|---|---|---|
| **`rand()%1 == 0`** | `LostSpace-AI/space.cpp:2023` | `rand()%1` **恒为 0** ⇒ "随机选策略"被实质禁用，`else` 分支是死代码，但代码留着没删 |
| **Python 残留** | `Miracle-AI/calculator.cpp:98`、`calculator.h:41` | C++ 里留着被注释的 Python `def __str__(self):` ⇒ 早期用 Python 写过同一套六边形几何库 |
| **`ai3.cpp` 未部署** | `Miracle-AI/makefile:2` | `g++ … ai2.cpp` —— **只编 ai2**；`ai3.cpp` 是孪生体，与 ai2 只差 **51 行且全是阈值/航点** ⇒ 人工 A/B 调参的物证 |
| **故意死循环 `TLE()`** | `AquaWar-AI/Action.cpp:23`，在 `:778/781/792/802` 调用 | 作者标注"TODO：理论不存在"——**想让评测判超时**的自毁分支 |
| **整块注释的旧版本** | `AquaWar-AI/Action.cpp:3097-3689` | 约 590 行旧版 `Act` 被整块注释 |
| **死代码** | Miracle 的 `mid_sword`/`mid_priest`/`mid_archer`/`last1_priest`（各只出现 2 次 = 声明+定义，**无调用点**）；RollMan 的 `not_used.cpp`（`mcts(true)` 永不执行）与 `main_pac.cpp`/`main_both.cpp`（git `19ebc7d "rollman instead of pac"`）；AntWar 的 `supp_expand`（调用被注释） | |
| **707 个 `ai.py` 快照** | `saiblo-30th-AI/Game2/deepclue_ai/` | 每版本一个目录的快照（单个 135–155KB）+ 日志前缀 `[n556y1]` 还留在新文件里 ⇒ 层层复制改写 |
| **名称与实现不符** | `RollMan-AI/README.md` 说 Pac 是 "MiniMax Search" | 实现是 **MCTS**，以代码为准 |
| **SDK 命名错位** | `Generals-AI` 的 SDK | `SAND`=规则的 BOG、`SWAMP`=规则的 MOUNTAIN（语义一致、命名错位） |
| **`Action_Assert` 自动改发 Null** | `AquaWar-AI/sdk/ai_client.hpp:1181` | SDK 层把非法/已暴露的断言**自动改成 Null** ⇒ 这就是"断言双向赌注"的安全网 |
| **`state_limit_exceed` 从未使用** | AquaWar | SDK 解析了 128 步上限，**AI 侧从没用过** ⇒ 没适配 |

---

## 6. 他自己踩过的坑 —— 和我们这次发现的**高度重合**

他的 `saiblo-30th-AI/docs/game1_known_discrepancies.md`（"当前仓库里已经确认存在的高影响冲突"）里记着：

| 他的记录 | 我们的对应发现 |
|---|---|
| 官方是"**同回合有序操作列表**"，他的 baseline **还不能表达 `sell -> lightning -> build`**，建议"搜索单位应从单操作改为同回合有序 op-list" | 我们确认的"变长有序列表 + **前缀依赖合法性**" |
| "**同回合后手会直接吃到前手 EMP**" | 我们确认的"轮内序贯，P1 能看到 P0 同轮操作" |
| 轻量模拟"默认忽略**每 10 回合周期随机移动机制**"，搜索分数偏离 native 真值（给了具体案例 `seed_0024, player 0, round 199`） | 我们确认的 `ANT_TELEPORT_INTERVAL=10` / `ANT_TELEPORT_RATIO=0.1` |
| 官方 `game` 二进制**不执行 `config.max_rounds`**（传 256、实跑 396~450 回合） | 我们没查到的工具链坑 |
| `ProducerMedic` 的 `support_range` 语义不统一 | 我们没查这么细 |
| `Ant-Game/README.md` **不是完整结算真值** | 我们那 20+ 条文档冲突清单 |

⇒ 他也专门维护了一份 **`game1_code_truth.md` + `game1_known_discrepancies.md`**，并写"**不要把单一文件当成完整真值**"。这印证了那 20+ 条不是过度解读，而是**这类比赛的系统性问题**。

---

## 7. 他 AntWar2 的高层策略（**可直接借鉴**）

当前最优是 `cpp_lure_v3`，策略名 **"lure"（诱饵）**。原文：

> 当前 Game1 的核心**不是**"尽量把普通蚂蚁全防住"，而是：先建立并维护 `C1` 主锚点；把主要精力放在**战斗蚂蚁控制**上；用**外圈廉价塔反复引诱战斗蚂蚁偏离基地**；在战斗蚂蚁即将白拆 lure 塔前**主动卖塔回收**；在合适时机用 `Lightning Storm` 清理积累的战斗蚂蚁。
> 也就是说，**lure 塔首先是路线控制工具，不是输出塔。**

而且他的策略灵感来自**人类高分回放**：反复引用 `Game1/good.json`（注明是"手打 replay"），提炼出"**少操作、重战斗蚁控制、重卖塔回收、闪电作收束工具**"。

---

## 8. 抽查记录（我逐条核实的，与子 agent 报告对照）

| 断言 | 我的核实 | 结论 |
|---|---|---|
| 我的关键词扫描 `genetic/GA/evolution/CEM=86` | `\bgenetic`=0、`\bevolution`=0、`\bCEM\b`=0、`\bGA\b`=0；而 `placements?`=**136**、`replacements?`=**125** | **我错了**：`CEM` 是 `pla·CEM·ent` 的子串，剩下的是二进制里的 `GA` 字节。真实命中**全为 0** |
| RollMan 是 MCTS、`ARG=1.414`、1 万节点 | `ai.h:20 MAX_TIME 0.9f`、`:22 MAX_NODE_COUNT 10000`、`:23 ARG 1.414f`；`:1712` UCB 公式原文 | ✅ 确认 |
| RollMan 只编 ghost 版 | `makefile:10 SOURCES = … main_ghost.cpp` | ✅ 确认（`not_used.cpp`/`main_pac.cpp`/`main_both.cpp` 被排除） |
| 他用 Codex 跑迭代闭环 | `scripts/codex_iterate_loop.sh` 存在，`:8 INTERVAL_SEC` 默认 **1800**、`:7` flock 锁文件；另有 anchor/saiblo 变体 + 配额保护 | ✅ 确认 |
| 候选生成不是 GA | `make_n623_candidates.py` 里 `population`/`fitness`/`cross`/`mutat`/`random` **各 0 命中** | ✅ 确认 |
| Miracle 只编 ai2 | `makefile:2` = `g++ … ai2.cpp -o main` | ✅ 确认 |
| LostSpace 的 `rand()%1` 化石 | `space.cpp:2023` 原文 | ✅ 确认 |
| Miracle 四函数是死代码 | 各只出现 2 次（声明+定义），无调用点 | ✅ 确认 |
| `calculator` 从 Python 移植 | `calculator.cpp:98` / `calculator.h:41` 有 `/*def __str__(self):` | ✅ 确认 |
| Generals 无 MCTS/α-β | `main.cpp` 里 `ucb/rollout/node/visit/mcts/alpha/beta/beam/prune` **全 0 命中** | ✅ 确认 |
| Generals `MAX_T1..T4` | `main.cpp:6-9` = 0.15/0.4/0.6/0.85 | ✅ 确认 |
| AquaWar 的 `TLE()` | `Action.cpp:23 void TLE(){`，在 `:778/781/792/802` 调用 | ✅ 确认 |
| Generals 自研 4,828 行 | **我实测 `main.cpp` = 4,646 行** | ⚠️ 口径差 182 行 |
| "`MAX_T0` 只在注释里" | `main.cpp:5` 有真实 `#define MAX_T0 0.1f` | ⚠️ 子 agent 小错 |

---

## 9. 不确定 / 未确认

1. **有 4 个仓库的 push 日期全是 `2022-03-24`**（Miracle/AquaWar/LostSpace/SnakeGo）——批量上传，**不能据此推断写作时间**，所以 §4 的时间线只能做弱推断。
2. 这些仓库的 git 历史基本被压成 1~2 个 commit，**看不到仓库内部的迭代顺序**。
3. AntWar2 最终上传的是 6 个 `ai_cpp_lure_v4_*_20260510.zip` 里的哪一个**不确定**；`_aistorm32/64`、`dp16` 后缀含义未查。
4. RollMan 的 pac 版是否也上过 Saiblo（git 有 `"rollman instead of pac"`）**不确定**。
5. `DeepClue` 的 82 个 `make_n5XX_candidates.py` 只精读了 1 个，其余按同一模式推断。
6. 与官方 SDK 未逐行 diff，只能说"AI 自有函数都在 `main.cpp`/`Action.cpp`"。

---

## 10. 对他的评价（一句话）

**他赢在"纪律"和"基础设施"，不是赢在算法新。** 算法上他就是"手写估值 + 少量搜索 + 海量手调常数"；但他有：① 固定的 agent 迭代 objective（只信代码真值、最小可验证改动、不引用历史结论）；② autolab 自动评测 + Elo 看板；③ 明确的"代码真值 vs 文档冲突"清单；④ 从人类高分回放反向提炼策略。**这套元方法（而不是算法）才是他长期霸榜的原因。**

---

## 附：证据与复现

- 仓库：本目录下 `Miracle-AI/` `AquaWar-AI/` `LostSpace-AI/` `SnakeGo-AI/` `ANTWar-AI/` `Generals-AI/` `RollMan-AI/` `SnowCup-AI/` `saiblo-30th-AI/`（均为子模块，pin 见 [`../../docs/other_versions_ai_sources.md`](../../docs/other_versions_ai_sources.md)）
- 他自述方法论的文档：`saiblo-30th-AI/docs/README.md`、`codex_objective_fixed.md`、`codex_iteration_prompt.md`、`codex_automation_runbook.md`、`game1_code_truth.md`、`game1_known_discrepancies.md`、`game1_lure_strategy.md`、`game1_random_search_strategy.md`
- 他对自己 AI 的分析：`saiblo-30th-AI/past_AIs/antwar_ai_deep_analysis.md`、`antwar_behavior_analysis.md`（含 500 局自对弈统计 `antwar_stats_summary.json`）
- 相关：本项目对 AAArena 12 个游戏的 RL 适配性评估见 [`../../docs/12游戏RL适配性评估.md`](../../docs/12游戏RL适配性评估.md)

## 变更记录

- 2026-10-10 建立。4 个子 agent 精读 + 我逐条抽查；**留档了我自己开头那次关键词侦察的错误结论**（GA/CEM 假阳性）。
