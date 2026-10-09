# AAArena 说明（`其他版本ai/AAArena/`）

> 整理时间：2026-10-09
> 整理方式：全部结论都来自实际拉取/解包/解析，命令和数字可复现；凡是我自己的推测都会标注「**推测**」。

---

## 0. 这是什么，为什么在这里

来源链：

1. 微信公众号文章 **《30年清华智能体大赛，迎来一批特殊的"新选手"》**（公众号「酒井科协」= 清华计算机系学生科协，2026-10-09 21:00）
   - 全文存档：[`../docs/微信文章存档_30年清华智能体大赛.md`](../docs/微信文章存档_30年清华智能体大赛.md)
2. 论文 **《Can AI Agents Learn Their Way to the Top? Evaluating Heuristic Learning in a Long-Running Game Agent Competition》**
   - arXiv: <https://arxiv.org/abs/2610.12341>（v1，2026-10-08）
3. 代码仓库：<https://github.com/THU-CST-SAST/AAArena>
   - 本文档对应的本地目录：`其他版本ai/AAArena/`
4. 项目主页：<https://aaarena.net>（源码仓库 `THU-CST-SAST/AAArena-Web`）

**AAArena（Agents for Agents Arena）是什么**：清华计算机系学生科协做的一个基准（benchmark），把自己办了近 30 年的智能体大赛的 12 个历届赛题 GamePack 集中起来，**把 AI Agent 放到人类选手的位置上**——读规则、写可执行策略、挑对手打小对局、看回放、改代码，基座模型权重冻结，最后按对冻结人类程序池的 Elo 和排名计分。论文把这件事形式化成 **Adversarial Heuristic Learning（AHL，对抗启发学习）**。

这个目录和 `saiblo-30th-AI`、`ant-war2-magica-v3` 等一样，属于"别人的仓库参考副本"。**和它们的关键区别见第 2 节**：AAArena 里的 antwar2 官方后端就是你本地 `Ant-Game` 的那套引擎。

---

## 1. 仓库基本信息

| 项 | 值 |
|---|---|
| 本地路径 | `其他版本ai/AAArena/` |
| HEAD | `fd52d2337b6711f0f3a239216f48720c125750be` |
| HEAD 日期/主题 | 2026-10-09，*Use the published paper title as the README heading* |
| 分支 | `main`（远端只有这一个分支） |
| 提交数 | 57 |
| 仓库体积 | **134.2 MB**（其中 `.git` 89.8 MB、`assets` 41.1 MB、`results` 1.8 MB、`src` 0.7 MB、`tests` 0.5 MB） |
| 许可证 | 框架 MIT；游戏资源各自保留自己的 provenance / 授权文件 |

顶层结构：

```text
assets/          12 个游戏的 tar.gz（引擎 + SDK + 允许放出的人类选手包 + 完整性清单）
configs/         论文实验预设（paper.json / distribution.json）
docs/            runtime、资源可见性、off-policy、评测 API 文档
environment/     控制器与选手环境的依赖规格
results/         冻结的人类选手可用性元数据 + 参考 Elo 榜
scripts/         安装器、实验启动器、回放目录生成
src/aa_arena/    评测器、对局记账、资源隔离、harness 适配
tests/           组件与集成测试
validation/      发布验证证据（含放出规则的口径文件）
```

### ⚠️ 运行平台限制（重要）

**只支持 x86-64 Linux，Windows / macOS / ARM 原生跑不了本地对局。**

reference host 实测配置（`validation/README.md`）：Debian 13 x86-64、glibc 2.41、GCC/G++ 14.2、控制器 Python 3.14.6、选手 Python 3.10.14；还需要 cgroup v2、可用的 systemd user manager、非特权 user namespace、Bubblewrap、Make/CMake/Ruby。

→ 想跑就得开 WSL2 或 Linux 虚拟机（且要能跑 Bubblewrap / systemd user，WSL2 下 systemd 要单独开）。

---

## 2. 关键发现：antwar2 后端 = 你本地 `Ant-Game` 的引擎

我把 AAArena 的 antwar2 C++ 后端（`games/antwar2/backend/logic/gamecode_logic/game/`，从 `assets/antwar2.tar.gz` 解出）和你本地 `Ant-Game/game/` 做了逐文件比对：

| 比对方式 | 结果 |
|---|---|
| 直接 SHA-256 | 23 个文件全部不同（**假象**：纯换行符 CRLF/LF 差异） |
| 归一化换行后比对 | **23 个 C++ 源文件/头文件中 21 个完全相同** |
| 差异文件 | 只有 `src/comm_judger.cpp` 和 `src/main.cpp` |

`comm_judger.cpp` / `main.cpp` 正是 AAArena 为了自己那套 evaluator 改的**接线文件**（通信 + 入口），不是游戏逻辑。

**含义**：
- `game.yaml` 确认 antwar2 就是 **第 30 届 Saiblo 游戏「AntWar2 / 蚁洋陷役2」**，19×19 offset-hex 网格双人回合制塔防，和你正在打的是同一个东西；
- AAArena 的 `evaluator/` 跑的就是**同一套规则**，可以当本地评测器用；
- 反过来说，**你把本地 AI 接到它的对手池上时，规则侧不存在"对不上"的风险**，唯一要处理的是候选接口（它用公开 Python SDK `AI.choose_operations` 入口）。

---

## 3. 12 个游戏与届次对应

`game.yaml` 的 `edition` 字段 + `players/publication.json` 实测：

| 游戏 | CLI id | 届次 | 人类程序池总数 | 放出源码数 | 扣留数 | 规则 AST 节点 | RA |
|---|---|---|---:|---:|---:|---:|---:|
| Pacman | `pacman` | legacy_ai9 | 44 | 18 | 26 | 1,127 | 219 |
| LOTA | `lota` | legacy_ai9 | 200 | 96 | 104 | 4,513 | 684 |
| MoneCraft | `monecraft` | legacy_ai9 | 112 | 52 | 60 | 1,823 | 347 |
| Dorado | `dorado` | legacy_ai9 | 323 | 157 | 166 | 4,241 | 616 |
| Miracle | `miracle` | **24** | 253 | 122 | 131 | 4,391 | 581 |
| LostSpace | `lostspace` | **25** | 111 | 51 | 60 | 2,297 | 426 |
| AquaWar | `aquawar` | **25** | 170 | 81 | 89 | 2,531 | 439 |
| SnakeGo | `snakego` | **26** | 141 | 66 | 75 | 1,538 | 308 |
| AntWar | `antwar` | **27** | 114 | 53 | 61 | 2,160 | 368 |
| Generals | `generals` | **28** | 197 | 94 | 103 | 2,683 | 461 |
| Rollman | `rollman` | **29** | 64 | 28 | 36 | 1,539 | 312 |
| **AntWar2** | `antwar2` | **30** | **191** | **91** | **100** | **6,368** | **1,008** |
| **合计** | | | **1,920** | **909** | **1,011** | | |

- 4 个游戏标成 `edition: legacy_ai9`（AI9 平台的老游戏，没有 Saiblo 届次编号）；其余覆盖**第 24–30 届**，其中第 25 届含 LostSpace + AquaWar 两个游戏。
- 所以 **不是全部 30 届**，是"最近 7 届赛题 + 4 个 legacy 游戏"。
- AST/RA 是论文用来度量规则复杂度的（抽象语法树节点数 / 可独立修改的规则命题数），**AntWar2 是 12 个游戏里规则最复杂的**（6,368 AST，是 Pacman 的 5.6 倍）。

---

## 4. 人类选手代码的放出规则：只有一半，而且前 8 名永远不给

官方口径（`docs/remote-evaluation.md`、`validation/player-publication.json`）：

```
policy: "rank > 8 and rank % 2 == 0"
```

即：**冻结测量的 Elo 榜里，名次 > 8 且为偶数**。

- 12 个游戏放出的**最小名次全部正好是 10**，一个例外都没有 → **第 1–8 名永远拿不到**；
- **奇数名次（9、11、13…）全部不放** → 用"只放偶数"来防止从放出集合反推被扣留的名次；
- 实际放出 **909 / 1,920 ≈ 47%**；
- `validation/README.md` 原话：*"The public release contains 909 permitted human strategies: measured-Elo ranks greater than eight with even rank numbers. The hosted service retains the full 1,920-player inventory."*

### 坑：目录名的 `rankNN` 前缀 ≠ 冻结榜名次

目录名带 `rankNN` 只是该提交自己的命名。**必须查 `players/publication.json` 或 `players/manifest.tsv`**。已核实的反例：

- `rank04__Aoraku__baba__v2` → `publication.json` 里参考名次是 **18**
- `rank07__Agony__骨__v8` → 目录名写 7，实际放出名次不是 7

### 元数据比源码公开得多

| 文件 | 披露了什么 |
|---|---|
| `results/elo/<game>/measured_ranking.tsv` | **完整榜**（antwar2 是 191 人）的匿名 ID + Elo + 场次 + 胜率 + anchor |
| `results/elo/<game>/measured_elo.json` | 同上，JSON 形式 |
| `results/availability/<game>/runnable.json` | antwar2 列了 **481 个**提交的 ID / rank / Elo / 可用性（其中 verified 191）；**481 里有 390 个是"只有元数据、没有源码"** 的 |

→ 也就是说**前 8 名的匿名 ID 和分数是公开的，只是不给代码**。

### 匿名化是部分的

antwar2 放出的 91 个里：**17 个用真实用户名**（如 `rank01__yyzsanyi__ai_storm__v37`、`CoffeLatte__ent_...__v13`、`ZhangZhiyuan__ent_...__v8`、`zzy25__ent_...__v1`），**74 个是 `anon__<hash>__<hash>__v<n>`**。

> 注：`zzy25__ent_a46f6d616d9cd9b3__v1` 看名字疑似是**你自己**的提交（未确认，请你核对）。

### `manifest.tsv` 的字段

```
player_id  dir  username  entity  rank  elo  version  lang  submission_id  split  named
```

- `lang` 列在 antwar2 的 91 行里**全是 `auto`**（语言自动探测），**不是**算法/方法标注；
- `rank` / `elo` **只有 6 行有值**（都在 train split 的 18 行里），其余 85 行为空。实测：总行数 91、rank 非空 6、elo 非空 6、split=train 18、test 73。

---

## 5. LLM AI 的「解」：找不到，没有公开

**结论：论文里那些 LLM agent 写出来的最终策略源码，没有任何一处公开。**

### 查过的地方（全部为空）

| 地方 | 结果 |
|---|---|
| AAArena 仓库 | 无 `runs/`、无策略/冠军目录；grep `champion` 只命中 controller 记账代码（`ledger.py` / `service.py` / `runtime.py` / `trajectory.py` 等）。**`validation/README.md` 末尾明写：`Private strategies and credentials are not included.`** |
| 仓库 git 历史 | 57 个 commit 全查了 `git log --diff-filter=D --name-only`，**没有被删掉的策略文件**（是从来没进过库，不是删了） |
| 论文 | 只有统计与图表（详见下） |
| 项目主页 + `AAArena-Web` 仓库 | 只有回放 + 哈希 + 统计表。唯一带 "champion" 的文件是 `assets/champion-algorithms.png` —— **一张图** |
| GitHub 组织 `THU-CST-SAST` | 只有 3 个仓库：`AAArena`、`AAArena-Web`、`sweet-agent-skill`（最后这个是同组织另一个项目：血糖管理 skill，**与 AAArena 完全无关**） |
| HuggingFace | <https://huggingface.co/papers/2610.12341> 只有论文条目，**没有挂任何 dataset / model** |

另外：公开代码 `src/aa_arena/core/decision_space.py:38` 引用了一个 **"B 仓"模块 `aa_arena_hl.adapters.transcript`**，但公开包是 `src/aa_arena/`，**不含 `aa_arena_hl`**。

同时 `docs/layout.md` 明确排除：*"Runtime environments, complete private pools, **paid experiment transcripts**, credentials and server state are not public distribution inputs."* —— 连实验 transcript（token 级对话记录）都不给。

### 能间接看到的替代材料（三层，都不足以还原代码）

**① 论文 Appendix E / Table 10：算法家族的占比统计**（不是代码）
→ 见第 6 节。

**② 论文 Appendix B.6 / Figure 11–13：完整策略演化轨迹图**（Elo/rank/token 曲线，不是代码）

**③ 论文 Section 4.2.2 / Table 3：唯一能看到 AI 真实代码的地方**

Table 3 的 caption：*"Dorado champion milestones. Mechanism descriptions with added (+) and deleted (−) **C++ diff excerpts**. Ellipses omit context."*

它贴了 GLM-5.3 在 Dorado 上从 rank 41 爬到 rank 2 的 9 个里程碑，每个给三两行带省略号的 C++：

```
+232  enemyBase != NULL && enemyBase->hp < 1800 && (int)myHeroes.size() >= 4;
+245  int count = seriousThreat ? (int)defenders.size() : 2;
+228  const bool seriousThreat = ... || baseThreat || recentBaseHit;
+343  if (alive && mineUsable(it->second, round)) occ[it->second]++;
-477  ... + 600LL  * occ[m];
+472  ... + 2000LL * occ[m];
+472  ... + 5000LL * occ[m];
+217  s_mineExistUntil[i] = round + exist->val[0] - 1;
+ 36  #define MINE_OPEN_OCCUPANCY_PENALTY 50000LL
+296  if (s_dutyId < 0 && (int)s_leveledCore.size() < 2 ...)
```

**总共就 12 行左右，只覆盖 12 个游戏里的 1 个。**

**④ 项目主页放了 5 条 AI 对局回放**（完整逐帧对局，不是代码）

我把 <https://aaarena.net> 整个静态站点扒下来了（`data.js` 7.4 KB / `details.js` 5.6 KB / `replays.js` 22 KB / `assets/replays/replay-data.js` **6.2 MB** / `assets/replays/antwar-data.js` 1.15 MB）：

| 游戏 | 模型 | AI Elo / 名次 | 人类对手 | seed |
|---|---|---|---|---|
| snakego | Opus5.5 | 1860.9 / **1** | `anon__710a5f8e...` (rank 1) | 42 |
| pacman | Opus5.5 | 2703.9 / **1** | `pacman__p43__ai895` (rank 1) | 42 |
| monecraft | GPT6-sol | 2418.2 / **1** | `monecraft__p672__ai6935` (rank 1) | 42 |
| dorado | GPT6-sol | 2271.5 / 2 | `dorado__p1859__ai22546` (rank 1) | 42 |
| antwar | GLM-5.3 | 1464.2 / 5 | **`rank01__omegafantasy__Magica__v13`** | 20240117 |

（前 4 个各带 2 个 variants；AntWar 在单独文件里。）

回放数据字段：`width / height / frames / winner / fps / aiSeat / finalScores / rawReplay`。AI 一侧**只给两个哈希**：

```
strategy_hash: 89b4a2a58b7baca2f1729dcdd9eb3fb41ef842f6caab4e644e863fb65ab649af
snapshot:      c2e01dcd07f3b68d6e4596a7563eae966d2eec5e08dee2c6a90a09c9f0f01d89
```

我全站搜过 `choose_operations` / `#include` / `std::vector` / `int main(` → **0 命中**。策略哈希的用途是**可验证性**（核对某次提交是不是同一份策略），不是还原代码。

> **有趣的交叉**：AntWar 那条回放的人类对手是 `rank01__omegafantasy__Magica__v13` —— 就是 `其他版本ai/saiblo-30th-AI` 那个作者（omegafantasy）的 ID。所以这等于能看到 **GLM-5.3（rank 5） vs Magica（rank 1）** 的整局逐帧对局。

### 想拿到 AI 的解只有一条路：自己跑

```bash
python3 scripts/setup_local.py          # 装控制器 + 选手环境，验证 12 个游戏合约（不调用模型）
source validation/local-setup/activate.sh
python scripts/configure_model.py --name default --model YOUR_MODEL_ID \
  --context-window 200000 --reasoning-effort max       # 配自己的模型 API
python scripts/run_experiments.py plan --suite main --profiles default \
  --harness codex --output plans/main.json
python scripts/run_experiments.py run --plan plans/main.json --jobs 4 --workers 8
```

- 正式 Elo 还要评测 token（`AA_ARENA_EVAL_TOKEN`，向维护者提 GitHub issue 申请，**有每人运行配额**），服务地址 `https://101.42.12.204`；
- 本地 `practice.py` 不需要 token，但只能对**放出的子集**打分，**子集分数不能当成完整池的论文结果**；
- 你复现出来的是**你自己那份 agent 的解**，跟论文里 Opus5.5 那份不是同一个东西。

---

## 6. LLM 选手用的方法：论文有系统统计，但**没有强化学习**

### 论文 Appendix E / Table 10 的六类算法家族

论文把 84 个"保留冠军"（7 个模型配置 × 12 个游戏）的代码，按**非注释代码行占比**分成六类：

| 家族 | 内容 |
|---|---|
| Rule-based control | 手写 if–else 优先级、状态机、固定开局/脚本 |
| Heuristic scoring | 手挑特征给候选动作/目标打分 + 贪心 |
| Graph search | BFS / DFS / Dijkstra / A* / 距离图 / flood fill / 领地划分 |
| Lookahead search | rollout、beam search、minimax、expectimax、**MCTS** |
| Combinatorial optimization | 指派与匹配、DP、背包式分配 |
| Inference & randomization | 隐藏信息下的信念跟踪、对手建模、混合/随机决策 |

关键数据（论文原文）：

- 规则 + 打分 **79%**，图搜索 **13%**，其余三类合计 **8%**；
- **前瞻搜索在 84 个冠军里只有 7 个超过 10%，全部集中在 AntWar、AntWar2、Generals**；
- Pacman 图搜索 33%、Rollman 28%；AquaWar 推断 30%；**MoneCraft 和 LOTA 几乎纯规则**；
- Opus5.5 规则占比最低（47%）、打分 27%、前瞻 5%；GPT6-sol 图搜索最高（18%）；其余模型规则占 58–65%；
- 非规则占比与 Elo 的 Spearman 相关平均只有 **+0.35**，论文特意强调这是相关、**不能推出"少写规则就更强"**；
- 分类是**自动且近似**的，且是 **endpoint 分类**（只描述最终策略构成，**不归因**哪次改动带来了提升）。

### 「用过强化学习吗？」—— 没有

证据链：

1. **六类家族里根本没有 RL**，也没有任何"学习/训练"这一类；
2. 全文 **0 次**出现 `network` / `neural net` / `trained` / `learning rate` / `backprop` / `policy network`（逐词数过）；
3. "reinforcement learning" 只出现在三处：摘要/引言的背景铺垫、Related Work 的参考文献、以及 **Future Work**；
4. Future Work 原文：*"We aim to compare how code revisions and **parameter updates** learn from experience... Hybrid learners could use HL to revise policy structure and **reinforcement learning to train its components**."* → **RL 是被明确写成未来计划的**；
5. 范式自我定位就是反梯度方法：挂在 Weng (2026) *Learning Beyond Gradients* 的启发式学习脉络下，且基座模型**权重冻结**（*"The base model's weights remain fixed."*）。

> **我的推测（非论文结论）**：LLM 选手不用 RL 更像是**预算和环境的产物**——RL 需要大量自对弈 + 训练时间，而 agent 被卡在 128 小对局 / 16 次全池评测的固定交互预算里，策略还要在每步决策的 CPU 限额下跑；人类选手则有几周 + GPU 可以离线训网络。
> **未验证**：我没有去确认比赛/AAArena 是否**明文禁止**在策略里做训练。要往这个方向下注，这个规则得先查清楚。

### 每个游戏最强的是谁（论文 Table 2 原文数字）

| 游戏 | 最强配置 | Elo / 名次 |
|---|---|---|
| Pacman / SnakeGo / Rollman / LostSpace / AquaWar / Generals / Miracle / LOTA / AntWar2 | **Opus5.5 + Claude Code** | 各游戏最好 |
| MoneCraft | GPT6-sol | 2418.2 / 1（Opus 2387.1） |
| Dorado | GPT6-sol | 2271.5 / 2（Opus 2252.4） |
| AntWar | GLM-5.3 | 1464.2 / 5 |

金牌总数：Opus5.5 **6**、GPT6-sol **4**、GLM-5.3 **4**、Qwen 3.8 **3**、Kimi K3 **2**、DeepSeek V4 Pro **2**、LongCat 2.0 **1**（合计 22）。
**12 个游戏里最强的那些，一个用 RL 的都没有。**

---

## 7. 人类选手用的方法：没有标注，但源码可反推

- 论文把人类程序只当**冻结对手池**，没有分析人类程序用了什么算法；Appendix A 只描述 12 个游戏各自的决策问题；
- 全文 **MCTS / minimax 各只出现 1 次**，就是 Appendix E 给"前瞻搜索"下定义那句；
- `manifest.tsv` 的 `lang` 列全是 `auto`，**没有任何方法元数据**。

**但源码放出来了，可以读代码反推。** 我扫了 antwar2 的 91 个选手：

### ⚠️ 必须先剔除 SDK 共享文件，否则会得出假结论

直接对整个目录 grep，会得到 **87/91 的选手都用了 AlphaZero** 的假象——因为 `alphazero.py`、`training/` 这些是**官方 SDK 自带**的，所有人提交时都带一份。

按论文口径（*"excluding byte-identical SDK files"*）处理：

| 步骤 | 结果 |
|---|---|
| 全部文件 | 3,607 |
| 跨选手共享（哈希出现在 ≥20 个选手里）= SDK/脚手架 | **2,463** |
| 选手自己写的文件 | **1,144** |

在这 1,144 个文件里扫 RL / 搜索类关键词：

| 关键词 | 命中行数 |
|---|---:|
| AlphaZero | 1,256 |
| MCTS | 320 |
| self-play | 170 |
| `import torch` | 112 |
| `nn.Module` | 34 |
| `.backward(` | 31 |
| PPO | 23 |
| `from torch` | 22 |
| `torch.optim` | 9 |

**63 / 91 个选手**的自有代码里有这些痕迹。最重的 `anon__a510f76de171e535__46fa0fdbb0ff53b8__v57` 命中 252 处。例：`CoffeLatte__ent_a2383151e5f0a446__v13/SDK/` 下有 `alphazero.py`、`train_mcts.py`、`backend/model.py`。

论文侧也有旁证（`validation/README.md` 行为发现表）：*"A separate policy imports torch at startup and can approach or exceed the one-second first-turn deadline"*（Generals）。

### 结论对比

| | 人类选手 | LLM 选手 |
|---|---|---|
| 是否用 RL | **很多用了**（antwar2 至少 63/91 有 AlphaZero/torch 痕迹） | **没有**（论文零证据，且 RL 被列为 Future Work） |
| 源码 | 放出 909 份（偶数名次 > 8） | 不公开 |

---

## 8. 怎么用这份东西（落地方案）

### 8.1 把 antwar2 的对手池解出来当本地陪练

```powershell
cd "D:\2026智能体大赛_新2\其他版本ai\AAArena"
# 只解游戏定义和 SDK
tar -xzf assets/antwar2.tar.gz -C <目标目录> games/antwar2/game.yaml games/antwar2/rules.md `
    games/antwar2/decision_space.yaml games/antwar2/public_sdk`
# 解全部 91 个选手（约 3,694 个条目）
tar -xzf assets/antwar2.tar.gz -C <目标目录> games/antwar2/players
```

- `players/manifest.tsv` 给出 player_id / 名次 / split；
- `players/publication.json` 给出**冻结参考名次**（用它而不是目录名）；
- 候选接口权威是**公开 Python SDK**（入口 `AI.choose_operations`），规则权威是 C++ 后端。

### 8.2 评测协议要点（`docs/` + 论文 Appendix B）

- 小对局：每个游戏 128 个小对局单位（一次请求可能消耗多个单位），返回**回放**（稠密 / 二值两种 arm）；
- 全池评测：16 次，返回 **Elo + 名次 + 逐对手统计**，不给稠密轨迹；
- Elo 用**冻结池的 Bradley–Terry** 拟合（400 分制），并加一个对全池均值的虚拟平局项让估计有限；
- 名次 = 1 + 冻结池中严格高于该 Elo 的人数；
- **不同游戏的参考尺度不同，只能同游戏内比较**。

### 8.3 论文里对你有直接价值的几个实验结论

| 结论 | 数据（GLM-5.3，128/16 预算） |
|---|---|
| 稠密回放远胜二值反馈 | Pacman 2236.1 vs 1710.1；AntWar 1118.7 vs 880.6；Miracle 1528.2 vs 1461.1 |
| 对手选择要"阶梯式"而非挑战最强 | Pacman 阶梯 2127.5 最强；AntWar 阶梯 1311.7 最强；"只打前四"最弱 |
| 批大小不是越大越好 | Pacman 4 个对手最好（2703.9）；AntWar 2 个最好（1433.4） |
| 加 3 倍预算不一定有用 | Generals / LOTA 零提升；Miracle 1436.8→1486.8；AntWar2 1977.1→2117.0 |
| Off-policy 学别人的回放也有效 | Pacman #10→#1；AntWar #41→#19；Miracle #27→#15 |

### 8.4 顺手记下的 AntWar2 已知坑（`validation/README.md`）

这几条和蚂蚁后端/SDK 行为有关，写 AntWar2 AI 时可能踩到：

1. **原始选手策略在 AntWar2 上计算超时**：同一份冻结输入，reference host 11.58 s、hosted 机 10.98 s，**超过未改动的 10 s 限额**；
2. **原始 SDK 的枚举序列化是错的**：Python 3.10 下 enum 名不是合法数值协议字段，导致 6 场规则错误；**公开 SDK 改成显式整数**，Python 3.10 / 3.14 都通过；
3. **原始 SDK 的经济公式算错**：47 金币 + 3 座基础塔，其内置公式算出 54 金币返还，**官方公式只给 40**；两份原始 SDK 都会接受这个付不起的 90 金币动作。**公开 SDK 用的是官方公式。**

---

## 9. ⚠️ 合规提醒

这批代码是别人**公开释放**的，但"公开释放"**不等于**"可以搬进自己的参赛提交"。

旁证：AAArena 给自家 AI Agent 定的协议里明确有 `✗ No reading rivals' code`（见公众号文章里的框架图 Figure 2），说明赛方对"看对手代码"本身是有态度的。

**拿来当本地对手池、或读思路**，和**直接搬代码进 submission**，性质完全不同。后者建议先确认比赛规则怎么写。

---

## 10. 未解决问题 / 待办

- [ ] **是否允许在策略里做训练**（RL）—— 没查比赛/AAArena 规则
- [ ] **比赛规则是否禁止参考/使用他人代码** —— 没查
- [ ] AAArena 的 antwar2 **冻结榜**与第 30 届真实最终榜的关系 —— 未核对（论文榜是"各提交版本为独立测例"）
- [ ] `zzy25__ent_a46f6d616d9cd9b3__v1` 是否是你自己的提交 —— 待你确认
- [x] 把 AAArena 登记成**真子模块** —— 2026-10-09 已完成（见第 11 节）

---

## 11. 版本控制：已登记为真子模块（2026-10-09）

AAArena 按本仓库既有约定登记成了**真子模块**，和顶层 `Ant-Game` 以及 `saiblo-30th-AI` / `ant-war2-magica-v3` / `Antwat_2/Ant-Game` 一致（见 [`../docs/other_versions_ai_sources.md`](../docs/other_versions_ai_sources.md)）。

| 项 | 值 |
|---|---|
| `.gitmodules` 条目 | `[submodule "其他版本ai/AAArena"]` |
| url | `https://github.com/THU-CST-SAST/AAArena.git` |
| 钉住的提交 | `fd52d2337b6711f0f3a239216f48720c125750be`（= 上游 `main` 尖端，已 `git ls-remote` 核实） |
| gitlink 模式 | `160000` |
| 工作区状态 | **干净**（不像另外三个子模块分别有 2 / 6 / 4 处本地改动） |

登记命令（已执行）：

```bash
git submodule add https://github.com/THU-CST-SAST/AAArena.git 其他版本ai/AAArena
# 输出：Adding existing repo at '其他版本ai/AAArena' to the index
```

别人 clone 本仓库之后初始化：

```bash
git clone --recurse-submodules <本仓库url>     # 或者 clone 完再：
git submodule update --init --recursive
```

**登记之后仓库体积不受影响** —— 134 MB 只在本地，超级仓库里只存一个指针。

> 仍然适用：**千万别在这个仓库里直接跑 `git add -A` / `git add .`**（除 `.gitignore` 覆盖的之外，还有大量 `_tmp_*` 临时脚本和 `submission/` 下的产物是未跟踪的）。要提交被忽略的东西用 `git add -f <路径>`。
