# 外部标尺：注入 A/B **直接对打亚军本体**（2026-10-28）

> 工作目录 `D:\2026智能体大赛_新2_distill`。
> 研究问题：**"RL/ML 在这个游戏里到底管不管用"** —— 本文只回答其中一格：
> **"评估/搜索"这一环与"候选/先验"这一环，各占多大比重。**
>
> 相关文档：`docs/oracle_inject_ab_20261028.md`（自对弈注入 A/B：81.2%）、
> `docs/oracle_inject_design.md`（注入通道）、`docs/runnerup_oracle_protocol.md`（查询协议）、
> `docs/prior_repair_20261023.md` §30（历史"vs 亚军零传导"）、`docs/project_state_20261028.md`（状态梳理）。
>
> **本文件 §0-§2 是跑之前写下的（预注册），§3 之后是跑完回填的。**

---

## 0. 口径声明（读结论前必看，三条）

### 0.1 本研究**不使用**"导入亚军价值函数 −34pp"作为"评估是瓶颈"的证据

任务书里那句转述（"0.7344 → 0.3906 ⇒ 暗示评估是大问题"）有三处与原始记录不符，
以 `docs/experiment_log.md:11384-11396` 为准：

1. 那个实验导入的是**亚军的手写启发式价值**（`AZAI_VALUE_SRC=handcrafted`），
   **不是**它训练出来的价值网；
2. 基准对手是 **`rule_v4`（作者自己写的规则 AI，赛会第 16 名）**，**不是亚军本体**；
3. **"−34pp ⇒ 我们的评估器不行"是未经验证的推断**。同一份文档前面就记了一个同类工程 bug：
   `ValueNetwork` 归一化未反卷积 ⇒ 输出被放大 `1/y_std`（terminal 2.6×、rel 13×）⇒ 搜索直接失效。
   所以"手写价值接进去就崩"**首先**是**量纲/语义不匹配**（绝对 vs 相对血量、尺度、正负号）的嫌疑，
   不能反推"评估质量差"。

⇒ 本文的判读**只基于本次两臂 vs 亚军本体的实测数据**（§3），外部证据最多用作"待检验的假设"。

### 0.2 历史"vs 亚军"基线**与本次不可直接比较**（配置不同）

| 项 | 历史 `VS_RU_newcfg`（2026-10-23） | 历史 `VS_RUNNERUP_cells4`（2026-10-21） | **本次 A/B 两臂** |
|---|---|---|---|
| seed（token） | 7..14 ×（正/镜像）＝16 局 | 7..22 ×（正/镜像）＝32 局 | **同左（7..14 ×2 ＝16 局/臂）** |
| `AZAI_K` | 24 | 24 | **48** |
| `AZAI_ITERS` | 256 | 256 | 256 |
| `AZAI_FLAT_ROLLOUTS` | 16 | 16 | **64** |
| `AZAI_FLAT_HORIZON` | 64 | 64 | **16** |
| `AZAI_FLAT_ROUNDS` | 39000 | 39000 | **49152** |
| `AZAI_MC_HORIZON` | **64** | **256** | **16** |
| `AZAI_MC_SCORE` | net（4 采样） | quiet | net（4 采样） |
| 战绩 | **1W-15L = 6.2%**，血量差 **−22.56** | 1W-31L = 3.1% | 本次测（§3.2） |

**⇒ 本次 B 臂是"这套更贵配置下的新读数"，不是对历史 6.2% 的复测**。
两者的差异**同时**包含"配置不同"与"测量批次不同"，禁止把它读成"提升了 X pp"。
（任务书里"HP −26.3 / 1W-15L"是 `prior_repair §30` 记的**冠军臂**读数，也不是本文任何一臂。）

### 0.3 一个必须记住的机制限额：本实验测的是**上限**，不是"亚军在我们的位置会怎么下"

A 臂注入的是"**亚军在同一个局面（含它自己那份搜索+评估）算出来的那一手**"，
由**我们自己**的 rollout 评估器打分；赢不了就丢。所以 A 臂拿到的是
**"白拿对手的出招"这一信息优势 + 多一个候选的机制效应**的**合体**，
它**上界**地回答"如果候选/先验这一步做到亚军水平，能挽回多少"，
但**不能**单独分离出"先验质量"这一个因子（见 §4）。

---

## 1. 准备（三件查证，全部有出处）

### 1.1 历史"我们 vs 亚军"的 harness / seed / 配置

**harness（历史与本次同一个）**：`_tmp_ladder.py`（仓库根）→ 每局一个
`code/test_match/match_via_sdk.py --trace-ops --ai0=<我们> --ai1=<亚军 exe> <token>`，
`--jobs=8` 线程池，日志落 `match_results/ladder_logs/<tag>/<token>.log`，
并把每局两个 AI 的 stderr 复制到 `<tag>/stderr/`。

**token 语义（必须记牢）**：`7` = **AI0 先手**（我们 = ai0 ⇒ 我们执 P0）；
`7r` = **AI1 先手**（镜像，我们执 P1）。所以 `7`/`7r` 是**同一张图的两个座位**。

**历史批次（`code/test_match/_vs_runnerup.sh`、`_vs_ru_newcfg.sh`、`_vs_ru_k48rand.sh`）**：

| 脚本 | tag | seed | 局数 | 结果 |
|---|---|---|---|---|
| `_vs_runnerup.sh` | `VS_RUNNERUP_cells4` | `7 7r … 22 22r` | 32 | 1W-31L = 3.1% |
| `_vs_ru_newcfg.sh` | `VS_RU_newcfg` | `7 7r … 14 14r` | 16 | **1W-15L = 6.2%**（唯一胜局 = `9r`，512 回合 9:5） |
| `_vs_ru_k48rand.sh` | `VS_RU_k48rand` | 见脚本 | — | 历史 k=48 臂 |

⇒ **本次复用 `VS_RU_newcfg` 那一批（seeds 7..14 × 正/镜像 = 16 局/臂）**，
这样"同一批地图/座位"能与历史读数**并排**（但不是直接相减，见 §0.2）。

### 1.2 亚军 exe 路径（DLL 同级）

```
D:/2026智能体大赛_新2_distill/其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe
  ├─ libstdc++-6.dll (2608670B) / libgcc_s_seh-1.dll / libwinpthread-1.dll   ← 4 个文件都在
  └─ ai_cpp_lure_v4_custom.exe（另一个变体，本次不用）
```

**⚠️ 必须是 `D:/...` 形态**：`az_bridge_ai._oracle_init` 只把 `Path(exe).is_absolute()`
为真的路径当绝对路径；Git-bash 的 `/d/...` 在 Windows 上判 **False** ⇒ 被拼成 `<repo>/d/...`
⇒ `Popen` 抛 `FileNotFoundError` ⇒ **注入整局静默不启用，而 A/B 看起来照跑**（已踩过一次，
`oracle_inject_ab_20261028.md` §4.1.1 第 2 条）。脚本里用 `RU="$(pwd -W)/<相对路径>"` 生成。

### 1.3 注入"静默失效"的两种形态与本次的核查办法

| 形态 | 现象 | 本次对策 |
|---|---|---|
| 路径形态错（`/d/...`） | 每局 stderr 只有 `[oracle] ⚠️ 启用失败`，`asks=0`，两臂行为**逐字节相同** | 跑前断言 `RU` 以 `D:/` 开头；跑后**逐局** grep `注入已启用` 且核对路径 |
| oracle 超时/错位 | `[oracle] ⚠️ 第 N 回合查询失败`，之后**永久停用**（读端错位不可重连）⇒ A 臂后半局退化成 B 臂 | `AZAI_ORACLE_TIMEOUT=15`（自对弈那批 p95≈1.4s，超时是调度抖动）；跑后逐局断言 `timeout=0 dead=0` 且 `asks` 覆盖整局回合数 |

**判定规则（预注册）**：A 臂某局若 ① 没有 `注入已启用`、或 ② `asks==0`、
或 ③ 出现 `注入已停用` / `永久停用` / `timeout>0` / `dead>0` / `OracleDesync` / `OracleDead`
⇒ **该局不计入 A 臂，标记无效并重跑**。报告里给出"**有效注入局数 / 总局数**"。

---

## 2. 实验设计（跑之前登记）

### 2.1 两臂（唯一变量 = `AZAI_ORACLE_INJECT`）

```
BASE=AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;
     AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;
     AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;
     AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;
     AZAI_FLAT_BATCH=32;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;
     AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;
     AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;
     AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53
A 臂 = BASE + AZAI_ORACLE_INJECT=<亚军 exe D:/ 绝对路径> + AZAI_ORACLE_TIMEOUT=15
B 臂 = BASE
```

* 与自对弈注入 A/B（81.2%）**逐字同一套配置** ⇒ 两批可互相参照（但对手不同，不是同一读数）。
* **不开** `AZAI_NO_DOWNGRADE` / `AZAI_BAN_OPS`（与注入组合的一类非法 op 源；已有整块闸门，但没必要冒险）。
* 对手 = 亚军本体（同一个 exe），两臂同一批 seed、同一坐位镜像。

### 2.2 规模、驱动、超时

| 项 | 值 |
|---|---|
| 局数 | 每臂 16 局 = seeds `7..14` × 2 座位（`7`/`7r`）⇒ **共 32 局** |
| 并行 | `--jobs 8`（16 核；每局 3 个进程：我们 + 对手 + A 臂的 oracle 实例） |
| 驱动 | `_tmp_ladder.py`（历史同一个）× 2 次（先 A 后 B），自带的单局 `TIMEOUT=7200 s` |
| 续跑 / 重试 | 脚本内实现：跳过已有 `verdict=engine` 的 token 重跑无效局，最多 2 轮 |
| 日志 | `bash code/run_logged.sh inject_vs_ru bash code/test_match/_inject_vs_runnerup.sh` |
| Python | `D:/anaconda3/envs/pytorch-gpu/python.exe` |
| 单局预算 | 历史同驱动 15-29 min/局（轻配置）；本次配置重 2-3 倍 ⇒ 估 40-70 min/局，2 波/臂 |

### 2.3 主读数与判据（预注册）

**主读数（两臂各自）**：vs 亚军的**胜率**（1/0.5/0，Wilson 95%CI）、**血量差**
（我方基地 HP − 对方基地 HP，均值 + t 95%CI）、**按座位拆分**。
**配对读数**：同一 token 上 A−B 的胜负差（McNemar 精确检验）与血量差差（配对 t / bootstrap）。

**判据（写在跑之前）**：

| 观测 | 判读 |
|---|---|
| A ≈ B（差 < ~2·SE，且 A 仍 ≤ 2/16 胜） | "白拿对手的出招"**也**救不了 ⇒ **评估/搜索是主导**（候选不是） |
| A 明显 > B（例如 ≥ 6/16 胜，或配对 A−B ≥ +4 局） | 候选/先验这条**能**兑现 ⇒ 差距主要在**候选**，评估的份额要按 §4 的替代解释打折 |
| B 已明显高于历史 6.2% | 说明"贵配置"本身就把绝对水平抬了 ⇒ 任何 A 增益都要先扣掉这一层 |

---

## 3. 结果（跑完回填）

_待填_

## 4. 替代解释（必须与结果一起读）

_待填_

## 5. 判读：评估/搜索 vs 候选/先验

_待填_

## 6. 复现清单

_待填_

## 7. 没做 / 没覆盖的

_待填_
