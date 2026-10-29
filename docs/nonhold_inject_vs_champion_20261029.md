# 对手无关性：同一套"非 HOLD 注入"去打冠军（2026-10-29）

> **一句话**：E3 已证明"只注入亚军的**非 HOLD** 动作"对**亚军**能把胜率 16.7% → **45.8%**
> （配对 +29.2pp，p=0.039）。这一轮把**对手换成冠军**、**oracle 仍然是亚军**，看增益还在不在
> ⇒ 决定它是"**知识**"（不依赖对手风格）还是"**针对亚军的风格补偿**"。

---

## 0. 预注册（**跑之前登记，不许事后改**）

> ⏱️ 登记时间：**2026-10-29 21:40**（开跑前 2 分钟；当时只有 E3 批的读数，本批**一局都没跑**）。

### 0.1 两臂（都带 `AZAI_ALWAYS_HOLD=1`）

| 臂 | 环境 | 含义 |
|---|---|---|
| **B**（基准） | `AZAI_ALWAYS_HOLD=1` | HOLD 恒为候选，不注入 |
| **A**（被测） | `AZAI_ALWAYS_HOLD=1` + `AZAI_ORACLE_SKIP_HOLD=1` + `AZAI_ORACLE_INJECT=<**亚军** exe>` | HOLD 恒为候选 **+** 只注入**亚军**的非 HOLD 动作 |

⚠️ **oracle 仍是亚军，只把对手换成冠军**——A/B 的差别与 E3 逐字相同（`docs/nonhold_inject_ab_20261029.md` §0.1），
唯一变的是**对手**。

### 0.2 规模与配置

| 项 | 值 |
|---|---|
| 对手 | 冠军 `其他版本ai/ant-war2-magica-v3/magica_v3_O0.exe` |
| seeds | **7..18**（12 张图）× 2 座位 = **每臂 24 局**（共 48） |
| 并发 | 两臂**并行**，每臂 5（总 10） |
| BASE | `_us_vs_external.sh` 的 BASE（与 `_inject_vs_runnerup.sh:45` 逐字相同） |
| oracle | `AZAI_ORACLE_TIMEOUT=15`；`AZAI_ORACLE_REBUILD_MAX=6`；不开主动重启 |
| 不开 | 安慰剂 / 升级候选 / PIN_K / MC_UPG |
| 超时/韧性 | `LADDER_TIMEOUT=**21600**`（冠军 O0 慢 + 夜里可能休眠）、`MVS_TIMEOUT_SECONDS=300`、`MVS_STDERR_SUFFIX=1`、`--retry-invalid 2 --final-sweep --watchdog-stall-min 90` |
| 保活 | **不起** `_keepawake.ps1`（夜里断电；主动接受休眠 + 重试补跑） |

### 0.3 判读（三档）

| 档 | 条件 | 结论 |
|---|---|---|
| **① 是知识** | **A − B ≥ +12pp 且 CI 不跨 0** | 外部动作的价值**不依赖对手风格** |
| **② 风格补偿** | **A ≈ B** | 只对亚军有用（补的是亚军风格） |
| **③ 有害** | **A < B** | 亚军的非 HOLD 动作对冠军有害（风格冲突） |

### 0.4 必须报的指标

1. 配对胜负/血量（A−B，逐 token）+ 胜率 Wilson CI + 配对 Δhp 的 t 95%CI 与**符号检验**；
2. **按座位**拆分；
3. 注入构成：`asks / skipped_hold / injected_nonhold / new / hit_new`（**含非 HOLD 采纳率**）；
4. 两臂的 **HOLD 率**与**升级次数**（E3 里 A 臂升级 126 次、B 臂 0 次）；
5. `verdict` / `illegal` 健康度；
6. **冠军侧**的 op 数与升级数（对照我们的变化：我们多动了，它有没有被压住）。

---

## 1. 参照：E3（对亚军）的读数

| 臂 | 胜-负 | 胜率 [Wilson] | 血量差 [t95] |
|---|---|---|---|
| A（ALWAYS_HOLD + 非HOLD注入） | 11W-13L | **45.8%** [27.9, 64.9] | −0.04 [−7.78, +7.69] |
| B（ALWAYS_HOLD） | 4W-20L | 16.7% [6.7, 35.9] | −14.62 [−20.88, −8.37] |
| 配对 | 只 A 赢 8 / 只 B 赢 1 | McNemar **p=0.0391** | **+14.58 [+6.88, +22.28]** |

同一套注入、只把对手换成冠军 ⇒ 与上表逐 token 对照即可。

---

## 2. 结果

<!-- NHCH_RESULTS -->

---

## 3. 判读与"外部动作知识"的最终定性

<!-- NHCH_CONCLUSION -->

---

## 4. 复现

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
JOBS=10 bash code/run_logged.sh nonhold_vs_champion bash code/test_match/_nonhold_vs_champion_ab.sh

for T in NONHOLD_VS_CHAMPION_A NONHOLD_VS_CHAMPION_B; do
  $P -u code/test_match/_batch_table.py --tag $T \
      --stderr match_results/us_vs_external/$T/stderr --opp-label champion \
      --seeds 7 8 9 10 11 12 13 14 15 16 17 18 \
      --markdown-out match_results/us_vs_external/$T/table.md
  $P -u code/test_match/analyze_inject_vs_runnerup.py \
      --arm $T=match_results/ladder_logs/$T \
      --stderr $T=match_results/us_vs_external/$T/stderr
done
$P -u code/test_match/analyze_inject_vs_runnerup.py \
    --arm A=match_results/ladder_logs/NONHOLD_VS_CHAMPION_A \
    --stderr A=match_results/us_vs_external/NONHOLD_VS_CHAMPION_A/stderr \
    --arm B=match_results/ladder_logs/NONHOLD_VS_CHAMPION_B \
    --stderr B=match_results/us_vs_external/NONHOLD_VS_CHAMPION_B/stderr \
    --pair A B
$P -u code/test_match/_ahold_report.py --tag-a NONHOLD_VS_CHAMPION_A --tag-b NONHOLD_VS_CHAMPION_B \
    --out match_results/us_vs_external/_nonhold_vs_champion_report.md
```
