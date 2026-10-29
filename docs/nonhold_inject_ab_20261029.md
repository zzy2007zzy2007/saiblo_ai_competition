# 非 HOLD 外部动作的定价：`AZAI_ORACLE_SKIP_HOLD` A/B（2026-10-29）

> **一句话**：上一轮把"恒为候选的 HOLD"单独做出来只拿回 **+12.5pp**（≈ 亚军注入 +37.5pp 的 1/3）。
> 这一轮把 HOLD 的功劳**控制住**（两臂都开 `AZAI_ALWAYS_HOLD=1`），只让**非 HOLD 的外部动作**进候选
> ⇒ 两臂之差 = "真外部动作"的净价值。**结果见 §4/§5**（本文件在开跑前已写好 §0 预注册）。

---

## 0. 预注册（**跑之前登记，不许事后改**）

### 0.1 两臂

| 臂 | 环境 | 含义 |
|---|---|---|
| **A**（被测） | `AZAI_ALWAYS_HOLD=1` + `AZAI_ORACLE_INJECT=<亚军 exe 绝对路径>` + `AZAI_ORACLE_SKIP_HOLD=1` | HOLD 恒为候选 **+** 只注入亚军的**非 HOLD** 动作 |
| **B**（基准） | `AZAI_ALWAYS_HOLD=1` | HOLD 恒为候选，不注入 |

* `AZAI_ORACLE_SKIP_HOLD=1`（**本轮新增，默认 0**）：注入路径里若 oracle 给的是空 bundle（HOLD），
  **不注入、只记一条 `skipped_hold` 计数**；其余（时间戳/重建/回放）行为不变。
* 两臂都开 `AZAI_ALWAYS_HOLD=1` ⇒ **HOLD 可用性被控制住**，差异只来自"非 HOLD 外部动作"。

### 0.2 规模与配置

| 项 | 值 |
|---|---|
| seeds | **7..18**（12 个地图）× 2 座位 = **每臂 24 局**（共 48 局） |
| 并发 | 两臂**并行**，每臂 5（总 10） |
| 对手 | 亚军本体 `cpp_lure_v4` |
| BASE | `_inject_vs_runnerup.sh:45` 那串（`_us_vs_external.sh` 内有逐字断言），`FLAT_SEED=53` |
| 不开 | 安慰剂 / 升级候选 / PIN_K / MC_UPG / oracle 主动重启（亚军无状态，不需要） |
| 超时/韧性 | `LADDER_TIMEOUT=10800`、`MVS_TIMEOUT_SECONDS=300`、`MVS_STDERR_SUFFIX=1`、`--retry-invalid 2 --final-sweep --watchdog-stall-min 90` |
| 保活 | **不起** `_keepawake.ps1`（AC 供电） |

### 0.3 判读（三档）

| 档 | 条件 | 结论 |
|---|---|---|
| **① 有独立价值** | **A − B ≥ +12pp 且 CI 不跨 0** | 非 HOLD 外部动作确有价值 ⇒ "知识注入"叙事**部分成立** |
| **② 基本没用** | **A ≈ B**（`|Δhp| < 5` 且 CI 跨 0） | +37.5pp **基本全是"能空过"的功劳**，外部动作知识价值很小 |
| **③ 有害** | **A < B** | 外部非 HOLD 动作有害（要考虑亚军风格与我们不兼容） |

### 0.4 必须报的六项指标

1. 配对胜负/血量差（A−B，逐 token）+ 胜率 Wilson CI + **符号检验 p**；
2. **按座位拆分**（上一轮增益全在 @P1：1/8 → 4/8）；
3. **注入内容构成**：`asks / skipped_hold / new / dup / hit_new` + **非 HOLD 注入的实际采纳率**；
4. 两臂的 **HOLD 率**、**降级 churn**；
5. `verdict` 全 `engine`、`illegal=0`；
6. **零影响证明**：`AZAI_ORACLE_SKIP_HOLD=0` 时与现役（历史 `INJECT_VS_RU_A`）逐字段相同。

---

## 1. 为什么这一步是必须的（上一轮的遗留）

| 已知 | 读数 | 出处 |
|---|---|---|
| 亚军注入臂 12.5% → 50.0%（+37.5pp） | 16 局配对，`Δhp +20.7` | `docs/inject_vs_runnerup_20261028.md` |
| 其中注入的内容 **84.5% 是 HOLD** | 8117 次注入 / 6857 次 `()`；`hit_new` 里 70.5% 是 HOLD | 父任务解析 `match_results/inject_vs_ru/A/stderr/` |
| "恒为候选的 HOLD"单独只值 **+12.5pp** | A 25.0% vs B 12.5%，McNemar p=0.625 | `docs/always_hold_ab_20261029.md` |
| ⇒ 剩下 ~2/3 必须来自**非 HOLD 的外部动作**（那 15.5%）或交互 | 未测 | **本文件** |

---

## 2. 实现（默认关、逐位不变）

| 位置 | 改动 |
|---|---|
| `az_bridge_ai.py` `_ORACLE`/`_ORACLE_B` | 新增 `skip_hold` / `skipped_hold` 两个字段 |
| `_oracle_init` / `_oracle_b_init` | 读 `AZAI_ORACLE_SKIP_HOLD`（B 会话另有 `AZAI_ORACLE_SKIP_HOLD_B` 覆盖） |
| `_oracle_step` / `_oracle_b_ask` | oracle 回空 bundle 且开关开 ⇒ **不注入**、`skipped_hold+1`、把通道清空（让 `_ahold_step` 只留 HOLD） |
| `_oracle_stats_text` | 新增 `skipped_hold=` 与 `injected_nonhold=`（只在非 0 时出现 ⇒ 老日志逐字不变） |
| `_us_vs_external.sh` | 新增 `EXP=E3`（注入亚军 vs 亚军，`AZAI_ORACLE_TIMEOUT=15` 与历史一致）；抓取/回填 `AZAI_ORACLE_SKIP_HOLD`、`AZAI_ALWAYS_HOLD` |
| `_nonhold_inject_ab.sh` | **新增**：两臂并行 + 逐臂读数 + 配对 + 六项指标 |

---

## 3. 零影响证明（指标 6）

**做法**：用**当前代码** + `AZAI_ORACLE_SKIP_HOLD=0`（其余与历史 `INJECT_VS_RU_A` 逐字相同：`EXP=E3`
= 注入亚军 vs 亚军、`BASE`、seed 7），与历史日志逐字节比较。

| 比较对象 | 结果 |
|---|---|
| `RESULT` 行 | **完全相同**：`winner=runnerup rounds=512 terminal=True base_hp=14,28 coins=138,91 illegal=0` |
| 我方（p0）逐回合 op | **512/512 回合完全相同**（第一个不同的回合 = 无） |
| 对手（p1）逐回合 op | **512/512 回合完全相同** |

⇒ `AZAI_ORACLE_SKIP_HOLD=0` 时行为与改动前**逐字节相同**（连"什么时候注入 HOLD"都没变）；
而开关打开时，同一局的注入通道从"每回合都可能注入"变成"只在 oracle 给非 HOLD 时注入"
（§4.3 的 `skipped_hold` 计数为证）。

复现：
```bash
TAG=ZEROIMPACT_NH EXP=E3 OPP=runnerup SEEDS=7 JOBS=1 RETRY=0 \
  LADDER_TIMEOUT=10800 MVS_TIMEOUT_SECONDS=300 MVS_STDERR_SUFFIX=1 AZAI_ORACLE_SKIP_HOLD=0 \
  bash code/test_match/_us_vs_external.sh
# 与 match_results/ladder_logs/INJECT_VS_RU_A/7.log 逐回合比较
```

---

## 4. 六项指标

### 4.1 指标 1：胜负与血量差（配对）

| 臂 | 局数 | 胜-负 | 胜率 | Wilson 95%CI | 血量差 | t 95%CI | `verdict` | `illegal` |
|---|---|---|---|---|---|---|---|---|
| **A**（ALWAYS_HOLD + 非HOLD注入） | 24 | **11W-13L** | **45.8%** | [27.9%, 64.9%] | **−0.04**（sd 18.31） | [−7.78, +7.69] | 24/24 `engine` | **0** |
| **B**（ALWAYS_HOLD） | 24 | **4W-20L** | **16.7%** | [6.7%, 35.9%] | **−14.62**（sd 14.80） | [−20.88, −8.37] | 24/24 `engine` | **0** |

**配对（同 token，A − B）**：

| 量 | 读数 |
|---|---|
| 胜负表 | 两臂都赢 **3** / **只 A 赢 8** / 只 B 赢 **1** / 都输 **12** |
| **McNemar 精确双侧** | **p = 0.0391** |
| 胜率差 | **+29.2pp**（16.7% → 45.8%） |
| **配对血量差** | **+14.58**（sd 18.23，SE 3.72），t 95%CI **[+6.88, +22.28]** ⇒ **不跨 0** |
| **配对 Δhp 符号检验** | 正 **17** / 负 6 / 平 1 ⇒ **p = 0.0347** |

⇒ 落在预注册的 **①「有独立价值」**：`A − B = +29.2pp ≥ +12pp` 且区间不跨 0。

### 4.2 指标 2：按座位拆分

| 臂 | @P0（我们执先手） | @P1（我们执后手） |
|---|---|---|
| **A** | 4W-8L = **33.3%**（n=12），hp −6.25 [−19.00, +6.50] | 7W-5L = **58.3%**（n=12），hp **+6.17** [−3.14, +15.47] |
| **B** | **0W-12L = 0.0%**（n=12），hp −21.92 [−28.40, −15.43] | 4W-8L = **33.3%**（n=12），hp −7.33 [−17.14, +2.47] |

⇒ 这次**两个座位都有增益**（@P0 0/12 → 4/12，@P1 4/12 → 7/12），@P1 仍然更强。
与上一轮"增益全在 @P1"**不同**——上一轮那条"座位假设"在 24 局上**没有复现**（@P0 也涨了）。
⚠️ n=12/座位，仍然只能当方向看。

### 4.3 指标 3：注入内容构成（**剂量**）

| 量 | A 臂（24 局合计） | 说明 |
|---|---|---|
| `asks` | **8180** | 问了 8180 次（每回合一次） |
| 它给 HOLD | **6584（80.5%）** | 亚军 80.5% 的回合是 HOLD |
| **`skipped_hold`** | **6584** | **全部被跳过**（= 不注入、只计数）✅ |
| **`inject`（非 HOLD）** | **1596（19.5% of asks）** | 这就是本实验的**剂量**：每局 ≈ 66 个"真外部动作"候选 |
| `new` / `dup` | 1596 / 0 | 全部是新候选（不与我们的 48 条重复） |
| `hit_new` | **407** | 其中 407 次被评估器选中 |
| **非 HOLD 采纳率** | **25.5%（407/1596）** | 每 4 个外部动作采纳 1 个 |

⇒ 与历史注入臂的构成一致（84.5% 是 HOLD），说明**跳过 HOLD 只跳掉了"噪音"**，
非 HOLD 那一小部分（19.5%）才是有效载荷。

### 4.4 指标 4：HOLD 率与降级 churn

| 臂 | 回合数 | HOLD 回合 | **HOLD 率** | 我方总 op | 降级 op | **降级占比** | 每回合降级 | 建塔 | **升级** | 闪电 |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | 12283 | 10335 | **84.1%** | 2967 | 1255 | 42.3% | **0.102** | 1277 | **126** | 308 |
| **B** | 12151 | 10629 | **87.5%** | 2489 | 1006 | 40.4% | **0.083** | 1185 | **0** | 298 |

⇒ 注入"真外部动作"让 HOLD 率从 **87.5% → 84.1%**（多动了 3.4pp）、
升级从 **0 → 126 次**（亚军会升级，我们采纳了它的一部分非 HOLD 动作）。
降级 churn **没有下降**（每回合 0.083 → 0.102，反而略升）⇒ 这一轮**不支持**"注入能减少 churn"。

### 4.5 指标 5：健康度

| 判据 | A | B |
|---|---|---|
| `verdict=engine` | **24/24** ✅ | **24/24** ✅ |
| 裁判侧 `illegal` | **0** ✅ | **0** ✅ |
| 我方 `illegal_total` | **0** ✅ | **0** ✅ |
| oracle 失败/重建 | 0 次失败（`dead=0`） | — |
| 每局墙钟（min） | min/med/max = 29.9 / 43.3 / 52.6（合计 1032.7 min） | 31.0 / 35.3 / 46.1（合计 860.8 min） |
| 我方决策耗时（每局最后一行取中位） | p50 **2.10s**（含一次 oracle 查询）/ p95 16.48 / max 31.75 | p50 **0.66s** / p95 20.07 / max 29.00 |

### 4.6 指标 6：零影响证明

见 §3：`AZAI_ORACLE_SKIP_HOLD=0` 时同 seed 与历史 `INJECT_VS_RU_A` **逐字节相同**（双方 512 回合 op 全同）。

---

## 5. 结论（落在哪一档 + 归因）

### 5.1 落在预注册的 **①「非 HOLD 外部动作有独立价值」**

| 档 | 门槛 | 实测 | 判定 |
|---|---|---|---|
| **① 有独立价值** | A − B ≥ +12pp **且** CI 不跨 0 | **+29.2pp**；配对 Δhp **+14.58 [+6.88, +22.28]**；McNemar **p=0.0391**；符号检验 **p=0.0347** | **✅** |
| ② 基本没用 | `|Δhp| < 5` 且 CI 跨 0 | Δhp 14.58、CI 不跨 0 | ✗ |
| ③ 有害 | A < B | A 45.8% > B 16.7% | ✗ |

⇒ **"知识注入"的叙事部分成立**：把 HOLD 的功劳控制住之后，**只注入亚军的非 HOLD 动作**
（19.5% 的回合、每局 ≈66 个候选、采纳 25.5%）就能把胜率从 **16.7% 拉到 45.8%**、
血量差从 **−14.62 拉到 −0.04**（配对 p=0.039 / 符号检验 p=0.035）。

### 5.2 归因（三方对照，**同 BASE、同对手、同 seeds**）

| 干预 | 候选集 | 胜率 | 相对 BASE | 出处 |
|---|---|---|---|---|
| BASE | 48 条网络候选 | 12.5%（2W-14L） | — | `INJECT_VS_RU_B`（历史） |
| **+ HOLD 恒为候选** | 48 + `()` | **16.7%**（4W-20L） | **+4.2pp**（另一批 16 局测到 25.0% ⇒ **+12.5pp**） | 本批 B / `ALWAYSHOLD_A` |
| **+ HOLD + 非HOLD 外部动作** | 48 + `()` + 亚军非HOLD手 | **45.8%**（11W-13L） | **+33.3pp**（配对 +29.2pp vs 本批 B） | **本批 A** |
| + HOLD + **全部**外部动作（含 HOLD 注入） | 48 + `()` + 亚军全部手 | 50.0%（8W-8L） | +37.5pp | `INJECT_VS_RU_A`（历史） |

**读法**：
1. **主要功劳在"非 HOLD 外部动作"**：控制住 HOLD 之后仍有 **+29.2pp（p=0.039）**；
   而"只补 HOLD"两次测量只有 **+4.2 / +12.5pp（都不显著）**。
2. **两者近似可加**：`12.5 + 29.2 = 41.7pp` 与"全量注入"的 `+37.5pp` 同量级
   ⇒ 没有明显交互（多出来的 HOLD 注入在全量臂里没带来额外收益）。
3. **上一轮"HOLD 解释 1/3"的估计偏高**：那时把"HOLD-only = +12.5pp"当分子；
   本批 24 局把 B 臂拉到 16.7% ⇒ HOLD-only 只有 +4.2pp。**HOLD 是"必要条件"（评估器要能空过），
   但不是主要增益来源**——真正值钱的是那 19.5% 的真实外部动作。
4. **副作用**：升级从 0 → **126 次**（亚军会升级，我们采纳了它的一部分非 HOLD 手）；
   HOLD 率 87.5% → 84.1%；降级 churn **没有下降**（每回合 0.083 → 0.102）。

### 5.3 下一轮建议

1. **把"非 HOLD 注入"变成常驻配置**（不是实验）：它现在有 p<0.05、CI 不跨 0 的读数支撑，
   而且实现就是一个开关（`AZAI_ORACLE_SKIP_HOLD=1` + `AZAI_ALWAYS_HOLD=1`）。
   ⚠️ 但它是"借外部 AI 的招"，部署时要注意对手依赖（换对手是否还有效未测）。
2. **验证对手无关性**：同一套注入（亚军）去打**冠军**，看增益是否还在
   —— 这是"知识"还是"针对亚军的风格补偿"的分水岭。
3. **拆"哪个动作类型值钱"**：126 次升级里我们采纳了多少？把非 HOLD 注入再拆成
   {升级 / 建塔 / 闪电 / 降级}，看增益集中在哪里（`[oracle]` 逐回合日志已经能算）。
4. 冠军 oracle 仍然不要碰（需要闭环执行）。


---

## 6. 复现

```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
# 冒烟（~3 min，两个 tag 各 1 局）
ROUNDS=12 SEEDS="7" JOBS_PER_ARM=1 bash code/test_match/_nonhold_inject_ab.sh
# 正式：48 局（每臂 24），两臂并行
JOBS=10 bash code/run_logged.sh nonhold_inject_ab bash code/test_match/_nonhold_inject_ab.sh

for T in NONHOLD_INJECT_A NONHOLD_INJECT_B; do
  $P -u code/test_match/_batch_table.py --tag $T \
      --stderr match_results/us_vs_external/$T/stderr --opp-label runnerup \
      --seeds 7 8 9 10 11 12 13 14 15 16 17 18 \
      --markdown-out match_results/us_vs_external/$T/table.md
  $P -u code/test_match/analyze_inject_vs_runnerup.py \
      --arm $T=match_results/ladder_logs/$T \
      --stderr $T=match_results/us_vs_external/$T/stderr
done
$P -u code/test_match/analyze_inject_vs_runnerup.py \
    --arm A=match_results/ladder_logs/NONHOLD_INJECT_A \
    --stderr A=match_results/us_vs_external/NONHOLD_INJECT_A/stderr \
    --arm B=match_results/ladder_logs/NONHOLD_INJECT_B \
    --stderr B=match_results/us_vs_external/NONHOLD_INJECT_B/stderr \
    --pair A B
$P -u code/test_match/_ahold_report.py --tag-a NONHOLD_INJECT_A --tag-b NONHOLD_INJECT_B \
    --out match_results/us_vs_external/_nonhold_report.md
```
