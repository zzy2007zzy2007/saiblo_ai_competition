# 统一候选生成 A/B（`AZAI_CAND_MODE=unified` vs `pin`）—— 24 局 × 2 臂

> 状态：**预注册（§0）已先写死，结果后填**。
> 设计：`docs/unified_cand_design_20261029.md`；实现：commit `3b306b5`（`unified_cand.py`）。
> 驱动：`code/test_match/_unified_cand_ab.sh`；分析：`code/test_match/analyze_unified_cand_ab.py`。

## 0. 预注册（登记时间：2026-10-30 15:20，**在看到任何胜负数据之前**）

> 登记时的可见信息：只有 A 臂前 ~5 局的**行为**计数（HOLD/BUILD/DOWNGRADE/UPGRADE 条数），
> **没有任何一局的 RESULT**（A 臂第一波 10 局尚未结束）⇒ 预注册未被结果污染。

### 0.1 实验定义（先钉死，跑完不改）

| 项 | 值 |
|---|---|
| A 臂 | `AZAI_CAND_MODE=unified`（β=3.0、τ=1.0、`STEPS_TOTAL=1024`、`STEPS=12`、`POOL=1.5`、k=48，全默认） |
| B 臂 | **不设** `AZAI_CAND_MODE`（= `pin`，现役基线） |
| 两臂共同 | `AZAI_CAND_VERBOSE=1`（只为日志；新增的 `[cand] mode=pin` / `[act]` 都是**默认关**的纯日志） |
| 对手 | 亚军 `其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe` |
| seeds | 7..18 × 2 座位（`7 7r … 18 18r`）= **每臂 24 局**，共 48 局 |
| 配置 | BASE 与 `_inject_vs_runnerup.sh` / `_us_vs_external.sh:68` **逐字相同**（脚本里有断言）；K=48 / FLAT_ROLLOUTS=64 / FLAT_ROUNDS=49152 / random+opponent+decay(1.0,3.0) / horizon16 / share_prefix=1 / FLAT_SEED=53 |
| 不开的开关 | ALWAYS_HOLD / 注入 / 安慰剂 / PIN_K / MC_UPG（unified 自己天然含 HOLD） |
| 批次参数 | `--jobs 10 --retry-invalid 2 --final-sweep --watchdog-stall-min 90`，`LADDER_TIMEOUT=10800`、`MVS_TIMEOUT_SECONDS=300`、`MVS_STDERR_SUFFIX=1` |
| 有效性 | 只算 `verdict=engine`（timeout/aborted/hp_tiebreak 无效） |

**A 臂不是 pin 的证明**（否则 48 局白跑）：驱动脚本在正式批次**之前**跑 2 局 6 回合前置校验，
断言 A 臂 stderr 出现 `[cand] mode=unified`、B 臂**没有** —— 实测 `A: unified 行=7 / pin 行=0`，
`B: unified 行=0 / pin 行=6` ✓。

### 0.2 判读规则（**写死，事后不改**）

| 判读 | 条件 |
|---|---|
| **① 正解**（候选生成是瓶颈） | A−B ≥ **+12pp** 且 Wilson CI **不跨 0** |
| **② 无差异**（候选修好了但没转化为强度） | A≈B：\|Δhp\| < **5** 且 Δhp 的 95% CI **跨 0** |
| **③ 负向**（统一候选引入噪声/位置网头部拖累） | A < B 且 Δhp CI 整段在 0 以下（或胜率显著低） |

主读数用**配对**（同 token 两臂都有效）：McNemar 精确双侧 p（胜负）+ Δhp 配对 t 的 95% CI
+ Δhp 符号检验。副读数：逐臂 Wilson 胜率 CI、按座位拆分（@P0/@P1 各 12 局）。

### 0.3 若落在 ②（null）——**必须写清的归因歧义**（用户特别强调）

A≈B **不能**直接读成"统一候选思路错"。因为 unified 每个**形状**的游标一次只推进
`AZAI_CAND_STEPS=12` 格、总步数上限 1024 ⇒ **位置搜索比"按类深挖"更浅**，对
**位置网头部（argmax 附近几名）的依赖反而更高**；而位置网头部现在很烂
（建塔 @1 ≈ 1.2×随机，见 `docs/why_effective_candidates_collapse_20261028.md`）。所以 null 有两个候选原因：

* **(a) 统一候选思路不对**（跨类×位置的候选结构本身没价值）；
* **(b) 位置网太烂，撑不起浅位置搜索**（候选结构对，但提议质量的短板在位置网）。

**预注册的 follow-up（按顺序验证，不在本批里做）**
1. `AZAI_CAND_STEPS_TOTAL` 1024 → **4096**（把每个形状的游标挖深）⇒ 若 A 变好，支持 (b)。
2. 位置提议**加温度采样**（argmax → softmax(位置图/τ_pos)）⇒ 若 A 变好，支持 (b)。
3. **位置网消融**：把位置提议换成"合法格里随机均匀"⇒ 若 A 与"用位置网"相当，说明位置网没贡献 (b)；
   若明显变差，说明位置网有用但不够强。
4. 若 1–3 都不动 ⇒ 才倾向于 (a)。

## 1. 逐臂健康度 / 总体

（结果后填）

## 2. 配对胜负 / 血量

（结果后填）

## 3. 位置拆分（@P0 / @P1）

（结果后填）

## 4. 行为差异（HOLD / churn / 升级 / n_cand）

（结果后填）

## 5. 判读与归因

（结果后填）

## 6. 成本 / 墙钟

（结果后填）
