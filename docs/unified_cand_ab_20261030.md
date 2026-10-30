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

> ⛔ **本批因负信号提前中止（2026-10-30 17:00）** —— 用户拍板：不再花 ~5h 跑满 48 局。
> A 臂（unified legacy）跑完的局全部留档在 `match_results/ladder_logs/UNIFIED_AB_A/`
> （9 局 `verdict=engine` + 若干半路文件，未删除）。
>
> **旧的 unified（legacy decode）负结果：0W-9L，Δhp 均值 −35.2（中位 −37），
> 打满 300–428 回合，劣于 pin 基线（−20.1，见 `docs/us_vs_champion_20261028.md` 的口径）。**
> ⇒ 该版本**不进入 A/B 正式比较**，B 臂（pin）没有开跑就停了。

| token | 座位 | verdict | 回合 | 我方 hp | 对手 hp | Δhp |
|---|---|---|---|---|---|---|
| 7 | P0 | engine | 393 | 0 | 30 | −30 |
| 7r | P1 | engine | 299 | 0 | 32 | −32 |
| 8 | P0 | engine | 356 | 0 | 39 | −39 |
| 8r | P1 | engine | 349 | 0 | 41 | −41 |
| 9 | P0 | engine | 408 | 0 | 37 | −37 |
| 9r | P1 | engine | 356 | 0 | 39 | −39 |
| 10 | P0 | engine | 380 | 0 | 29 | −29 |
| 10r | P1 | engine | 344 | 0 | 41 | −41 |
| 11r | P1 | engine | 428 | 0 | 29 | −29 |

健康度：9/9 `verdict=engine`、`illegal=0`、每局 45–51 min（`--jobs 10`，unified 每决策 10–20 s）。
9 局里**一局都没赢**，Δhp 均值 **−35.2**（中位 −37）—— 负信号方向一致、幅度大，
不需要跑满 24 局就能判"这个版本劣于 pin"。

> 判读：**这一版（legacy decode + 玻尔兹曼 Q + 完整回写）落在预注册的第 ③ 档（A<B）**，
> 但按用户要求**不再展开归因**，直接转向 split decode 与评估器/位置网方向（见 §5 与
> `docs/unified_cand_fix_20261029.md` §8/§9/§10）。

## 2. 配对胜负 / 血量

**未做**（B 臂未开跑就停了；A 臂 9 局 0W-9L、Δhp −35.2）。

## 3. 位置拆分（@P0 / @P1）

A 臂 9 局：P0 4 局（7/8/9/10）全负、Δhp −33.8；P1 5 局（7r/8r/9r/10r/11r）全负、Δhp −36.4。

## 4. 行为差异（HOLD / churn / 升级 / n_cand）

未做（对照组缺失）。行为读数改在 `docs/unified_cand_fix_20261029.md` §8/§9 里，用
"同对手的 1 局冒烟 + 同局面离线对照" 的方式给（含 `[candq]` 的 N[c]/候选构成/定价分析）。

## 5. 判读与归因

1. **本版（legacy decode）落在第 ③ 档（A<B）**：0W-9L、Δhp −35.2，且**打满 300–428 回合
   才输**（不是早期崩盘）⇒ 不是"非法/崩溃"那类问题，而是**决策质量**。
2. **不能就此断言"统一候选思路错"**（预注册 §0.3 的歧义仍在）：legacy decode 下 16 个塔通道
   在空格**都建 Basic**，别名冗余 + 升级几乎搜不到（含升级 5.6%）⇒ 这套候选本来就残废。
   split decode 就是为修这个做的（`docs/unified_cand_fix_20261029.md` §8：含升级 5.6%→10.9%，
   且**该局供给上界正好 10.9%**）。
3. **已定位到两条比"候选"更硬的线索**（详见同报告 §9/§10）：
   * `[candq]` 分析：选 HOLD 的 263 个回合里，**候选 100% 含非 HOLD**，且阶段 2 分
     "非HOLD最高 − HOLD" 中位 **−0.046**、47% 的回合 ≤ −0.05 ⇒ **是评估器系统性偏好 HOLD**，
     不是候选没提出；
   * 临界金币（70–89）定价：18 个回合里 14 个的最好非 HOLD 是建塔/升级，其中 **92.9%
     被评估器排到 HOLD 前面**（gap 中位 +0.020）⇒ **值网宁可花钱也不攒 90 放闪电**
     （本局闪电只放了 2 次 = 0.6%，pin 历史 1.6–2.7%）。
4. **下一步（按证据强度排序）**：① 评估器（值网定价/`random` rollout 的 decay 参数）
   ② 闪电的候选供给（affordable 的 12 个回合里只有 3 个回合的候选含闪电 —— bandit 把
   45% 的 N 花在 HOLD、30% 花在降级）③ 位置网。

## 6. 成本 / 墙钟

A 臂 9 局：45–51 min/局（`--jobs 10`）；unified 每决策 10–20 s（48 候选 × 64 采样 × 16 回合）。
⇒ 24 局 A 臂 ≈ 3.5 h、B 臂(pin) ≈ 1.5 h，共约 5 h（这也是"负信号提前中止"省下的时间）。
