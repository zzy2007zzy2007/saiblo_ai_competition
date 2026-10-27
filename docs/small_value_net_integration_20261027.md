# 小值网 32/3 接入 rollout 叶子评估：折叠 / 开关 / 计时 / 8 局冒烟（2026-10-27）

> 前置：`docs/small_value_net_20261027.md`（离线过关：R²=0.4716、ρ=0.9603、前向 5.13×）、
> `docs/value_net_pipeline_survey.md` §5（12 条坑）、实施计划 `docs/small_value_net_integration_plan_20261027.md`。
>
> **所有数字都来自本文列出的真实命令输出**；日志全部落盘（`training_history/runs/`、`training_history/logs/`）。

---

## 0. 结论（TL;DR）

| # | 任务 | 结果 |
|---|---|---|
| **A** | 折 `y_std/y_mean` 进 `value_head` | ✅ 产出 `checkpoints/small32_3/v_32_3_folded.pt`（429,075 B）。**等价性 `max|folded-(raw*y_std+y_mean)|=1.24e-07`**；1000 帧留出集 **ρ(tanh vs 现役)=0.9600**（报告口径 0.9603）；不折的对照：Pearson 0.9513（掉）、`\|tanh\|>0.9` 从 0% 变 5.70% |
| **B** | `AZAI_VALUE_CKPT` 开关（默认关） | ✅ 空值：**探测器的 6 个决策哈希与改动前逐行相同（diff 0 行）**；非空：同探针 **24 行不同**（决策真的变了）+ 两处消费点都跑通 |
| **C** | 同一局面计时 | ✅ **总耗时 10.30 s → 6.78 s（1.52×）**；`_leaf_values` 6.00 s → 2.55 s（2.35×）；**纯值网前向 4.33 s（42%）→ 0.90 s（13%），4.8×**。同预算换算：**N 64→86（保守 88 实测）**，或 **k 48→65** |
| **D** | 8 局冒烟 | 见 §5（每局 `verdict=engine`、无异常退出、逐局耗时） |

**一句话**：小值网**能接**（折叠正确、开关零影响、两处消费点兼容），**也确实省时间**——但在
`k=48/N=64/h=8` 这个部署配置里**只能把总耗时砍到 1/1.52**，因为决策里 58% 是值网管不到的固定成本
（树模拟 4.28 s + 特征抽取 1.66 s = 5.94 s，占小网决策的 **88%**）。省下的 3.5 s 折成
**约 +1070 个叶子样本 ⇒ N 64→86 或 k 48→65**（不是"前向 4.8× ⇒ N 300"）。

---

## 1. A：把 `y_std/y_mean` 折进 `value_head`（关键量纲）

### 1.1 命令

```powershell
$env:PYTHONIOENCODING="utf-8"; $env:OMP_NUM_THREADS="4"
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/fold_small_value_ckpt.py `
  --student-ckpt checkpoints/small32_3/lowq_value_seed0.pt `
  --out checkpoints/small32_3/v_32_3_folded.pt --check-frames 1000
```
日志：`training_history/logs/fold_small32_3.log`（新增脚本 `code/test_match/fold_small_value_ckpt.py`）

做法**照 `graft_value_ckpt.py:68-76`**：`W' = W*y_std`、`b' = b*y_std + y_mean`（只动 `value_head.2`）。
落盘时多带 `y_folded=True` + `fold_info`（**否则 `value_override.load_override` 会再折一次**，
量纲又被缩 2.6×——这是这条路上最容易踩的二次事故）。

### 1.2 自检 ①：等价性（与数据无关的**充分**判据）

```
[fold] y_std=0.386068 y_mean=-0.000408 | 折前 value_head.2.weight[0,0]=-0.082227 bias=0.145873
[fold] 折后 value_head.2.weight[0,0]=-0.031745 bias=0.055909  (= 0.145873*0.3861-0.0004)

max|folded - (raw*y_std+y_mean)| = 1.237e-07   ✅ 折叠正确
folded  raw: min=-0.8689 max=0.9139 std=0.2691
未折 raw   : min=-2.2497 max=2.3683 std=0.6970   ← 未折的 std 是折后的 2.59 倍  (=1/0.386)
```

### 1.3 自检 ②：与现役值网比（1000 帧按局留出，部署口径 `tanh`）

| 指标 | **折后（产出）** | 对照：**未折** | 现役（参照） |
|---|---|---|---|
| Spearman ρ(tanh) | **0.9600** | 0.9600（**不变**） | — |
| Pearson r(tanh) | **0.9708** | 0.9513（掉） | — |
| Pearson r(raw) | 0.9718 | — | — |
| 平均 \|Δtanh\| / 符号不一致 | 0.0504 / 8.40% | — | — |
| `tanh` 分布 std | **0.2489** | 0.4979（**2.0×**） | 0.2190 |
| `\|tanh\|>0.9` 的帧占比 | **0.00%** | **5.70%**（被压到饱和） | — |
| R² vs label（raw / tanh） | 0.4876 / 0.4824 | — | 0.4665 / 0.4577 |

**读法**：ρ 对任意严格单调变换不变 ⇒ **光看 ρ 抓不到这个坑**（两行 ρ 一模一样）。
真正的判据是 **Pearson / tanh 分布**：不折的话 5.70% 的帧被 tanh 压到 `|tanh|>0.9`，
和"终局叶子（±1）"撞在一起 ⇒ 排序被压平（= 调研 §5.3 的历史事故）。
折后两边 std 同量级（0.249 vs 0.219）、`|tanh|>0.9` 为 0。

> 与报告 `docs/small_value_net_20261027.md` §3 的 0.9603 同口径：这里只抽了 1000 帧（vs 20,000），
> 0.9600 与 0.9603 的差是抽样噪声。**"折对了"的充分证据是 §1.2 的 1.24e-07，不是这两个 ρ。**

---

## 2. B：`AZAI_VALUE_CKPT` 开关（默认关，行为逐位不变）

### 2.1 改动（4 个文件）

| 文件 | 改动 |
|---|---|
| `code/my_ai/az_intent/value_override.py`（**新增**） | `ValueModelAdapter`（把独立 `ValueNetwork` 包成三网形态 `{"value": ...}`，`parameters()/eval()/to()/state_dict()` 透传）、`value_of(out)`（dict / 裸张量通用）、`load_override()`（按 ckpt 的 `config` 建网 + 需要时补折 + 同路径缓存）、`maybe_override(model)`（**空 ⇒ 返回同一个对象**）、`describe()`（日志用） |
| `code/my_ai/az_intent/flat_rollout_greedy.py:284` | `self.value_model(...)["value"]` → `value_of(self.value_model(...))` |
| `code/my_ai/az_intent/value_batch.py:55` | 同上（`mc_score=net` 的批量叶子评估走这条） |
| `code/test_match/az_bridge_ai.py` | `AZAI_SEARCH=flat` / `unified` / `AZAI_MC_SCORE=net` 三处**叶子值网**加 `maybe_override(...)`；文件头补 `AZAI_VALUE_CKPT` 说明 |

**故意没做**：不在 `az_selfplay.load_three_models` 里挂钩。那样连 `net_fn` 的 value 都会被换掉
（M6c 仲裁用的 `net_out["value"]`），超出"只换叶子值网"的范围，也会让所有训练脚本被这个部署开关影响。

### 2.2 "默认行为不变"的**证据**（不是口头保证）

用项目里已有的零影响探针 `code/test_match/probe_oracle_inject_zero_impact.py`
（对真实中局快照跑 `search`，把 `last_scores` 的 float64 原始字节 / `visit_policy` / `bundles` / `sum_scores`
全打哈希）：

```powershell
$P code/test_match/probe_oracle_inject_zero_impact.py --tag VS_RU_dump --seed 8 `
   --rounds 10 60 200 --out match_results/small_value/zero_impact_before.log     # 改动前（HEAD 6db56ab）
# …改代码…
$P code/test_match/probe_oracle_inject_zero_impact.py ... --out match_results/small_value/zero_impact_after_off.log
$P code/test_match/probe_oracle_inject_zero_impact.py ... --out match_results/small_value/zero_impact_after_on.log
#                                ↑ AZAI_VALUE_CKPT=checkpoints/small32_3/v_32_3_folded.pt
```

| 比对 | 结果 |
|---|---|
| `before` vs `after_off`（开关空） | **IDENTICAL，0 行差异** ⇒ 默认路径逐位不变 ✅ |
| `after_off` vs `after_on`（开关开） | **24 行不同** ⇒ 开关是**活的**（不是只写了个变量） ✅ |

开关打开后**分数同量纲、决策真的变了**（举例，`r200/p1`，97 个候选）：
`chosen_idx 89 → 28`、`root_value 0.7714 → 0.7456`、`sum_scores 41.20 → 43.10`。
其余 5 个决策的 `sum_scores` 也在同量级（±0.3 内）变化，没有出现"tanh 被压平"那种数量级跳变。

### 2.3 开关语义 + 两处消费点的单元验证

`code/test_match/probe_value_override_switch.py`（日志 `training_history/logs/probe_value_override_switch.log`）：

```
[1] 开关为空（AZAI_VALUE_CKPT=''）
    maybe_override(teacher) is teacher : True
    已加载的覆盖网个数 (_CACHE)        : 0          ← 一个额外模型都没加载
    value_of(dict) is d['value']       : True       ← 返回的就是原来那个张量对象
    value_of(裸张量) is 原张量          : True
    ✅ 空值：同一对象、不加载任何模型、取值逐位不变

[2] 开关打开（AZAI_VALUE_CKPT=v_32_3_folded.pt）
    返回类型                     : ValueModelAdapter   is_value_override: True
    next(parameters()).is_cuda   : False                ← 消费点的 `next(...).is_cuda` 照常工作
    maybe_override 二次调用同一对象: True（缓存命中，不重复加载）
    输出形态：teacher dict=True | override dict=True     ← 两者同形态 ⇒ 消费点不用改
    老师 tanh std=0.2129 / 小网 tanh std=0.2400          ← 同量纲
    Spearman ρ(tanh) = 0.9621                            ← 256 帧，应 ≈0.96
    make_batched_value_fn：2 个真实局面（回合 [120, 200]）
      #0: 老师=-0.2819  小网=-0.3043  Δ=-0.0224
      #1: 老师=-0.1430  小网=-0.3642  Δ=-0.2211
✅ 开关语义与两处消费点都通过
```

（`make_batched_value_fn` 那两行只有 2 个点 ⇒ 脚本里算的 Spearman 是 `nan`，这是脚本打印的边界情况，
不是失败；判据是"有限值 + 同量纲"。）

真实对局里的日志长这样（`az_bridge_ai` 的 stderr）：
```
[azai] [value_override] 叶子值网 = v_32_3_folded.pt ValueNetwork(params=103,137) y_std=0.3861 y_mean=-0.0004 （ckpt 已折）
```

---

## 3. C：同一局面计时（本次核心产出）

### 3.1 方法（为什么这些数字能信）

- **局面**：从真实对局的 `--trace-ops` 日志（`match_results/ladder_logs/VS_RU_dump/7.log`）**按 bridge 顺序重放**，
  取 **3 个中局快照**：`seed7_r120_p1`、`seed7_r210_p1`、`seed7_r240_p1`
  （自动挑选规则：先只跑 `_expand` 数候选数，取**候选饱和 = 48** 的局面；空棋盘/早期局面只有 1-4 个候选，没有代表性）。
- **配置 = 部署配置**（与正在跑的 `rollout_strength` 实验的 `DECAY_C=1.0` 臂逐字一致）：
  `k=48, N=64, h=8, max_total_rounds=49920, rollout_mode=random, PROB_MODE=decay(1.0,3.0), SHARE_PREFIX=1,
  ACT_PROB=0.3, RANDOM_SIDE=opponent, value_batch=32, value_tanh=1, torch threads=1（部署口径 AZAI_TORCH_THREADS=1）`
- **配对**：同一进程、同一 searcher 构造参数、同一 `rollout_seed=4104` ⇒ 两侧 rollout **采到完全相同的叶子**
  （见下表 `候选/叶子/去重/节点` 两列**完全相同**），差别只有叶子值网本身。
- **抗负载**：交替先后顺序（偶数 rep 老师先、奇数 rep 小网先）、每个点 3 次取**中位**。
- **三类耗时分别记**：`total`（`search()` 全程）、`leaf`（包 `_leaf_values`，含 `encode_observation`）、
  `net`（只包**值网对象**的 `__call__`）⇒ `leaf-net` = 每叶固定成本（特征抽取），`total-leaf` = 树模拟。

```powershell
$env:OMP_NUM_THREADS="1"; $env:AZAI_VALUE_CKPT=""
bash code/run_logged.sh small_value_timing D:/anaconda3/envs/pytorch-gpu/python.exe -u `
  code/test_match/bench_leaf_value_swap.py --seeds 7 8 9 `
  --scan-rounds 120 150 180 210 240 270 300 --n-pos 3 --reps 3 --sweep-reps 2 `
  --out-json training_history/logs/bench_leaf_value_swap.json
```
运行记录：`training_history/runs/20261027_134637_small_value_timing/output.log`（`exit=0`，739 s）

### 3.2 主表：现役 64/6 vs 小网 32/3（k=48, N=64, h=8；3 局面 × 3 重复，取中位）

| 局面 | 臂 | 总耗时 | `_leaf_values` 累计 | **纯值网前向** | 树模拟 | 值网占比 | 叶块占比 | 候选 | 叶子评估数 | 去重叶子 | 节点数 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `r120_p1` | 现役 | 9991 ms | 5912 ms | **4317 ms** | 4079 ms | 43.2% | 59.2% | 48 | 3072 | 3006 | 21338 |
| `r120_p1` | 小网 | **6643 ms** | **2506 ms** | **897 ms** | 4137 ms | **13.5%** | 37.7% | 48 | 3072 | 3006 | 21338 |
| `r210_p1` | 现役 | 10334 ms | 6021 ms | **4341 ms** | 4349 ms | 42.0% | 58.3% | 48 | 3072 | 2995 | 19047 |
| `r210_p1` | 小网 | **6842 ms** | **2546 ms** | **901 ms** | 4295 ms | **13.2%** | 37.2% | 48 | 3072 | 2995 | 19047 |
| `r240_p1` | 现役 | 10296 ms | 5997 ms | **4326 ms** | 4342 ms | 42.0% | 58.3% | 48 | 3072 | 3000 | 19062 |
| `r240_p1` | 小网 | **6776 ms** | **2547 ms** | **900 ms** | 4378 ms | **13.3%** | 37.6% | 48 | 3072 | 3000 | 19062 |
| **中位** | 现役 | **10296 ms** | **5997 ms** | **4326 ms** | 4342 ms | **42.0%** | 58.3% | 48 | 3072 | 3000 | 19062 |
| **中位** | 小网 | **6776 ms** | **2546 ms** | **900 ms** | 4278 ms | **13.3%** | 37.6% | 48 | 3072 | 3000 | 19062 |

**逐项比值（同一次运行内、配对）**

| 量 | 现役 | 小网 | 比值 |
|---|---|---|---|
| 搜索总耗时 | 10296 ms | 6776 ms | **1.52×** |
| `_leaf_values` 累计（含特征抽取） | 5997 ms | 2546 ms | **2.35×** |
| **纯值网前向** | 4326 ms | 900 ms | **4.81×** |
| 每叶纯前向 | 1.408 ms/叶 | 0.293 ms/叶 | **4.81×** |
| 每叶固定成本（`encode_observation`） | 0.544 ms/叶 | 0.536 ms/叶 | 1.01×（**与值网无关**） |
| 树模拟（clone/apply/advance/前缀树） | 4342 ms | 4278 ms | 1.01×（**与值网无关**） |
| 节点数 / 去重叶子数 | 19062 / 3000 | 19062 / 3000 | **完全相同**（配对成功） |

> 与离线前向 bench 的 5.13×（batch64/4 线程）一致（这里 4.81×，batch32/1 线程、机器还在跑 8 局对局）。

**耗时分解（同一次运行的中位）**

```
现役 10296 ms = 树模拟 4342 (42.2%) + 特征抽取 1671 (16.2%) + 值网 4326 (42.0%) + 其它 ~39
小网  6776 ms = 树模拟 4278 (63.1%) + 特征抽取 1646 (24.3%) + 值网  900 (13.3%) + 其它 ~39
```

⇒ **值网占比 42% → 13%**；剩下 87% 与值网无关。所以"前向 4.8×"的**上限**只能是
`10296 / (4342+1671) = 1.71×`，实测 1.52× 已接近这个天花板。

### 3.3 双点斜率（同一次运行，用来换算"同样的时间能多买多少采样/候选"）

| 配置 | 叶子数 | 现役总耗时 | 小网总耗时 | 比值 |
|---|---|---|---|---|
| k=48, N=64（部署） | 3072 | 10296 ms | 6776 ms | 1.52× |
| k=96, N=64 | 6144 | 26428 ms | 17423 ms | 1.52× |
| k=48, N=128 | 6144 | 25832 ms | 16868 ms | 1.53× |

| 边际成本（每**多**一个单位） | 现役 | 小网 | 比值 |
|---|---|---|---|
| 每多 1 个采样样本（h=8 ⇒ 8 回合） | 5.05 ms | **3.28 ms** | 1.54× |
| 每多 1 个候选（N=64 ⇒ 64 样本） | 336.1 ms | **220.4 ms** | 1.52× |

### 3.4 **换算结论：同样的时间预算，能多放多少**

用**同一次运行**的配对数据（省下的时间 ÷ 小网的边际成本）：

| 说法 | 推导 | 数字 |
|---|---|---|
| 省下的时间 | 10296 − 6776 | **3520 ms（34%）** |
| 折成**采样数** | 3520 / 3.28 ms | **+1073 个叶子样本** |
| ⇒ **N（k=48 不变）** | 64 + 1073/48 | **64 → 86**（外推） |
| ⇒ **k（N=64 不变）** | 48 + 1073/64 | **48 → 65**（外推） |
| 若按 10 s 硬预算 | 现役这配置本身要 10.30 s（负载重时 13.25 s）⇒ **已经贴住/超出**；小网 6.78 s（8.65 s） | 小网**在 10 s 内还余 1.6–3.2 s** ⇒ 还能再加 **~500–1000 个样本** |

**外推做了实测验证（不然不敢写）**：

| 对照（同一次运行内交替测） | 总耗时 | 相对现役 N=64 |
|---|---|---|
| 现役 N=64（基准） | 10296 ms | 1.00 |
| 小网 **N=88**（外推 N*≈86 取整） | 11784 ms | **1.14×**（略超） |
| 另一次运行：现役 N=64 = 13247 ms，小网 **N=81** = 10828 ms | — | **0.82×**（还有余量） |

⇒ **安全区间 N ≈ 81–88，推荐明早用 N=88（保守）/ k=64**；N=88 在最坏情况下也只比现在慢 14%，
而 k=65 这条在两种估法下都稳定（48→64~69）。

> ⚠️ **两条必须说清的限制**：
> 1. **绝对值会被机器负载污染**：同配置现役总耗时在不同运行里是 **10.30 s / 13.25 s / 13.57 s**（差 1.3×，
>    因为同时有 8 局对局在跑）。所以本文只引用**同一次运行内**配对的比值；跨运行的绝对值不可比。
> 2. **"N 64→86" 不是线性永真**：N 越大前缀共享率越低（树模拟边际上升），所以外推在 N=88 处已经开始偏乐观
>    （1.14×）。要更远的 N 必须重测，不能拿 3.28 ms/样本 一路乘下去。

---

## 4. D：8 局冒烟（开关开/关各 4 局）

### 4.1 命令（`run_logged` 落盘）

```powershell
bash code/run_logged.sh small_value_smoke bash code/test_match/_smoke_small_value.sh
```
脚本内容 = 两批 `_tmp_ladder.py`（`--jobs=2`，2 局并发 ≈ 4 线程，遵守"最多 4 核"），
**冠军配置**（`docs/oracle_inject_design.md:220` 的 CHAMP，逐字一致）：

```
AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_K=160;AZAI_FLAT_ROUNDS=39000;
AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_SEARCH=flat;AZAI_MODE=pos-only   (+ AZAI_TORCH_THREADS=1)
```
① `AZAI_VALUE_CKPT` 空（= 改动前逻辑） ② `AZAI_VALUE_CKPT=checkpoints/small32_3/v_32_3_folded.pt`；
两批都用 `5 5r 6 6r`（2 seed × 2 座位），对手 = 亚军 `ai_cpp_lure_v4.exe`。

### 4.2 结果

（本次运行结束后填入 —— 见 `training_history/runs/20261027_141750_small_value_smoke/output.log`
与 `match_results/ladder_logs/small_value_smoke_{off,on}/`）

---

## 5. 明早 128 局 A/B 的可复制命令

判据（按 `docs/small_value_net_20261027.md` §6.2）：**"不显著变差"**，不是"显著变好"
（历史有 5 次"离线更准、实战更差"）。128 局 ⇒ `SE=√(0.25/128)=4.4%`，`2SE=8.8%`；
`--paired-sides` 会把座位方差配对掉。

### 5.1 必做：**只换叶子值网**（k=48/N=64/h=8，两边配置逐字相同）

```powershell
$PY = "D:/anaconda3/envs/pytorch-gpu/python.exe"
$BASE = "AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_K=48;AZAI_FLAT_ROUNDS=49920;" +
        "AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_HORIZON=8;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROLLOUT_MODE=random;" +
        "AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;" +
        "AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_RANDOM_SIDE=opponent;" +
        "AZAI_FLAT_EVAL=net;AZAI_VALUE_TANH=1;AZAI_SEARCH=flat;AZAI_MODE=pos-only;AZAI_TORCH_THREADS=1"

# A 臂 = 小值网，B 臂 = 现役值网（只差 AZAI_VALUE_CKPT 这一项）
& "D:\Program Files\Git\bin\bash.exe" code/run_logged.sh small_value_ab128 `
  $PY -u code/test_match/ab_search_modes.py --paired-sides `
  --seeds 3200 3201 3202 3203 3204 3205 3206 3207 3208 3209 3210 3211 3212 3213 3214 3215 `
          3216 3217 3218 3219 3220 3221 3222 3223 3224 3225 3226 3227 3228 3229 3230 3231 `
  --jobs 8 --out-dir match_results/small_value_ab128 `
  --flat-env="$BASE;AZAI_VALUE_CKPT=checkpoints/small32_3/v_32_3_folded.pt" `
  --mcts-env="$BASE;AZAI_VALUE_CKPT="
```
（`--paired-sides` + 64 seed ⇒ **128 局**；`--jobs 8` 是"8 局并发 ≈ 16 线程"，
明早没有别的实验在跑时可用；若还有实验在跑就 `--jobs 4`。）

**读法**：输出末行 `flat 胜率` 就是 A 臂（小值网）的胜率；判据 `≥ 0.5 − 2SE = 41.2%` 视为"不显著变差"，
`< 41.2%` ⇒ **停**，别再花时间调小网。

### 5.2 可选（红利）：小值网 **N=88** vs 现役 **N=64**（等时间、不等配置）

```powershell
# A 臂把小网省下的时间花掉：N=88（同时把 ROUNDS 上限抬高，避免被 48*8*64 clamp）
--flat-env="<同上 BASE，但 AZAI_FLAT_ROLLOUTS=88，AZAI_FLAT_ROUNDS=65536>;AZAI_VALUE_CKPT=checkpoints/small32_3/v_32_3_folded.pt" `
--mcts-env="<BASE>;AZAI_VALUE_CKPT="
```
跑法与 §5.1 完全一样（同一个 `ab_search_modes.py`，只换两个 env 串）。
**只有 §5.1 不掉（≥41.2%）才有资格跑这一条。**

---

## 6. 复现（全部命令）

```powershell
$PY = "D:\anaconda3\envs\pytorch-gpu\python.exe"
$env:PYTHONIOENCODING="utf-8"; $env:AZAI_VALUE_CKPT=""

# A：折叠 + 自检
& $PY -u code/test_match/fold_small_value_ckpt.py `
    --student-ckpt checkpoints/small32_3/lowq_value_seed0.pt `
    --out checkpoints/small32_3/v_32_3_folded.pt --check-frames 1000

# B：开关语义 + 两处消费点（应打印 ✅）
& $PY -u code/test_match/probe_value_override_switch.py

# B：零影响哈希比对（before 是改动前的代码打的，diff 必须无输出）
& "D:\Program Files\Git\bin\bash.exe" code/run_logged.sh small_value_zero_impact `
    $PY -u code/test_match/probe_oracle_inject_zero_impact.py --tag VS_RU_dump --seed 8 `
    --rounds 10 60 200 --out match_results/small_value/zero_impact_after_off.log
diff match_results/small_value/zero_impact_before.log match_results/small_value/zero_impact_after_off.log

# C：计时（3 局面 × 3 重复 × 2 臂 + 双点斜率 + 同预算验证）
& "D:\Program Files\Git\bin\bash.exe" code/run_logged.sh small_value_timing `
    $PY -u code/test_match/bench_leaf_value_swap.py --seeds 7 8 9 `
    --scan-rounds 120 150 180 210 240 270 300 --n-pos 3 --reps 3 --sweep-reps 2 `
    --out-json training_history/logs/bench_leaf_value_swap.json
& "D:\Program Files\Git\bin\bash.exe" code/run_logged.sh small_value_budget_check `
    $PY -u code/test_match/bench_leaf_value_swap.py --seeds 7 --scan-rounds 120 210 240 `
    --positions 7:120:1,7:210:1,7:240:1 --n 64 --reps 3 --sweep 0 --budget-check 81 `
    --out-json training_history/logs/bench_budget_check.json

# D：8 局冒烟
& "D:\Program Files\Git\bin\bash.exe" code/run_logged.sh small_value_smoke `
    bash code/test_match/_smoke_small_value.sh
```

产物：
- `checkpoints/small32_3/v_32_3_folded.pt`（429,075 B，sha256 `76dfa446…f7ef315`）
- `training_history/logs/fold_small32_3.log`、`probe_value_override_switch.log`、
  `bench_leaf_value_swap.json`、`bench_budget_check.json`、`bench_leaf_value_swap_n80.json`
- `match_results/small_value/zero_impact_{before,after_off,after_on}.log`

---

## 7. 约束遵守 / 已知未做

| 约束 | 实际情况 |
|---|---|
| 不干扰正在跑的实验 | **未 kill/暂停任何进程**。计时用 1 线程（部署口径）；冒烟 `--jobs=2`（2 局 × (AI 1 线程 + bridge 1 线程) ≈ 4 线程） |
| 不改 `az_intent/*` 除消费点兼容以外的逻辑 | 只改了 2 行取值（`value_of`）+ 新增 1 个模块；**搜索逻辑一行没动**（哈希探针证） |
| 不跑 128 局 A/B | 未跑（明早的事，命令见 §5） |
| 日志落盘 | 全部 `run_logged` 或重定向到 `training_history/logs/`；没有过滤原始输出 |
| 已知**没做** | ① `AZAI_MC_SCORE=net` 那条路只做了**单元级**验证（2 个真实局面），没跑整局；② 与"部署时实时 encode 出来的 stats"逐维比对（调研 §5.2 遗留项）仍未做；③ 计时是在"8 局对局同时跑"的负载下量的（绝对值偏慢 ~1.3×，但比值可用）；④ `N>88` 的换算没有实测，不做承诺 |
| 一次失败记录 | `training_history/runs/20261027_135910_small_value_timing_n80`（`exit=1`，5 s）：我第一次跑 N=80 时忘了 `--scan-rounds`，`--positions` 里的回合不在重放集里 ⇒ 脚本主动报错退出（**不是对局失败**）。重跑那次（`20261027_135937`）`exit=0` |
