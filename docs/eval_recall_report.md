# 统一的按动作类型 recall 报告：`eval_recall_report.py`（2026-10-27）

> 一句话：把「按动作类型的 recall 统计」从三份重叠脚本收敛到
> `code/test_match/eval_recall_report.py`，一次跑出**两套口径**（亚军决策点 / 我们自己的留出局），
> 共用一个表格 + 随机基线 + JSON 生成器；旧脚本 `eval_per_op_recall.py`、`eval_multi_prior.py`
> 已标 `DEPRECATED` 但保留（可能还有调用方）。

---

## 一、为什么合并

合并前，同一件事有三份实现，**动作类型覆盖、支撑集门槛、基线列数都不同**：

| 脚本 | 口径 | 实际会打印的动作类型 | min-cells | 随机基线 |
|---|---|---|---|---|
| `compare_priors_on_runnerup.py` | 亚军决策点 | build / lightning / downgrade / evasion（**n=0 的类直接 `continue` 跳过**） | 无 | 只有 @1 / @24 |
| `eval_per_op_recall.py` | 我们留出 | build / downgrade / lightning | 2 | 只有 1 行汇总 |
| `eval_multi_prior.py` | 我们留出 | build / lightning / downgrade | **硬编码 3** | 只有 @1 / @24 |
| `my_ai/train_prior_z.py:recall_eval` | 训练内环 | build / lightning / downgrade | **硬编码 3** | 无 |

后果：`min-cells=3` 会把**降级（支撑集中位 2）整类筛掉**，看起来像"降级没样本"；
`emp(22)` / `deflector(23)` / `evasion(24)` 在任何一份旧报告里都从未出现过
——而它们在留出局里占 **3,118 / 4,749 = 66%** 的决策点（见第四节口径 B）。
换句话说，**旧的"按动作类型"结论只覆盖了 1/3 的数据**。

---

## 二、怎么用

```powershell
cd D:\2026智能体大赛_新2_distill
$env:PYTHONIOENCODING="utf-8"
& "D:\anaconda3\envs\pytorch-gpu\python.exe" code\test_match\eval_recall_report.py `
    --json-out docs\eval_recall_report_20261027.json
```

默认网络（可用 `--nets` 覆盖）：`posnet_A` / `prior_hybrid_z_only` / `prior_hard_ce_w1` / `prior_hard_ce_w3`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--nets name=path,...` | 上面 4 个 | 逗号分隔；路径不存在会直接报错退出 |
| `--min-cells` | `2` | 支撑集门槛；`1` = 不设门槛（可复现旧 A 脚本）；实测 `3` 会吃掉整类降级 |
| `--insufficient-n` | `20` | `n` 低于此值的行尾标注 `<- 样本不足` |
| `--only` | `all` | `A` / `B` / `all` |
| `--logs-dir` / `--n-logs` | `match_results/ladder_logs/VS_RUNNERUP_cells4` / `16` | 口径 A 用前 N 局（`0` = 全部 32 局）；也可直接给位置参数指定日志 |
| `--holdout` | `g9376,g9377,g9378,g9379` | 口径 B 的留出局 stem（`training_history/prior_labels/`） |
| `--device` / `--threads` | `cpu` / `4` | 默认 CPU 是为了和旧脚本**逐位可比**；线程上限默认 4，避免干扰在跑的实验 |
| `--json-out` | 无 | 写一份 JSON（`meta` + `calibers.A/B`，每个网络每个动作类型都有 `n/hit/recall/random`） |

其他用法：

```powershell
# 只跑亚军口径，复现旧脚本（无门槛）
... eval_recall_report.py --only A --min-cells 1

# 换成任意一组网络
... eval_recall_report.py --nets "a=checkpoints/posnet_A_k5_m32.pt,b=checkpoints/prior_z_best.pt"
```

---

## 三、两套口径的语义（**先看这张表，别横比**）

| 项 | 口径 A：亚军决策点 | 口径 B：我们自己的留出局 |
|---|---|---|
| 数据 | `VS_RUNNERUP_cells4` 日志重放（复用 `compare_priors_on_runnerup.py` 的 `parse_log`/`collect_decisions`） | `prior_labels/g9376-g9379.npz`（复用 `my_ai.train_prior_z.load_games`） |
| 状态 | 重放到该轮的我方视角 | npz 里的对局窗口 |
| **支撑集** | 该动作类的**合法格数**（position mask） | npz 里该动作组**出现过的候选格数** |
| **目标** | **亚军实际走法**（模仿目标）；只在"重放后该格仍合法"的决策点上统计 | **教师 argmax 候选**（rollout 最优的那一格） |
| 排名域 | 全 361 格里把非法格压成 `-1e9` 再 `argsort`（沿用旧脚本写法） | 只在候选格内 `argsort` |
| 实测支撑集中位 | build 32、downgrade 2、lightning 271、evasion 271 | build 33、downgrade 2、lightning 206、emp 35、deflector 37、evasion 26 |

两套口径**不可横比**：目标不同（实际走法 vs 教师最优）、数据分布不同（亚军对局 vs 我方对局）。
本脚本的意义是"同一份代码、同一套基线、同一套样本不足标注"下把两者并排看，而不是求两者一致。

---

## 四、一次真实运行（4 个网络，`--min-cells 2`）

命令见第二节；日志 `training_history/logs/eval_recall_report_20261027.log`；
JSON `docs/eval_recall_report_20261027.json`（39 KB，含全部 hit 计数）。
耗时：A 重放 2.3 s + A 评估 7.1 s + B 25.9 s（本机 CPU，4 线程，全程 < 1 分钟）。

`⚠` = `n < 20`，样本不足；`随机@k` 该行支撑集上的逐决策点均值（`1/n`、`min(4,n)/n`、`min(24,n)/n`）。

### 口径 A：亚军决策点（16 局，1,383 个决策点，min-cells 丢掉 153 个）

| 动作类型 | 网络 | n | 支撑中位 | @1% | @4% | @24% | 随机@1% | 随机@4% | 随机@24% |
|---|---|---|---|---|---|---|---|---|---|
| **build** | posnet_A | 303 | 32 | 5.0 | 18.2 | 74.9 | 3.1 | 12.3 | 74.1 |
|  | prior_hybrid_z_only | 303 | 32 | 5.0 | 16.2 | 78.2 | 3.1 | 12.3 | 74.1 |
|  | prior_hard_ce_w1 | 303 | 32 | 8.6 | 21.5 | 76.9 | 3.1 | 12.3 | 74.1 |
|  | prior_hard_ce_w3 | 303 | 32 | 8.3 | 20.1 | 77.6 | 3.1 | 12.3 | 74.1 |
| **downgrade** | posnet_A | 866 | 2 | 53.9 | 100.0 | 100.0 | 46.5 | 100.0 | 100.0 |
|  | prior_hybrid_z_only | 866 | 2 | 55.4 | 100.0 | 100.0 | 46.5 | 100.0 | 100.0 |
|  | prior_hard_ce_w1 | 866 | 2 | 54.4 | 100.0 | 100.0 | 46.5 | 100.0 | 100.0 |
|  | prior_hard_ce_w3 | 866 | 2 | 53.9 | 100.0 | 100.0 | 46.5 | 100.0 | 100.0 |
| **lightning** | posnet_A | 55 | 271 | 5.5 | 20.0 | 67.3 | 0.4 | 1.5 | 8.9 |
|  | prior_hybrid_z_only | 55 | 271 | 7.3 | 16.4 | 72.7 | 0.4 | 1.5 | 8.9 |
|  | prior_hard_ce_w1 | 55 | 271 | 7.3 | 27.3 | 78.2 | 0.4 | 1.5 | 8.9 |
|  | prior_hard_ce_w3 | 55 | 271 | 3.6 | 25.5 | 63.6 | 0.4 | 1.5 | 8.9 |
| **emp**（op=22） | — | n=0 | — | — | — | — | — | — | — |
| **deflector**（op=23） | — | n=0 | — | — | — | — | — | — | — |
| **evasion** | posnet_A ⚠ | 6 | 271 | 0.0 | 0.0 | 0.0 | 0.4 | 1.5 | 8.9 |
|  | prior_hybrid_z_only ⚠ | 6 | 271 | 0.0 | 0.0 | 0.0 | 0.4 | 1.5 | 8.9 |
|  | prior_hard_ce_w1 ⚠ | 6 | 271 | 0.0 | 16.7 | 33.3 | 0.4 | 1.5 | 8.9 |
|  | prior_hard_ce_w3 ⚠ | 6 | 271 | 0.0 | 0.0 | 16.7 | 0.4 | 1.5 | 8.9 |
| 总体（6 类合并） | posnet_A | 1230 | 2 | 39.4 | 75.8 | 91.9 | 33.5 | 73.5 | 89.1 |
| 总体（6 类合并） | prior_hybrid_z_only | 1230 | 2 | 40.6 | 75.1 | 92.9 | 33.5 | 73.5 | 89.1 |
| 总体（6 类合并） | prior_hard_ce_w1 | 1230 | 2 | 40.7 | 77.0 | 93.0 | 33.5 | 73.5 | 89.1 |
| 总体（6 类合并） | prior_hard_ce_w3 | 1230 | 2 | 40.2 | 76.5 | 92.4 | 33.5 | 73.5 | 89.1 |

> **口径 A 的两个坑**（新脚本都打印出来了，旧脚本没有）：
> ① `emp` / `deflector` 在这 16 局里 **n=0**——亚军首操作从未出现这两类，所以旧 A 脚本"按动作类型"永远只有 4 类；
> ② 降级类占 **866/1230 = 70%** 的决策点，支撑集中位只有 2（随机 @1 就有 46.5%），
> 所以 A 的"总体"行基本是降级类在说话，**只有分动作类型的行才有信息量**。

### 口径 B：我们自己的 80 局留出（g9376-g9379，4,860 个决策点，min-cells 丢掉 111 个）

| 动作类型 | 网络 | n | 支撑中位 | @1% | @4% | @24% | 随机@1% | 随机@4% | 随机@24% |
|---|---|---|---|---|---|---|---|---|---|
| **build** | posnet_A | 1577 | 33 | 14.0 | 35.6 | 91.4 | 3.2 | 12.7 | 73.9 |
|  | prior_hybrid_z_only | 1577 | 33 | 12.2 | 34.7 | 92.9 | 3.2 | 12.7 | 73.9 |
|  | prior_hard_ce_w1 | 1577 | 33 | 10.6 | 32.1 | 91.8 | 3.2 | 12.7 | 73.9 |
|  | prior_hard_ce_w3 | 1577 | 33 | 10.5 | 31.5 | 90.2 | 3.2 | 12.7 | 73.9 |
| **downgrade** | posnet_A ⚠ | 2 | 2 | 100.0 | 100.0 | 100.0 | 50.0 | 100.0 | 100.0 |
|  | prior_hybrid_z_only ⚠ | 2 | 2 | 50.0 | 100.0 | 100.0 | 50.0 | 100.0 | 100.0 |
|  | prior_hard_ce_w1 ⚠ | 2 | 2 | 100.0 | 100.0 | 100.0 | 50.0 | 100.0 | 100.0 |
|  | prior_hard_ce_w3 ⚠ | 2 | 2 | 50.0 | 100.0 | 100.0 | 50.0 | 100.0 | 100.0 |
| **lightning** | posnet_A | 52 | 206 | 1.9 | 11.5 | 48.1 | 0.5 | 1.9 | 11.5 |
|  | prior_hybrid_z_only | 52 | 206 | 1.9 | 9.6 | 55.8 | 0.5 | 1.9 | 11.5 |
|  | prior_hard_ce_w1 | 52 | 206 | 3.8 | 9.6 | 50.0 | 0.5 | 1.9 | 11.5 |
|  | prior_hard_ce_w3 | 52 | 206 | 5.8 | 9.6 | 46.2 | 0.5 | 1.9 | 11.5 |
| **emp** | posnet_A | 526 | 35 | 6.5 | 20.2 | 75.9 | 4.0 | 13.9 | 69.8 |
|  | prior_hybrid_z_only | 526 | 35 | 7.6 | 18.8 | 74.0 | 4.0 | 13.9 | 69.8 |
|  | prior_hard_ce_w1 | 526 | 35 | 5.9 | 21.5 | 76.6 | 4.0 | 13.9 | 69.8 |
|  | prior_hard_ce_w3 | 526 | 35 | 6.8 | 20.9 | 77.4 | 4.0 | 13.9 | 69.8 |
| **deflector** | posnet_A | 1298 | 37 | 3.7 | 12.7 | 63.0 | 2.9 | 11.4 | 65.4 |
|  | prior_hybrid_z_only | 1298 | 37 | 3.5 | 13.9 | 70.9 | 2.9 | 11.4 | 65.4 |
|  | prior_hard_ce_w1 | 1298 | 37 | 4.9 | 16.4 | 76.5 | 2.9 | 11.4 | 65.4 |
|  | prior_hard_ce_w3 | 1298 | 37 | 4.5 | 16.2 | 79.1 | 2.9 | 11.4 | 65.4 |
| **evasion** | posnet_A | 1294 | 26 | 5.5 | 18.3 | 90.6 | 4.0 | 16.1 | 90.1 |
|  | prior_hybrid_z_only | 1294 | 26 | 7.0 | 22.0 | 91.3 | 4.0 | 16.1 | 90.1 |
|  | prior_hard_ce_w1 | 1294 | 26 | 8.3 | 24.3 | 94.4 | 4.0 | 16.1 | 90.1 |
|  | prior_hard_ce_w3 | 1294 | 26 | 9.7 | 26.5 | 93.7 | 4.0 | 16.1 | 90.1 |
| 总体（6 类合并） | posnet_A | 4749 | 33 | 7.9 | 22.7 | 81.3 | 3.4 | 13.3 | 74.9 |
| 总体（6 类合并） | prior_hybrid_z_only | 4749 | 33 | 7.8 | 23.5 | 84.0 | 3.4 | 13.3 | 74.9 |
| 总体（6 类合并） | prior_hard_ce_w1 | 4749 | 33 | 7.9 | 24.3 | 86.2 | 3.4 | 13.3 | 74.9 |
| 总体（6 类合并） | prior_hard_ce_w3 | 4749 | 33 | 8.2 | 24.6 | 86.2 | 3.4 | 13.3 | 74.9 |

> **口径 B 的降级类基本不可评估**：84 个降级候选组里 82 个支撑集只有 1（被 min-cells=2 丢掉），
> 剩下 n=2 —— 所以旧 `eval_multi_prior.py`（min-cells=3）里降级直接变成 n=0 完全合理，
> 但它**同时**把 emp/deflector/evasion 也漏了（这三类是正常样本量），那才是真问题。

### 终端原样输出（节选，口径 A build / 口径 B build）

```text
  -- 建塔 build（op=11）  n=303  支撑集中位=32（31-33）
    网络                       n  支撑中位      @1      @4     @24   随机@1  随机@4  随机@24
    posnet_A                 303        32    5.0%   18.2%   74.9%     3.1%   12.3%    74.1%
    prior_hybrid_z_only      303        32    5.0%   16.2%   78.2%     3.1%   12.3%    74.1%
    prior_hard_ce_w1         303        32    8.6%   21.5%   76.9%     3.1%   12.3%    74.1%
    prior_hard_ce_w3         303        32    8.3%   20.1%   77.6%     3.1%   12.3%    74.1%

  -- EMP emp（op=22）  n=0
    （无样本）               n=0   -- 该口径样本中未出现此动作类型；仍列出以保持 6 类动作覆盖
```

参数/环境行、`NOTE 降级类（op=13）样本量通常极小…n<20 的行标注「样本不足」` 和"总体"行都在实际日志里，
见 `training_history/logs/eval_recall_report_20261027.log`。

---

## 五、和旧脚本数字对得上的证据（逐位核对，不是"差不多"）

### 核对 1：口径 A 完全复现 `compare_priors_on_runnerup.py`

```powershell
# 旧脚本（4 个旧网络，无 min-cells 门槛）
$env:OMP_NUM_THREADS=4; & $py code\test_match\compare_priors_on_runnerup.py   # 日志 deprecated_compare_runnerup_20261027.log
# 新脚本：--min-cells 1 等价于"无门槛"
& $py code\test_match\eval_recall_report.py --only A --min-cells 1 --nets "posnet_A=checkpoints/posnet_A_k5_m32.pt" `
    --json-out training_history\logs\eval_recall_report_mc1_A.json               # 日志 ..._crosscheck_mc1_A_20261027.log
```

| 指标 | 旧脚本 | 新脚本 `--min-cells 1` |
|---|---|---|
| build n / 支撑中位 / @1 / @4 / @24 | 303 / 32 / 5.0% / 18.2% / 74.9% | 303 / 32 / **5.0% / 18.2% / 74.9%** |
| build 随机 @1 / @24 | 3.1% / 74.1% | 3.1% / 74.1% |
| downgrade n / @1 / 随机@1 | 1019 / 60.8% / 54.5% | 1019 / **60.8% / 54.5%** |
| lightning n / @1 / @4 / @24 | 55 / 5.5% / 20.0% / 67.3% | 55 / **5.5% / 20.0% / 67.3%** |
| evasion n / @1 / @4 / @24 | 6 / 0.0% / 0.0% / 0.0% | 6 / **0.0% / 0.0% / 0.0%** |

（新脚本多给了 @4 与逐行随机基线；重叠列**逐位相同**。注意 `posnet_A` 建塔 `@1=5.0%`，
而 `docs/prior_ceiling_20261026.md` §九 的表里写的是 `6.3%` —— 那是 `f10` 的数，见第六节 ⑥。）

### 核对 2：口径 B（`--min-cells 2`）完全复现 `eval_per_op_recall.py`

| 指标 | 旧 `eval_per_op_recall.py` | 新 `--only B`（默认 min-cells=2） |
|---|---|---|
| build n / 支撑中位 | 1577 / 33 | 1577 / 33 |
| build posnet_A / z_only / w1 / w3 @1 | 14.0 / 12.2 / 10.6 / 10.5 % | **14.0 / 12.2 / 10.6 / 10.5 %** |
| build @4 | 35.6 / 34.7 / 32.1 / 31.5 % | **35.6 / 34.7 / 32.1 / 31.5 %** |
| build @24 | 91.4 / 92.9 / 91.8 / 90.2 % | **91.4 / 92.9 / 91.8 / 90.2 %** |
| lightning n / 随机@24 | 52 / 11.5% | 52 / 11.5% |
| lightning z_only @24 | 55.8% | 55.8% |
| downgrade n | 2 | 2 |

### 核对 3：口径 B（`--min-cells 3`）复现 `docs/day_summary_20261027.md` 的 `eval_multi_prior.py` 旧表

```powershell
& $py code\test_match\eval_recall_report.py --only B --min-cells 3 --json-out training_history\logs\eval_recall_report_mc3_B.json
```

| 网络 | 文档（旧 `eval_multi_prior.py`，min-cells=3） | 新脚本 `--min-cells 3` |
|---|---|---|
| posnet_A 总体 @1/@4/@24 | 7.8% / 22.4% / 81.2%（n 未打印） | **7.8% / 22.4% / 81.2%**（n=4734） |
| z_only 总体 @1/@4/@24 | 7.6% / 23.3% / 83.9% | **7.6% / 23.3% / 83.9%** |
| 建塔 posnet_A @1 | 14.0% | **14.0%** |
| 闪电 @24 posnet_A / z_only | 48.1% / 55.8% | **48.1% / 55.8%** |
| 降级 n | （整类消失，文档里没有降级行） | **n=0，被 min-cells 丢弃 84** |

⇒ 三处独立核对全部逐位相等；默认 `min-cells=2` 与 3 的差别**只是** 15 个决策点（4,749 vs 4,734）。

---

## 六、发现的口径不一致（这是本次重构最有价值的部分）

1. **`min-cells` 三套并存（无 / 2 / 3）**。同一份留出局，降级类在 `min-cells=2` 下有 2 个决策点、
   在 `3` 下 **n=0**；口径 A 里降级类有 1,019 个决策点，其中 **153 个支撑集只有 1**（`@1` 天然命中，
   被旧脚本算进去 ⇒ 降级 `@1` 从 53.9% 被抬到 60.8%，**虚高 6.9pp**）。新脚本默认 `2` 并打印被丢弃数。
2. **动作类型覆盖不同，旧结论只看了 1/3 的数据**。口径 A 的旧脚本对 n=0 的类 `continue`，
   所以 `emp`/`deflector` 从未出现（这 16 局里确实是 0）；口径 B 的旧脚本 `NAME`/循环只有
   build/downgrade/lightning，而 `emp`/`deflector`/`evasion` 合计 **3,118/4,749 = 66%** 的决策点。
   **"按动作类型"的历史结论其实漏掉了三分之二的样本。**
3. **随机基线的给法不同**。旧 A 脚本只给随机 `@1`/`@24`；旧 `eval_per_op_recall.py` 把基线压成
   一行单独打印（数值上等价，但视觉上像"只有基线网络有基线"）；`eval_multi_prior.py` 只给
   `@1`/`@24`。新脚本**每行都给** `@1`/`@4`/`@24` 三条基线。
4. **支撑集定义不同（不是 bug，但不可横比）**：A = 合法格数（lightning/evasion 恒 271，build 31-33），
   B = npz 候选格数（lightning 178-239，build 6-33）。降级类最典型：A 有 866-1,019 个决策点，
   B 只有 **2** 个 ⇒ "降级样本少"这句话在两套口径下差三个数量级。
5. **目标定义不同**：A 是"亚军实际走法"（模仿），B 是"教师 argmax"（rollout 最优）。
   直接后果见第七节：**同一网络在 build `@1` 上两套口径给出相反的符号**。
6. **文档与脚本输出不一致（记录级）**：`docs/prior_ceiling_20261026.md` §九 把
   `posnet_A` 的建塔 `@1` 记成 `6.3%`，实跑旧脚本是 `5.0%`（`6.3%` 是 `f10` 那一行）。
   以脚本输出为准；本次核对顺手发现了这处笔误。

---

## 七、4 个网络在两套口径下的关键差异

**口径 A（亚军决策点，目标=实际走法）**

- **建塔**：`hard_ce_w1` 最好（`@1` 8.6% vs `posnet_A` 5.0%，n=303 ⇒ 多命中 **11** 个决策点，
  噪声量级约 2pp，属边缘信号）；`w3` 8.3%；纯 z-score 的 `z_only` 头部没变（5.0%），但 `@24` 最好（78.2%）。
- **闪电**：`hard_ce_w1` 的 `@24` 最高（78.2% vs 67.3%，n=55 ⇒ 多 6 个命中，**不足以区分**）；
  `w3` 反而最差（63.6%）。**符号与口径 B 相反**，样本太少，不要当结论。
- **降级 / evasion**：`@4`/`@24` 全是平凡值（支撑集 2-3），evasion n=6 → 均不可区分。

**口径 B（我们自己的留出局，目标=教师 argmax）**

- **建塔 `@1` 单调下降**：14.0 → 12.2 → 10.6 → 10.5%（−53 个命中/1,577，约 −3.4pp），
  `@24` 却持平甚至略升（91.4 → 92.9%）⇒ 训练**把排序整体抬了一点，但弄钝了头部**，与
  `docs/day_summary_20261027.md` 的旧结论一致。
- **`deflector` 是最大赢家**：`@24` 63.0 → 79.1%（`w3`，**+16.1pp / +209 个命中**），
  `@1` 也从 3.7 → 4.9%；`evasion` `@1` 5.5 → 9.7%（`w3`）、`@4` 18.3 → 26.5%，`@24` 已达 94.4%（`w1`，
  随机 90.1%，**接近上限**）；`emp` 基本无变化（75.9 → 77.4%）。
- **总体**：`@1` 7.9 → 8.2%、`@4` 22.7 → 24.6%、`@24` 81.3 → 86.2%（`hard_ce_*` 两版几乎并列）。

**两套口径的分歧（重点）**

| 现象 | 口径 A | 口径 B |
|---|---|---|
| build `@1`：`hard_ce_w1` vs `posnet_A` | **更好 +3.6pp**（8.6% vs 5.0%） | **更差 −3.4pp**（10.6% vs 14.0%） |
| build `@24` 相对随机 | +0.8~+4.1pp（几乎没学到东西） | +16.3~+19.0pp（明显学到东西） |
| 降级 | n=866，可看（但 `@4/@24` 平凡） | n=2，不可看 |

⇒ 结论：**任何"训练有没有用 / 哪个网更好"的判断都必须写清是哪套口径**。
口径 A 偏"模仿亚军"，口径 B 偏"对齐 rollout 教师"，两者在 build `@1` 上符号相反；
而 build `@24` 上 A 几乎贴着随机（74.1% vs 74.9-78.2%）、B 明显高于随机（73.9% vs 90-93%），
说明**旧文档"先验没多少可挖"的结论主要来自口径 A 的这个特定指标**，换到 B 上信号强得多。
这也和"离线指标预测不了实战强度"（`docs/prior_ceiling_20261026.md`）不矛盾：
口径只是换了"离线"的定义，实战仍需另行验证。

---

## 八、复现清单

| 内容 | 路径 |
|---|---|
| 统一脚本 | `code/test_match/eval_recall_report.py` |
| 已废弃（保留） | `code/test_match/eval_per_op_recall.py`、`code/test_match/eval_multi_prior.py`（头部 `# DEPRECATED: ...`） |
| 主运行日志 | `training_history/logs/eval_recall_report_20261027.log` |
| 主运行 JSON | `docs/eval_recall_report_20261027.json` |
| 核对 1 日志（A，min-cells=1） | `training_history/logs/eval_recall_report_crosscheck_mc1_A_20261027.log` |
| 核对 3 日志（B，min-cells=3） | `training_history/logs/eval_recall_report_crosscheck_mc3_B_20261027.log` |
| 旧脚本基线日志 | `training_history/logs/deprecated_compare_runnerup_20261027.log`、`..._deprecated_eval_per_op_recall_20261027.log` |
| 计划文档 | `docs/eval_recall_report_plan.md` |

未改动：`code/my_ai/train_prior_z.py:recall_eval`（训练内环，任务明确要求不动）。
