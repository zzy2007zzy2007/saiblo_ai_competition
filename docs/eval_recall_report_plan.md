# 计划：把"按动作类型的 recall 统计"统一到一个脚本（2026-10-27）

## 现状

仓库里有三份重叠的统计代码（不含训练期用的第四份）：

| 文件 | 口径 | 覆盖动作类型 | 随机基线 | min-cells |
|---|---|---|---|---|
| `code/test_match/compare_priors_on_runnerup.py` | 亚军决策点（日志重放） | build/lightning/downgrade/evasion（缺 emp/deflector） | 只给 @1/@24 | 无 |
| `code/test_match/eval_per_op_recall.py` | 我们 80 局留出（npz） | build/downgrade/lightning（缺 emp/deflector/evasion） | 只 1 行汇总 | 2 |
| `code/test_match/eval_multi_prior.py` | 我们 80 局留出（npz） | build/lightning/downgrade | 逐行给 @1/@24 | **硬编码 3** |
| `code/my_ai/train_prior_z.py:recall_eval` | 训练期 | build/lightning/downgrade | 无 | **硬编码 3**（不动） |

问题：三种动作类型覆盖、两种支撑集门槛（2 vs 3）、基线列不一致 ⇒ 同一件事的数字不可直接对比；
`min-cells=3` 会把**降级（支撑集中位 2）整类筛掉**，导致"降级看不到"被误读成"降级没样本"。

## 方案

新增 `code/test_match/eval_recall_report.py`，一次跑出**两套口径**，共用同一套表格/基线/JSON 生成代码：

- **口径 A（亚军决策点）**：直接 `import compare_priors_on_runnerup`，复用其 `parse_log` / `collect_decisions`
  重放与口径（**不重写解析**）。支撑集 = 该动作类在该局面的合法格数（position mask）；
  目标 = 亚军实际走法。排名沿用其 `-1e9` 掩码 + `argsort` 写法，保证与旧脚本逐位一致。
- **口径 B（我们 80 局留出）**：复用 `my_ai.train_prior_z.load_games`，取 `g9376-g9379`。
  支撑集 = npz 里该动作组出现过的候选格数；目标 = 教师 argmax 候选。

共同点（这是"统一"的实质）：

1. 动作类型固定 6 类：建塔 11 / 降级 13 / 闪电 21 / EMP 22 / deflector 23 / evasion 24，**n=0 也打印一行**；
2. `--min-cells` 可配（默认 2，可设 1 复现旧脚本"无门槛"）；
3. 每行都给随机基线（`1/n`、`min(4,n)/n`、`min(24,n)/n` 的均值）；
4. `n < 20` 的行标注 `样本不足`（降级类本来就极小，避免读成结论）；
5. `--nets name=path,...` 传网络列表；`--json-out` 落一份 JSON 便于以后对比。

## 两个旧脚本

`eval_per_op_recall.py` / `eval_multi_prior.py` 头部加 `# DEPRECATED: 已被 eval_recall_report.py 取代（2026-10-27）`，
**不删除**（可能还有脚本/文档在调）。

`train_prior_z.py:recall_eval` 不动（训练内环需要，且不属于"报告"）。

## 验收

1. 新脚本对 4 个默认网络跑通，两套口径各出一张表；
2. 与旧脚本数字对上：口径 A 用 `--min-cells 1 --only A` 与 `compare_priors_on_runnerup.py` 的 posnet_A 行逐位相等；
   口径 B 与 `eval_per_op_recall.py`（min-cells=2）逐位相等；
3. 报告 `docs/eval_recall_report.md` + `git commit`。
