# Q 网当搜索偏置：类先验 + s2 混合（2026-11-02）

## 0. 一句话

把上一轮训的动作价值 Q 网（`code/my_ai/az_intent/qnet_action.py`，commit 979460d）当
**搜索偏置**接进 unified 候选搜索，做两个实验：**A = Q 网当类先验**（改 bandit 的
`Q[c]` 初始化）、**B = s2 分数混合 Q 网值**（改候选排序）。每个实验都跑
**terminal 与 kgeo50 两个 Q 网 checkpoint**，并排看哪个标签臂当搜索偏置更好。

## 1. 背景（Q 网已知结论，见 docs/qnet_action_gradient_20261102.md）

- Q(s,a) = 冻结值网 encoder(局面 42 stats + board) + 24 维动作多重集向量 → 小 MLP；
  输出 = 终局血差（回归，clip ±1 量纲）。
- 4 个标签臂：terminal（终局血差）、kgeo50（abs+kgeo τ50）、abs50、relk50。
- 关键（commit 979460d 诊断）：**闪电 ∂Q/∂a 两臂都 100% 正**（terminal +0.040、
  kgeo50 +0.043）；"闪电排名#1" 90.2%（terminal）/ 96.6%（kgeo50）。
  - 建塔/升级的相对排序两臂不同：terminal **升级(+0.0065) > 建塔(-0.0070)**；
    kgeo50 **建塔(+0.0033) > 升级(-0.0014)**（"闪电>>建塔>升级" 58.6%）。
  - kgeo50 回归 r 更高（+0.927 vs terminal +0.648）⇒ 用户想两个都试。
- **用直接求值 `Q(s,a)`，不用偏导**（候选动作类/多重集确定了，直接前向）。

## 2. 实验 A：Q 网当类先验（改 bandit 的 Q[c] 初始化）

- unified 阶段1 的类 UCB bandit `Q[c]` 现在初始全相等（`self.Q={c: []}` ⇒ `_q_util` 恒 0）。
- 改成：每类初始 `Q[c] = [Q(s, a_c)]`，`a_c` = 第 c 类=1 的 24 维 one-hot（其余 0）。
  - 注意 UCB 的 `n<=0 → +inf` 分支**不动**（默认路径零影响）；类先验通过 `_q_util` 的
    玻尔兹曼加权影响"已试过臂"的利用项，让 bandit 在探索一遍后偏向值钱类（闪电）。
- 开关：`AZAI_QNET_CLASS_PRIOR=1` + `AZAI_QNET_CKPT=<qnet ckpt 路径>`；默认关、零影响。

## 3. 实验 B：s2 分数混合 Q 网值（改候选排序）

- 每条候选：`s2' = zscore(s2) + w × zscore(Q(s, a_candidate))`。
  - `s2` = 阶段2 flat rollout 分数（`FlatRolloutGreedy.search` 的 `scores`）。
  - `Q(s,a_candidate)` = 该候选 bundle 的 24 维动作多重集（`encode_ops`）直接前向。
  - z-score **在本次决策的候选集内做（全局）**；std=0 / 含 NaN 时安全处理（→ 0）。
- 用 `s2'` 做 argmax 选 bundle（退化保护、`last_scores` 上报口径不变）。
- 开关：`AZAI_QNET_S2_MIX=w`（float）+ `AZAI_QNET_CKPT=<qnet ckpt 路径>`；默认 0 = 关、零影响。

## 4. 实现位置

- `code/my_ai/az_intent/qnet_action.py`：新增 `QNetScorer`（批量直接前向）+ `load_qnet_scorer()`。
- `code/my_ai/az_intent/unified_cand.py`：`UnifiedCandidateGen` 加 `class_prior_fn`；
  `UnifiedCandFlatSearch.__init__` 读 `AZAI_QNET_CLASS_PRIOR` 注入先验。
- `code/my_ai/az_intent/flat_rollout_greedy.py`：`FlatRolloutGreedy` 加 `qnet_scorer` 参数 +
  读 `AZAI_QNET_S2_MIX`，在 `search()` 的 argmax 前混合。
- `code/test_match/az_bridge_ai.py`：读 `AZAI_QNET_CKPT` 加载 scorer 并传给搜索器。

## 5. 实验设计

- 对手 = **pin**（`AZAI_CAND_MODE=pin` + BASE 自带 random rollout，现役）。
- 阶段1/阶段2 其余参数与 BASE（`_us_vs_external.sh:68`）**逐字一致**（归因干净）。
- 16 局 = seeds 100..107 × 2 座位（`--paired-sides` 镜像配对）。
- 先小批量（4 局）看方向，有明显差异再补 16 局。

### 臂矩阵

| 臂 | 开关 | Q 网 | 权重 |
|---|---|---|---|
| A-term | `AZAI_QNET_CLASS_PRIOR=1` | `qnet_terminal.pt` | — |
| A-kgeo | `AZAI_QNET_CLASS_PRIOR=1` | `qnet_kgeo50.pt` | — |
| B-term | `AZAI_QNET_S2_MIX=w` | `qnet_terminal.pt` | w ∈ {0.1, 0.3, 0.5, 1.0} |
| B-kgeo | `AZAI_QNET_S2_MIX=w` | `qnet_kgeo50.pt` | w ∈ {0.1, 0.3, 0.5, 1.0} |

- B 的 w 先扫小批量（4 局 × {0.1,0.3,0.5,1.0}），挑"闪电被顶上来又不破坏整体"的 w 补 16 局。

## 6. 预注册判据

- **有效**：A/B 任一臂 16 局胜率比 BASE（unified vs pin 历史基线）明显更高，或
  行为上"放闪电次数"显著上升且胜率不降。
- **无效**：胜率与 BASE 无差异，且"放闪电次数/攒到90占比"没变化 ⇒ Q 网先验没转成行为。

## 7. 交付

1. 代码（两个开关 + 文档，默认关零影响）。
2. 实验 A/B（terminal/kgeo50）各自的 16 局胜率/血差 + 行为构成（放闪电次数、攒到90占比、动作分布）。
3. 结论：Q 网当类先验 / s2 混合有没有让 unified 打 pin 变好？"闪电>>建塔/升级"先验有没有转成行为？
4. 本 doc + commit（仓库身份 + Co-authored-by + Session-Id + Session-Parent）。
