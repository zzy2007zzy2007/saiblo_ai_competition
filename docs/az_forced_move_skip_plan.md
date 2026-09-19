# 计划：单候选（强制着）跳过迭代 —— 把"没得搜"回合的成本真正省掉

2026-09-15，用户提出，我实现。

## 一、动机与实测依据

前面两轮实测（`docs/experiment_log.md` 的 `search_axis_ablation` 与 `search_speed_bench`）：

1. **搜索增益 100% 在"位置"维**：`pos-only(argmax)` 候选空间只有 2.8%（97% 的回合
   root 只剩 1 个候选），却完整复现 joint 的 93.8%（同样 30W/2L）。
2. 但"只搜位置"只省 **12%**——因为限制搜索轴减的是**候选生成**，钱花在别处。
3. 成本模型（三条 arm 全部命中，误差 <3%）：

   | 项 | 实测 | joint 占比 |
   |---|---|---|
   | 一次前向（encode 0.34 + 模型 2.20）| **2.539 ms** | — |
   | 360×`sample_bundle`（一次展开的候选生成）| **~1.2 ms** | — |
   | **256 次迭代 = 256 次前向** | **653 ms** | **~85%** |
   | 一次展开合计（1 前向 + 360 采样）| 3.7 ms | 49.9 次 ≈ 185 ms |

   ⇒ 搜索成本 ≈ **257 × 2.539ms + 展开次数 × 3.7ms**。
   验算：joint 836ms（实测 850）、pos-only 741（748）、skip-hold 172（178）。

> ⚠️ **2026-09-19 更正**：上表的"一次展开 3.7ms"与"256 次迭代 = 653ms（85%）"**都偏低了**。
> 直接实测（root 有满 24 个候选、`skip_single_candidate=False`）：**k=24/迭代 256 的一次完整
> 搜索 = 2.30 s**（一次前向 4.39 ms）。差的倍数来自**展开**：`_expand` 每次要跑
> `k×sample_mult = 360` 次 `sample_bundle`，而每次 `sample_bundle` 内部对 3 个头各调一次
> `decode_head` ⇒ **每次展开一千多次解码**；而 256 次迭代里**几乎每次都在展开一个新节点**
> （最深 4 轮）。所以大头是"展开 × 解码"，不是"256 次前向"。
> 本文档当时之所以"误差 <3%"，是因为那台子上 `skip_single_candidate` 让 ~97% 的回合只有 1 个
> 候选、**只发生 1 次展开**，展开项被掩盖了。
> ⇒ **引用搜索成本时请用 2.3 s/次（k=24/迭代 256），不要用 0.65 s。**

**所以真正的大头是"每次都跑满 256 次迭代"**，而 MCTS 跑这 256 次，是为了在
多个候选里挑一个。**如果 root 只有一个候选，这 256 次迭代对结果毫无影响。**

## 二、为什么"单候选跳过"是严格等价的，不是近似

`BundleMCTS.search` 的出口是 `visits = [该候选的访问数]`（只有一个元素），于是

- `policy = _policy_from_visits(visits, T)` → 无论 T 是多少都是 `[1.0]`；
- `chosen = argmax(visits)` = 0（或 `_sample([1.0])` 也必返回 0）；
- `chosen_bundle = root.bundles[0]`；
- `intent_counts = root.intent_counts`。

这些都**只依赖 `_expand(root)` 的产物**，与迭代次数无关。迭代只影响
`root.visits / root.value_sum`（即 `root_value`）和子树。

⇒ 跳过 256 次迭代，**选动作、visit 策略目标（自对弈写入 npz 的 `visit`）、
bundle 列表、intent 计数全部逐位不变**；唯一变的是 `root_value` 与 `last_root` 的子树。

### ⚠️ 等价的边界：单次搜索内 vs 整局（重要，别读过头）

上面那条是**"同一 RNG 状态下的单次搜索"**层面的恒等式。**整局层面并不逐位相同**：

被跳过的那 256 次迭代本来会消耗 `self.rng`（每次展开都要 `rng.choice` 采候选）。
同一个 `BundleMCTS` 对象在一局里被复用 ⇒ 跳过之后 RNG 流推进的步数变了
⇒ **后面回合采到的候选不一样** ⇒ 实现出来的那几局对不上。

**决策规则与分布没变，实现的具体对局会变。** 两个实际后果：

1. **跨这个开关不保证逐局可复现**（不能拿开关前的某局去对开关后的同一局）；
2. `az_selfplay` 的 `random_action_prob` 用的是 `mcts.rng`，也跟着变（概率不变、实现不同）。

所以判据只能是：
- **等价性** = 同 RNG、单次搜索逐位比对（见 §五，已通过）；
- **强度是否不变** = 两边各打 N 局比胜率（32 局起），**不是**比逐局结果。
"8 局逐局相同"既不该期待、也不构成证明。

## 三、影响面调查（已做）

| 字段 | 消费方 | 影响 |
|---|---|---|
| `chosen_bundle` | `az_selfplay.py:238`、`eval.py:140`、`az_search_vs_search.py:69`、`diagnose_bundle.py`、`diag_native_crash*.py` | 逐位不变 ✅ |
| `bundles` | `az_selfplay.py:239`（写入数据）、`eval.py:142`（room 诊断）、`_bench_depth_cost.py` | 逐位不变 ✅ |
| `intent_counts` | `az_selfplay.py:239`（写入数据）、`eval.py:149` | 逐位不变 ✅ |
| `visit_policy` | `az_selfplay.py:240`（**训练用的策略目标**）、`_smoke_test.py`（断言 sum=1） | `[1.0]`，逐位不变 ✅ |
| **`root_value`** | **只有 `_smoke_test.py` 打印**；`az_selfplay`/`eval`/`svs`/训练全都不读 | 语义从"访问均值"变为"根网络值"，**无实质消费方** ✅ |
| `last_root` | `diagnose_search.py`（调试工具，打印树统计） | 跳过时子树不展开、`visits=0` ⇒ **该调试工具的数字会失真**（可接受，会在文档里写明） |

## 四、改法

`BundleMCTS.__init__(..., skip_single_candidate: bool = False)`；
`search()` 在 `self._expand(root, self.k)` 之后：

```python
if self.skip_single_candidate and len(root.bundles) <= 1:
    policy = np.ones(1, dtype=np.float32) if root.bundles else np.zeros(0, dtype=np.float32)
    return BundleSearchResult(
        bundles=list(root.bundles), visit_policy=policy, chosen_index=0,
        chosen_bundle=root.bundles[0] if root.bundles else (),
        root_value=float(root.net_value), intent_counts=list(root.intent_counts))
```

**默认 `False`（opt-in），CLI 为 `--skip-single-candidate`**（`eval.py`、`az_selfplay.py`）。

### 为什么最终决定"默认关"（而不像原计划那样验证后翻成默认开）

原本打算验证通过就默认开启。但**跳过会扰动 `self.rng` 的推进步数**（见 §二 的边界），
于是：

- **joint 模式**：只有 ~3% 的回合命中 ⇒ 收益约 1.03×，却仍然改变了 RNG 流
  ⇒ **为 3% 的加速破坏"同一 commit + 同一命令 → 同一数字"的复现性，不划算**；
- **pos-only 模式**：97% 回合命中 ⇒ 25.8×，值得；而 pos-only 本身就是 opt-in 的。

⇒ 让两个开关都 opt-in、成对使用，现有脚本的行为与随机流一字不变。

## 五、验证计划

1. **等价性（决定性）**：取 ≥300 个真实局面，同一局面、同一 seed（**保证同一 RNG 状态**），
   `skip=False` 与 `skip=True` 两次 `search()`，断言 `chosen_bundle`、`bundles`、
   `visit_policy`、`intent_counts` **逐位相等**；并另测 `temperature=1.0` 的探索路径。
   —— 已完成：330 局面 × 2 模式 = 660 次比对 **0 差异**，`temperature=1.0` 0/60。
2. **端到端（只当冒烟 + 测速，不作等价性证据）**：小规模跑通不崩、量出实际加速比。
   **不比对逐局结果**（见上节：RNG 流不同 ⇒ 对局本就会变）。
3. **速度**：同批局面横比，检验"单候选回合 ≈ 4-5ms vs 748ms"的预测。
   —— 已完成：joint 3.0% 单候选中位 775→5.0ms；pos-only 97.3% 单候选整体 **25.83×**。
4. **记录**：结果补进 `docs/experiment_log.md`（`run_logged` 名 `forced_move_skip`）。

## 六、预期收益（先写下来，免得到时候顺着结果编解释）

- 单候选回合：**~748ms → ~4ms（1 前向 + 360 采样）**，约 6-7×；
- 与 `pos-only(argmax)` 组合（97% 回合是单候选）：
  平均 ≈ 0.97×4ms + 0.03×748ms ≈ **26 ms/回合**，对比 joint 的 850ms ⇒ **~30×**；
- 对比 `--skip-hold-search`（81% 回合 0 成本、19% 回合付全价 ≈ 162ms/回合）
  ⇒ 预估本方案**比它再快 ~6 倍**，且**保留位置搜索**（skip-hold 是整店不打烊式跳过）。

## 七、风险 / 边界

- `root_value` 语义变化：目前无消费方，但**若将来有人拿它当价值目标就会踩坑**，
  已在代码注释里标注。
- `last_root` 跳过时不展开子树 ⇒ `diagnose_search.py` 的树统计失真（调试工具，接受）。
- 单候选时**仍然要付 1 次前向 + 360 次候选生成**（~4ms）。理论上可以更省
  （先判 argmax 是否 HOLD 直接返回 0 成本，即 `--skip-hold-search`），
  但那会丢掉"位置维"的搜索（本方案的收益恰恰来自保留位置搜索）。
- **前提是这个模型的性质**："argmax 类常不可执行 ⇒ 单候选"来自类头坍缩。
  类头若训好，单候选占比会下降、加速比随之下降（功能仍然正确，只是没那么赚）。
