# unified_noupg：候选层删升级（2026-11-02）

> 状态：**实现完成、默认关；A/B 排队中（lightning 跑完后串行）**。
> 一句话：`AZAI_CAND_MODE=unified_noupg` —— 在 **unified 阶段1 结束、top-k 已生成、阶段2 之前**，
> 把候选里含升级（op 12）的**全部删掉**，剩下的进阶段2（flat rollout）里择优。

## 1. 语义（用户强调）

- **删除时机** = 阶段1 结束、top-k 候选已生成之后、阶段2（flat rollout）之前 ⇒ **不影响阶段1**（bandit+游标照跑）。
- 是"**候选层删升级**"，**不是**"阶段2 后检查 chosen 是不是升级、是就改成 HOLD"。
  用户要的是"让搜索在**除升级外的候选里选最好**"，不是"选到升级就换 HOLD"。
- **兜底**：删完一个候选都不剩 ⇒ 退化为 HOLD（空束）；理论上有 `AZAI_CAND_FORCE_HOLD=1` 兜底所以罕见。

## 2. 实现（只改"阶段1→阶段2 之间的候选过滤"这一处）

- `code/my_ai/az_intent/unified_cand.py`：
  - `UnifiedCandFlatSearch.__init__`：`self.cand_no_upgrade = (cand_mode == "unified_noupg")`；且
    阶段1 生成器 `UnifiedCandidateGen` 的 `mode` 传 `"unified"`（与 unified **逐字相同**，不 shape_pin）。
  - `UnifiedCandFlatSearch._expand`：`generate()` 返回 top-k 后、写 `node.bundles` 前，加过滤：
    ```python
    if self.cand_no_upgrade:
        _n_before = len(bundles)
        bundles = [b for b in bundles if not any(int(o[0]) == 12 for o in (b or ()))]
        if not bundles:
            bundles = [()]
        self.last_cand_stats["n_upg_removed"] = _n_before - len(bundles)
    ```
- `code/test_match/az_bridge_ai.py`：`AZAI_CAND_MODE` 合法值扩 `unified_noupg`（`_is_unified_cand` 也含），
  走 `make_unified_searcher_class()`、`cand_mode="unified_noupg"`。
- 阶段1 bandit/cursor、阶段2 flat rollout **一字不动**；参数与 BASE 逐字一致。

## 3. 实验设计

| 项 | 值 |
|---|---|
| A 臂 | `AZAI_CAND_MODE=unified_noupg`（+ `AZAI_CAND_VERBOSE=1`） |
| B 臂 | `AZAI_CAND_MODE=pin`（现役 pin+flat） |
| rollout | 两臂都 `random`（对手建塔/降级，BASE 默认）—— **不含 lightning** |
| 对局 | 16 局 = seeds 100..107 × 2 座位，`--jobs 8` |
| 驱动 | `code/test_match/_noupg_ab.sh` → `ab_search_modes.py`；stderr 唯一后缀 `_noupg` |
| 有效性 | 只算 `verdict=engine` |

## 4. 结果

> 16 局 A/B 跑完后填：unified_noupg vs pin 的胜率（Wilson CI）+ 血差 Δhp（t CI）。

_待补_

## 5. 关键观察（升级被删后选了啥、攒钱/闪电有没有变多）

> 跑完后填：A 臂 op 构成（op11/12/13/21/HOLD）+ 金币≥90 决策占比，与 pin / lightning 臂对照。

_待补_

## 6. commit

- 见提交记录（`unified_noupg` 实现 + 驱动 + 分析器）。
