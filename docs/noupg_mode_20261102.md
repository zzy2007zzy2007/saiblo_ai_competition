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

## 4. 结果（16 局 = seeds 100..107 × 2 座位，`verdict=engine` 16/16、illegal 全 0）

| 主读数 | 值 |
|---|---|
| **胜率** | **0W-16L / 16 = 0.0%**，Wilson 95% = **[0.0%, 19.4%]** |
| **血差 Δhp = A − B** | 按 seed 配对（n=8 对）：**mean = −22.4**，**t95 = [−26.1, −18.8]**（se=1.5） |
| 血差（16 局不配对） | mean −22.4，median −23.5，min −33，max −6 |
| 座位拆分 | A 坐 P0（n=8）：Δhp −25.9；坐 P1（n=8）：Δhp −19.0（都负） |

逐局 Δhp（A − B）：−31 / −21 / −25 / −20 / −33 / −25 / −26 / −6 / −27 / −15 / −28 / −14 / −22 / −30 / −15 / −21。
16 局里 A 的基地血量全是 0（被平推）。

**结论**：`AZAI_CAND_MODE=unified_noupg`（unified 候选里删掉升级）**显著劣于** pin+flat（胜率 0%、Δhp 配对
t95 整段在 0 以下）。删升级**没有**救回 unified，反而比 unified 更差（Δhp −22.4 vs lightning 臂的 −20.9，
但注意两者 rollout 不同，不能直接比）。

## 5. 关键观察（升级被删后选了啥、攒钱/闪电有没有变多）

（口径：A 臂 5573 决策 vs B 臂 5573 决策，同批对局 stderr 聚合）

**① 删了多少升级**：541/5573 决策（9.7%）的 top-k 里含升级被删，共删 **7023 条**升级候选（平均 1.26/决策）。

**② 行为对比**：

| 行为 | A（unified_noupg） | B（pin） |
|---|---|---|
| 升级 op12 发出 | **0** | 0 |
| 闪电 op21 发出 | **36** | 139 |
| 建塔 op11 | 1172 | 1186 |
| 降级 op13 | 1017 | 1107 |
| 偏转 op23 | **15** | 0 |
| 闪避 op24 | **5** | 0 |
| HOLD（空束）回合 | 3822 | 4069 |
| **金币≥90 的决策（攒到 90）** | **121/5573 = 2.2%** | **2644/5573 = 47.4%** |

**结论：删升级没有让搜索更爱攒钱/放闪电，反而是更回避。** 删掉升级后：
- 攒到 90 的决策掉到 **2.2%**（pin 47.4%；也比 lightning 臂的 4.2% 更低）；
- 闪电掉到 **36 次**（pin 139）；
- 替代 = **偏转(15) + 闪避(5) 这些超武**（pin 从不用）+ 建塔/降级/HOLD（与 pin 同量级）。

即"候选层删升级"释放出来的预算**没有**流到闪电/攒钱，而是流到超武（偏转/闪避）和继续 HOLD。这进一步支持
`docs/unified_cand_fix_20261029.md` §9/§10 的判读：**瓶颈是评估器（值网定价）系统性偏好"别攒钱别放闪电"**，
不是"升级候选把闪电挤掉了"。

## 6. commit

- 见提交记录（`unified_noupg` 实现 + 驱动 + 分析器）。
