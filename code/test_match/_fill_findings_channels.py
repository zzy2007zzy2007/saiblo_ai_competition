"""把「位置网通道质量（轴序修正后）」与「统一搜索两个 bug 修正」写进 findings 文档。"""
from pathlib import Path

p = Path("docs/findings_flat_rollout_vs_tree.md")
t = p.read_text(encoding="utf-8", errors="replace")

section = """
### (e) **轴序 bug 与位置通道质量重测（2026-10-21 深夜）**

**Bug**：`make_position_masks` 与网络 `action_map` 的空间轴都是 **`[x][y]`**
（`decoder.decode` 里是 `x, y = np.unravel_index(argmax, ...)`），
但 `unified_action_search.py` 早期版本写成 `[ch, y, x]` ⇒ **全部转置**。
症状：33 个可建格里只枚举出 **4** 个建塔（应 31）。修正后 31 与引擎判定一致。
**该 bug 影响到的旧读数（已作废）**：
- "统一搜索在真实中局选中升级" —— 那是转置先验下的候选集；修正后该局面选的是建塔 `(11,4,9)`；
- "建塔通道与 adv 反相关（−0.247）" —— 见下，修正后是 +0.046。

**重测（18 个真实中局局面，`adv` = 打完不推进的 1-ply 值网差分）**：

| 通道 | top-1 与 1-ply 值网一致 | **位置图 vs adv 的 Spearman** | 判读 |
|---|---|---|---|
| 闪电(17) | 0% | **+0.738**（均值 +0.765，n=6）| **强信号** |
| 建塔(0) | 0% | **+0.046**（均值 +0.049，n=12）| **基本无信号** |

⇒ **除闪电外，位置图不能用于排序**（用户 2026-10-21 的判断成立）。

**对各实验解释的影响**：

| 实验 | 是否受影响 | 说明 |
|---|---|---|
| T2（cells4 > cells0，75%）| ⚠️ | `mc_cells` 用 `action_map` 取前 4 格，而经济类排序无信号 ⇒ 那 4 格≈随机。增益更可能来自**类比较的样本变多（边际更稳）**或**引入我方塔格（多表达升级/降级）**，不能解释为"选格更准" |
| flat 胜 mcts（86%）| ⚠️ 部分 | flat 候选来自 `_expand` 采样（依赖位置图）⇒ 经济类候选覆盖差；胜因更可能在评估器 |
| 统一搜索 | ✅ | 通道预算已改为**类网 softmax**；通道内排序虽用坏通道，但建塔格**全部枚举**（cap 不生效）⇒ 影响很小 |
| 仲裁评估器对照 | ✅ | 只换评估器 |

**重训优先级据此调整**：第一优先是**建塔通道**（不是"改进"而是"从零建立信号"），
目标用 `adv`（冻结值网的 1-ply 优势），覆盖 16 个塔意图通道。
"""

anchor = "### (d) 待办"
if anchor in t:
    t = t.replace(anchor, section + "\n" + anchor, 1)
else:
    t = t.rstrip() + "\n" + section
p.write_text(t, encoding="utf-8")
print("已写入 findings (e) 节")
