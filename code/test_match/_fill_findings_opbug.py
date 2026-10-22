"""把「int('op') bug」的根因写进 findings 文档。"""
from pathlib import Path

p = Path("docs/findings_flat_rollout_vs_tree.md")
t = p.read_text(encoding="utf-8")

sec = """
### (f) **根因：`int('op')` 一行 bug 让「升级」从未被执行（2026-10-22）**

对亚军的 32 局里我们 **0 次升级**（对手 8.2%）。我先后归因为"值网不认可升级""需要重训"
"需要照抄亚军的硬编码奖励"——**全部是错的**。真正原因是 `bundle_mcts.py` 里的一行：

```python
_ccls = (int(_c[0]) if isinstance(_c, (tuple, list)) else int(_c))   # 先执行
if isinstance(_c, (tuple, list)) and _c[0] == "op":                  # 才判断
    self.mc_last_explicit = (...)
```

显式升级候选的形式是 `("op", 12, tower_id, target)`，`_c[0]` 是**字符串 "op"** ⇒
`int("op")` 抛 `ValueError` ⇒ 被宽 `except Exception` 吞掉 ⇒ 走 fallback（`pinned = 类头 argmax`）
⇒ **`mc_last_explicit` 永远是 None ⇒ 桥接永远拿不到升级 op ⇒ 升级动作自落地起一次都没被执行过**。

**修复后（30 个真实中局、双方都有塔的局面）**：

| 仲裁模式 | 升级被选中 | op 中位排名 |
|---|---|---|
| quiet256（现状，**不加 bonus**）| **12/30（40%）** | 2.0 |
| net32 | 5/30（17%）| 3.0 |
| quiet + bonus 0.05 / 0.10 / 0.20 | 47% / **63%** / 73% | 1.0 |

⇒ **升级通路本来是通的，只是执行环节被异常吞掉了**；bonus 只是把它推得更高，不是必需。
⇒ 之前"5000 局低质量数据里塔类型 0 占 99.7%"的观察仍成立（那些局由 M6c 裸策略生成），
但它**不是**当前 0% 升级率的原因。
⇒ 教训：**宽 `except` 会吞掉"选择成功但执行失败"这类错误**，表现为"某个动作类永远不出现"，
看起来像模型/数据问题，实际是一行 API 误用。
"""

anchor = "### (d) 待办"
if anchor in t:
    t = t.replace(anchor, sec + "\n" + anchor, 1)
else:
    t = t.rstrip() + "\n" + sec
p.write_text(t, encoding="utf-8")
print("已写入 findings (f)")
