"""给 _tree_rollout_value 加"树规模"统计：节点数 / 叶子数（开关 AZAI_ROLLOUT_SAMPLES=1）。"""
from pathlib import Path

p = Path("code/my_ai/az_intent/flat_rollout_greedy.py")
t = p.read_text(encoding="utf-8")

if "last_tree_stats" not in t:
    # 在 _tree_rollout_value 返回前记录（它 return 的是单个 float）
    old = "    def _tree_rollout_value(self, child, player: int, rngs) -> float:"
    assert old in t
    t = t.replace(old, old + "\n        self.last_tree_stats = None", 1)
    # 找到该函数里的 return 语句（返回均值）
    # 用更稳的方式：在函数体内任何 "return float(" / "return np.mean" 前插记录
    import re
    pattern = re.compile(r"(    def _tree_rollout_value\(.*?\n)(.*?)(\n    def )", re.S)
    m = pattern.search(t)
    assert m, "找不到函数体"
    body = m.group(2)
    # 在 body 的最后一个 return 前插入
    idx = body.rfind("return")
    assert idx > 0, "body 里没有 return"
    ins = ("if _REC_SAMPLES:\n            self.last_tree_stats = (len(states), len(leaf_states))\n        ")
    body2 = body[:idx] + ins + body[idx:]
    t = t[:m.start(2)] + body2 + t[m.end(2):]
    p.write_text(t, encoding="utf-8")
    print("  已加 last_tree_stats（节点数/叶子数）")
else:
    print("  已有 last_tree_stats")
