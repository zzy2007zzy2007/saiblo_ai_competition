"""给 flat_rollout_greedy 加 PROB_MODE=reverse（时间反演，2026-10-27）。

现状：decay ⇒ p(r) = C·exp(-r/k)          （r=0 最大，早期反击）
新增：reverse ⇒ p(r) = C·exp(-(H-r)/k)     （r=H 最大，末期反击）
两者期望动作总数相同（同一串指数反序），区别只在**分叉发生在早期还是末期**。
前缀树共享的是早期路径 ⇒ 末期分叉应显著减少节点数。
"""
from pathlib import Path

p = Path("code/my_ai/az_intent/flat_rollout_greedy.py")
t = p.read_text(encoding="utf-8")

if '"reverse"' not in t:
    old = "        _p = self.rollout_decay_c * _math.exp(-round_idx / self.rollout_decay_k)"
    assert old in t, "decay 分支锚点未找到"
    new = (old + "\n"
           "        elif getattr(self, \"rollout_prob_mode\", \"\") == \"reverse\":\n"
           "            # 时间反演：p(r) = C·exp(-(H-r)/k)，末期才反击 ⇒ 前缀树节点更少\n"
           "            _p = self.rollout_decay_c * _math.exp(-(self.horizon - round_idx) / self.rollout_decay_k)")
    # 注意：原代码是 if/elif 链，需要在 decay 之后插 elif；先看看原分支结构
    t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")
    print("  已加 reverse 分支（需人工核对 if/elif 链）")
else:
    print("  已有 reverse 分支")
