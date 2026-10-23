"""把前缀树接到 search() 的 rollout 循环里（第二步）。

原循环（每样本独立）：
    for _k in range(n_rollouts):
        sim = child.clone()
        ... 逐回合 _random_actions + advance ...
        leaves.append(sim)
    scores[index] = mean(leaf_values(leaves))

新：若 `rollout_share_prefix` 且 rollout_mode != idle ⇒ 走 `_tree_rollout_value`。
"""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "my_ai/az_intent/flat_rollout_greedy.py"
t = P.read_text(encoding="utf-8")

old = """        for index, child in enumerate(children):
            leaves = []
            for _k in range(n_rollouts):"""
new = """        for index, child in enumerate(children):
            # 2026-10-23：前缀树记忆化（随机 rollout 专用）。
            #   随机 rollout 要大 N 才抹得平噪声，而朴素实现成本 ∝ N×H；
            #   树版按 (父节点, 本回合 ops) 复用状态：共享率 43%~94% ⇒ 省 21%~91%。
            if _rctx is not None and self.rollout_share_prefix:
                _rngs = [np.random.default_rng(
                    (int(self.rollout_seed) * 1000003 + self._decision_idx) * 100003
                    + int(index) * 101 + int(_k)) for _k in range(n_rollouts)]
                _t0 = _time.perf_counter()
                scores[index] = self._tree_rollout_value(child, player, _rngs)
                total_rounds += self.horizon * n_rollouts
                continue
            leaves = []
            for _k in range(n_rollouts):"""
assert old in t, "锚点未命中"
t = t.replace(old, new, 1)
P.write_text(t, encoding="utf-8")
print("前缀树已接入 search()")
