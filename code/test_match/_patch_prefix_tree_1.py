"""实现**前缀树 rollout**（2026-10-23 用户论点：随机 rollout 需要大 N，必须靠记忆化压成本）。

实测依据：
  · clone 31.9 µs，advance ~61.5-93 µs；
  · N=32、H=16 时朴素成本 ≈ 42 ms/候选；前缀树在共享率 85% 时只要 8.4 ms（省 80%）；
  · 共享率实测：const 27%、decay(C=.6,k=3) 43%、decay(C=.3,k=1) 85%、first 94%。

实现方式（最小改动）：
  1. 把 `_random_actions` 拆成 **只决策不落子** 的 `_decide_actions()`（返回 ops 列表）
     + 复用现有逻辑的 `_apply_actions()`；
  2. 新增 `_tree_rollout_value()`：按 (父节点, 本回合 ops token) 缓存子状态，
     命中则直接跳到缓存状态（省掉 clone+apply+advance）；
  3. `search()` 里若 `rollout_share_prefix=True` 且 rollout_mode != idle，走树版。
"""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "my_ai/az_intent/flat_rollout_greedy.py"
t = P.read_text(encoding="utf-8")

# ── ① 参数 ────────────────────────────────────────────────────────────────
old = "                 rollout_decay_k: float = 3.0,        # 衰减常数 k：p(r)=C*exp(-r/k)"
new = ("                 rollout_decay_k: float = 3.0,        # 衰减常数 k：p(r)=C*exp(-r/k)\n"
       "                 # 2026-10-23（用户论点）：随机 rollout 要大 N 才管用 ⇒ 必须做**前缀树记忆化**。\n"
       "                 #   实测共享率 const 27% / decay(.6,3) 43% / decay(.3,1) 85% / first 94%。\n"
       "                 rollout_share_prefix: bool = False,")
assert old in t, "① 锚点未命中"
t = t.replace(old, new, 1)
t = t.replace("        self.rollout_decay_k = max(1e-6, float(rollout_decay_k))",
              "        self.rollout_decay_k = max(1e-6, float(rollout_decay_k))\n"
              "        self.rollout_share_prefix = bool(rollout_share_prefix)", 1)

# ── ② 把 _random_actions 改成"只决策"，并新增 _apply_actions ─────────────
start = t.index("    def _random_actions(self, sim, rng, ctx, root_player=None, round_idx: int = 0) -> None:")
end = t.index("    # ── 叶子评估 ──")
old_block = t[start:end]
new_block = '''    def _decide_actions(self, sim, rng, ctx, root_player=None, round_idx: int = 0) -> list:
        """只**决定**本回合双方各做什么（不落子），返回 [(player, t, a0, a1), ...]。

        2026-10-23：从 `_random_actions` 拆出来，供前缀树实现复用（先决策、查缓存、再决定是否真模拟）。
        概率按 `rollout_prob_mode`（const / decay / first）逐回合计算。
        """
        if self.rollout_prob_mode == "first":
            _p = self.rollout_decay_c if round_idx == 0 else 0.0
        elif self.rollout_prob_mode == "decay":
            import math as _math
            _p = self.rollout_decay_c * _math.exp(-round_idx / self.rollout_decay_k)
        else:
            _p = self.rollout_act_prob
        if _p <= 0.0:
            return []

        from SDK.utils.constants import OperationType

        _sides = (0, 1)
        if self.rollout_random_side == "opponent" and root_player is not None:
            _sides = (1 - int(root_player),)
        out = []
        for _pl in _sides:
            if rng.random() >= _p:
                continue
            _did = False
            if rng.random() < self.rollout_build_ratio:
                try:
                    _cost = int(sim.build_tower_cost(sim.tower_count(_pl)))
                    _coins = int(sim.coins[_pl])
                except Exception:
                    _cost, _coins = -1, -1
                if _cost >= 0 and _coins >= _cost:
                    _occ = {(t.x, t.y) for t in sim.towers_of(_pl)}
                    _cand = [c for c in ctx["cells"][_pl] if c not in _occ]
                    if _cand:
                        _x, _y = _cand[int(rng.integers(len(_cand)))]
                        out.append((_pl, int(OperationType.BUILD_TOWER), int(_x), int(_y)))
                        _did = True
            if not _did:
                _tws = list(sim.towers_of(_pl))
                if _tws:
                    _tw = _tws[int(rng.integers(len(_tws)))]
                    out.append((_pl, int(OperationType.DOWNGRADE_TOWER), int(_tw.tower_id), -1))
        return out

    def _random_actions(self, sim, rng, ctx, root_player=None, round_idx: int = 0) -> list:
        """决策 + 立刻落子（旧的单轨迹路径用）。返回 applied ops 列表。"""
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType
        ops = self._decide_actions(sim, rng, ctx, root_player, round_idx)
        for (_pl, _t, _a0, _a1) in ops:
            try:
                sim.apply_operation_list(int(_pl),
                                         [Operation(OperationType(int(_t)), int(_a0), int(_a1))])
            except Exception:
                pass
        return ops

    def _tree_rollout_value(self, child, player: int, rngs) -> float:
        """前缀树版 rollout：按 (父节点, 本回合 ops) 复用子状态，省掉重复的 clone/apply/advance。"""
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType, HIGHLAND_CELLS

        ctx = {"cells": {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}}
        # node_id -> state（state 已推进到该节点回合结束）
        states = {0: child.clone()}
        edges = {}          # (parent_id, token) -> child_id
        next_id = 1
        leaf_values = []

        for rng in rngs:
            cur = 0
            for _r in range(self.horizon):
                st = states[cur]
                if st.terminal:
                    break
                ops = self._decide_actions(st, rng, ctx, player, _r)
                token = tuple(sorted(ops)) if ops else ()
                key = (cur, token)
                nxt = edges.get(key)
                if nxt is None:
                    sim = states[cur].clone()
                    for (_pl, _t, _a0, _a1) in ops:
                        try:
                            sim.apply_operation_list(int(_pl),
                                                     [Operation(OperationType(int(_t)), int(_a0), int(_a1))])
                        except Exception:
                            pass
                    sim.advance_round()
                    nxt = next_id
                    next_id += 1
                    states[nxt] = sim
                    edges[key] = nxt
                cur = nxt
            leaf_values.append(float(self._leaf_values([states[cur]], player)[0]))
        self.last_tree_nodes = next_id
        return float(np.mean(leaf_values)) if leaf_values else 0.0

'''
t = t[:start] + new_block + t[end:]

P.write_text(t, encoding="utf-8")
print("已加入 _decide_actions / _apply 路径 / _tree_rollout_value")
