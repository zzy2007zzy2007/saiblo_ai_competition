"""给 flat rollout 加「随机出招」（2026-10-23，用户指出：rollout 里双方必须真的出招）。

背景：`advance_round()` 是确定性的（双方不出招）⇒ N 次采样得到同一个叶子，
实测 samples=1/16/64 的分数逐位相同、耗时 177/919/3548 ms ⇒ 采样纯浪费。
亚军的 `simulate_round(rng)` 是带随机性的。

本脚本把三处改动写进 flat_rollout_greedy.py：
  1. search() 里准备 rollout 上下文（候选格列表，**只算一次**）
  2. rollout 循环：每样本独立 rng + 每回合调用 _random_actions()
  3. 新增 _random_actions()：双方各以 p 概率随机建塔/降级
"""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "my_ai/az_intent/flat_rollout_greedy.py"
t = P.read_text(encoding="utf-8")

# ── ① ctx ────────────────────────────────────────────────────────────────
old1 = """        _t_after_expand = _time.perf_counter()
        priors = np.asarray([c.prior for c in root.children], dtype=np.float64)"""
new1 = """        _t_after_expand = _time.perf_counter()
        priors = np.asarray([c.prior for c in root.children], dtype=np.float64)

        # 2026-10-23：随机出招 rollout 的上下文（**只在这里算一次**，rollout 内复用）
        _rctx = None
        if self.rollout_mode == "random":
            try:
                from SDK.utils.constants import HIGHLAND_CELLS as _HC
                _rctx = {"cells": {0: list(_HC[0]), 1: list(_HC[1])}}
            except Exception:
                _rctx = None"""
assert old1 in t, "① 锚点未命中"
t = t.replace(old1, new1, 1)

# ── ② rollout 循环 ──────────────────────────────────────────────────────
old2 = """            for _ in range(n_rollouts):
                sim = child.clone()
                for _ in range(self.horizon):
                    if sim.terminal:
                        break
                    sim.advance_round()
                    total_rounds += 1
                leaves.append(sim)"""
new2 = """            for _k in range(n_rollouts):
                sim = child.clone()
                _rng = None
                if _rctx is not None:
                    _rng = np.random.default_rng(self.rollout_seed + index * 100003 + _k)
                for _ in range(self.horizon):
                    if sim.terminal:
                        break
                    if _rng is not None:
                        self._random_actions(sim, _rng, _rctx)
                    sim.advance_round()
                    total_rounds += 1
                leaves.append(sim)"""
assert old2 in t, "② 锚点未命中"
t = t.replace(old2, new2, 1)

# ── ③ 新方法 ────────────────────────────────────────────────────────────
old3 = "    # ── 叶子评估 ────────────────────────────────────────────────────────────"
new3 = '''    def _random_actions(self, sim, rng, ctx) -> None:
        """给双方各做一次**随机**出招（2026-10-23）。

        为什么：`advance_round()` 是确定性的（双方都不出招）⇒ N 次采样得到同一个叶子。
        亚军的 `simulate_round(rng)` 带随机性；这里用"每方以 p 概率随机建塔/降级"近似。
        成本控制：候选格列表在 search() 里算一次，这里只做廉价抽样 + apply。
        """
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType
        for _pl in (0, 1):
            if rng.random() >= self.rollout_act_prob:
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
                        try:
                            sim.apply_operation_list(
                                _pl, [Operation(OperationType.BUILD_TOWER, int(_x), int(_y))])
                            _did = True
                        except Exception:
                            pass
            if not _did:
                _tws = list(sim.towers_of(_pl))
                if _tws:
                    _tw = _tws[int(rng.integers(len(_tws)))]
                    try:
                        sim.apply_operation_list(
                            _pl, [Operation(OperationType.DOWNGRADE_TOWER, int(_tw.tower_id), -1)])
                    except Exception:
                        pass

    # ── 叶子评估 ────────────────────────────────────────────────────────────'''
assert old3 in t, "③ 锚点未命中"
t = t.replace(old3, new3, 1)

P.write_text(t, encoding="utf-8")
print("已写入 3 处改动：ctx / rollout 循环 / _random_actions")
