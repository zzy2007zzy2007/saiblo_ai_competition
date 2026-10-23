"""**扁平 rollout 搜索**：候选来自动作网（只算一次），评估靠「真实引擎 rollout + 叶值网」。

## 为什么做这个（对照对象）
现有 M6c = `BundleMCTS`（depth 4、256 iterations）。实测每决策要 **257 次 `net_fn`**，
占搜索时间 **81.8%**（2.45 s 里的 1.95 s）。而亚军是**扁平 UCB bandit**：
根节点一次动作网、每个候选跑固定 h 回合 rollout，网络开销 O(1)。

本类就是"亚军的结构 + 我们的零件"：
```
根：一次 net_fn（走 BundleMCTS._expand，得到 k 个候选 bundle，和 M6c 完全相同的候选集）
每个候选：克隆 → 跑 N 次「h 回合真实引擎 rollout」→ 叶子用值网评估 → 取均值
决策：argmax（并列/退化时回落到先验 argmax）
```

## 与 M6c 的唯一差别
**只换评估器**：树搜索（跨迭代复用信息）→ 扁平 N×h 采样（独立、无复用）。
候选生成、类判定、值网、特征提取全部不变 ⇒ 这是干净的"树 vs 扁平等预算"消融。

## 时间预算自调节
`max_total_rounds`（默认 39_000）≈ 2.45 s（完整引擎 62.7 µs/回合）。
候选多时会自动把每候选的采样数 N 压下来，保证**每决策总回合数不超预算** ⇒
和 M6c 做等时间对照才公平。

## 用法
    from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
    searcher = FlatRolloutGreedy(net_fn, k=24, rollouts_per_candidate=32, horizon=64,
                                 value_model=vmodel, feature_extractor=feat)
    res = searcher.search(state, player, temperature=0.0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for _p in (_REPO, _CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from my_ai.az_intent.bundle_mcts import BundleMCTS, BundleNode, BundleSearchResult  # noqa: E402

# 完整引擎 pybind 实测：advance_round ≈ 62.7 µs ⇒ 39_000 回合 ≈ 2.45 s
DEFAULT_MAX_TOTAL_ROUNDS = 39_000


class FlatRolloutGreedy(BundleMCTS):
    """一次扩张出候选 + 每候选 N 次 h 回合 rollout 打分（无树）。"""

    def __init__(self, *args,
                 rollouts_per_candidate: int = 32,
                 horizon: int = 64,
                 value_model=None,
                 feature_extractor=None,
                 value_batch: int = 256,
                 value_tanh: bool = True,
                 eval_mode: str = "net",
                 # 2026-10-23（用户指出）：rollout 里双方**必须真的出招**，否则
                 # `advance_round()` 是确定性的 ⇒ N 次采样得到同一个叶子，采样纯浪费
                 # （实测 samples=1/16/64 的分数逐位相同，耗时 177/919/3548 ms）。
                 # 亚军的 `simulate_round(rng)` 就是带随机性的；这里用"随机出招"近似。
                 rollout_mode: str = "idle",        # idle=现状(双方不出招) / random=随机出招
                 rollout_random_side: str = "opponent",  # opponent=只随机对手(推荐) / both=双方
                 # 2026-10-23：随机概率随回合衰减（避免整段 churn 污染叶子状态）
                 rollout_prob_mode: str = "const",   # const | decay | first
                 rollout_decay_c: float = 0.6,        # 衰减模式的首回合概率 C
                 rollout_decay_k: float = 3.0,        # 衰减常数 k：p(r)=C*exp(-r/k)
                 # 2026-10-23（用户论点）：随机 rollout 要大 N 才管用 ⇒ 必须做**前缀树记忆化**。
                 #   实测共享率 const 27% / decay(.6,3) 43% / decay(.3,1) 85% / first 94%。
                 rollout_share_prefix: bool = False,
                 rollout_act_prob: float = 0.5,     # 每回合每方出招概率
                 rollout_build_ratio: float = 0.7,  # 出招时选"建塔"的比例（否则降级）
                 rollout_seed: int = 0,
                 max_total_rounds: int = DEFAULT_MAX_TOTAL_ROUNDS,
                 verbose: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert eval_mode in ("net", "hand"), eval_mode
        self.rollouts_per_candidate = int(rollouts_per_candidate)
        self.horizon = int(horizon)
        self.value_model = value_model
        self.feature_extractor = feature_extractor
        self.value_batch = int(value_batch)
        self.value_tanh = bool(value_tanh)
        self.eval_mode = eval_mode
        assert rollout_mode in ("idle", "random"), rollout_mode
        self.rollout_mode = rollout_mode
        self.rollout_act_prob = float(rollout_act_prob)
        self.rollout_build_ratio = float(rollout_build_ratio)
        self.rollout_seed = int(rollout_seed)
        assert rollout_random_side in ("both", "opponent"), rollout_random_side
        self.rollout_random_side = rollout_random_side
        assert rollout_prob_mode in ("const", "decay", "first"), rollout_prob_mode
        self.rollout_prob_mode = rollout_prob_mode
        self.rollout_decay_c = float(rollout_decay_c)
        self.rollout_decay_k = max(1e-6, float(rollout_decay_k))
        self.rollout_share_prefix = bool(rollout_share_prefix)
        self.max_total_rounds = int(max_total_rounds)
        self.verbose = bool(verbose)
        self.last_scores = None
        self.last_timing = {}
        self.last_rollouts = 0
        self.last_rounds = 0
        if eval_mode == "net" and value_model is None:
            raise ValueError("eval_mode='net' 需要 value_model（load_three_models 的第三个返回值）")
        if eval_mode == "net" and feature_extractor is None:
            raise ValueError("eval_mode='net' 需要 feature_extractor（SDK.utils.features.FeatureExtractor）")

    def _decide_actions(self, sim, rng, ctx, root_player=None, round_idx: int = 0) -> list:
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
        leaf_states = []

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
            leaf_states.append(states[cur])
        self.last_tree_nodes = getattr(self, "last_tree_nodes", 0) + next_id
        if not leaf_states:
            return 0.0
        _lv = self._leaf_values(leaf_states, player)     # ← 批量评估（与朴素路径一致）
        return float(np.mean(_lv))

    # ── 叶子评估 ────────────────────────────────────────────────────────────
    def _leaf_values(self, states: list, root_player: int) -> np.ndarray:
        """给一批叶子状态打分（从 root_player 视角，量纲 ≈ [-1, 1]）。"""
        values = np.zeros(len(states), dtype=np.float64)
        pending, pending_index = [], []
        for index, state in enumerate(states):
            if state.terminal:
                values[index] = self._terminal_value(state, root_player)
            else:
                pending.append(state)
                pending_index.append(index)
        if not pending:
            return values

        if self.eval_mode == "hand":
            from my_ai.distill_top2.lure_value_eval import evaluate_terminal
            for index, state in zip(pending_index, pending):
                raw = float(evaluate_terminal(state, root_player))
                # 手写价值量纲 ~1e4，线性归一到 [-1,1]（与值网同尺度，便于共用阈值）
                values[index] = float(np.clip((raw - 10_000.0) / 2_000.0, -1.0, 1.0))
            return values

        import torch
        zeros = np.zeros(96)
        for start in range(0, len(pending), self.value_batch):
            chunk = pending[start:start + self.value_batch]
            boards, stats = [], []
            for state in chunk:
                obs = self.feature_extractor.encode_observation(state, root_player, zeros)
                boards.append(obs["board"])
                stats.append(obs["stats"])
            board = torch.from_numpy(np.stack(boards)).float()
            stat = torch.from_numpy(np.stack(stats)).float()
            if next(self.value_model.parameters()).is_cuda:
                board, stat = board.cuda(), stat.cuda()
            with torch.no_grad():
                out = float("nan")
                raw = self.value_model(board, stat)["value"].squeeze(-1)
            if self.value_tanh:
                raw = torch.tanh(raw)
            for offset, value in enumerate(raw.detach().cpu().numpy().tolist()):
                values[pending_index[start + offset]] = float(value)
        return values

    # ── 主搜索 ──────────────────────────────────────────────────────────────
    def search(self, state, player: int, *, temperature: float = 0.0) -> BundleSearchResult:
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType

        import time as _time
        _t_expand = _time.perf_counter()
        root = BundleNode(state=state.clone(), player=player)
        self.last_root = root
        # ⚠️ 必须和 BundleMCTS.search 一样挂上 `_mc_root`：
        #    `_expand` 里的 MC 类仲裁用 `_is_root = (self._mc_root is not None and node is self._mc_root)`
        #    做门控（`mc_root_only=1` 时只在根仲裁）。不设它 ⇒ 仲裁被静默跳过 ⇒
        #    两侧就不是"只换 mcts"了（实测这是 0.7344 基线口径的核心一环）。
        self._mc_root = root
        # 2026-10-23（用户要求）：随机出招**必须可复现**。
        #   种子只由 (rollout_seed, 决策序号, 候选序号, 采样序号) 决定，不依赖时间/哈希随机化：
        #   同一局、同一配置重跑 ⇒ 完全相同的对局。
        self._decision_idx = int(getattr(self, "_decision_idx", 0)) + 1
        self._expand(root, self.k)                 # ← 每决策唯一一次 net_fn（含根节点的 MC 类仲裁）
        bundles = list(root.bundles)
        if not bundles:
            self.last_scores = np.zeros(0)
            self.last_rollouts = 0
            self.last_rounds = 0
            return BundleSearchResult(bundles=[], visit_policy=np.array([], dtype=np.float32),
                                      chosen_index=0, chosen_bundle=(),
                                      root_value=float(root.net_value), intent_counts=[])

        _t_after_expand = _time.perf_counter()
        priors = np.asarray([c.prior for c in root.children], dtype=np.float64)

        # 2026-10-23：随机出招 rollout 的上下文（**只在这里算一次**，rollout 内复用）
        _rctx = None
        if self.rollout_mode == "random":
            try:
                from SDK.utils.constants import HIGHLAND_CELLS as _HC
                _rctx = {"cells": {0: list(_HC[0]), 1: list(_HC[1])}}
            except Exception:
                _rctx = None

        # 候选的子节点（与 BundleMCTS._expand 的约定一致：P1 的父节点要推进一回合）
        children = []
        for key in bundles:
            ops = [Operation(OperationType(int(x[0])), int(x[1]), int(x[2])) for x in key]
            child = state.clone()
            if ops:
                child.apply_operation_list(player, ops)
            if player == 1:
                child.advance_round()
            children.append(child)

        # 预算自调节：总回合数 = 候选数 × N × h ≤ max_total_rounds
        n_cand = len(children)
        per_candidate_rounds = max(1, self.max_total_rounds // max(1, n_cand * max(1, self.horizon)))
        n_rollouts = int(min(self.rollouts_per_candidate, max(1, per_candidate_rounds)))
        if n_rollouts < 1:
            n_rollouts = 1

        _t_rollout = _time.perf_counter()
        scores = np.zeros(n_cand, dtype=np.float64)
        total_rounds = 0
        for index, child in enumerate(children):
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
            for _k in range(n_rollouts):
                sim = child.clone()
                _rng = None
                if _rctx is not None:
                    # 种子 = f(rollout_seed, 决策序号, 候选序号, 采样序号) ⇒ 完全可复现
                    _rng = np.random.default_rng(
                        (int(self.rollout_seed) * 1000003 + self._decision_idx) * 100003
                        + int(index) * 101 + int(_k))
                _round_idx = 0
                for _ in range(self.horizon):
                    if sim.terminal:
                        break
                    if _rng is not None:
                        self._random_actions(sim, _rng, _rctx, player, _round_idx)
                    sim.advance_round()
                    _round_idx += 1
                    total_rounds += 1
                leaves.append(sim)
            scores[index] = float(self._leaf_values(leaves, player).mean())

        _t_after_rollout = _time.perf_counter()
        self.last_timing = {"expand_s": _t_after_expand - _t_expand,
                            "rollout_s": _t_after_rollout - _t_rollout}
        self.last_scores = scores
        self.last_rollouts = n_rollouts
        self.last_rounds = total_rounds
        if self.verbose:
            print(f"[flat] 候选 {n_cand} 个 × {n_rollouts} 次 × {self.horizon} 回合 "
                  f"= {total_rounds} 回合（约 {total_rounds * 62.7 / 1e6:.2f} s）", flush=True)

        # 退化保护：全 NaN / 全相等时退回先验（沿用 Depth0Greedy 的做法）
        if not np.isfinite(scores).any() or np.nanmax(scores) - np.nanmin(scores) < 1e-9:
            best = int(np.argmax(priors)) if priors.size else 0
        else:
            best = int(np.nanargmax(np.where(np.isfinite(scores), scores, -np.inf)))

        policy = np.zeros(n_cand, dtype=np.float32)
        policy[best] = 1.0
        return BundleSearchResult(
            bundles=bundles,
            visit_policy=policy,
            chosen_index=best,
            chosen_bundle=bundles[best],
            root_value=float(scores[best]),
            intent_counts=list(root.intent_counts),
        )
