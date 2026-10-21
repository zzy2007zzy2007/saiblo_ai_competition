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
        self._expand(root, self.k)                 # ← 每决策唯一一次 net_fn
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
            leaves = []
            for _ in range(n_rollouts):
                sim = child.clone()
                for _ in range(self.horizon):
                    if sim.terminal:
                        break
                    sim.advance_round()
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
