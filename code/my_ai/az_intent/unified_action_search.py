"""统一动作搜索：把「MC 类仲裁」和「位置 rollout」合并成一个 bandit 过程。

## 与现有两条路线的关系
- `BundleMCTS`（M6c）：根节点一次 net_fn → MC 仲裁选**类** → 树/扁平搜索只在该类内选**位置**。
  缺点：类比较只用"该类位置图 argmax 那一个格"当代表；且**升级动作几乎从不进入候选**
  （实测对亚军 1,171 个 op 里升级 = 0，对手 9.0%）。
- 本类：**直接把原子动作枚举出来**（建塔/升级/降级/闪电/HOLD），每个动作用 rollout 打分，
  类分数 = 同类动作的**边际均值**（这就是用户 2026-10-21 提的"两份 rollout 融合"的落地）。

## 过程
1. 一次 `net_fn` 拿 `action_map`（当各动作的**先验**，用于排序/预算分配）。
2. **阶段一（探索）**：枚举全部合法原子动作，每个动作跑 1 次 rollout（h 回合）→ 得到每个动作的
   1 个样本；同时按"类"聚合出类分数。
3. **阶段二（复评）**：把剩余预算加在**当前 top-k 动作**上（复用阶段一的样本，不丢弃）。
4. argmax（并列/退化时回落到先验）。

单次 rollout 的噪声 σ≈110、真实分差中位仅 33 ⇒ 阶段二的复评是**必须**的
（否则单样本 argmax 选错率约 42%）；阶段一负责"广度"（我们实测覆盖率是主瓶颈）。

## 相对现状的关键改善
**升级动作第一次会被真正评估**：`TOWER_UPGRADE_TREE` 里每个可行的一步都单独成为一个候选，
不依赖类头的先验大小，也不依赖 decoder"在已有塔格输出降级"的旧行为。

用法：
    from my_ai.az_intent.unified_action_search import UnifiedActionSearch
    s = UnifiedActionSearch(net_fn, horizon=32, value_model=vm, feature_extractor=feat)
    res = s.search(state, player, temperature=0.0)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for _p in (_REPO, _CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from my_ai.az_intent.bundle_mcts import BundleNode, BundleSearchResult  # noqa: E402
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy  # noqa: E402

DEFAULT_TOTAL_ROUNDS = 39_000  # ≈2.45 s（完整引擎 62.7 µs/回合）


class UnifiedActionSearch(FlatRolloutGreedy):
    """原子动作枚举 + 逐动作 rollout 打分 + top-k 复评（无树、无两阶段类仲裁）。"""

    def __init__(self, *args,
                 horizon: int = 32,
                 candidate_cap: int = 256,
                 reeval_topk: int = 8,
                 total_rounds: int = DEFAULT_TOTAL_ROUNDS,
                 min_samples_per_candidate: int = 30,
                 max_lightning: int = 32,
                 max_other_weapon: int = 8,
                 verbose: bool = False,
                 **kwargs):
        # 复用 FlatRolloutGreedy 的叶子评估（_leaf_values / _terminal_value）
        super().__init__(*args, horizon=horizon, rollouts_per_candidate=1,
                         max_total_rounds=total_rounds, verbose=verbose, **kwargs)
        self.horizon = int(horizon)
        self.candidate_cap = int(candidate_cap)
        self.reeval_topk = int(reeval_topk)
        self.total_rounds = int(total_rounds)
        self.min_samples = int(min_samples_per_candidate)
        self.max_lightning = int(max_lightning)
        self.max_other_weapon = int(max_other_weapon)
        self.verbose = bool(verbose)
        self.last_actions = []
        self.last_class_scores = {}

    # ── 动作枚举 ────────────────────────────────────────────────────────────
    def enumerate_actions(self, state, player: int, action_map, class_mask, pos_masks):
        from SDK.backend.model import Operation
        from SDK.utils.constants import HIGHLAND_CELLS, TOWER_UPGRADE_TREE, OperationType

        AM = np.asarray(action_map, dtype=np.float32)
        actions = []          # (ops_tuple, prior, tag)
        tower_at = {(t.x, t.y): t for t in state.towers_of(player)}

        def legal(op) -> bool:
            try:
                return bool(state.can_apply_operation(player, op))
            except Exception:
                return False

        # ① 建塔：每个高地空格（先验 = 类 0「Basic」的位置图）
        build_cost = None
        try:
            build_cost = state.build_tower_cost(state.tower_count(player))
        except Exception:
            pass
        for (x, y) in HIGHLAND_CELLS[player]:
            if (x, y) in tower_at:
                continue
            if build_cost is not None and state.coins[player] < build_cost:
                continue
            op = Operation(OperationType.BUILD_TOWER, int(x), int(y))
            if legal(op):
                actions.append(((int(op.op_type), int(x), int(y)), float(AM[0, y, x]), "build"))

        # ② 升级：我方每座塔的每个可行一步（先验 = 该类意图在塔格上的位置图值取最大）
        for tower in state.towers_of(player):
            steps = TOWER_UPGRADE_TREE.get(tower.tower_type, ())
            for step in steps:
                op = Operation(OperationType.UPGRADE_TOWER, int(tower.tower_id), int(step))
                if not legal(op):
                    continue
                prior = 0.0
                for ch in range(16):
                    try:
                        if self._channel_target(ch) == step:
                            prior = max(prior, float(AM[ch, tower.y, tower.x]))
                    except Exception:
                        continue
                actions.append(((int(op.op_type), int(tower.tower_id), int(step)),
                                prior, "upgrade"))

        # ③ 降级：我方每座塔
        for tower in state.towers_of(player):
            op = Operation(OperationType.DOWNGRADE_TOWER, int(tower.tower_id), -1)
            if legal(op):
                actions.append(((int(op.op_type), int(tower.tower_id), -1),
                                float(AM[16, tower.y, tower.x]), "downgrade"))

        # ④ 闪电：冷却为 0 时枚举合法中心（先验 = 类 17 的位置图）
        #    ⚠️ 合法中心很多（实测 235 个），若全放进来会挤掉建塔/升级 ⇒ 按先验取前 N 个
        lightning = []
        for x in range(19):
            for y in range(19):
                op = Operation(OperationType.USE_LIGHTNING_STORM, x, y)
                if legal(op):
                    lightning.append(((int(op.op_type), int(x), int(y)),
                                      float(AM[17, y, x]), "lightning"))
        lightning.sort(key=lambda a: -a[1])
        actions.extend(lightning[: int(self.max_lightning)])

        # ④b 其余超级武器：EMP(22) / Deflector(23) / Evasion(24)
        #     ⚠️ 2026-10-21 补：之前只枚举了闪电，漏了这三种 ⇒ "列出所有动作"并不成立。
        #     它们的位置同样几乎任意格合法 ⇒ 每种按先验取前 `max_other_weapon` 个。
        for op_type_value, channel, tag in (
                (22, 18, "emp"), (23, 19, "deflector"), (24, 20, "evasion")):
            picked = []
            for x in range(19):
                for y in range(19):
                    op = Operation(OperationType(int(op_type_value)), x, y)
                    if legal(op):
                        prior = float(AM[channel, y, x]) if AM.shape[0] > channel else -1e3
                        picked.append(((int(op.op_type), int(x), int(y)), prior, tag))
            picked.sort(key=lambda a: -a[1])
            actions.extend(picked[: int(self.max_other_weapon)])

        # ④c 基地升级：出蚁速度(31) / 出蚁等级(32) —— 无位置，各一个候选
        for op_type_value, channel, tag in ((31, 21, "up_speed"), (32, 22, "up_ant")):
            op = Operation(OperationType(int(op_type_value)))
            if legal(op):
                prior = float(np.max(AM[channel])) if AM.shape[0] > channel else -1e3
                actions.append(((int(op.op_type), int(op.arg0), int(op.arg1)), prior, tag))

        # ⑤ HOLD
        actions.append(((-1, -1, -1), float(AM[23].max()) if AM.shape[0] > 23 else -1e3, "hold"))
        return actions

    @staticmethod
    def _channel_target(channel: int):
        try:
            from my_ai.decoder import CHANNEL_TO_TOWER_TYPE
            return CHANNEL_TO_TOWER_TYPE.get(int(channel))
        except Exception:
            return None

    # ── 主搜索 ──────────────────────────────────────────────────────────────
    def search(self, state, player: int, *, temperature: float = 0.0) -> BundleSearchResult:
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType

        t_start = time.perf_counter()
        net_out = self.net_fn(state, player)           # ← 每决策唯一一次前向
        actions = self.enumerate_actions(state, player, net_out["action_map"], None, None)
        if not actions:
            return BundleSearchResult(bundles=[], visit_policy=np.array([], dtype=np.float32),
                                      chosen_index=0, chosen_bundle=(),
                                      root_value=0.0, intent_counts=[])
        # 先按先验降序（先验只用于**顺序**与截断，不参与最终打分）
        actions.sort(key=lambda a: -a[1])
        actions = actions[: max(1, self.candidate_cap)]
        n = len(actions)

        per_round = self.horizon
        budget_rounds = self.total_rounds
        n_phase1 = min(n, max(1, budget_rounds // max(1, per_round)))
        samples = [[] for _ in range(n)]

        def rollout_state(index: int):
            """打一次动作 + 推进 h 回合，返回**叶子状态**（不评估；评估必须批量化）。"""
            ops = actions[index][0]
            sim = state.clone()
            if ops[0] >= 0:
                sim.apply_operation_list(player, [Operation(OperationType(int(ops[0])),
                                                           int(ops[1]), int(ops[2]))])
            if player == 1:
                sim.advance_round()
            for _ in range(self.horizon):
                if sim.terminal:
                    break
                sim.advance_round()
            return sim

        def evaluate(indices) -> None:
            """批量评估叶子并写回 samples。
            ⚠️ 必须批量：单条 `encode_observation` ≈ 414 µs，逐条会让每回合成本从 63 µs 涨到 ~500 µs。"""
            if not indices:
                return
            leaves = [rollout_state(int(i)) for i in indices]
            values = self._leaf_values(leaves, player)
            for i, value in zip(indices, values):
                samples[int(i)].append(float(value))

        # 阶段一：每个动作 1 次（64 个一批评估）
        phase1 = list(range(n_phase1))
        for start in range(0, len(phase1), 64):
            evaluate(phase1[start:start + 64])
        rounds_used = n_phase1 * per_round

        # 阶段二：剩余预算加在 top-k 上（复用已有样本）
        means = np.array([np.mean(s) if s else -np.inf for s in samples], dtype=np.float64)
        order = np.argsort(-means)[: max(1, self.reeval_topk)]
        remaining = max(0, budget_rounds - rounds_used)
        per_cand = remaining // max(1, len(order) * per_round) if order.size else 0
        per_cand = int(min(per_cand, max(0, self.min_samples - 1)))
        for _ in range(per_cand):
            batch = []
            for i in order:
                if rounds_used + per_round > budget_rounds:
                    break
                batch.append(int(i))
                rounds_used += per_round
            if not batch:
                break
            evaluate(batch)

        means = np.array([np.mean(s) if s else -np.inf for s in samples], dtype=np.float64)
        best = int(np.nanargmax(np.where(np.isfinite(means), means, -np.inf)))

        # 类分数（边际均值）——只做观测/诊断
        by_tag = {}
        for i, (_, _, tag) in enumerate(actions):
            by_tag.setdefault(tag, []).append(means[i])
        self_last = {k: float(np.mean(v)) for k, v in by_tag.items()}
        self.last_actions = [(actions[i][0], actions[i][2], float(means[i]),
                              len(samples[i])) for i in range(n)]
        self.last_class_scores = self_last
        self.last_rounds = rounds_used
        self.last_timing = {"expand_s": 0.0, "rollout_s": time.perf_counter() - t_start}

        chosen = actions[best][0]
        bundle = () if chosen[0] < 0 else (chosen,)
        policy = np.zeros(n, dtype=np.float32)
        policy[best] = 1.0
        if self.verbose:
            tag = actions[best][2]
            print(f"[unified] 动作 {n} 个，阶段一 {n_phase1}，复评 top{len(order)}×{per_cand}，"
                  f"共 {rounds_used} 回合，选中 {tag} {bundle} "
                  f"（分数 {means[best]:+.3f}，样本 {len(samples[best])}）", flush=True)
        return BundleSearchResult(
            bundles=[bundle], visit_policy=policy, chosen_index=best,
            chosen_bundle=bundle, root_value=float(means[best]), intent_counts=[])
