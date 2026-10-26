"""统一动作搜索：把「MC 类仲裁」和「位置 rollout」合并成一个 bandit 过程。

## 2026-10-21 深夜修订（一个致命 bug + 一个设计错误）

**Bug（致命）：空间轴序搞反了。**
`make_position_masks` 与网络 `action_map` 都是 **`[x][y]`** 顺序
（`decoder.decode` 里就是 `x, y = np.unravel_index(argmax, ...)`）。
上一版写的是 `AM[ch, y, x]` / `masks[ch, y, x]` ⇒ 全部转置，
导致"33 个可建格只枚举出 4 个建塔"，升级候选也大量漏掉。
⇒ 本版统一改为 `[ch, x, y]`（`np.where` 返回的也是 [x, y]）。

**设计错误：跨通道比较位置图分数（用户 2026-10-21 指出）。**
上一版把所有动作按各自通道的分数做**全局排序**；但不同通道的位置图语义/尺度不可比。
⇒ 改为三层分工：
   · **通道间**（给哪个类分多少候选预算）→ **类网 head_logits 的 softmax**（3 头取平均）
   · **通道内**（同类里先测哪个位置）→ 该类自己的位置图
   · **最终打分**（跨类可比）→ rollout 均值（值网输出，同一量纲）

## 过程
1. 一次 `net_fn` 拿 `action_map` + `head_logits`；用 decoder 掩码定合法动作（要求真买得起）。
2. 按通道分组枚举：建塔(0) / 升级(目标类型对应通道) / 降级(16) / 超级武器(17-20) / 基地升级(21-22) / HOLD(23)。
3. 类网 softmax ⇒ 每通道预算；通道内按位置图取前 N。
4. 阶段一：每个候选 1 次 rollout（h 回合）；阶段二：剩余预算加在当前 top-k 的已有样本上。
5. argmax（rollout 均值）。
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

from my_ai.az_intent.bundle_mcts import BundleSearchResult  # noqa: E402
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy  # noqa: E402

DEFAULT_TOTAL_ROUNDS = 39_000  # ≈2.45 s（完整引擎 62.7 µs/回合）

CHANNEL_TAG = {16: "downgrade", 17: "lightning", 18: "emp", 19: "deflector",
               20: "evasion", 21: "up_speed", 22: "up_ant", 23: "hold",
               100: "oracle"}   # 100 = 额外注入候选（亚军出招，见 docs/oracle_inject_design.md）


class UnifiedActionSearch(FlatRolloutGreedy):
    """原子动作枚举 + 逐动作 rollout + top-k 复评；类网只负责**跨通道预算分配**。"""

    def __init__(self, *args,
                 horizon: int = 32,
                 candidate_cap: int = 256,
                 reeval_topk: int = 8,
                 total_rounds: int = DEFAULT_TOTAL_ROUNDS,
                 min_samples_per_candidate: int = 30,
                 max_lightning: int = 32,
                 max_other_weapon: int = 8,
                 min_per_class: int = 2,
                 class_prior_temp: float = 1.0,
                 verbose: bool = False,
                 **kwargs):
        super().__init__(*args, horizon=horizon, rollouts_per_candidate=1,
                         max_total_rounds=total_rounds, verbose=verbose, **kwargs)
        self.horizon = int(horizon)
        self.candidate_cap = int(candidate_cap)
        self.reeval_topk = int(reeval_topk)
        self.total_rounds = int(total_rounds)
        self.min_samples = int(min_samples_per_candidate)
        self.max_lightning = int(max_lightning)
        self.max_other_weapon = int(max_other_weapon)
        self.min_per_class = int(min_per_class)
        self.class_prior_temp = float(class_prior_temp)
        self.verbose = bool(verbose)
        self.last_actions = []
        self.last_class_scores = {}
        self.last_class_probs = {}
        self._t_rollout = 0.0
        self._t_value = 0.0

    # ── ① 枚举：按通道分组，通道内按该类位置图降序 ──────────────────────────
    def enumerate_by_channel(self, state, player: int, action_map, class_mask, pos_masks):
        """返回 {channel: [(ops_tuple, within_prior), ...]}（组内已降序）。空间索引一律 [x, y]。"""
        from SDK.utils.constants import HIGHLAND_CELLS, TOWER_UPGRADE_TREE, OperationType
        from my_ai.decoder import CHANNEL_TO_TOWER_TYPE

        AM = np.asarray(action_map, dtype=np.float32)
        masks = np.asarray(pos_masks)
        cmask = np.asarray(class_mask)
        groups: dict[int, list] = {}
        tower_at = {(t.x, t.y): t for t in state.towers_of(player)}

        def put(channel: int, ops: tuple, prior: float) -> None:
            groups.setdefault(int(channel), []).append((ops, float(prior)))

        # 建塔 → 通道 0
        for (x, y) in HIGHLAND_CELLS[player]:
            if (x, y) in tower_at or not bool(masks[0, x, y]):
                continue
            put(0, (int(OperationType.BUILD_TOWER), int(x), int(y)), float(AM[0, x, y]))

        # 升级 → 通道 = 目标类型对应的意图类（在同一格上取最大者）
        for tower in state.towers_of(player):
            for step in TOWER_UPGRADE_TREE.get(tower.tower_type, ()):
                best_ch, best_v = None, -np.inf
                for ch in range(16):
                    if CHANNEL_TO_TOWER_TYPE.get(ch) != step:
                        continue
                    if not bool(masks[ch, tower.x, tower.y]):
                        continue
                    v = float(AM[ch, tower.x, tower.y])
                    if v > best_v:
                        best_ch, best_v = ch, v
                if best_ch is not None:
                    put(best_ch, (int(OperationType.UPGRADE_TOWER),
                                  int(tower.tower_id), int(step)), best_v)

        # 降级 → 通道 16
        for tower in state.towers_of(player):
            if bool(masks[16, tower.x, tower.y]):
                put(16, (int(OperationType.DOWNGRADE_TOWER), int(tower.tower_id), -1),
                    float(AM[16, tower.x, tower.y]))

        # 超级武器 → 通道 17/18/19/20（np.where 返回 [x, y]）
        op_of = {17: OperationType.USE_LIGHTNING_STORM, 18: OperationType.USE_EMP_BLASTER,
                 19: OperationType.USE_DEFLECTOR, 20: OperationType.USE_EMERGENCY_EVASION}
        for channel in (17, 18, 19, 20):
            if not bool(cmask[channel]):
                continue
            xs, ys = np.where(masks[channel])
            for x, y in zip(xs, ys):
                put(channel, (int(op_of[channel]), int(x), int(y)),
                    float(AM[channel, x, y]))

        # 基地升级 → 通道 21/22（无位置）
        for channel, op_type in ((21, 31), (22, 32)):
            if bool(cmask[channel]):
                put(channel, (int(op_type), 0, 0), float(np.max(AM[channel])))

        # HOLD → 通道 23
        put(23, (-1, -1, -1), float(np.max(AM[23])) if AM.shape[0] > 23 else 0.0)

        caps = {17: self.max_lightning, 18: self.max_other_weapon,
                19: self.max_other_weapon, 20: self.max_other_weapon}
        for channel in list(groups):
            groups[channel].sort(key=lambda kv: -kv[1])
            cap = caps.get(channel)
            if cap is not None:
                groups[channel] = groups[channel][: max(1, int(cap))]
        return groups

    # ── ② 通道间：用类网 softmax 分配候选预算 ───────────────────────────────
    def allocate(self, groups: dict, head_logits) -> list:
        channels = sorted(groups)
        if len(channels) == 1:
            return [(channels[0], ops) for ops, _ in groups[channels[0]]]

        stacked = np.mean(np.stack([np.asarray(h, dtype=np.float64) for h in head_logits[:3]]),
                          axis=0)
        legal = np.array(channels, dtype=int)
        masked = np.full_like(stacked, -np.inf)
        masked[legal] = stacked[legal] / max(1e-6, self.class_prior_temp)
        masked -= np.max(masked)
        probs = np.exp(masked)
        probs /= probs.sum()
        self.last_class_probs = {int(c): float(probs[c]) for c in channels}

        budget, spare = {}, self.candidate_cap
        for c in channels:
            n = max(self.min_per_class, int(round(self.candidate_cap * probs[c])))
            n = min(n, len(groups[c]))
            budget[c] = n
            spare -= n
        while spare > 0:
            progressed = False
            for c in channels:
                if spare <= 0:
                    break
                if budget[c] < len(groups[c]):
                    budget[c] += 1
                    spare -= 1
                    progressed = True
            if not progressed:
                break

        flat = []
        for c in channels:
            for ops, _ in groups[c][: budget[c]]:
                flat.append((c, ops))
        return flat

    # ── ③ 搜索 ──────────────────────────────────────────────────────────────
    def search(self, state, player: int, *, temperature: float = 0.0,
               extra_bundles=None) -> BundleSearchResult:
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType
        from my_ai.decoder import make_class_mask, make_position_masks

        t_start = time.perf_counter()
        self._t_rollout = self._t_value = 0.0
        net_out = self.net_fn(state, player)         # 每决策唯一一次前向
        masks = make_position_masks(state, player, intent_decoding=False)
        cmask = make_class_mask(state, player, position_mask=masks, intent_decoding=False)
        groups = self.enumerate_by_channel(state, player, net_out["action_map"], cmask, masks)
        flat = self.allocate(groups, net_out["head_logits"])
        # ── 额外候选注入（默认关；`None`/空 ⇒ 一个字节都不多看）─────────────────────
        #   内部候选是**单原子动作**，而一个 bundle 可以含多个 op ⇒ 统一先用 `bundle_of`
        #   表示（HOLD = `()`）。额外候选追加在末尾，既有候选的下标/预算都不变。
        n_net = len(flat)
        bundle_of = [((ops,) if int(ops[0]) >= 0 else ()) for _c, ops in flat]
        _extras_in = self.extra_bundles if extra_bundles is None else extra_bundles
        _extras = self._collect_extras(_extras_in, bundle_of)
        for _b in _extras:
            flat.append((100, _b[0] if _b else (-1, -1, -1)))
            bundle_of.append(_b)
        self.last_extra = len(_extras)
        self.last_chosen_is_extra = False
        if not flat:
            return BundleSearchResult(bundles=[], visit_policy=np.array([], dtype=np.float32),
                                      chosen_index=0, chosen_bundle=(),
                                      root_value=0.0, intent_counts=[])
        n = len(flat)
        actions = [ops for _, ops in flat]
        chans = [c for c, _ in flat]

        per_round = self.horizon
        budget_rounds = self.total_rounds
        n_phase1 = min(n, max(1, budget_rounds // max(1, per_round)))
        samples = [[] for _ in range(n)]

        def rollout_state(index: int):
            sim = state.clone()
            if bundle_of[index]:
                sim.apply_operation_list(
                    player, [Operation(OperationType(int(t[0])), int(t[1]), int(t[2]))
                             for t in bundle_of[index]])
            if player == 1:
                sim.advance_round()
            for _ in range(self.horizon):
                if sim.terminal:
                    break
                sim.advance_round()
            return sim

        def evaluate(indices) -> None:
            if not indices:
                return
            _t0 = time.perf_counter()
            leaves = [rollout_state(int(i)) for i in indices]
            _t1 = time.perf_counter()
            values = self._leaf_values(leaves, player)
            self._t_rollout += _t1 - _t0
            self._t_value += time.perf_counter() - _t1
            for i, value in zip(indices, values):
                samples[int(i)].append(float(value))

        evaluate(list(range(n_phase1)))
        rounds_used = n_phase1 * per_round

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

        by_chan = {}
        for i, c in enumerate(chans):
            by_chan.setdefault(c, []).append(means[i])
        self.last_class_scores = {int(c): float(np.mean(v)) for c, v in by_chan.items()}
        # 额外候选（channel 100）在 `last_actions` 里放**整个 bundle**（它可能含多个 op，
        # 只放第一个 op 会误导读数）；普通候选仍放单 op 三元组（既有消费者按 (ops, tag, mean, k) 读）。
        self.last_actions = [(bundle_of[i] if i >= n_net else actions[i],
                              CHANNEL_TAG.get(chans[i], f"ch{chans[i]}"),
                              float(means[i]), len(samples[i])) for i in range(n)]
        self.last_rounds = rounds_used
        self.last_timing = {"expand_s": 0.0, "rollout_s": self._t_rollout,
                            "value_s": self._t_value,
                            "total_s": time.perf_counter() - t_start}

        chosen = bundle_of[best]
        policy = np.zeros(n, dtype=np.float32)
        policy[best] = 1.0
        self.last_chosen_is_extra = bool(best >= n_net)
        if self.verbose:
            tag = self.last_actions[best][1]
            print(f"[unified] 候选 {n}（{len(groups)} 通道），复评 top{len(order)}×{per_cand}，"
                  f"共 {rounds_used} 回合，选中 {tag} {chosen} "
                  f"（{means[best]:+.3f}，样本 {len(samples[best])}）", flush=True)
        return BundleSearchResult(
            bundles=[chosen], visit_policy=policy, chosen_index=best,
            chosen_bundle=chosen, root_value=float(means[best]), intent_counts=[])
