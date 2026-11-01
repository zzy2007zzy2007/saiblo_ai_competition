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

import os as _os
import numpy as np

_REC_SAMPLES = _os.environ.get("AZAI_ROLLOUT_SAMPLES", "0") == "1"

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for _p in (_REPO, _CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from my_ai.az_intent.bundle_mcts import BundleMCTS, BundleNode, BundleSearchResult  # noqa: E402
# 取值兼容：`AZAI_VALUE_CKPT` 的独立 `ValueNetwork` 返回裸张量，三网返回 `{"value": ...}`。
# 默认（环境变量为空）时没有任何额外模型被加载 ⇒ 行为逐位不变（见 value_override.py）。
from my_ai.az_intent.value_override import value_of  # noqa: E402

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
                 rollout_mode: str = "idle",        # idle=双方不出招 / random=对手随机建塔·降级 / lightning=我方随机放闪电
                 rollout_random_side: str = "opponent",  # opponent=只随机对手(推荐) / both=双方
                 # 2026-10-23：随机概率随回合衰减（避免整段 churn 污染叶子状态）
                 rollout_prob_mode: str = "const",   # const | decay | first
                 rollout_decay_c: float = 0.6,        # 衰减模式的首回合概率 C
                 rollout_decay_k: float = 3.0,        # 衰减常数 k：p(r)=C*exp(-r/k)
                 # 2026-10-23（用户论点）：随机 rollout 要大 N 才管用 ⇒ 必须做**前缀树记忆化**。
                 #   实测共享率 const 27% / decay(.6,3) 43% / decay(.3,1) 85% / first 94%。
                 rollout_share_prefix: bool = False,
                 # 2026-10-23：按轨迹去重加权（每条不同轨迹只算一次），避免大 N 被重复的"静止"轨迹稀释
                 rollout_dedupe_weight: bool = False,
                 rollout_act_prob: float = 0.5,     # 每回合每方出招概率
                 rollout_build_ratio: float = 0.7,  # 出招时选"建塔"的比例（否则降级）
                 rollout_seed: int = 0,
                 # ── 2026-10-27：**引擎 RNG 消费**这个正交随机源（开关式，默认关）──
                 #   动机：现在的唯一随机源是"对手出招随机"，decay(.3,1) 下 N=64 只有
                 #   17~20 条不同轨迹（大量分支整段没触发）。引擎自己在每次
                 #   `advance_round()` 里本来就要抽 12~17 次 48-bit LCG（蚂蚁走位 softmax
                 #   + 刷怪 + 传送 + 武器漂移），在 rollout 里**主动多抽**几次就能让分支
                 #   从第一回合起就分开，而且几乎不花时间（LCG 一次 ~2 ns，见
                 #   docs/engine_rng_impl.md）。
                 #   ⚠️ 不消费时（默认）代码路径与改动前逐位一致（`rollout_rng_perturb=False`）。
                 #   ⚠️ 引擎必须支持 `consume_rng` ⇒ 见 `my_ai/az_intent/rng_engine.py`
                 #   （`AZAI_FLAT_RNG_PERTURB=1` 时会在 __init__ 里自动 enable）。
                 rollout_rng_perturb: bool = None,   # None ⇒ 读 env AZAI_FLAT_RNG_PERTURB
                 # 每回合独立 Bernoulli(p(r)) 命中时消费几次（"位置"固定在第 r 回合
                 # ops 施加之后、advance_round() 之前；"次数"由这个参数给）
                 rollout_rng_consume: int = 1,
                 # 前缀树缓存键里是否带引擎 rng_state（"state"=正确、默认；
                 # "none" 只是**诊断用**：会错误合并 rng_state 不同的分支，
                 # 用来量化"把 rng_state 加进键到底多花多少节点"）
                 rollout_rng_key: str = "state",
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
        assert rollout_mode in ("idle", "random", "lightning"), rollout_mode
        self.rollout_mode = rollout_mode
        self.rollout_act_prob = float(rollout_act_prob)
        self.rollout_build_ratio = float(rollout_build_ratio)
        self.rollout_seed = int(rollout_seed)
        assert rollout_random_side in ("both", "opponent"), rollout_random_side
        self.rollout_random_side = rollout_random_side
        assert rollout_prob_mode in ("const", "decay", "first", "reverse"), rollout_prob_mode
        self.rollout_prob_mode = rollout_prob_mode
        self.rollout_decay_c = float(rollout_decay_c)
        self.rollout_decay_k = max(1e-6, float(rollout_decay_k))
        self.rollout_share_prefix = bool(rollout_share_prefix)
        self.rollout_dedupe_weight = bool(rollout_dedupe_weight)
        self.last_effective_samples = None
        self.max_total_rounds = int(max_total_rounds)
        self.verbose = bool(verbose)
        # ── 统一候选生成（2026-10-29，`AZAI_CAND_MODE`；**默认 "pin" ⇒ 下面两处钩子恒不执行**）──
        #   `cand_mode` 只在 `unified_cand.UnifiedCandFlatSearch` 里被显式设成 "unified"；
        #   `pin` 时它的**唯一**后果是 `search()` 里一个 `is not None` 判空（见 §"阶段 2 复用"），
        #   赋值路径（预算公式 / rollout / 叶值网 / 选择规则）逐字不变。
        self.cand_mode = "pin"
        self.cand_verbose = _os.environ.get("AZAI_CAND_VERBOSE", "0") == "1"
        self.last_cand_stats = {}
        self.last_reused = 0          # 阶段 1 样本被阶段 2 复用的条数（诊断）
        # ── 额外候选注入（2026-10-26，开关式）：见 docs/oracle_inject_design.md ──────────
        #   动机：把"亚军（cpp_lure_v4）在这个局面会怎么走"并进候选，由**我们自己的**
        #   rollout+值网 给它打分（不信任它的分数）。
        #   ⚠️ 默认 `None` ⇒ `search()` 一个字节都不多看 ⇒ 数值行为与改动前逐位一致
        #   （已由 `code/test_match/probe_oracle_inject_zero_impact.py` 的哈希比对验证）。
        self.extra_bundles = None      # list[tuple[(op_type, arg0, arg1), ...]]；HOLD = ()
        self.extra_prior = 0.0         # 额外候选的 prior（只影响"全 NaN/全相等"的退化回退分支）
        self.last_extra = 0            # 本次决策**真正追加**的额外候选数（去重后）
        self.last_chosen_is_extra = False   # 选中的是不是额外候选（注入命中率靠它统计）
        # 2026-10-28（E4；`AZAI_ONLY_EXTRA=1`，**默认 False = 改动前行为**）：
        #   候选集**只保留**额外塞进来的 bundle（网络那 K 条被类轴 pinned 的候选不进候选集）。
        #   动机：E4 要测"把冠军/亚军的动作原样当候选、由同一个评估器裁决"的上限
        #   ⇒ 除了候选集，别的（预算公式/rollout/叶值网/选择规则）一个都不改。
        #   由 `code/test_match/az_bridge_ai.py` 的 `set_only_extra(True)` 打开。
        self.only_extra = False
        self.last_scores = None
        self.last_samples = None   # list[list[float]]：每候选每分支的分数（开关打开时记录）
        self.last_timing = {}
        self.last_rollouts = 0
        self.last_rounds = 0
        # ── 前缀树 dump（**只读诊断**，默认关；`AZAI_FLAT_DUMP_TREE=1` 或 `self._dump_tree=True` 才记录）──
        #   动机：`AZAI_FLAT_ROLLOUTS=64` 全看不可能，但要把 flat rollout 的**前缀树**（共享前缀、
        #   分叉点、每条 rollout 的叶子值网分）扒出来解释"为什么某候选 s2 低"。
        #   ⚠️ 默认关 ⇒ 对现役搜索的数值行为逐位零影响（只是 `_tree_rollout_value` 里多一次 bool 判空）。
        self._dump_tree = False
        self._tree_dump = None
        # ── 引擎 RNG 消费开关（默认关 ⇒ 下面所有 `if self._rng_on` 分支都不执行）──
        #   `AZAI_FLAT_RNG_PERTURB=1` 时自动打开（命令行/构造参数也可显式给）。
        if rollout_rng_perturb is None:
            rollout_rng_perturb = _os.environ.get("AZAI_FLAT_RNG_PERTURB", "0") == "1"
        self.rollout_rng_perturb = bool(rollout_rng_perturb)
        self.rollout_rng_consume = max(1, int(rollout_rng_consume))
        assert rollout_rng_key in ("state", "none"), rollout_rng_key
        self.rollout_rng_key = rollout_rng_key
        self._rng_on = False          # 真正生效（引擎支持 consume_rng）才置 True
        self._rng_warned = False
        self.last_rng_consumed = 0    # 本决策实际消费的 RNG 次数（诊断/验收用）
        self.last_rng_missing = 0     # 想消费但引擎不支持而跳过的次数（应为 0）
        self._rng_enabled_engine = None
        if self.rollout_rng_perturb:
            from my_ai.az_intent import rng_engine as _rng_engine
            self._rng_enabled_engine = _rng_engine.enable(verbose=self.verbose or True)
            self._rng_on = bool(self._rng_enabled_engine)
        if eval_mode == "net" and value_model is None:
            raise ValueError("eval_mode='net' 需要 value_model（load_three_models 的第三个返回值）")
        if eval_mode == "net" and feature_extractor is None:
            raise ValueError("eval_mode='net' 需要 feature_extractor（SDK.utils.features.FeatureExtractor）")

    # ── 每回合的"随机化强度"p(r)：动作随机与 RNG 消费**共用**同一套调度 ──────
    def _prob_for_round(self, round_idx: int) -> float:
        """`rollout_prob_mode` 决定的、第 round_idx 回合的随机化概率。

        ⚠️ 这段逻辑原来是内联在 `_decide_actions` 里的，2026-10-27 抽出来供
        "引擎 RNG 消费"复用同一套 C/k（行为逐位不变：同样的输入返回同样的 p）。
        """
        if self.rollout_prob_mode == "first":
            return float(self.rollout_decay_c) if round_idx == 0 else 0.0
        if self.rollout_prob_mode == "decay":
            import math as _math
            return float(self.rollout_decay_c * _math.exp(-round_idx / self.rollout_decay_k))
        if self.rollout_prob_mode == "reverse":
            # 时间反演：p(r) = C*exp(-(H-r)/k)，末期才反击 => 前缀树节点更少
            import math as _math
            _p = self.rollout_decay_c * _math.exp(
                -(self.horizon - round_idx) / self.rollout_decay_k)
            return float(min(1.0, max(0.0, _p)))
        return float(self.rollout_act_prob)

    # ── 引擎 RNG 消费（只在 rollout 的**克隆**上动手，绝不碰裁判/真实对局）──────
    #   RNG 是 48-bit LCG：s ← (25214903917*s + 11) & (2^48-1)（官方 game.cpp:25-27）。
    #   这里只把"消费后 rng_state 是多少"算出来当**缓存键**用（一次乘法），
    #   真正的消费由引擎的 `consume_rng()` 做。两者一致性由
    #   `probe_engine_rng.py` 的 smoke/parity 断言（consume_rng(k) == 手推 LCG）。
    _RNG_MULT = 25214903917
    _RNG_INC = 11
    _RNG_MASK = (1 << 48) - 1

    @classmethod
    def _rng_after(cls, state: int, count: int) -> int:
        for _ in range(int(count)):
            state = (cls._RNG_MULT * state + cls._RNG_INC) & cls._RNG_MASK
        return state

    def _rng_state_of(self, sim):
        """读 sim 的引擎 rng_state（读不到返回 None）。"""
        g = getattr(sim, "_g", sim)
        fn = getattr(g, "get_rng_state", None)
        if fn is None:
            return None
        return int(fn())

    def _consume_rng(self, sim, count: int) -> bool:
        """在 sim（一个 rollout 克隆）上消费 count 次引擎 RNG。不改变任何局面量。"""
        g = getattr(sim, "_g", sim)
        fn = getattr(g, "consume_rng", None)
        if fn is None:
            if not self._rng_warned:
                self._rng_warned = True
                print("[flat] ⚠️ 引擎不支持 consume_rng（native_game 不是 rng_build 版本）"
                      "⇒ 本次 rollout 不消费 RNG；先跑 build_rng_variant.py 并 enable()",
                      flush=True)
            self.last_rng_missing += 1
            return False
        fn(int(count))
        self.last_rng_consumed += int(count)
        return True

    def _decide_actions(self, sim, rng, ctx, root_player=None, round_idx: int = 0) -> list:
        """只**决定**本回合双方各做什么（不落子），返回 [(player, t, a0, a1), ...]。

        2026-10-23：从 `_random_actions` 拆出来，供前缀树实现复用（先决策、查缓存、再决定是否真模拟）。
        概率按 `rollout_prob_mode`（const / decay / first）逐回合计算。
        """
        if self.rollout_mode not in ("random", "lightning"):
            # 2026-10-27：只有 `rollout_mode="random"` 才出随机招；2026-11-02 加 `lightning`。
            # 原来这里没判 rollout_mode（因为前缀树路径只在 random 时进），现在
            # "仅 RNG 消费"也要走前缀树 ⇒ 必须显式判。对现有配置是 no-op
            # （_tree_rollout_value 只在 rollout_mode=="random" 时被调用）。
            return []
        _p = self._prob_for_round(round_idx)
        if _p <= 0.0:
            return []

        from SDK.utils.constants import OperationType

        _sides = (0, 1)
        if self.rollout_mode == "lightning":
            # 2026-11-02：lightning 模式的随机侧**固定为我方**（root_player）——忽略
            # `rollout_random_side`。语义 = "我方在 rollout 里随机放闪电"，对手不动。
            if root_player is not None:
                _sides = (int(root_player),)
        elif self.rollout_random_side == "opponent" and root_player is not None:
            _sides = (1 - int(root_player),)
        out = []
        if self.rollout_mode == "lightning":
            # 2026-11-02：`AZAI_FLAT_ROLLOUT_MODE=lightning` —— **我方**随机放闪电
            # （class 17 = `USE_LIGHTNING_STORM`）。与 random 的差别有**两处**：
            # ① 侧：random = 对手(1-root_player)随机建塔/降级；lightning = 我方(root_player)随机放闪电；
            # ② 类：建塔/降级 → 闪电（落点 = 该侧合法高地格随机，复用 build 的 `ctx["cells"]` 随机位置逻辑）。
            # 闪电有 cd==0 + 金币≥90 的合法性约束 ⇒ 不满足时**空过**（这一模式只有闪电一种动作，
            # 不建塔也不降级）。概率调度（decay(1,3)）/ seed 与 random 逐字共用。
            from SDK.utils.constants import SUPER_WEAPON_STATS, SuperWeaponType
            _sw = int(SuperWeaponType.LIGHTNING_STORM)
            _cost = int(SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost)
            for _pl in _sides:
                if rng.random() >= _p:
                    continue
                try:
                    _cd = int(sim.weapon_cooldowns[_pl][_sw])
                    _coins = int(sim.coins[_pl])
                except Exception:
                    _cd, _coins = 1, -1
                if _cd != 0 or _coins < _cost:
                    continue
                _cand = list(ctx["cells"][_pl])
                if _cand:
                    _x, _y = _cand[int(rng.integers(len(_cand)))]
                    out.append((_pl, int(OperationType.USE_LIGHTNING_STORM), int(_x), int(_y)))
            return out
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
        self.last_tree_stats = None
        """前缀树版 rollout：按 (父节点, 本回合 ops) 复用子状态，省掉重复的 clone/apply/advance。"""
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType, HIGHLAND_CELLS

        # ── 前缀树 dump（只读诊断，默认关 ⇒ 与改动前逐位一致）────────────────
        _dump = bool(getattr(self, "_dump_tree", False)) or _os.environ.get("AZAI_FLAT_DUMP_TREE", "0") == "1"
        if _dump:
            self._tree_dump = {"nodes": {0: {"parent": -1, "depth": 0, "token": None}},
                               "paths": []}

        ctx = {"cells": {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}}
        # node_id -> state（state 已推进到该节点回合结束）
        states = {0: child.clone()}
        edges = {}          # (parent_id, token) -> child_id
        next_id = 1
        leaf_states = []

        leaf_tokens = []
        _rng_key_on = bool(self._rng_on and self.rollout_rng_key == "state")
        for _item in rngs:
            # 2026-10-27：`rngs` 里的元素可以是 Generator，也可以是
            # (action_rng, rng_perturb_rng) 二元组（后者给"引擎 RNG 消费"抽 Bernoulli）。
            if isinstance(_item, tuple):
                rng, rng_p = _item
            else:
                rng, rng_p = _item, None
            cur = 0
            _seq = []
            _path = [] if _dump else None
            for _r in range(self.horizon):
                st = states[cur]
                if st.terminal:
                    break
                ops = self._decide_actions(st, rng, ctx, player, _r)
                token = tuple(sorted(ops)) if ops else ()
                # ── 引擎 RNG 消费（在 apply(ops) 之后、advance_round() 之前）──
                #   `_k` 只影响"这一步子状态用哪个 rng_state"，不改变任何局面量。
                _k = 0
                if self._rng_on and rng_p is not None:
                    if rng_p.random() < self._prob_for_round(_r):
                        _k = self.rollout_rng_consume
                if _rng_key_on:
                    # 键里的 rng_state = **本回合 ops 施加后、advance_round() 之前**
                    # 的引擎 rng_state（= 消费了 _k 次之后）。它和 (父状态, ops) 一起
                    # 唯一决定子状态 ⇒ 精确键；(父状态, ops) 相同但消费回合组合不同
                    # 的分支，rng_state 必不同 ⇒ 不会被错误合并。
                    _parent_rng = self._rng_state_of(st)
                    if _parent_rng is None:
                        key = (cur, token)
                    else:
                        key = (cur, token, self._rng_after(_parent_rng, _k))
                else:
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
                    if _k:
                        self._consume_rng(sim, _k)
                    sim.advance_round()
                    nxt = next_id
                    next_id += 1
                    states[nxt] = sim
                    edges[key] = nxt
                    if _dump:
                        self._tree_dump["nodes"][nxt] = {
                            "parent": int(cur), "depth": int(_r) + 1,
                            "token": [list(t) for t in token]}
                cur = nxt
                if _dump:
                    _path.append((int(_r), [list(t) for t in token], int(cur)))
                # 去重加权按"轨迹签名"分组：消费次数也必须进签名（否则两条
                # 消费模式不同的轨迹会被当成同一条）
                _seq.append((token, _k) if _k else token)
            leaf_states.append(states[cur])
            leaf_tokens.append(tuple(_seq))
            if _dump:
                self._tree_dump["paths"].append(_path)
        self.last_tree_nodes = getattr(self, "last_tree_nodes", 0) + next_id
        if not leaf_states:
            return 0.0
        _lv = self._leaf_values(leaf_states, player)     # ← 批量评估（与朴素路径一致）
        if _dump:
            self._tree_dump["leaf_values"] = [float(v) for v in _lv]
            self._tree_dump["score"] = float(np.mean(_lv))
        if _REC_SAMPLES:
            self.last_tree_stats = (len(states), len(leaf_states), len(set(leaf_tokens)))
            self.last_leaf_tokens = list(leaf_tokens)
        if not self.rollout_dedupe_weight:
            self.last_effective_samples = (self.last_effective_samples or 0) + len(_lv)
            return float(np.mean(_lv))
        # 去重加权：相同 token 序列的样本只算一次
        groups: dict = {}
        for _tok, _v in zip(leaf_tokens, _lv):
            groups.setdefault(_tok, []).append(float(_v))
        reps = [float(np.mean(v)) for v in groups.values()]
        self.last_effective_samples = (self.last_effective_samples or 0) + len(reps)
        return float(np.mean(reps))

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
                raw = value_of(self.value_model(board, stat)).squeeze(-1)
            if self.value_tanh:
                raw = torch.tanh(raw)
            for offset, value in enumerate(raw.detach().cpu().numpy().tolist()):
                values[pending_index[start + offset]] = float(value)
        return values

    @staticmethod
    def _size_hist(bundles) -> dict:
        """候选长度直方图（只给 `AZAI_CAND_VERBOSE` 的日志用，纯只读）。"""
        out: dict = {}
        for b in (bundles or ()):
            n = len(b or ())
            out[n] = out.get(n, 0) + 1
        return dict(sorted(out.items()))

    # ── 额外候选注入（默认关）─────────────────────────────────────────────────
    def idle_rollout_values(self, state, player: int, bundles: list, horizon: int | None = None):
        """**确定性 idle rollout**（双方都不出招）：每条 bundle 各 1 次，返回 np.ndarray。

        2026-10-29：给统一候选生成的**阶段 1** 用（`AZAI_CAND_MODE=unified`）。
        语义与 `search()` 里 `rollout_mode="idle"` 的那条路径**逐位一致**：
        `clone → apply(ops) → (P1 才 advance_round) → horizon 次 advance_round → 叶值网`。
        ⇒ 阶段 1 的分数与"把该候选交给 idle 评估器"得到的是同一个数。

        ⚠️ 本方法**不改变任何状态**、不被 `pin` 路径调用（只在统一生成器里当 `rollout_fn`）⇒
        默认配置下它对既有读数零影响。`AZAI_FLAT_RNG_PERTURB=1` 时 idle **不再是确定性的**
        （引擎 RNG 会被消费）⇒ 这里显式**不消费**（阶段 1 要的就是"确定性"；签名里带上
        `_rng_on` 让阶段 2 拒绝复用，见 `unified_cand.UnifiedCandFlatSearch._p1_signature`）。
        """
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType

        h = int(self.horizon if horizon is None else horizon)
        leaves = []
        for key in bundles:
            ops = [Operation(OperationType(int(x[0])), int(x[1]), int(x[2])) for x in (key or ())]
            sim = state.clone()
            if ops:
                sim.apply_operation_list(int(player), ops)
            if int(player) == 1:
                sim.advance_round()
            for _ in range(h):
                if sim.terminal:
                    break
                sim.advance_round()
            leaves.append(sim)
        if not leaves:
            return np.zeros(0, dtype=np.float64)
        return self._leaf_values(leaves, int(player))

    def set_extra_bundles(self, bundles=None) -> None:
        """设置**下一次**决策要注入的额外候选（`[( (op,arg0,arg1), ... ), ...]`；HOLD = `()`）。

        `None` / 空 ⇒ 关闭（= 改动前的行为）。调用方负责每回合显式设置或清空。
        """
        if not bundles:
            self.extra_bundles = None
            return
        out = []
        for b in bundles:
            out.append(tuple(tuple(int(x) for x in op) for op in (b or ())))
        self.extra_bundles = out

    def set_only_extra(self, flag: bool = False) -> None:
        """2026-10-28（E4；`AZAI_ONLY_EXTRA=1`）：候选集**只保留**额外候选。默认 `False`。

        打开后 `search()` 里 `bundles = 额外候选`（网络候选不参与），`priors`/`intent_counts`
        跟着对齐；预算公式、rollout、叶值网、选择规则（含全 NaN 时退回先验）**一律不变**。
        关闭时 `search()` 走的是与改动前**逐字相同**的那几行（`probe_*_zero_impact` 口径）。
        """
        self.only_extra = bool(flag)

    def _collect_extras(self, extra_bundles, existing) -> list:
        """把调用方给的额外候选规整成 bundle 元组列表：**去重**（与已有候选、与彼此）+ 丢畸形项。

        去重的意义：重复候选会白烧一份 rollout 预算，而且 `argmax` 在多份相同分数上
        是任意取的（`np.argmax` 取第一个，其实还好），但预算浪费是真的。
        """
        if not extra_bundles:
            return []
        seen = {tuple(tuple(int(x) for x in op) for op in (b or ())) for b in existing}
        out = []
        for b in extra_bundles:
            # 容错：允许直接传一个 op 三元组（= 单 op 候选）
            if isinstance(b, (list, tuple)) and len(b) == 3 and all(isinstance(x, (int, np.integer))
                                                                  for x in b):
                b = (b,)
            try:
                key = tuple(tuple(int(x) for x in op) for op in (b or ()))
            except (TypeError, ValueError):
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    # ── 主搜索 ──────────────────────────────────────────────────────────────
    def search(self, state, player: int, *, temperature: float = 0.0,
               extra_bundles=None) -> BundleSearchResult:
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
        # 每次决策重置 RNG 消费计数（诊断/验收用；不开 perturb 时恒为 0）
        self.last_rng_consumed = 0
        self.last_rng_missing = 0
        self._expand(root, self.k)                 # ← 每决策唯一一次 net_fn（含根节点的 MC 类仲裁）
        bundles = list(root.bundles)
        n_net = len(bundles)                      # 网络候选数（注入前的边界；命中判定用它）
        # ── 额外候选注入（默认关；`None`/空 ⇒ 下面一行也不执行）────────────────────
        #   追加在**末尾**：既有候选的下标/展开顺序/`_leaf_values` 批组成都不变
        #   ⇒ 不开注入时数值路径与改动前一致；开了也只有"末尾多几个候选"这一个变化。
        _extras_in = self.extra_bundles if extra_bundles is None else extra_bundles
        # 2026-10-28（E4）：`only_extra` ⇒ **候选集 = 只有额外候选**（绕过网络/类轴那一支）。
        #   默认 False ⇒ 走的是与改动前**逐字相同**的两行（`_collect_extras(_, bundles)` +
        #   `bundles.extend`），数值路径一个字节都不变。
        #   ⚠️ only_extra 时额外候选只在**彼此之间**去重（不跟网络候选比），否则"冠军这一手刚好
        #   也是网络候选"会被悄悄丢掉 —— 那样候选集就不再是"冠军动作 + 亚军动作"了。
        _only_extra = bool(getattr(self, "only_extra", False))
        _extras = self._collect_extras(_extras_in, [] if _only_extra else bundles)
        if _only_extra:
            bundles = list(_extras)
            n_net = 0                     # 选中任何候选都算"额外候选"（命中统计口径）
        else:
            bundles.extend(_extras)
        self.last_extra = len(_extras)
        if not bundles:
            self.last_scores = np.zeros(0)
            self.last_rollouts = 0
            self.last_rounds = 0
            return BundleSearchResult(bundles=[], visit_policy=np.array([], dtype=np.float32),
                                      chosen_index=0, chosen_bundle=(),
                                      root_value=float(root.net_value), intent_counts=[])

        _t_after_expand = _time.perf_counter()
        # priors 只用于「全 NaN / 全相等」的退化回退；额外候选没有网络先验 ⇒ 补 `extra_prior`。
        # only_extra 时候选集里没有网络候选 ⇒ priors 只有额外候选那几项（长度必须与 bundles 对齐）。
        if _only_extra:
            priors = np.asarray([self.extra_prior] * self.last_extra, dtype=np.float64)
        else:
            priors = np.asarray([c.prior for c in root.children] + [self.extra_prior] * self.last_extra,
                                dtype=np.float64)

        # 2026-10-23：随机出招 rollout 的上下文（**只在这里算一次**，rollout 内复用）
        # 2026-11-02：`lightning` 也用 `_rctx`（随机落点要读 `cells`）；其余语义与 random 一致。
        _rctx = None
        if self.rollout_mode in ("random", "lightning"):
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
        _samples_buf = [None] * n_cand if _REC_SAMPLES else None
        total_rounds = 0
        # ── 两阶段复用（2026-10-29，`AZAI_CAND_MODE=unified` 专用；**pin 时恒 None**）──────
        #   设计 `docs/unified_cand_design_20261029.md` §"rollout 缓存与两阶段复用"：阶段 1
        #   （bandit 生成候选时的 rollout）已经算过的结果，直接当阶段 2 的样本，只补足采样数。
        #   ⚠️ 只有**两个阶段的估计器完全一致**才允许（`UnifiedCandFlatSearch._p1_signature`
        #   == `_p2_signature`）。默认阶段 1 = idle（确定性）、阶段 2 = random ⇒ 不一致 ⇒
        #   不复用（把 idle 的分混进 random 的样本均值是错的）。`AZAI_FLAT_ROLLOUT_MODE=idle`
        #   时两者一致 ⇒ 复用生效（phase 2 的 N 次全是同一个确定性值 ⇒ 省 N-1 次）。
        _reuse = None
        if getattr(self, "cand_mode", "pin") == "unified":
            _reuse = getattr(self, "_phase2_cache", None)
        for index, child in enumerate(children):
            # 2026-10-23：前缀树记忆化（随机 rollout 专用）。
            #   随机 rollout 要大 N 才抹得平噪声，而朴素实现成本 ∝ N×H；
            #   树版按 (父节点, 本回合 ops) 复用状态：共享率 43%~94% ⇒ 省 21%~91%。
            # 2026-10-27：**仅引擎 RNG 消费**（rollout_mode=idle + perturb 开）也要走
            #   树，否则 N 条会各自 clone 到底、白跑 N×H 回合。默认（perturb 关）条件
            #   与改动前完全一致 ⇒ 走哪条路不变。
            _use_tree = self.rollout_share_prefix and (_rctx is not None or self._rng_on)
            if _use_tree:
                _base = ((int(self.rollout_seed) * 1000003 + self._decision_idx) * 100003
                         + int(index) * 101)
                _rngs = [np.random.default_rng(_base + int(_k))
                         for _k in range(n_rollouts)]
                if self._rng_on:
                    # 消费决策用**独立**的随机流（不和动作抽签抢 draw，否则开/关
                    # perturb 会改变动作随机的抽样序列 ⇒ 两种随机源不再正交）
                    _rngs = [(g, np.random.default_rng(_base + 500003 + int(_k)))
                             for _k, g in enumerate(_rngs)]
                _t0 = _time.perf_counter()
                scores[index] = self._tree_rollout_value(child, player, _rngs)
                total_rounds += self.horizon * n_rollouts
                continue
            # 阶段 1 已有的样本（`_reuse` 非 None ⇒ 估计器一致 ⇒ 直接当阶段 2 的样本用）
            _pre: list = []
            if _reuse is not None:
                from my_ai.az_intent.unified_cand import canon_ops as _canon
                _pre = list(_reuse.get(_canon(bundles[index]), ()))[:n_rollouts]
            leaves = []
            for _k in range(len(_pre), n_rollouts):
                sim = child.clone()
                _rng = None
                _rng_p = None
                if self._rng_on:
                    _rng_p = np.random.default_rng(
                        (int(self.rollout_seed) * 1000003 + self._decision_idx) * 100003
                        + int(index) * 101 + int(_k) + 500003)
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
                    # 引擎 RNG 消费：ops 已施加、advance_round() 之前
                    if _rng_p is not None and _rng_p.random() < self._prob_for_round(_round_idx):
                        self._consume_rng(sim, self.rollout_rng_consume)
                    sim.advance_round()
                    _round_idx += 1
                    total_rounds += 1
                leaves.append(sim)
            _lv = self._leaf_values(leaves, player)
            if _pre:
                _lv = np.concatenate([np.asarray(_pre, dtype=np.float64), _lv])
                self.last_reused = int(getattr(self, "last_reused", 0)) + len(_pre)
            if _REC_SAMPLES:
                _samples_buf[index] = [float(v) for v in _lv]
            scores[index] = float(_lv.mean())

        _t_after_rollout = _time.perf_counter()
        self.last_timing = {"expand_s": _t_after_expand - _t_expand,
                            "rollout_s": _t_after_rollout - _t_rollout}
        self.last_scores = scores
        if _REC_SAMPLES:
            self.last_samples = _samples_buf
        self.last_rollouts = n_rollouts
        self.last_rounds = total_rounds
        # 2026-10-31：`AZAI_CAND_VERBOSE=1` 时**pin 臂也打一行**同格式的读数
        #   （`AZAI_CAND_MODE` 的 A/B 要对比 n_cand：unified 侧由 `UnifiedCandFlatSearch.search`
        #   自己打 `mode=unified`，这里补 `mode=pin`）。**纯日志、默认关**（`cand_verbose`
        #   由 `AZAI_CAND_VERBOSE` 决定，不设 ⇒ 一个字节都不多打 ⇒ 既有读数逐位不变）。
        if getattr(self, "cand_verbose", False) and self.cand_mode != "unified":
            print(f"[cand] mode=pin n_cand={n_cand} n_rollouts={n_rollouts} "
                  f"决定_ms=0 长度分布={self._size_hist(bundles)}",
                  file=sys.stderr, flush=True)
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
        self.last_chosen_is_extra = bool(best >= n_net)
        _ic = [] if _only_extra else list(root.intent_counts)
        if self.last_extra:
            # 长度必须与 `bundles` 对齐（额外候选没有网络 intent 计数 ⇒ 补空字典）
            _ic = _ic + [[dict(), dict(), dict()] for _ in range(self.last_extra)]
        return BundleSearchResult(
            bundles=bundles,
            visit_policy=policy,
            chosen_index=best,
            chosen_bundle=bundles[best],
            root_value=float(scores[best]),
            intent_counts=_ic,
        )
