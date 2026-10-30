"""Self-play collection for bundle-MCTS AlphaZero training (T1).

Both players use the bundle MCTS.  At each decision we record a training sample:
board/stats (inputs), player, legal masks, the k sampled bundles with their
per-head intent sample counts, the temperature-scaled visit distribution, the
network's recorded policy outputs (anchor for §3.2), and the terminal
HP-difference value target.

Samples are saved per game (pickle) so training can stream / load them.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
import sys
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _guard_no_three_net(ckpt: dict, caller: str) -> None:
    """Fail loudly if a 3-net checkpoint is fed to a 1-/2-net loader.

    A 3-net checkpoint deliberately has NO ``model_state`` (so a stale, untrained
    action_map can never be read silently) — see docs/az_three_net_split_plan.md.
    """
    if "pos_state" in ckpt or ("class_state" in ckpt and "model_state" not in ckpt):
        raise ValueError(
            f"{caller}: this checkpoint is a 3-net (class/pos/value) checkpoint; "
            "use load_three_models() / make_net_fn_from_ckpt() instead. "
            "Reading it with a 1-/2-net loader would silently return an untrained "
            "action_map.")


def load_model_from_ckpt(ckpt_path: str):
    from my_ai.network import create_model, model_kwargs_from_ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _guard_no_three_net(ckpt, "load_model_from_ckpt")
    model = create_model(**model_kwargs_from_ckpt(ckpt))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_split_models(ckpt_path: str):
    """Load a split checkpoint into (policy_model, value_model).

    Both networks share the same architecture.  For a single-model checkpoint
    (model_state only, e.g. gen0120_warm) both nets warm-start from the same
    weights; for a split checkpoint (has ``value_state``) each loads its own.
    """
    from my_ai.network import create_model, model_kwargs_from_ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _guard_no_three_net(ckpt, "load_split_models")
    _kw = model_kwargs_from_ckpt(ckpt)
    # A mixed checkpoint may record a separate normalization for the value net
    # (e.g. GroupNorm value + BatchNorm policy); older checkpoints omit these
    # keys and both nets share the policy's config.
    _vkw = dict(_kw)
    if "value_no_bn" in ckpt or "value_gn" in ckpt:
        _vkw["no_bn"] = bool(ckpt.get("value_no_bn", False))
        _vkw["gn"] = bool(ckpt.get("value_gn", False))
        _vkw["gn_groups"] = int(ckpt.get("value_gn_groups", 8))
    _vkw["value_pool"] = str(ckpt.get("value_value_pool", _kw.get("value_pool", "gap")))

    def _build():
        return create_model(**_kw)

    def _build_value():
        return create_model(**_vkw)

    policy_model = _build()
    policy_model.load_state_dict(ckpt["model_state"])
    value_model = _build_value() if "value_state" in ckpt else _build()
    value_model.load_state_dict(ckpt.get("value_state", ckpt["model_state"]))
    policy_model.eval()
    value_model.eval()
    return policy_model, value_model


def load_three_models(ckpt_path: str):
    """Load a 3-net checkpoint into (class_model, pos_model, value_model).

    Class net  -> ``head_logits`` (the class distribution; the frozen/anchored half)
    Pos net    -> ``action_map``  (per-class position maps; the half we train)
    Value net  -> ``value``

    All three are plain ``AntWarNetwork`` instances (which always build every head);
    only the head each one is responsible for is read.  Slimming them is a later step
    (docs/az_three_net_split_plan.md §3).
    """
    from my_ai.network import create_model, model_kwargs_from_ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "class_state" not in ckpt or "pos_state" not in ckpt:
        raise ValueError("load_three_models: checkpoint lacks class_state/pos_state")
    _kw = model_kwargs_from_ckpt(ckpt)
    _vkw = dict(_kw)
    if "value_no_bn" in ckpt or "value_gn" in ckpt:
        _vkw["no_bn"] = bool(ckpt.get("value_no_bn", False))
        _vkw["gn"] = bool(ckpt.get("value_gn", False))
        _vkw["gn_groups"] = int(ckpt.get("value_gn_groups", 8))
    _vkw["value_pool"] = str(ckpt.get("value_value_pool", _kw.get("value_pool", "gap")))

    class_model = create_model(**_kw)
    class_model.load_state_dict(ckpt["class_state"])
    pos_model = create_model(**_kw)
    pos_model.load_state_dict(ckpt["pos_state"])
    value_model = create_model(**_vkw)
    value_model.load_state_dict(ckpt.get("value_state", ckpt["pos_state"]))
    for m in (class_model, pos_model, value_model):
        m.eval()
    return class_model, pos_model, value_model


class ThreeNetPolicy:
    """Anchor-recording façade for 3-net checkpoints.

    ``collect_game`` only needs ``num_heads`` and ``__call__(board, stats) ->
    {action_map, head1..N_logits}``; this forwards the class net for the logits and
    the pos net for the action map, so the collection code path stays unchanged.
    """

    def __init__(self, class_model, pos_model):
        self.class_model = class_model
        self.pos_model = pos_model
        self.num_heads = class_model.num_heads

    def __call__(self, board, stats):
        with torch.no_grad():
            c_out = self.class_model(board, stats)
            p_out = self.pos_model(board, stats)
        out = {"action_map": p_out["action_map"]}
        for i in range(self.num_heads):
            out[f"head{i + 1}_logits"] = c_out[f"head{i + 1}_logits"]
        return out

    def eval(self):  # parity with nn.Module call sites
        return self


def make_three_net_fn(class_model, pos_model, value_model, feature_extractor,
                      max_actions: int = 96, value_tanh: bool = True,
                      value_rel_to_abs: float = 0.0, q_model=None,
                      value_src: str = "net", value_offset: float = 0.0,
                      value_scale: float = 1.0):
    """net_fn over three independent networks — same interface as make_split_net_fn.

    ``action_map`` comes from the POS net, ``head_logits`` from the CLASS net,
    ``value`` from the VALUE net (3 forwards instead of 2).

    ``value_src="handcrafted"``：不用 value 网，改调**亚军的手写价值**
    `lure_value_eval.evaluate_terminal(state, player)`，并做 `(v - offset) / scale`
    线性归一到搜索的尺度（亚军价值 ~10000 量级，搜索期望 ~[-1,1]）。
    """
    from my_ai.az_intent.mcts import HP_SCALE

    def net_fn(state, player):
        from my_ai.az_intent.timing import T
        with T("search.net"):
            with T("net.encode"):
                obs = feature_extractor.encode_observation(state, player, np.zeros(max_actions))
                board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
                stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
            with torch.no_grad():
                with T("net.class"):
                    c_out = class_model(board, stats)
                with T("net.pos"):
                    p_out = pos_model(board, stats)
                with T("net.value"):
                    v_out = value_model(board, stats)
        heads = [c_out[f"head{i + 1}_logits"].squeeze(0).numpy()
                 for i in range(class_model.num_heads)]
        if value_src == "handcrafted":
            from my_ai.distill_top2.lure_value_eval import evaluate_terminal
            value = (float(evaluate_terminal(state, player)) - value_offset) / value_scale
        else:
            v_t = v_out["value"].squeeze(0)
            value = float(torch.tanh(v_t).item()) if value_tanh else float(v_t.item())
            if value_rel_to_abs > 0.0:
                value = value / value_rel_to_abs + float(obs["stats"][1]) / HP_SCALE
        out = {
            "action_map": p_out["action_map"].squeeze(0).numpy(),
            "head_logits": heads,
            "value": value,
        }
        # 可选的「类条件价值」打分（方法侧 M2，见 docs/prereg_20261005_qhead_class_decision.md）：
        # 用**冻结主干**的 state_emb（class 网的）拼上类 one-hot，过一个小 MLP ⇒ 24 个类的 Q 值。
        if q_model is not None:
            try:
                emb = c_out["state_emb"].squeeze(0).detach().cpu().numpy()
                out["state_emb"] = emb
                out["q_classes"] = q_model.score_np(emb)
            except Exception as _exc:  # noqa: BLE001
                # 2026-10-22（审计）：原来静默 ⇒ pos_pin="q" 会悄悄退化成 argmax，零观测。
                if not out.get("_q_warned"):
                    out["_q_warned"] = True
                    print(f"[q_classes] ⚠️ 类条件 Q 值不可用（首次）：{type(_exc).__name__}: {_exc}",
                          flush=True)
        return out

    return net_fn


def make_net_fn_from_ckpt(ckpt_path: str, feature_extractor, value_tanh: bool = True,
                          value_rel_to_abs: float = 0.0, q_ckpt: str = "",
                          value_src: str = "net", value_offset: float = 0.0,
                          value_scale: float = 1.0):
    """Build (anchor_model, net_fn) for a checkpoint — single or split.

    ``anchor_model`` is the policy network (used to record the anchor outputs
    during self-play); ``net_fn`` is the search interface (one or two forwards).
    ``value_rel_to_abs`` (0 = off): reconstruct an absolute value from a head
    trained on relative labels (see ``train.make_net_fn``).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "class_state" in ckpt and "pos_state" in ckpt:
        class_model, pos_model, value_model = load_three_models(ckpt_path)
        _q = None
        if q_ckpt:
            from my_ai.az_intent.q_head import QHead
            _q = QHead.load(q_ckpt)
        return (ThreeNetPolicy(class_model, pos_model),
                make_three_net_fn(class_model, pos_model, value_model, feature_extractor,
                                  value_tanh=value_tanh,
                                  value_rel_to_abs=value_rel_to_abs, q_model=_q,
                                  value_src=value_src, value_offset=value_offset,
                                  value_scale=value_scale))
    if "value_state" in ckpt:
        from my_ai.az_intent.train import make_split_net_fn
        policy_model, value_model = load_split_models(ckpt_path)
        return policy_model, make_split_net_fn(policy_model, value_model, feature_extractor,
                                               value_tanh=value_tanh,
                                               value_rel_to_abs=value_rel_to_abs)
    from my_ai.az_intent.train import make_net_fn
    model = load_model_from_ckpt(ckpt_path)
    return model, make_net_fn(model, feature_extractor, value_tanh=value_tanh,
                              value_rel_to_abs=value_rel_to_abs)


def make_initial_state(seed: int, native_engine: bool = False):
    """Create a fresh game state — pure-Python engine or the C++ engine.

    The C++ engine (``native_game`` + GameStateFacade) matches the official game
    binary; the Python SDK has known detail deviations.  Switching engines
    changes the simulated game, so results are only comparable within the same
    engine.  The C++ engine's advance_round is ~22x faster (MCTS bottleneck).
    """
    if native_engine:
        cpp_dir = Path(__file__).resolve().parents[2] / "cpp_engine"
        if str(cpp_dir) not in sys.path:
            sys.path.insert(0, str(cpp_dir))
        from my_ai.az_intent.game_state_facade import GameStateFacade
        return GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    from SDK.backend.engine import GameState
    return GameState.initial(seed=seed, cold_handle_rule_illegal=True)


def _fmt_op(t: int, a0: int, a1: int) -> str:
    """动作命名表示（与 code/test_match/diagnose.py 的 op_desc 一致），如 BUILD(6,9)。"""
    from SDK.utils.constants import OperationType
    name = {
        OperationType.BUILD_TOWER: "BUILD",
        OperationType.UPGRADE_TOWER: "UPGRADE",
        OperationType.DOWNGRADE_TOWER: "DOWNGRADE",
        OperationType.USE_LIGHTNING_STORM: "LIGHTNING",
        OperationType.USE_EMP_BLASTER: "EMP",
        OperationType.USE_DEFLECTOR: "DEFLECTOR",
        OperationType.USE_EMERGENCY_EVASION: "EVASION",
        OperationType.UPGRADE_GENERATION_SPEED: "UP_SPEED",
        OperationType.UPGRADE_GENERATED_ANT: "UP_ANT_HP",
    }.get(OperationType(t))
    if name is None:
        return f"OP_{t}"
    if t == OperationType.BUILD_TOWER:
        return f"{name}({a0},{a1})"
    if t == OperationType.UPGRADE_TOWER:
        return f"{name}(id={a0}->type={a1})"
    if t == OperationType.DOWNGRADE_TOWER:
        return f"{name}(id={a0})"
    if OperationType.USE_LIGHTNING_STORM <= t <= OperationType.USE_EMERGENCY_EVASION:
        return f"{name}({a0},{a1})"
    return name


def _random_legal_bundle(model, out, state, player, rng, pm, cm, n_samples: int = 12):
    """A uniformly-random LEGAL bundle: uniform class dist per head (position
    sampling made ~uniform via a huge t_pos), then one bundle picked at random.
    """
    from my_ai.az_intent.bundle_mcts import sample_bundle

    head_logits = np.stack([out[f"head{i + 1}_logits"].squeeze(0).numpy()
                            for i in range(model.num_heads)])
    net_out = {"action_map": out["action_map"].squeeze(0).numpy(),
               "head_logits": head_logits}
    uni = []
    for h in range(model.num_heads):
        n_cls = head_logits[h].shape[-1]
        uni.append(np.ones(n_cls, dtype=np.float32) / n_cls)
    class_ids = [rng.choice(len(p), size=n_samples, p=p).tolist() for p in uni]
    bundles = []
    for j in range(n_samples):
        ops_j, _ = sample_bundle(
            net_out, state, player, t_class=1.0, t_pos=1e6, rng=rng,
            position_mask=pm, class_mask=cm,
            class_probs=uni, class_ids=[ci[j] for ci in class_ids])
        if ops_j:
            bundles.append(tuple((int(o.op_type), o.arg0, o.arg1) for o in ops_j))
    if not bundles:
        return None
    return bundles[int(rng.integers(len(bundles)))]


def _example_legal_bundle(state, player, feat):
    """**ExampleAI 的做法**：`ai_example.py` 逐字逻辑（`bundles[1:8]` 里取
    (score, -len(ops)) 最大者）。**不做任何过滤** —— 用户 2026-09-19 明确要求保持原样。

    用途（`--inject-example-prob`）：以概率 p 把搜索选的动作换成它 —— 与
    `--random-action-prob`（均匀随机 bundle = 纯噪声）不同，**这是"合理动作"**，
    所以既能把对局推向"有塔的局面"（给价值网/类头铺覆盖），又不会把对局搞乱。

    ⚠️ 已知行为（2026-09-19 直接测量，`_tmp_probe_example_pick.py`）：金币耗尽时
    `ActionCatalog.build()` 只剩 `[hold(0.0), downgrade...(-3.3)]`，而这里取
    `bundles[1:8]` ⇒ **跳过 `bundles[0]` 那个 hold** ⇒ 它会**拆一座塔换钱**，
    下回合又有钱建塔 ⇒ 官方参考实现自身就有"建↔拆循环"成分（真正出招的决策点里
    26%~39% 是降级）。我们**故意保留**这个行为：注入率只有 2%，且这套数据要覆盖的
    正是"含建塔/升级/降级的混合局面"。
    """
    from SDK.utils.actions import ActionCatalog
    from SDK.utils.features import FeatureExtractor
    cat = ActionCatalog(max_actions=96, feature_extractor=feat or FeatureExtractor(max_actions=96))
    b = cat.build(state, player)
    if not b:
        return None
    short = b[1:min(len(b), 8)] or b
    best = max(short, key=lambda x: (x.score, -len(x.operations)))
    # 注意：`chosen` 在别处是**元组的元组** (op_type, arg0, arg1)，不是 Operation 对象
    ops = [(int(o.op_type), int(o.arg0), int(o.arg1)) for o in best.operations]
    return ops or None


def _lightning_ops(state, player: int, feat) -> list[tuple] | None:
    """给"强制放闪电"找一个**合法**的落点，返回 ops 元组列表（`(op_type, x, y)`）。

    两条路（先精确、后兜底）：
      1. 直接问 `ActionCatalog`：它已经在 `_superweapon_candidates` 里做了
         "敌方蚂蚁/塔为圆心 + `_storm_value` 打分 + `can_apply_operation` 校验"
         （`SDK/utils/actions.py:158-176`）。命中就用它的落点（= 最划算那一手）。
      2. 兜底：目录里的 storm 被 `value > 1.5` 的门槛挡掉时（例如只有一只残血蚂蚁），
         自己按"半径内敌方单位最多"挑一格，再用 `can_apply_operation` 校验。
    都拿不到就返回 None（规则这一步不注入，让 AI 自己走）—— **绝不发非法 op**。
    """
    from SDK.backend.model import Operation
    from SDK.utils.actions import ActionCatalog
    from SDK.utils.constants import OperationType, SUPER_WEAPON_STATS, SuperWeaponType
    from SDK.utils.features import FeatureExtractor
    from SDK.utils.geometry import hex_distance

    cat = ActionCatalog(max_actions=96, feature_extractor=feat or FeatureExtractor(max_actions=96))
    for b in cat.build(state, player):
        if "storm" in getattr(b, "tags", ()):
            ops = [(int(o.op_type), int(o.arg0), int(o.arg1)) for o in b.operations]
            if ops:
                return ops
    # 兜底：自己挑落点
    radius = int(SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].attack_range)
    enemy = 1 - player
    units = [(int(a.x), int(a.y)) for a in state.ants_of(enemy)]
    units += [(int(t.x), int(t.y)) for t in state.towers_of(enemy)]
    if not units:
        return None
    best = None
    for (x, y) in set(units):
        hits = sum(1 for (ux, uy) in units if hex_distance(x, y, ux, uy) <= radius)
        if best is None or hits > best[0]:
            best = (hits, x, y)
    op = Operation(OperationType.USE_LIGHTNING_STORM, best[1], best[2])
    if state.can_apply_operation(player, op):
        return [(int(op.op_type), int(op.arg0), int(op.arg1))]
    return None


def _coin_rule_decision(state, player: int, feat, cd_near: int = 5,
                        hold_lo: int = 60) -> tuple[str, list[tuple]] | None:
    """「金币 × 冷却」手写规则的**唯一决策点**（`--inject-coin-rule`）。

    条件与动作（`docs/value_net_retrain_collection_plan_20261031.md` §2）：

    | 条件 | 覆盖动作 | 目的 |
    |---|---|---|
    | `coins ≥ 90` 且 闪电 `cd == 0` | **LIGHTNING_STORM**（落点见 `_lightning_ops`）| 造"放闪电"的正例 |
    | `coins ∈ [hold_lo, 90)` 且 `cd ≤ cd_near` | **HOLD**（空 bundle，攒钱）| 造"攒到 90"的正例 |

    返回 `(kind, ops)`，kind ∈ {"lightning", "hold"}；`None` = 本条规则不适用。
    ⚠️ 只**读** state（金币/冷却），不写；不消耗任何 RNG（概率由调用方抽）。
    """
    from SDK.utils.constants import SUPER_WEAPON_STATS, SuperWeaponType

    st = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM]
    cost = int(st.cost)
    cd = int(state.weapon_cooldowns[player][int(SuperWeaponType.LIGHTNING_STORM)])
    coins = int(state.coins[player])
    if coins >= cost and cd == 0:
        ops = _lightning_ops(state, player, feat)
        return ("lightning", ops) if ops else None
    if hold_lo <= coins < cost and cd <= cd_near:
        return ("hold", [])
    return None


def collect_game(net_fn, model, feature_extractor, mcts, seed, *,
                 max_rounds: int = 512, temp_rounds: int = 30,
                 progress_path: str | None = None,
                 native_engine: bool = False,
                 random_action_prob: float = 0.0,
                 inject_example_prob: float = 0.0,
                 skip_hold_search: bool = False,
                 inject_coin_rule: bool = False,
                 inject_coin_rule_p: float = 0.5,
                 inject_coin_rule_cd: int = 5,
                 inject_coin_rule_hold_lo: int = 60,
                 stats_out: dict | None = None,
                 dump_cand: bool = False,
                 mc_horizon: int = 100, mc_every: int = 1,
                 mc_root_only: int = 0, mc_bidir: int = 0) -> list[dict]:
    """Play one self-play game (both sides = bundle MCTS), record training samples.

    If ``progress_path`` is given, a per-round text log (round index + each
    player's chosen operations) is written there and flushed every round, so the
    game's progress can be watched while it runs (games take ~2h at 128/depth4).

    ``random_action_prob`` (x): with probability x a decision ignores the
    search's choice and plays a UNIFORMLY random legal bundle (uniform over
    class + position within the legal masks).  Purpose: break the
    two-equal-searchers grind so games end with larger HP differences.  Keep x
    small (~0.05): random blunders are symmetric, so a small rate adds decisive
    outcomes without destroying the state->outcome signal.  The recorded policy
    target (visit distribution) is still the search's, so only the played move
    and the resulting states change.
    """
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.decoder import make_class_mask, make_position_masks
    from my_ai.az_intent.mcts import HP_SCALE

    state = make_initial_state(seed, native_engine)
    samples: list[dict] = []
    pfile = open(progress_path, "w", encoding="utf-8") if progress_path else None
    # 「金币×冷却」规则注入的计数器 + **独立随机流**：不消耗 `mcts.rng` ⇒
    # 开关关着时对搜索的随机序列零影响；开着时注入率也不被搜索抽随机干扰。
    coin_rule_rng = np.random.default_rng(int(seed) ^ 0x5EEDC01E) if inject_coin_rule else None
    st = {"decisions": 0, "elig_hold": 0, "fire_hold": 0, "elig_lightning": 0,
          "fire_lightning": 0, "light_total": 0, "light_injected": 0, "light_ai": 0,
          # 每决策墙钟（秒）：采集规模的唯一成本口径（`--search unified` 的 p50/p95 决定采多少局）
          "times": [], "coins_ge90": 0, "rounds": 0,
          # B：官方 replay 用的**每回合双方完整 op 序列**（种子 + 它 ⇒ 可逐位重建任意时刻）
          "replay_steps": []}
    for round_idx in range(max_rounds):
        if state.terminal:
            break
        temperature = 1.0 if round_idx < temp_rounds else 1e-6
        round_ops: list[list] = [[], []]
        for player in (0, 1):
            if state.terminal:
                break
            _t_dec = time.time()
            # 采集成本读数：本决策时刻我方金币是否 ≥90（"够买闪电"的状态占比）
            if int(state.coins[player]) >= 90:
                st["coins_ge90"] += 1
            # recorded policy outputs (anchor) + masks (training denominators)
            obs = feature_extractor.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            recorded_am = out["action_map"].squeeze(0).numpy().astype(np.float16)
            recorded_hl = np.stack(
                [out[f"head{i + 1}_logits"].squeeze(0).numpy()
                 for i in range(model.num_heads)]
            ).astype(np.float16)
            pm = make_position_masks(state, player, intent_decoding=True)
            cm = make_class_mask(state, player, position_mask=pm, intent_decoding=True)

            # HOLD-skip optimisation: the MCTS is the bottleneck (advance_round), and
            # ~91% of decisions end up HOLD — almost always because every action class
            # is masked out (cooldown / no valid position / unaffordable).  When the
            # *legal* class argmax is HOLD for every head the search can only pick HOLD
            # too (measured agreement: 2 wrong skips in 10,002 decisions), so skip it.
            skip = False
            if skip_hold_search:
                votes = []
                for i in range(model.num_heads):
                    lg = out[f"head{i + 1}_logits"].squeeze(0).numpy().astype(np.float64)
                    lg = np.where(cm, lg, -1e9)
                    votes.append(int(np.argmax(lg)))
                skip = all(v == 23 for v in votes)  # 23 = HOLD
            if skip:
                res = None
                chosen = ()
                bundles, intent_counts, visit = [()], None, np.ones(1, dtype=np.float32)
            else:
                res = mcts.search(state, player, temperature=temperature)
                chosen = res.chosen_bundle
                bundles, intent_counts, visit = (list(res.bundles), res.intent_counts,
                                                 res.visit_policy.astype(np.float32))
                rb = None
                if inject_example_prob > 0.0 and mcts.rng.random() < inject_example_prob:
                    rb = _example_legal_bundle(state, player, feature_extractor)
                elif random_action_prob > 0.0 and mcts.rng.random() < random_action_prob:
                    rb = _random_legal_bundle(model, out, state, player, mcts.rng, pm, cm)
                if rb:
                    chosen = rb

            # ── 「金币 × 冷却」规则注入（2026-10-31，`--inject-coin-rule`）─────────────
            # 放在**所有选择之后**：它覆盖 search/example/random 的任何结果（含 skip 分支——
            # 我们 AI 在"钱够 90 且 cd 已好"时大概率本来就选 HOLD，若只在 search 分支注入，
            # 最该注入的那类决策反而永远注入不到）。
            # ⚠️ 开关关着时**整段不执行**（不读 state、不建 catalog、不抽随机数）⇒ 逐位零影响。
            inject_src = 0
            if inject_coin_rule:
                rule = _coin_rule_decision(state, player, feature_extractor,
                                           cd_near=inject_coin_rule_cd,
                                           hold_lo=inject_coin_rule_hold_lo)
                if rule is not None:
                    kind, rule_ops = rule
                    st["elig_" + kind] += 1
                    if coin_rule_rng.random() < inject_coin_rule_p:
                        chosen = rule_ops if kind == "lightning" else ()
                        inject_src = 1
                        st["fire_" + kind] += 1
            # 统计：本决策最终是不是"放闪电"（注入的 / AI 自己的分开计）
            n_light = sum(1 for k in chosen if int(k[0]) == int(OperationType.USE_LIGHTNING_STORM))
            if n_light:
                st["light_total"] += n_light
                st["light_injected" if inject_src else "light_ai"] += n_light
            st["decisions"] += 1
            st["times"].append(time.time() - _t_dec)
            # ── A：本决策的"候选 + 两阶段打分"快照（只有 unified/flat 搜索有；BundleMCTS 没有）──
            cand_rec = None
            if dump_cand and res is not None and hasattr(mcts, "last_cand_stats"):
                _cs = dict(getattr(mcts, "last_cand_stats", None) or {})
                _s1 = list(_cs.get("scores") or [])
                _s2 = getattr(mcts, "last_scores", None)
                _nc = len(bundles)
                _s2l = None
                if _s2 is not None:
                    _s2l = [float(x) for x in np.asarray(_s2, dtype=np.float64).reshape(-1).tolist()]
                    if len(_s2l) != _nc:      # 长度对不上就不硬塞（宁缺勿错）
                        _s2l = None
                cand_rec = {
                    "n_cand": int(_nc),
                    # 进阶段 2 的 top-k 候选（每个 bundle 是 op 三元组的列表；HOLD = []）
                    "cand": [[[int(t[0]), int(t[1]), int(t[2])] for t in (b or ())] for b in bundles],
                    # 阶段 1（idle rollout）分数，与 cand 逐条对齐
                    "s1": [float(x) for x in _s1[:_nc]] if _s1 else None,
                    # 阶段 2（flat 随机化 rollout）分数，与 cand 逐条对齐
                    "s2": _s2l,
                    "chosen_idx": int(res.chosen_index),
                    "chosen": [[int(t[0]), int(t[1]), int(t[2])] for t in (chosen or ())],
                    "root_value": float(res.root_value),
                    # 阶段 1 结束时的每类 Q[c]/N[c]（"合规类"可能 <24）
                    "Q": {str(k): round(float(v), 5) for k, v in (_cs.get("Q") or {}).items()},
                    "N": {str(k): int(v) for k, v in (_cs.get("Nmap") or {}).items()},
                    "N_total": int(_cs.get("N_total", 0)),
                    "has_hold": int(any(len(b or ()) == 0 for b in bundles)),
                    "n_upg": int(sum(1 for b in bundles if any(int(t[0]) == 12 for t in (b or ())))),
                }
            samples.append(
                {
                    "board": obs["board"].astype(np.float16),
                    "stats": obs["stats"].astype(np.float16),
                    "player": player,
                    "class_mask": cm,
                    "position_mask": pm,
                    "bundles": bundles,
                    "intent_counts": intent_counts,
                    "visit": visit,
                    "recorded_action_map": recorded_am,
                    "recorded_head_logits": recorded_hl,
                    "search_skipped": skip,
                    # 0 = AI 自己的选择；1 = 「金币×冷却」规则注入的（供按来源分层）
                    "inject_src": int(inject_src),
                    # A：候选 + 两阶段打分（None = 该决策没走搜索 / 不是 flat 家族搜索）
                    "cand_rec": cand_rec,
                    # ── 完整局面字段（按决策快照，供"一一对应"重建 + 实体模型）──
                    "towers": [[int(t.tower_id), int(t.player), int(t.x), int(t.y),
                                int(t.tower_type), int(t.hp)] for t in state.towers],
                    "ants": [[int(a.ant_id), int(a.player), int(a.x), int(a.y), int(a.hp),
                              int(a.max_hp), int(a.kind), int(a.age), int(a.behavior)]
                             for a in state.ants],
                    "effects": [[int(e.weapon_type), int(e.player), int(e.x), int(e.y),
                                 int(e.remaining_turns)] for e in state.active_effects],
                    "weapon_cd": [[int(x) for x in row] for row in state.weapon_cooldowns],
                    "weapon_usage": [int(x) for x in state.super_weapon_usage],
                    "coins": [int(x) for x in state.coins],
                    "hp_us": int(state.bases[player].hp),
                    "hp_opp": int(state.bases[1 - player].hp),
                    "die": [int(x) for x in getattr(state, "die_count", [0, 0])],
                    "old": [int(x) for x in getattr(state, "old_count", [0, 0])],
                    "round": int(state.round_index),
                }
            )
            ops = [
                Operation(OperationType(int(k[0])), int(k[1]), int(k[2]))
                for k in chosen
            ]
            round_ops[player] = [(int(o.op_type), o.arg0, o.arg1) for o in ops]
            state.apply_operation_list(player, ops)
        if player == 1 and not state.terminal:
            state.advance_round()
        # B：官方 replay 的每回合记录（完整 op 参数；空列表 = HOLD）。
        #   写的是**实际被执行**的 ops（含规则注入/随机注入的结果），所以重放出来的就是这一局。
        st["replay_steps"].append({
            "round": int(round_idx),
            "op0": [[int(t), int(a0), int(a1)] for (t, a0, a1) in round_ops[0]],
            "op1": [[int(t), int(a0), int(a1)] for (t, a0, a1) in round_ops[1]],
        })
        if pfile is not None:
            hp0 = state.bases[0].hp
            hp1 = state.bases[1].hp
            g0 = state.coins[0]
            g1 = state.coins[1]
            p0 = "[" + ", ".join(_fmt_op(*op) for op in round_ops[0]) + "]" if round_ops[0] else "[]"
            while len(p0) < 32:
                p0 += " "
            p1 = "[" + ", ".join(_fmt_op(*op) for op in round_ops[1]) + "]" if round_ops[1] else "[]"
            pfile.write(f"round={round_idx} HP0={hp0} HP1={hp1} G0={g0} G1={g1} "
                        f"P0={p0} P1={p1}\n")
            pfile.flush()

    diff = state.bases[0].hp - state.bases[1].hp
    v_p0 = float(np.clip(diff / HP_SCALE, -1.0, 1.0))
    if diff == 0 and state.winner is not None:
        v_p0 = 0.1 if state.winner == 0 else -0.1
    for s in samples:
        s["value_target"] = v_p0 if s["player"] == 0 else -v_p0
    if pfile is not None:
        pfile.write(f"terminal winner={state.winner} hp={[b.hp for b in state.bases]} "
                    f"rounds={state.round_index}\n")
        pfile.close()
    if stats_out is not None:
        st["rounds"] = int(state.round_index)
        # replay 的终局信息（官方 replay 语义里的 result 段）
        st["replay_result"] = {
            "winner": (None if state.winner is None else int(state.winner)),
            "hp": [int(state.bases[0].hp), int(state.bases[1].hp)],
            "rounds": int(state.round_index),
            "terminal": bool(state.terminal),
            "value_target_p0": float(v_p0),
        }
        stats_out.update(st)
    return samples


def _collect_and_save(seed: int, out_dir: str, ckpt_path: str, iterations: int,
                      max_depth_rounds: int, t_class: float, t_pos: float,
                      k: int, sample_mult: int, max_rounds: int, temp_rounds: int,
                      native_engine: bool = False, c_puct: float = 1.25,
                      random_action_prob: float = 0.0,
                      inject_example_prob: float = 0.0,
                      skip_hold_search: bool = False,
                      write_npz: bool = False,
                      search_mode: str = "joint", pos_pin: str = "argmax",
                      class_pin_random_prob: float = 0.0,
                      reserve_coins: int = 180,
                      skip_single_candidate: bool = False,
                      inject_coin_rule: bool = False,
                      inject_coin_rule_p: float = 0.5,
                      inject_coin_rule_cd: int = 5,
                      inject_coin_rule_hold_lo: int = 60,
                      search: str = "bundle",
                      search_total_rounds: int = 0,
                      cand_k: int = 48,
                      flat_rollouts: int = 64,
                      flat_horizon: int = 16,
                      flat_batch: int = 32,
                      flat_act_prob: float = 0.5,
                      flat_build_ratio: float = 0.7,
                      flat_mode: str = "random",
                      flat_random_side: str = "opponent",
                      flat_prob_mode: str = "decay",
                      flat_decay_c: float = 1.0,
                      flat_decay_k: float = 3.0,
                      flat_share_prefix: bool = True,
                      flat_seed: int = 53,
                      dump_cand: bool = True,
                      dump_replay: bool = True,
                    mc_horizon: int = 100, mc_every: int = 1,
                    mc_root_only: int = 0, mc_bidir: int = 0) -> dict:
    torch.set_num_threads(1)  # avoid thread thrash across parallel workers
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    model, net_fn = make_net_fn_from_ckpt(ckpt_path, feat)
    # ── 搜索器选择（2026-10-31；2026-11-01 修正）────────────────────────────────────
    #   bundle          = BundleMCTS（旧行为，逐位不变）
    #   flat            = FlatRolloutGreedy（部署的 pin 候选 flat 搜索）
    #   **unified       = 新版统一候选 `unified_cand.make_unified_searcher_class()`
    #                     （`UnifiedCandFlatSearch`：候选 = 类 UCB bandit + 位置游标，
    #                      评估 = flat 随机化 rollout）—— **这就是 `AZAI_CAND_MODE=unified`
    #                      在 `az_bridge_ai.py:1359-1361` 用的同一个入口**。
    #   unified-legacy  = 老版 `UnifiedActionSearch`（`unified_action_search.py`，
    #                     2026-10-31 之前的"统一动作搜索"；**不是**部署用的那个新版）
    #   ⚠️ flat/unified* 都要一个**真 torch 模块**当 value_model（`_leaf_values` 会调
    #      `.parameters()`），而 `make_net_fn_from_ckpt` 返回的 `ThreeNetPolicy` 门面没有
    #      ⇒ 这里单独把 value 网 load 出来（6.5 MB、一次性）。
    _cls_name = "BundleMCTS"
    _rounds = int(search_total_rounds if search_total_rounds > 0 else 49152)   # 部署 BASE: AZAI_FLAT_ROUNDS
    if search in ("flat", "unified", "unified-legacy"):
        _cm, _pm, _vm = load_three_models(ckpt_path)
        _vm.eval()
        # BundleMCTS 的那批公共参数（flat/unified 继承它；pin/unified 只用到其中一部分）
        _kw = dict(iterations=iterations, max_depth_rounds=max_depth_rounds,
                   sample_mult=sample_mult, t_class=t_class, t_pos=t_pos,
                   c_puct=c_puct, seed=seed, search_mode=search_mode,
                   pos_pin=pos_pin, class_pin_random_prob=class_pin_random_prob,
                   reserve_coins=reserve_coins,
                   mc_horizon=mc_horizon, mc_every=mc_every,
                   mc_root_only=mc_root_only, mc_bidir=mc_bidir,
                   skip_single_candidate=skip_single_candidate,
                   value_model=_vm, feature_extractor=feat, eval_mode="net")
        if search == "unified-legacy":
            from my_ai.az_intent.unified_action_search import UnifiedActionSearch
            _cls_name = "UnifiedActionSearch(legacy)"
            _k = dict(_kw)
            if search_total_rounds > 0:
                _k["total_rounds"] = int(search_total_rounds)   # 老版：total_rounds = 每决策预算
            mcts = UnifiedActionSearch(net_fn, k=k, **_k)
        else:
            from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
            if search == "unified":
                from my_ai.az_intent.unified_cand import make_unified_searcher_class
                _cls = make_unified_searcher_class()
                _cls_name = "UnifiedCandFlatSearch(unified_cand)"
            else:
                _cls = FlatRolloutGreedy
                _cls_name = "FlatRolloutGreedy(pin)"
            # flat 家族参数：**逐项对齐部署 BASE**（`_us_vs_external.sh:68` 的 `BASE` 字符串）：
            #   AZAI_FLAT_BATCH=32 / AZAI_FLAT_HORIZON=16 / AZAI_FLAT_SHARE_PREFIX=1 /
            #   AZAI_K=48 / AZAI_FLAT_ROLLOUTS=64 / AZAI_FLAT_ROLLOUT_MODE=random /
            #   AZAI_FLAT_RANDOM_SIDE=opponent / AZAI_FLAT_PROB_MODE=decay /
            #   AZAI_FLAT_DECAY_C=1.0 / AZAI_FLAT_DECAY_K=3.0 / AZAI_FLAT_ROUNDS=49152 /
            #   AZAI_FLAT_SEED=53
            #   ⚠️ 阶段 2 **必须是 random（对手随机出招）+ opponent-side + decay(1.0,3.0)**；
            #      `idle` 只是 `unified_cand` **阶段 1** 的候选打分方式（在 `cand_gen.rollout_fn` 里写死），
            #      与这里的 `rollout_mode` 无关。用错（idle/const/16 rollouts）会让评估器偏"少动"⇒ 对局失真。
            #   `k` 在 unified 下 = **候选条数**（AZAI_K=48）⇒ 用 cand_k，不沿用 bundle 的 k。
            mcts = _cls(
                net_fn,
                k=int(cand_k),
                rollouts_per_candidate=int(flat_rollouts),
                horizon=int(flat_horizon),
                value_batch=int(flat_batch),
                max_total_rounds=_rounds,
                verbose=False,
                rollout_mode=str(flat_mode),
                rollout_act_prob=float(flat_act_prob),
                rollout_build_ratio=float(flat_build_ratio),
                rollout_seed=int(flat_seed),
                rollout_random_side=str(flat_random_side),
                rollout_prob_mode=str(flat_prob_mode),
                rollout_decay_c=float(flat_decay_c),
                rollout_decay_k=float(flat_decay_k),
                rollout_share_prefix=bool(flat_share_prefix),
                rollout_dedupe_weight=False,
                **_kw)
        print(f"  [search] seed={seed} cls={_cls_name} k={cand_k} "
              f"rollouts={flat_rollouts} horizon={flat_horizon} batch={flat_batch} "
              f"rounds={_rounds if search != 'unified-legacy' else search_total_rounds} "
              f"mode={flat_mode} side={flat_random_side} prob={flat_prob_mode}"
              f"({flat_decay_c:g},{flat_decay_k:g}) seed={flat_seed} share_prefix={int(flat_share_prefix)} "
              f"cand_mode_env={__import__('os').environ.get('AZAI_CAND_MODE', '')!r} "
              f"cand_verbose={__import__('os').environ.get('AZAI_CAND_VERBOSE', '0')!r}",
              flush=True)
    elif search == "bundle":
        mcts = BundleMCTS(net_fn, iterations=iterations, max_depth_rounds=max_depth_rounds,
                          k=k, sample_mult=sample_mult, t_class=t_class, t_pos=t_pos,
                          c_puct=c_puct, seed=seed, search_mode=search_mode,
                          pos_pin=pos_pin, class_pin_random_prob=class_pin_random_prob,
                          reserve_coins=reserve_coins,
                          # ⚠️ 部署配置必须一起传：否则 mc_quiet 会在**每个节点**仲裁（~20 min/局）
                          mc_horizon=mc_horizon, mc_every=mc_every,
                          mc_root_only=mc_root_only, mc_bidir=mc_bidir,
                          skip_single_candidate=skip_single_candidate)
    else:
        raise SystemExit(f"--search {search!r} 不认识（bundle|flat|unified|unified-legacy）")
    gst: dict = {}
    samples = collect_game(net_fn, model, feat, mcts, seed,
                           max_rounds=max_rounds, temp_rounds=temp_rounds,
                           progress_path=str(Path(out_dir) / f"az_progress_seed{seed:05d}.txt"),
                           native_engine=native_engine,
                           random_action_prob=random_action_prob,
                           inject_example_prob=inject_example_prob,
                           skip_hold_search=skip_hold_search,
                           inject_coin_rule=inject_coin_rule,
                           inject_coin_rule_p=inject_coin_rule_p,
                           inject_coin_rule_cd=inject_coin_rule_cd,
                           inject_coin_rule_hold_lo=inject_coin_rule_hold_lo,
                           dump_cand=dump_cand,
                           stats_out=gst)
    path = Path(out_dir) / f"az_selfplay_seed{seed:05d}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_npz:
        # Light value-only format: drop bundles/visit/intent_counts/masks (heavy,
        # only needed for policy training).  The recorded anchors are kept AS-IS —
        # no re-forwarding — because they come from THIS collection model, which
        # for the iterative loop is the (fixed) policy we keep.
        import numpy as _np
        npz = path.with_suffix(".npz")
        if samples:
            # A：紧凑字段（完整候选列表在 `.cand.jsonl`；这里只放便于快速过滤的标量）
            _crecs = [s.get("cand_rec") for s in samples]
            _n_cand = _np.asarray([(c or {}).get("n_cand", -1) for c in _crecs], dtype=_np.int16)
            _ch_idx = _np.asarray([(c or {}).get("chosen_idx", -1) for c in _crecs], dtype=_np.int16)
            _ch_s1 = _np.full(len(samples), _np.nan, dtype=_np.float32)
            _ch_s2 = _np.full(len(samples), _np.nan, dtype=_np.float32)
            _root_v = _np.full(len(samples), _np.nan, dtype=_np.float32)
            for _i, _c in enumerate(_crecs):
                if not _c:
                    continue
                _root_v[_i] = _c.get("root_value", _np.nan)
                _k = _c.get("chosen_idx", -1)
                _s1, _s2 = _c.get("s1"), _c.get("s2")
                if _s1 is not None and 0 <= _k < len(_s1):
                    _ch_s1[_i] = _s1[_k]
                if _s2 is not None and 0 <= _k < len(_s2):
                    _ch_s2[_i] = _s2[_k]
            _np.savez_compressed(
                npz,
                board=_np.stack([s["board"] for s in samples], axis=0),
                stats=_np.stack([s["stats"] for s in samples], axis=0),
                action_map=_np.stack([s["recorded_action_map"] for s in samples], axis=0),
                head_logits=_np.stack([s["recorded_head_logits"] for s in samples], axis=0),
                player=_np.asarray([s["player"] for s in samples], dtype=_np.int8),
                value_target=_np.asarray([s["value_target"] for s in samples], dtype=_np.float32),
                inject_src=_np.asarray([s.get("inject_src", 0) for s in samples], dtype=_np.int8),
                search_skipped=_np.asarray([int(s.get("search_skipped", False)) for s in samples],
                                           dtype=_np.int8),
                n_cand=_n_cand, chosen_idx=_ch_idx,
                chosen_s1=_ch_s1, chosen_s2=_ch_s2, root_value=_root_v,
            )
            # 完整局面字段另存 JSONL（与 npz 逐条同序；含塔/蚁/效果/武器/金币/HP/死伤/回合）
            import json as _json
            ents_path = npz.with_suffix(".ents.jsonl")
            _keys = ("round", "player", "towers", "ants", "effects", "weapon_cd",
                     "weapon_usage", "coins", "hp_us", "hp_opp", "die", "old", "value_target",
                     "inject_src")
            with open(ents_path, "w", encoding="utf-8") as _fh:
                for _s in samples:
                    _fh.write(_json.dumps({_k: _s.get(_k) for _k in _keys},
                                          ensure_ascii=False) + "\n")
            # A：候选 + 两阶段打分（与 npz 行 1:1 同序；变长所以用 JSONL 而不是 npz 数组）
            if dump_cand:
                cand_path = npz.with_suffix(".cand.jsonl")
                with open(cand_path, "w", encoding="utf-8") as _fh:
                    for _s in samples:
                        _rec = {"round": _s.get("round"), "player": _s.get("player"),
                                "skipped": int(bool(_s.get("search_skipped"))),
                                "inject_src": int(_s.get("inject_src", 0))}
                        _rec.update(_s.get("cand_rec") or {})
                        _fh.write(_json.dumps(_rec, ensure_ascii=False) + "\n")
        path = npz
        # B：官方 replay（seed + 每回合双方完整 ops + 终局）⇒ 可用引擎逐位重建任意时刻
        if dump_replay:
            import json as _json
            rp = Path(out_dir) / f"az_selfplay_seed{seed:05d}.replay.json"
            _cfg = {"search": search, "cand_k": cand_k, "flat_rollouts": flat_rollouts,
                    "flat_horizon": flat_horizon, "flat_batch": flat_batch,
                    "flat_mode": flat_mode, "flat_random_side": flat_random_side,
                    "flat_prob_mode": flat_prob_mode, "flat_decay_c": flat_decay_c,
                    "flat_decay_k": flat_decay_k, "flat_share_prefix": bool(flat_share_prefix),
                    "flat_seed": flat_seed,
                    "rounds_budget": int(search_total_rounds if search_total_rounds > 0 else 49152),
                    "iterations": iterations, "max_depth_rounds": max_depth_rounds,
                    "native_engine": bool(native_engine), "max_rounds": max_rounds,
                    "temp_rounds": temp_rounds, "inject_coin_rule": bool(inject_coin_rule),
                    "inject_coin_rule_p": inject_coin_rule_p,
                    "inject_coin_rule_cd": inject_coin_rule_cd,
                    "inject_coin_rule_hold_lo": inject_coin_rule_hold_lo,
                    "skip_single_candidate": bool(skip_single_candidate),
                    "ckpt": str(ckpt_path)}
            _payload = {"format": "antwar2-replay/1", "engine": "cpp_native" if native_engine else "py_sdk",
                        "collector": "az_selfplay", "seed": int(seed),
                        "config": _cfg, "steps": gst.get("replay_steps", []),
                        "result": gst.get("replay_result", {})}
            rp.write_text(_json.dumps(_payload, ensure_ascii=False), encoding="utf-8")
    else:
        with open(path, "wb") as f:
            pickle.dump({"seed": seed, "samples": samples}, f)
    print(f"  [collect] seed={seed} samples={len(samples)} -> {path.name}", flush=True)
    _t = [float(x) for x in gst.get("times", [])]
    if _t:
        import statistics as _stat
        _t_sorted = sorted(_t)
        _p = lambda q: _t_sorted[min(len(_t_sorted) - 1, int(q * len(_t_sorted)))]  # noqa: E731
        _dec = max(gst.get("decisions", 1), 1)
        print(f"  [cost] seed={seed} 回合={gst.get('rounds',0)} 决策={_dec} "
              f"总={sum(_t)/60:.1f} min | 每决策 中位 {_stat.median(_t)*1000:.0f} ms "
              f"p95 {_p(0.95)*1000:.0f} ms 均值 {sum(_t)/_dec*1000:.0f} ms", flush=True)
    if inject_coin_rule or gst.get("light_total") or gst.get("coins_ge90"):
        _dec = max(gst.get("decisions", 1), 1)
        print(f"  [inject] seed={seed} 决策={_dec} "
              f"| 规则放闪电: 可注入 {gst.get('elig_lightning',0)} 次 → 命中 {gst.get('fire_lightning',0)}"
              f" | 规则攒钱(HOLD): 可注入 {gst.get('elig_hold',0)} 次 → 命中 {gst.get('fire_hold',0)}"
              f" | 闪电 op 总数 {gst.get('light_total',0)}（注入 {gst.get('light_injected',0)}"
              f" / AI 自己 {gst.get('light_ai',0)}）"
              f" | coins≥90 决策 {gst.get('coins_ge90',0)}/{_dec}"
              f"（{100.0*gst.get('coins_ge90',0)/_dec:.1f}%）", flush=True)
    # ── 游标推进深度汇总（`AZAI_CAND_VERBOSE=1` 时搜索器才累计；默认空 ⇒ 不打）──
    _agg = getattr(mcts, "cursor_agg", None) or {}
    if _agg:
        _steps_cap = int(getattr(getattr(mcts, "cand_gen", None), "steps_per_shape", 0) or 0)
        _rows = []
        for k, v in _agg.items():
            ts = [int(x) for x in v.get("t", [])]
            if not ts:
                continue
            ts_sorted = sorted(ts)
            med = ts_sorted[len(ts_sorted) // 2]
            p90 = ts_sorted[min(len(ts_sorted) - 1, int(0.9 * len(ts_sorted)))]
            at_cap = (sum(1 for x in ts if _steps_cap and x >= _steps_cap) / len(ts))
            _rows.append((v.get("n_trials", 0), k, v.get("n_dec", 0), med,
                          max(ts), p90, at_cap, len(ts)))
        _rows.sort(reverse=True)
        _hot = [r for r in _rows if r[0] >= 5]
        print(f"  [candcursor-agg] seed={seed} steps_per_shape={_steps_cap} "
              f"形状数={len(_rows)} 被选>=5步的形状={len(_hot)}", flush=True)
        print(f"    {'形状':<14} {'累计步':>7} {'决策数':>6} {'t中位':>6} {'t最大':>6} "
              f"{'t_p90':>6} {'卡满占比':>8}", flush=True)
        for n_tr, k, n_dec, med, mx, p90, at_cap, n in _rows[:20]:
            print(f"    {k:<14} {n_tr:>7} {n_dec:>6} {med:>6} {mx:>6} {p90:>6} {at_cap:>8.0%}",
                  flush=True)
        if _hot:
            _med_hot = sorted(r[3] for r in _hot)
            _at_cap_hot = sum(r[6] * r[7] for r in _hot) / max(sum(r[7] for r in _hot), 1)
            print(f"    ⇒ 被选>=5步的形状：t 中位={_med_hot[len(_med_hot)//2]} "
                  f"（t 的样本加权「卡满 steps_per_shape」占比={_at_cap_hot:.0%}）", flush=True)
    out = {"seed": seed, "samples": len(samples)}
    # 只把**标量**统计带回去（`times` 是 list，逐局已打印 p50/p95，不必回传）
    for _k, _v in gst.items():
        if isinstance(_v, bool) or not isinstance(_v, (int, float)):
            continue
        out[f"inj_{_k}"] = int(_v)
    return out


def collect_games_parallel(ckpt_path: str, seeds: list[int], out_dir: str, workers: int,
                           iterations: int, max_depth_rounds: int, t_class: float,
                           t_pos: float, k: int, sample_mult: int, max_rounds: int,
                           temp_rounds: int, native_engine: bool = False,
                           c_puct: float = 1.25,
                           random_action_prob: float = 0.0,
                           inject_example_prob: float = 0.0,
                           skip_hold_search: bool = False,
                           write_npz: bool = False,
                           search_mode: str = "joint", pos_pin: str = "argmax",
                           class_pin_random_prob: float = 0.0,
                           reserve_coins: int = 180,
                           skip_single_candidate: bool = False,
                           inject_coin_rule: bool = False,
                           inject_coin_rule_p: float = 0.5,
                           inject_coin_rule_cd: int = 5,
                           inject_coin_rule_hold_lo: int = 60,
                           search: str = "bundle",
                           search_total_rounds: int = 0,
                           cand_k: int = 48,
                           flat_rollouts: int = 64,
                           flat_horizon: int = 16,
                           flat_batch: int = 32,
                           flat_act_prob: float = 0.5,
                           flat_build_ratio: float = 0.7,
                           flat_mode: str = "random",
                           flat_random_side: str = "opponent",
                           flat_prob_mode: str = "decay",
                           flat_decay_c: float = 1.0,
                           flat_decay_k: float = 3.0,
                           flat_share_prefix: bool = True,
                           flat_seed: int = 53,
                           dump_cand: bool = True,
                           dump_replay: bool = True,
                           # ⚠️ 2026-10-07：mc_* 必须一路透传到 BundleMCTS，否则 mc_quiet 会在**每个节点**仲裁
                           mc_horizon: int = 100, mc_every: int = 1,
                           mc_root_only: int = 0, mc_bidir: int = 0) -> list[Path]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(s, out_dir, ckpt_path, iterations, max_depth_rounds, t_class, t_pos,
             k, sample_mult, max_rounds, temp_rounds, native_engine, c_puct,
             random_action_prob, inject_example_prob, skip_hold_search, write_npz,
             search_mode, pos_pin,
             class_pin_random_prob,
             reserve_coins,
             skip_single_candidate,
             inject_coin_rule, inject_coin_rule_p, inject_coin_rule_cd,
             inject_coin_rule_hold_lo, search, search_total_rounds,
             cand_k, flat_rollouts, flat_horizon, flat_batch, flat_act_prob, flat_build_ratio,
             flat_mode, flat_random_side, flat_prob_mode, flat_decay_c, flat_decay_k,
             flat_share_prefix, flat_seed, dump_cand, dump_replay,
             mc_horizon, mc_every, mc_root_only, mc_bidir) for s in seeds]
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.starmap(_collect_and_save, jobs)
    else:
        results = [_collect_and_save(*j) for j in jobs]
    total = sum(r["samples"] for r in results)
    print(f"[selfplay] collected {total} samples from {len(results)} games", flush=True)
    if any("inj_decisions" in r for r in results):
        keys = ("inj_decisions", "inj_elig_lightning", "inj_fire_lightning", "inj_elig_hold",
                "inj_fire_hold", "inj_light_total", "inj_light_injected", "inj_light_ai")
        agg = {k: sum(int(r.get(k, 0)) for r in results) for k in keys}
        print(f"[selfplay] 注入汇总：决策 {agg['inj_decisions']} | "
              f"放闪电 可注入 {agg['inj_elig_lightning']} → 命中 {agg['inj_fire_lightning']} | "
              f"攒钱 可注入 {agg['inj_elig_hold']} → 命中 {agg['inj_fire_hold']} | "
              f"闪电 op 总 {agg['inj_light_total']}（注入 {agg['inj_light_injected']}"
              f" / AI {agg['inj_light_ai']}）| 每局 {agg['inj_light_total']/max(len(results),1):.1f} 次",
              flush=True)
    return [Path(out_dir) / f"az_selfplay_seed{s:05d}.pkl" for s in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle-MCTS AlphaZero self-play collection (T1)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="training_history/az_intent/az_selfplay_data")
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--max-depth-rounds", type=int, default=4)
    parser.add_argument("--t-class", type=float, default=0.5)
    parser.add_argument("--t-pos", type=float, default=1.0,
                        help="position sampling temperature (after z-scoring over legal "
                             "cells).  Default 1.0 = start of the strength plateau; "
                             "0.3 is a very sharp prior that measures only 1.6%% vs rule_v4 "
                             "(vs 20.3%% at 1.0).  See docs/az_t_pos_default_fix.md")
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--sample-mult", type=int, default=15)
    parser.add_argument("--max-rounds", type=int, default=512)
    parser.add_argument("--temp-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native-engine", action="store_true",
                        help="use the C++ engine (native_game) instead of the Python SDK engine")
    parser.add_argument("--inject-example-prob", type=float, default=0.0,
                        help="x: with probability x OUR play ignores the search and plays "
                             "**ExampleAI 的动作**（动作目录里 score 最高的 bundle；合理动作、"
                             "大概率是建塔；不过滤）。与 --random-action-prob（均匀随机=纯噪声）"
                             "不同：它把对局推向'有塔的局面'给价值网/类头铺覆盖，又不打乱对局。"
                             "见 docs/az_class_axis_plan.md §5(a)")
    parser.add_argument("--random-action-prob", type=float, default=0.0,
                        help="x: with probability x a decision plays a uniformly random "
                             "LEGAL bundle instead of the search's choice (diversity / "
                             "decisive-game injection; keep small, e.g. 0.05)")
    parser.add_argument("--c-puct", type=float, default=1.25,
                        help="MCTS PUCT exploration constant (higher = more exploration)")
    parser.add_argument("--write-npz", action="store_true",
                        help="write a light value-only npz per game instead of the heavy pkl "
                             "(drops bundles/visit/masks; keeps the recorded anchors AS-IS). "
                             "Used by the iterative loop: az_selfplay npz -> merge -> train")
    parser.add_argument("--skip-hold-search", action="store_true",
                        help="skip the MCTS at decisions where the LEGAL class argmax is "
                             "HOLD for every head (the search can only pick HOLD there too; "
                             "measured 2 wrong skips in 10,002 decisions).  ~91%% of decisions "
                             "qualify, cutting collection cost by ~10x.  Value-only use.")
    parser.add_argument("--search-mode", type=str, default="joint",
                        choices=["joint", "class-only", "pos-only"],
                        help="bundle-MCTS candidate axis.  pos-only (pin the class to its "
                             "argmax, sample only positions) reproduces joint's strength "
                             "exactly (93.8%% vs raw, same 30W/2L) while giving the search a "
                             "single candidate on ~97%% of turns -> pairs with "
                             "skip_single_candidate for ~26x cheaper collection.  See "
                             "docs/az_forced_move_skip_plan.md")
    parser.add_argument("--pos-pin", type=str, default="argmax",
                        choices=["argmax", "playable", "reserve", "reserve_up",
                                 # ⚠️ 2026-10-07 补：BundleMCTS 早就支持这几个，但这里的 choices 忘了加
                                 #    ⇒ 无法用 `--pos-pin mc_quiet` 采集"部署策略"的数据（闭环重训的前提）
                                 "mc_light", "mc_quiet", "masked", "q"],
                        help="pos-only: which class to pin (argmax = raw argmax; "
                             "playable = highest-logit class that actually executes; "
                             "mc_quiet = 安静 rollout 的蒙特卡洛类决策〔部署用〕)")
    parser.add_argument("--class-pin-random-prob", type=float, default=0.0,
                        help="「随机钉类」(docs/az_posnet_random_class_plan.md)：pos-only 下**按头独立**"
                             "以该概率把钉住的类换成一个均匀随机的**合法非闪电类**"
                             "(c 不在 {超武17-20, HOLD23} 且 class_mask 与 position_mask 都允许)，"
                             "位置仍由搜索做价值排序。用途 = **采集**时给数据加'有塔局面'的多样性；"
                             "部署侧不要开。默认 0.0 ⇒ 行为与旧版逐位相同。建议 0.02。")
    parser.add_argument("--skip-single-candidate", action="store_true",
                        help="OPT-IN forced-move shortcut: when the root has a single candidate, "
                             "skip the (result-irrelevant) iterations.  Per-search bit-identical "
                             "chosen_bundle/bundles/visit_policy/intent_counts (verified on 660 "
                             "comparisons) and ~150x cheaper on those turns; in pos-only mode that "
                             "is ~97%% of turns => ~26x cheaper collection overall.  No effect in "
                             "joint mode (~3%% of turns).  NOTE: it changes which draws are taken "
                             "from mcts.rng afterwards, so realized games are NOT reproducible "
                             "across this flag (the decision distribution is unchanged).  See "
                             "docs/az_forced_move_skip_plan.md")
    parser.add_argument("--mc-horizon", type=int, default=100,
                        help="mc_quiet 的 rollout 回合上限（部署用 256）")
    parser.add_argument("--mc-every", type=int, default=1,
                        help="mc_quiet 的仲裁门控：每 N 回合在根节点比一次（部署用 5）")
    parser.add_argument("--mc-root-only", type=int, default=0,
                        help="1 = 只在搜索根节点仲裁（部署用 1；0 = 每个节点都仲裁，极慢）")
    parser.add_argument("--mc-bidir", type=int, default=0,
                        help="K>0 ⇒ K 步双向 rollout（预注册 prereg_20261006_rollout_ladder.md）")
    parser.add_argument("--reserve-coins", type=int, default=180,
                        help="pos_pin=reserve/reserve_up 时的闪电储备门槛（金币）。"
                             "用途：采集「带储备规则的类策略」数据（蒸馏用），部署侧不要开。")
    # ── 「金币 × 冷却」规则注入（2026-10-31，`docs/value_net_retrain_collection_plan_20261031.md`）──
    parser.add_argument("--inject-coin-rule", action="store_true",
                        help="按**手写规则**覆盖动作（造'花 vs 攒'的分叉对照；只用于值网数据采集）：\n"
                             "  · coins ≥ 90 且 闪电 cd == 0 ⇒ 强制放 LIGHTNING_STORM\n"
                             "  · coins ∈ [hold_lo, 90) 且 cd ≤ --inject-coin-rule-cd ⇒ 强制 HOLD（攒钱）\n"
                             "以概率 p 覆盖（--inject-coin-rule-p）。默认关 ⇒ 行为与旧版逐位相同。")
    parser.add_argument("--inject-coin-rule-p", type=float, default=0.5,
                        help="规则命中的决策里，真的覆盖动作的概率。0.5 = 保留一半'AI 乱花钱'的负例。")
    parser.add_argument("--inject-coin-rule-cd", type=int, default=5,
                        help="'闪电 cd 将好'的阈值（回合）：cd ≤ 该值才触发'攒钱'规则。0 = 只在已好时攒。")
    parser.add_argument("--inject-coin-rule-hold-lo", type=int, default=60,
                        help="'攒钱'规则的下界（金币）：coins ∈ [lo, 90) 才触发。")
    parser.add_argument("--search", type=str, default="bundle",
                        choices=["bundle", "flat", "unified", "unified-legacy"],
                        help="用哪种搜索采集（默认 bundle = 旧行为/逐位不变）。\n"
                             "  **unified** = 新版统一候选 `unified_cand.make_unified_searcher_class()`\n"
                             "   （= `AZAI_CAND_MODE=unified` 在 bridge 里用的同一个类；阶段 2 是 flat rollout）；\n"
                             "  flat = FlatRolloutGreedy（pin 候选）；\n"
                             "  unified-legacy = 老版 `UnifiedActionSearch`（2026-10-31 之前那个，**不是**部署用的）。")
    parser.add_argument("--search-total-rounds", type=int, default=0,
                        help="flat/unified：每决策 rollout 回合预算（0 = 部署默认 39000 ≈ 2.45 s/决策）；"
                             "unified-legacy：每决策总回合预算（0 = 类默认）。冒烟可设 2000~4000 换速度。")
    parser.add_argument("--cand-k", type=int, default=48,
                        help="unified 的**候选条数** k（bridge 默认 48；**不是** bundle 的束长 --k）。")
    parser.add_argument("--flat-rollouts", type=int, default=64,
                        help="flat/unified 的每候选 rollout 次数（部署 BASE: AZAI_FLAT_ROLLOUTS=64）")
    parser.add_argument("--flat-horizon", type=int, default=16,
                        help="flat/unified 的 rollout 视界（部署 BASE: AZAI_FLAT_HORIZON=16）")
    parser.add_argument("--flat-batch", type=int, default=32,
                        help="flat/unified 的叶值网批大小（部署 BASE: AZAI_FLAT_BATCH=32）")
    parser.add_argument("--flat-act-prob", type=float, default=0.5,
                        help="rollout 里每回合每方出招概率（部署默认 0.5；prob_mode=const 时才用）")
    parser.add_argument("--flat-build-ratio", type=float, default=0.7,
                        help="rollout 出招时选建塔的比例（部署默认 0.7）")
    parser.add_argument("--flat-mode", type=str, default="random", choices=["idle", "random"],
                        help="阶段 2 rollout 的出招模式（部署 BASE: AZAI_FLAT_ROLLOUT_MODE=**random**）。\n"
                             "⚠️ idle 只是 unified_cand **阶段 1** 的候选打分方式（写死在 cand_gen 里），\n"
                             "   阶段 2 用 idle 会让评估器偏'不动' ⇒ 对局失真。")
    parser.add_argument("--flat-random-side", type=str, default="opponent", choices=["opponent", "both"],
                        help="random 模式下谁随机出招（部署 BASE: AZAI_FLAT_RANDOM_SIDE=opponent）")
    parser.add_argument("--flat-prob-mode", type=str, default="decay",
                        choices=["const", "decay", "first", "reverse"],
                        help="出招概率随回合的衰减方式（部署 BASE: AZAI_FLAT_PROB_MODE=decay）")
    parser.add_argument("--flat-decay-c", type=float, default=1.0,
                        help="decay 的首回合概率 C（部署 BASE: AZAI_FLAT_DECAY_C=1.0）")
    parser.add_argument("--flat-decay-k", type=float, default=3.0,
                        help="decay 的常数 k：p(r)=C·exp(-r/k)（部署 BASE: AZAI_FLAT_DECAY_K=3.0）")
    parser.add_argument("--flat-share-prefix", type=int, default=1, choices=[0, 1],
                        help="rollout 前缀树共享（部署 BASE: AZAI_FLAT_SHARE_PREFIX=1）")
    parser.add_argument("--flat-seed", type=int, default=53,
                        help="rollout 随机种子（部署 BASE: AZAI_FLAT_SEED=53）")
    # ── A：每决策的候选 + 两阶段打分落盘（默认开；只在 unified/flat 下有内容）──
    parser.add_argument("--dump-cand", type=int, default=1, choices=[0, 1],
                        help="1（默认）= 每局额外写 `.cand.jsonl`（候选 + 阶段1/阶段2 分 + Q/N + chosen），"
                             "并在 npz 里加 n_cand / chosen_idx / chosen_s1 / chosen_s2 / root_value 紧凑字段")
    # ── B：官方 replay（seed + 每回合双方 ops）落盘（默认开）──
    parser.add_argument("--dump-replay", type=int, default=1, choices=[0, 1],
                        help="1（默认）= 每局写 `az_selfplay_seedNNNNN.replay.json`（官方 replay 语义："
                             "seed + 每回合双方完整 op 序列 + 终局），可用 code/test_match/verify_replay.py 逐位重放校验")
    args = parser.parse_args()

    seeds = [args.seed * 10000 + g for g in range(args.games)]
    print(f"[selfplay] collecting {args.games} games ({args.workers} workers, "
          f"{args.iterations} iters / depth {args.max_depth_rounds}) "
          f"[engine={'C++' if args.native_engine else 'python'}, c_puct={args.c_puct}, "
          f"random_action_prob={args.random_action_prob}, "
          f"inject_example_prob={args.inject_example_prob}, "
          f"search={args.search}, inject_coin_rule={args.inject_coin_rule}"
          f"(p={args.inject_coin_rule_p}, cd≤{args.inject_coin_rule_cd}, "
          f"hold_lo={args.inject_coin_rule_hold_lo})]...", flush=True)
    paths = collect_games_parallel(
        args.checkpoint, seeds, args.out_dir, args.workers,
        args.iterations, args.max_depth_rounds, args.t_class, args.t_pos,
        args.k, args.sample_mult, args.max_rounds, args.temp_rounds,
        native_engine=args.native_engine, c_puct=args.c_puct,
        random_action_prob=args.random_action_prob,
        inject_example_prob=args.inject_example_prob,
        skip_hold_search=args.skip_hold_search,
        write_npz=args.write_npz,
        mc_horizon=args.mc_horizon, mc_every=args.mc_every,
        mc_root_only=args.mc_root_only, mc_bidir=args.mc_bidir,
        search_mode=args.search_mode,
        pos_pin=args.pos_pin,
        class_pin_random_prob=args.class_pin_random_prob,
        reserve_coins=args.reserve_coins,
        skip_single_candidate=args.skip_single_candidate,
        inject_coin_rule=args.inject_coin_rule,
        inject_coin_rule_p=args.inject_coin_rule_p,
        inject_coin_rule_cd=args.inject_coin_rule_cd,
        inject_coin_rule_hold_lo=args.inject_coin_rule_hold_lo,
        search=args.search,
        search_total_rounds=args.search_total_rounds,
        cand_k=args.cand_k,
        flat_rollouts=args.flat_rollouts,
        flat_horizon=args.flat_horizon,
        flat_batch=args.flat_batch,
        flat_act_prob=args.flat_act_prob,
        flat_build_ratio=args.flat_build_ratio,
        flat_mode=args.flat_mode,
        flat_random_side=args.flat_random_side,
        flat_prob_mode=args.flat_prob_mode,
        flat_decay_c=args.flat_decay_c,
        flat_decay_k=args.flat_decay_k,
        flat_share_prefix=bool(args.flat_share_prefix),
        flat_seed=args.flat_seed,
        dump_cand=bool(args.dump_cand),
        dump_replay=bool(args.dump_replay),
    )
    print(f"[selfplay] done -> {len(paths)} files", flush=True)


if __name__ == "__main__":
    main()
