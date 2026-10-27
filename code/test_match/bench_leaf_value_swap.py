#!/usr/bin/env python3
"""**同一局面**上对比"现役值网 vs 小值网(32/3)"的 rollout 叶子评估耗时（2026-10-27）。

## 为什么必须"同一局面 + 同进程背靠背"

值网前向只占决策预算的一部分，`_leaf_values` 里还有 `encode_observation`（每叶固定成本），
所以"前向 5.1×"**不可能**等比例变成"叶子 5.1×"（`docs/small_value_net_20261027.md` §4 的警告）。
要回答"同样的时间是能多放采样还是多放候选"，只能在**真实部署配置**下量。

做法：
  * 局面来自**真实对局**的 `--trace-ops` 日志（`match_results/ladder_logs/<tag>/<seed>.log`），
    按 bridge 的确切顺序重放 ⇒ 有塔/有蚁/有钱的**中局**，不是"空棋盘"；
  * 两侧**同一个进程、同一个 searcher 构造参数、同一个 rollout_seed** ⇒
    rollout 采到的**叶子完全相同**（`_decision_idx` 每次由新建 searcher 归 1），
    差别只有叶子值网本身 ⇒ 干净的配对比较；
  * 交替先后顺序、`--reps` 次取**中位**（机器上还有 8 局对局在跑，绝对值偏慢且抖，比值才是可用量）。

## 三类耗时（分别记，不混在一起）

| 计时项 | 怎么来的 |
|---|---|
| `total` | `search()` 全程 |
| `leaf` | 包 `FlatRolloutGreedy._leaf_values`（含 `encode_observation` + 前向 + 后处理） |
| `net`  | 只包**值网对象**的 `__call__`（纯前向，把每叶固定成本分离出来）⇒ `leaf - net` = 特征抽取等 |
| `sim`  | `total - leaf`（clone/apply/advance/前缀树） |

另记节点数 / 去重叶子数：包 `_tree_rollout_value` 累加 `last_tree_stats`
（`AZAI_ROLLOUT_SAMPLES=1` 时才有：`(节点数, 叶子数, 去重叶子数)`）。

用法：
  python code/test_match/bench_leaf_value_swap.py --tag VS_RU_dump --seeds 8 9 \\
      --scan-rounds 60 100 140 180 220 260 300 --n-pos 3 --reps 3 \\
      --student-ckpt checkpoints/small32_3/v_32_3_folded.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402

# 记录 `last_tree_stats`（节点数/去重叶子数）必须开这个；两侧同开 ⇒ 不影响比值
os.environ.setdefault("AZAI_ROLLOUT_SAMPLES", "1")

from my_ai.az_intent.az_selfplay import load_three_models, make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.bundle_mcts import BundleNode                                  # noqa: E402
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy                   # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade                       # noqa: E402
from my_ai.az_intent.value_override import load_override, describe                  # noqa: E402
from SDK.utils.features import FeatureExtractor                                     # noqa: E402

# ── 部署配置（与正在跑的 `rollout_strength` 实验的 `DECAY_C=1.0` 臂逐字一致）──────────
FLAT_KW = dict(
    rollouts_per_candidate=64,     # N
    horizon=8,                     # h
    max_total_rounds=49920,        # 49920 = k(48)*N(64)*h(8)
    value_batch=32,
    value_tanh=True,               # AZAI_VALUE_TANH=1
    eval_mode="net",
    rollout_mode="random",
    rollout_random_side="opponent",
    rollout_prob_mode="decay",
    rollout_decay_c=1.0,
    rollout_decay_k=3.0,
    rollout_share_prefix=True,
    rollout_act_prob=0.3,
    rollout_build_ratio=0.7,
    rollout_seed=4104,
)
MCTS_KW = dict(
    iterations=256, max_depth_rounds=4, sample_mult=15, seed=20261027,
    search_mode="pos-only", pos_pin="argmax", t_class=0.5, t_pos=1.0, c_puct=1.25,
    skip_single_candidate=False,
)

# ── 计时累加器（monkeypatch 的包装函数写这里，每测一次清零）──────────────────────────
_ACC: dict = {}


def _reset() -> None:
    _ACC.clear()
    _ACC.update(total_s=0.0, leaf_s=0.0, net_s=0.0, net_calls=0, net_rows=0,
                leaves=0, nodes=0, leaf_states=0, uniq=0, tree_calls=0)


_reset()

_ORIG_LEAF = FlatRolloutGreedy._leaf_values
_ORIG_TREE = FlatRolloutGreedy._tree_rollout_value


def _timed_leaf(self, states, root_player):
    t0 = time.perf_counter()
    out = _ORIG_LEAF(self, states, root_player)
    _ACC["leaf_s"] += time.perf_counter() - t0
    _ACC["leaves"] += len(states)
    return out


def _timed_tree(self, child, player, rngs):
    out = _ORIG_TREE(self, child, player, rngs)
    _st = getattr(self, "last_tree_stats", None)
    _ACC["tree_calls"] += 1
    if _st:
        _ACC["nodes"] += int(_st[0])
        _ACC["leaf_states"] += int(_st[1])
        _ACC["uniq"] += int(_st[2])
    return out


FlatRolloutGreedy._leaf_values = _timed_leaf            # type: ignore[assignment]
FlatRolloutGreedy._tree_rollout_value = _timed_tree     # type: ignore[assignment]


class TimedModel:
    """只给"值网对象"计时（把 `encode_observation` 的固定成本分离出来）。其余属性透传。"""

    def __init__(self, model):
        self._m = model

    def __call__(self, board, stats):
        t0 = time.perf_counter()
        out = self._m(board, stats)
        _ACC["net_s"] += time.perf_counter() - t0       # CPU 前向是同步的 ⇒ 计时有效
        _ACC["net_calls"] += 1
        _ACC["net_rows"] += int(board.shape[0])
        return out

    def parameters(self, *a, **k):
        return self._m.parameters(*a, **k)

    def __getattr__(self, name):
        return getattr(self._m, name)


def _load_rsc():
    spec = importlib.util.spec_from_file_location("_repro_search_crash",
                                                  REPO / "code/test_match/repro_search_crash.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_repro_search_crash"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _replay_snapshots(rsc, tag: str, seed: int, rounds: list[int]):
    """按 bridge 顺序重放该局，返回 [(round, player, state)]（每回合两个座位各一份）。"""
    rounds_ops = rsc.read_ops(tag, f"{seed}.log")
    if not rounds_ops:
        raise SystemExit(f"{tag}/{seed}.log 里没有 --trace-ops 行")
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    want = sorted(set(int(r) for r in rounds))
    snaps = []
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    for r in range(0, max(want) + 1):
        if r in want:
            snaps.append((r, 0, st.clone()))
        for pl in (0, 1):
            ops = rounds_ops.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
            if pl == 0 and r in want:
                snaps.append((r, 1, st.clone()))
        st.advance_round()
    return snaps


def _build_searcher(net_fn, feat, value_model, *, k: int, n: int, rounds: int):
    kw = dict(MCTS_KW)
    kw["k"] = int(k)
    flat = dict(FLAT_KW)
    flat["rollouts_per_candidate"] = int(n)
    flat["max_total_rounds"] = int(rounds)
    return FlatRolloutGreedy(net_fn, value_model=TimedModel(value_model),
                             feature_extractor=feat, **flat, **kw)


def _count_candidates(net_fn, feat, value_model, state, player, k: int) -> int:
    """只跑一次 `_expand`（不 rollout）⇒ 该局面在网络候选数（用来挑"候选饱和"的局面）。"""
    s = _build_searcher(net_fn, feat, value_model, k=k, n=1, rounds=10 ** 9)
    root = BundleNode(state=state.clone(), player=player)
    s._mc_root = root
    s._decision_idx = 1
    s._expand(root, int(k))
    return len(root.bundles)


def _measure(net_fn, feat, model_, state, player, *, k: int, n: int, rounds: int) -> dict:
    _reset()
    s = _build_searcher(net_fn, feat, model_, k=k, n=n, rounds=rounds)
    t0 = time.perf_counter()
    res = s.search(state.clone(), int(player), temperature=0.0)
    total = time.perf_counter() - t0
    return dict(
        total_s=total, leaf_s=_ACC["leaf_s"], net_s=_ACC["net_s"],
        sim_s=total - _ACC["leaf_s"],
        net_calls=_ACC["net_calls"], net_rows=_ACC["net_rows"],
        leaves=_ACC["leaves"], nodes=_ACC["nodes"], leaf_states=_ACC["leaf_states"],
        uniq=_ACC["uniq"], tree_calls=_ACC["tree_calls"],
        n_cand=len(res.bundles), n_roll=s.last_rollouts, rounds=s.last_rounds,
        expand_s=float((s.last_timing or {}).get("expand_s", 0.0)),
        chosen=res.chosen_index, root_value=float(res.root_value),
        effective_samples=int(s.last_effective_samples or 0),
    )


def _agg(runs: list[dict]) -> dict:
    """逐项取中位（只有 total 需要中位；其余取中位以保证与 total 同一次运行的代表性）。"""
    out = {}
    for key in ("total_s", "leaf_s", "net_s", "sim_s", "nodes", "uniq", "leaves",
                "net_calls", "net_rows", "expand_s", "n_cand", "n_roll", "rounds",
                "effective_samples", "tree_calls", "leaf_states"):
        vals = [r[key] for r in runs]
        out[key] = float(st.median(vals)) if vals else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="VS_RU_dump")
    ap.add_argument("--seeds", type=int, nargs="+", default=[8, 9])
    ap.add_argument("--scan-rounds", type=int, nargs="+",
                    default=[60, 100, 140, 180, 220, 260, 300])
    ap.add_argument("--positions", default="",
                    help="直接指定局面 'seed:round:player,...'（默认空 ⇒ 按候选数自动挑）")
    ap.add_argument("--n-pos", type=int, default=3)
    ap.add_argument("--min-cand", type=int, default=40,
                    help="自动挑局面时要求的最少候选数（k=48 ⇒ 饱和≈48）")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sweep-reps", type=int, default=2,
                    help="双点斜率/验证用的重复次数（只有主表要求 ≥3）")
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=49920)
    ap.add_argument("--teacher-ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    ap.add_argument("--student-ckpt", default="checkpoints/small32_3/v_32_3_folded.pt")
    ap.add_argument("--threads", type=int, default=1, help="部署口径 = 1（AZAI_TORCH_THREADS）")
    ap.add_argument("--sweep", type=int, default=1, help="1 ⇒ 额外跑 k=96 / N=128 双点斜率")
    ap.add_argument("--budget-check", type=int, default=0,
                    help=">0 ⇒ 同一进程内交替测「老师 N=--n vs 小网 N=该值」，"
                         "用来在**同一次运行**里验证等预算换算（机器负载会漂移 ±60%%，跨运行不可比）")
    ap.add_argument("--out-json", default="training_history/logs/bench_leaf_value_swap.json")
    a = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, a.threads))
    print(f"[bench] torch={torch.__version__} threads={torch.get_num_threads()} "
          f"| 部署配置 k={a.k} N={a.n} h={FLAT_KW['horizon']} rounds={a.rounds} "
          f"decay=({FLAT_KW['rollout_decay_c']},{FLAT_KW['rollout_decay_k']}) "
          f"share_prefix={FLAT_KW['rollout_share_prefix']} batch={FLAT_KW['value_batch']}", flush=True)

    tc = Path(a.teacher_ckpt)
    tc = tc if tc.is_absolute() else REPO / tc
    sc = Path(a.student_ckpt)
    sc = sc if sc.is_absolute() else REPO / sc
    if not sc.exists():
        raise SystemExit(f"学生 ckpt 不存在: {sc}（先跑 fold_small_value_ckpt.py）")

    feat = FeatureExtractor(max_actions=96)
    _cm, _pm, teacher = load_three_models(str(tc))
    teacher.eval()
    student = load_override(str(sc))
    student.net.eval()
    print(f"[bench] 现役(老师) {tc.name}: AntWarNetwork value_state "
          f"params={sum(p.numel() for p in teacher.parameters()):,}", flush=True)
    print(f"[bench] 小网(学生) {sc.name}: {describe(student)}", flush=True)
    _, net_fn = make_net_fn_from_ckpt(str(tc), feat, value_tanh=True, value_rel_to_abs=0.0)

    # ── ① 取局面（真实对局重放）────────────────────────────────────────────
    rsc = _load_rsc()
    snaps = []
    for seed in a.seeds:
        for (rnd, pl, state) in _replay_snapshots(rsc, a.tag, seed, a.scan_rounds):
            snaps.append((f"seed{seed}_r{rnd}_p{pl}", seed, rnd, pl, state))
    print(f"[bench] 重放 {a.tag} seeds={a.seeds} ⇒ {len(snaps)} 个候选局面", flush=True)

    if a.positions:
        want = [tuple(int(x) for x in p.split(":")) for p in a.positions.split(",") if p.strip()]
        picked = []
        for (seed, rnd, pl) in want:
            hit = [s for s in snaps if s[1] == seed and s[2] == rnd and s[3] == pl]
            if not hit:
                raise SystemExit(f"--positions 里的 {seed}:{rnd}:{pl} 不在重放结果里")
            picked.append(hit[0])
    else:
        print(f"[bench] 扫候选数（只跑 _expand，不 rollout；k={a.k}）……", flush=True)
        scan = []
        for (label, seed, rnd, pl, state) in snaps:
            nc = _count_candidates(net_fn, feat, teacher, state, int(pl), a.k)
            scan.append((label, seed, rnd, pl, state, nc))
            print(f"   {label:<18} 候选={nc}", flush=True)
        ok = [x for x in scan if x[5] >= a.min_cand]
        pool = ok if len(ok) >= a.n_pos else scan
        # 确定性挑选：候选多的优先（并列按 label）
        pool.sort(key=lambda x: (-x[5], x[0]))
        picked = [(x[0], x[1], x[2], x[3], x[4]) for x in pool[:a.n_pos]]
        print(f"[bench] 自动挑出 {len(picked)} 个局面（候选≥{a.min_cand} 的优先）: "
              f"{[p[0] for p in picked]}", flush=True)

    pos = [(lbl, state, int(pl)) for (lbl, _s, _r, pl, state) in picked]

    # ── ② 主表：部署配置（k=48, N=64）────────────────────────────────────────
    def run_cfg(name: str, k: int, n: int, rounds: int, arms, reps: int) -> dict:
        out = {arm: [] for arm in arms}
        for rep in range(reps):
            order = list(arms) if rep % 2 == 0 else list(reversed(arms))
            for arm in order:
                model_ = teacher if arm == "teacher" else student
                for (lbl, state, pl) in pos:
                    r = _measure(net_fn, feat, model_, state, pl, k=k, n=n, rounds=rounds)
                    r.update(pos=lbl, arm=arm, rep=rep)
                    out[arm].append(r)
                    print(f"  [{name}] rep{rep} {arm:<8} {lbl:<18} "
                          f"总 {r['total_s']*1000:8.0f} ms | 叶块 {r['leaf_s']*1000:8.0f} "
                          f"| 值网 {r['net_s']*1000:8.0f} | 模拟 {r['sim_s']*1000:8.0f} "
                          f"| 候选 {r['n_cand']:.0f} N={r['n_roll']:.0f} 叶子 {r['leaves']:.0f} "
                          f"去重 {r['uniq']:.0f} 节点 {r['nodes']:.0f}", flush=True)
        return out

    print(f"\n===== A. 主表：k={a.k}, N={a.n}, h={FLAT_KW['horizon']}, "
          f"rounds={a.rounds}（{len(pos)} 局面 × {a.reps} 重复，交替先后）=====", flush=True)
    main_runs = run_cfg("main", a.k, a.n, a.rounds, ("teacher", "student"), a.reps)

    def med_by_pos(runs: list[dict]) -> dict:
        out = {}
        for (lbl, _st, _pl) in pos:
            out[lbl] = _agg([r for r in runs if r["pos"] == lbl])
        out["ALL"] = {k: float(st.median([out[l][k] for l in out])) for k in out[next(iter(out))]}
        return out

    main_med = {arm: med_by_pos(main_runs[arm]) for arm in ("teacher", "student")}

    print(f"\n{'局面':<18} {'臂':<9} {'总ms':>8} {'叶块ms':>8} {'值网ms':>8} "
          f"{'模拟ms':>8} {'值网%':>6} {'叶块%':>6} {'候选':>4} {'叶子':>5} {'去重':>5} {'节点':>5}")
    for lbl in list(main_med["teacher"].keys()):
        for arm in ("teacher", "student"):
            m = main_med[arm][lbl]
            tot = max(1e-9, m["total_s"])
            print(f"{lbl:<18} {arm:<9} {m['total_s']*1000:8.0f} {m['leaf_s']*1000:8.0f} "
                  f"{m['net_s']*1000:8.0f} {m['sim_s']*1000:8.0f} "
                  f"{100*m['net_s']/tot:5.1f}% {100*m['leaf_s']/tot:5.1f}% "
                  f"{m['n_cand']:4.0f} {m['leaves']:5.0f} {m['uniq']:5.0f} {m['nodes']:5.0f}")

    T_t = main_med["teacher"]["ALL"]["total_s"]
    T_s = main_med["student"]["ALL"]["total_s"]
    net_t = main_med["teacher"]["ALL"]["net_s"]
    net_s = main_med["student"]["ALL"]["net_s"]
    print(f"\n[ALL 中位] 总耗时：老师 {T_t*1000:.0f} ms → 小网 {T_s*1000:.0f} ms "
          f"（加速 {T_t/max(1e-9,T_s):.2f}×）")
    print(f"[ALL 中位] 值网：老师 {net_t*1000:.0f} ms（{100*net_t/T_t:.1f}%）→ "
          f"小网 {net_s*1000:.0f} ms（{100*net_s/T_s:.1f}%），纯前向加速 "
          f"{net_t/max(1e-9,net_s):.2f}×")
    print(f"[ALL 中位] 叶块（含特征抽取）：老师 "
          f"{main_med['teacher']['ALL']['leaf_s']*1000:.0f} ms / "
          f"小网 {main_med['student']['ALL']['leaf_s']*1000:.0f} ms ⇒ "
          f"每叶固定成本 ≈ {(main_med['student']['ALL']['leaf_s']-main_med['student']['ALL']['net_s'])*1000/max(1,main_med['student']['ALL']['leaves']):.3f} ms")

    # ── ③ 双点斜率：k=96 / N=128（同一批局面，两侧都测）──────────────────────
    res_all = {"config": dict(k=a.k, n=a.n, rounds=a.rounds, h=FLAT_KW["horizon"],
                              threads=a.threads, tag=a.tag, seeds=a.seeds,
                              positions=[p[0] for p in pos], reps=a.reps),
               "main": main_med}
    slopes = {}
    if a.sweep:
        print(f"\n===== B. 双点斜率：k={a.k}→{2*a.k}（N={a.n} 不变）=====", flush=True)
        k96 = run_cfg("k96", 2 * a.k, a.n, a.rounds, ("teacher", "student"), a.sweep_reps)
        k96_med = {arm: med_by_pos(k96[arm]) for arm in ("teacher", "student")}
        print(f"\n===== C. 双点斜率：N={a.n}→{2*a.n}（k={a.k} 不变）=====", flush=True)
        n128 = run_cfg("N128", a.k, 2 * a.n, 10 ** 9, ("teacher", "student"), a.sweep_reps)
        n128_med = {arm: med_by_pos(n128[arm]) for arm in ("teacher", "student")}
        res_all["k96"], res_all["N128"] = k96_med, n128_med

        for arm in ("teacher", "student"):
            # 每个局面各自拟合 a + b*x，再取中位斜率（对外部负载更稳健）
            bk, bn = [], []
            for lbl in main_med[arm]:
                if lbl == "ALL":
                    continue
                m0, m1 = main_med[arm][lbl], k96_med[arm][lbl]
                dk = m1["n_cand"] - m0["n_cand"]
                if dk > 0:
                    bk.append((m1["total_s"] - m0["total_s"]) / dk)
                m2 = n128_med[arm][lbl]
                dn = m2["n_cand"] * m2["n_roll"] - m0["n_cand"] * m0["n_roll"]
                if dn > 0:
                    bn.append((m2["total_s"] - m0["total_s"]) / dn)
            slopes[arm] = {"per_candidate_s": float(st.median(bk)) if bk else float("nan"),
                           "per_sample_s": float(st.median(bn)) if bn else float("nan")}
        print(f"\n[斜率] 老师：每多一个候选 {slopes['teacher']['per_candidate_s']*1000:.1f} ms，"
              f"每多一个采样 {slopes['teacher']['per_sample_s']*1000:.2f} ms")
        print(f"[斜率] 小网：每多一个候选 {slopes['student']['per_candidate_s']*1000:.1f} ms，"
              f"每多一个采样 {slopes['student']['per_sample_s']*1000:.2f} ms")
        res_all["slopes"] = slopes

        # ── ④ 换算：同样的时间预算，小网能多放多少 N / k ──────────────────────
        print("\n===== D. 换算（同一时间预算）=====")
        conv = {}
        for lbl in main_med["teacher"]:
            if lbl == "ALL":
                continue
            t_t, t_s = main_med["teacher"][lbl]["total_s"], main_med["student"][lbl]["total_s"]
            nc0 = main_med["student"][lbl]["n_cand"]
            n0 = main_med["student"][lbl]["n_roll"]
            bn = slopes["student"]["per_sample_s"]
            bk = slopes["student"]["per_candidate_s"]
            save = t_t - t_s
            extra_samples = save / bn if bn and bn == bn else float("nan")
            d_n = extra_samples / max(1.0, nc0) if extra_samples == extra_samples else float("nan")
            d_k = extra_samples / max(1.0, n0) if extra_samples == extra_samples else float("nan")
            conv[lbl] = dict(t_teacher=t_t, t_student=t_s, save_s=save,
                             n_teacher=n0, n_star=n0 + d_n,
                             k_teacher=nc0, k_star=nc0 + d_k)
            print(f"  {lbl:<18} 省 {save*1000:7.0f} ms ⇒ N: {n0:.0f} → {n0+d_n:.0f}"
                  f"（+{d_n:.0f}） | k: {nc0:.0f} → {nc0+d_k:.0f}（+{d_k:.0f}）")
        res_all["conversion"] = conv

        # 经验验证：把学生放到 N=N*，看总耗时是否≈老师（而不是只靠外推）
        n_star = st.median([v["n_star"] for v in conv.values()])
        n_star_i = int(max(a.n + 8, round(n_star / 8) * 8))
        print(f"\n===== E. 验证外推：小网 N={n_star_i}（外推 N*≈{n_star:.0f}）vs 老师 N={a.n} =====",
              flush=True)
        verify = []
        for rep in range(a.sweep_reps):
            for (lbl, state, pl) in pos:
                r = _measure(net_fn, feat, student, state, pl, k=a.k, n=n_star_i, rounds=10 ** 9)
                r.update(pos=lbl, arm="student", rep=rep)
                verify.append(r)
                print(f"  [N*] rep{rep} student  {lbl:<18} 总 {r['total_s']*1000:8.0f} ms "
                      f"| N={r['n_roll']:.0f} 叶子 {r['leaves']:.0f} "
                      f"| 老师同局面 {main_med['teacher'][lbl]['total_s']*1000:8.0f} ms",
                      flush=True)
        v_med = med_by_pos(verify)["ALL"]["total_s"]
        print(f"[N*] 小网 N={n_star_i} 总耗时中位 {v_med*1000:.0f} ms vs 老师 N={a.n} "
              f"{T_t*1000:.0f} ms ⇒ 比值 {v_med/T_t:.2f}（≈1.0 就说明外推可信）")
        res_all["verify_Nstar"] = dict(n_star=n_star_i, total_s=v_med,
                                       teacher_total_s=T_t, ratio=v_med / T_t,
                                       runs=verify)

    # ── ⑤ 同预算自洽检查（可选，同一进程内交替）─────────────────────────────
    if a.budget_check:
        nb = int(a.budget_check)
        print(f"\n===== F. 同预算自洽检查：**同一进程内交替**测 "
              f"老师 N={a.n} / 小网 N={nb} =====", flush=True)
        rows_bc = []
        for rep in range(a.reps):
            seq = [("teacher", a.n), ("student", nb), ("student", a.n), ("teacher", nb)]
            if rep % 2:
                seq = list(reversed(seq))
            for (arm, nn) in seq:
                model_ = teacher if arm == "teacher" else student
                for (lbl, state, pl) in pos:
                    r = _measure(net_fn, feat, model_, state, pl, k=a.k, n=nn, rounds=10 ** 9)
                    r.update(pos=lbl, arm=arm, rep=rep, n_req=nn)
                    rows_bc.append(r)
                    print(f"  [bc] rep{rep} {arm:<8} N={nn:<4} {lbl:<18} "
                          f"总 {r['total_s']*1000:8.0f} ms | 值网 {r['net_s']*1000:7.0f} "
                          f"| 叶子 {r['leaves']:.0f}", flush=True)
        bc = {}
        for (arm, nn) in (("teacher", a.n), ("student", a.n), ("student", nb), ("teacher", nb)):
            sel = [r["total_s"] for r in rows_bc if r["arm"] == arm and r["n_req"] == nn]
            bc[f"{arm}_N{nn}"] = float(st.median(sel)) if sel else float("nan")
        print(f"\n  [bc] 中位：老师 N={a.n} = {bc[f'teacher_N{a.n}']*1000:.0f} ms | "
              f"小网 N={a.n} = {bc[f'student_N{a.n}']*1000:.0f} ms | "
              f"小网 N={nb} = {bc[f'student_N{nb}']*1000:.0f} ms | "
              f"老师 N={nb} = {bc[f'teacher_N{nb}']*1000:.0f} ms")
        print(f"  [bc] 同预算（老师 N={a.n}）下小网 N={nb} 的比值 = "
              f"{bc[f'student_N{nb}'] / bc[f'teacher_N{a.n}']:.3f}（≈1.0 ⇒ 等价）")
        res_all["budget_check"] = dict(n_base=a.n, n_star=nb, medians_s=bc, rows=rows_bc)

    # ── ⑥ 落盘 ────────────────────────────────────────────────────────────
    out = Path(a.out_json)
    out = out if out.is_absolute() else REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res_all, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print(f"\n[bench] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
