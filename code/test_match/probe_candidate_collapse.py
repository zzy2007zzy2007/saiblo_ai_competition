"""**只读探针**：48 个候选束（`AZAI_K=48`）为什么在 66.5% 的决策里"去重后只剩 1 条"？

把现役配置的候选构造管线拆成三段，**在真实对局重放出来的中局状态上**逐步计数：

  (a) 类选择    —— `bundle_mcts.py:695-912` 的 MC 菜单（`mc_quiet` 仲裁）里到底有几个**不同的类**？
                    （`_econ_rank` 排了 17 个类，但只有 top-1 进菜单；`mc_cells>0` 那条分支追加的是
                     `(同一个类, 不同格)`，并且 `_ccls = int(_c[0])`（`:858-870`）把 `(x,y)` **丢掉**）
  (b) 位置枚举  —— `_sample_position`（`decoder.py:231-274`）在**该类的全部合法格**上抽样，
                    720 次采样（`k=48 × sample_mult=15`）能解出几个**不同的 bundle**？
  (c) 去重后    —— `bundle_mcts.py:1048-1061`：`agg[key]`（key = 全部 op 三元组的元组）→ top-k 截断，
                    这就是**送进 flat rollout 评估器的**候选数（= `[upg] sel … n_cand=` 那个数）。

另外给三组"放开后能到多少"的估算（**不跑对局、不改现役代码**）：
  * 只放开类（允许前 K 个经济类）：`|∪_{c∈topK} keys(c)|`
  * 只放开位置（单一类，但采满）：`|keys(c*)|`（= 现在的行为，用来证明"位置不是瓶颈"）
  * 全部合法类 0..16：`|∪_{c 合法} keys(c)|`
  * 上界：引擎认定的**合法原子 op 数**（建塔格 + 升级 + 降级 + 闪电）

设计纪律（`docs/` 里所有探针都遵守）：
  * **不改** `code/my_ai/**`；所有"如果那样会怎样"都在本进程内用**新 rng** 调**原函数**
    （`bundle_mcts.sample_bundle` / `decoder.make_position_masks`），不碰 `mcts.rng`。
  * "现役值"必须来自**真实代码路径**：直接调 `mcts._expand(root, mcts.k)` 然后读 `root.bundles`
    （这正是 `FlatRolloutGreedy.search()` 的第一步，`flat_rollout_greedy.py:492`），
    只省掉后面的 64×16×n_cand 回合 rollout（对候选集**零影响**）。
  * 监控点用**猴补**：`bundle_mcts.sample_bundle`（记录每次采样的 class_ids + 解出的 key）、
    `my_ai.az_intent.mc_quiet.mc_quiet_choose`（记录 MC 菜单）。两者都是 `_expand` 里
    **调用时**才 import 的模块属性 ⇒ 补丁生效且不改源码。

用法：
    python code/test_match/probe_candidate_collapse.py --tag INJECT_VS_RU_B \
        --games 7,10,12 --max-dec 12 --json match_results/collapse/probe_B.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO, REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402

from SDK.backend.model import Operation                                    # noqa: E402
from SDK.utils.constants import OperationType, TOWER_UPGRADE_TREE          # noqa: E402
from my_ai.decoder import (                                                # noqa: E402
    CHANNEL_TO_TOWER_TYPE,
    make_class_mask,
    make_position_masks,
)
from my_ai.az_intent.game_state_facade import GameStateFacade              # noqa: E402

UPG = int(OperationType.UPGRADE_TOWER)
BLD = int(OperationType.BUILD_TOWER)
DNG = int(OperationType.DOWNGRADE_TOWER)
LGT = int(OperationType.USE_LIGHTNING_STORM)

OPLINE_RE = re.compile(r"\[round (\d+)\] (p[01])\(\S+?\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OPOBJ_RE = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")
TOWERS_RE = re.compile(r"^\s*\[round (\d+)\] towers=\[(.*)\]\s*$")


# ────────────────────────── 日志解析 / 重放（与 probe_upgrade_class_axis.py 同口径）──
def parse_ladder(path: Path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    ops: dict[int, list[list[tuple[int, int, int]]]] = {}
    for m in OPLINE_RE.finditer(txt):
        rnd, side = int(m.group(1)), (0 if m.group(2) == "p0" else 1)
        row = ops.setdefault(rnd, [[], []])
        row[side] = [(int(o.group(2)), int(o.group(3)), int(o.group(4)))
                     for o in OPOBJ_RE.finditer(m.group(4))]
    tws: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    for line in txt.splitlines():
        m = TOWERS_RE.match(line)
        if not m:
            continue
        rows = []
        for t in re.finditer(r"\(([^()]*)\)", m.group(2)):
            parts = [p.strip() for p in t.group(1).split(",")]
            if len(parts) >= 6:
                rows.append(tuple(int(float(p)) for p in parts[:6]))
        tws.setdefault(int(m.group(1)), []).extend(rows)
    return ops, tws


def our_side(token: str) -> int:
    return 1 if token.strip().endswith("r") else 0


# ────────────────────────── 猴补：记录 (b) 段的每次采样 ──────────────────────────
class Rec:
    """一次 `_expand` 期间所有观测。"""

    def __init__(self):
        self.samples: list[tuple[tuple, tuple]] = []   # (class_ids 三元组, 解出的 key)
        self.mc: list[dict] = []                       # MC 菜单记录
        self.enabled = False

    def reset(self):
        self.samples = []
        self.mc = []


REC = Rec()


def install_patches():
    from my_ai.az_intent import bundle_mcts as BM
    from my_ai.az_intent import mc_quiet as MQ

    _orig_sb = BM.sample_bundle
    _orig_mc = MQ.mc_quiet_choose

    def sb_wrap(net_out, state, player, **kw):
        ops, intents = _orig_sb(net_out, state, player, **kw)
        if REC.enabled:
            # `_expand` 传的是 `[cid[idx] for cid in class_ids]` = 3 个**类号 int**（`bundle_mcts.py:1046`）
            cid = kw.get("class_ids")
            key = tuple((int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops)
            REC.samples.append((tuple(int(x) for x in cid) if cid else (), key))
        return ops, intents

    def mc_wrap(state, player, cand_classes, *a, **kw):
        best, scores = _orig_mc(state, player, cand_classes, *a, **kw)
        if REC.enabled:
            try:
                _sc = {str(k): float(v) for k, v in dict(scores).items()}
            except Exception:  # noqa: BLE001
                _sc = {}
            REC.mc.append({"cands": [list(c) if isinstance(c, (tuple, list)) else int(c)
                                     for c in cand_classes],
                           "best": (list(best) if isinstance(best, (tuple, list)) else best),
                           "scores": _sc})
        return best, scores

    BM.sample_bundle = sb_wrap
    MQ.mc_quiet_choose = mc_wrap
    return _orig_sb


# ────────────────────────── (b) 段的"每类能产出多少不同 bundle" ──────────────────
def keys_for_class(orig_sb, net_out, st, side, cid: int, n: int, seed: int,
                   pos_mask, cls_mask) -> set:
    """把整束**钉在单个类**上，采 n 次，返回解出的**不同 key** 集合（真实解码器口径）。"""
    rng = np.random.default_rng(int(seed))
    out = set()
    for _ in range(int(n)):
        ops, _ = orig_sb(net_out, st, side, t_class=0.5, t_pos=1.0, rng=rng,
                         intent_decoding=True, position_mask=pos_mask, class_mask=cls_mask,
                         class_probs=None, class_ids=[int(cid)] * 3)
        out.add(tuple((int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops))
    return out


def op_kind(key: tuple) -> str:
    """把一个 bundle 归类成字符串，用于报告里的"样例"。"""
    if not key:
        return "HOLD(空束)"
    names = {BLD: "BUILD", UPG: "UPGRADE", DNG: "DOWNGRADE", LGT: "LIGHTNING",
             21: "LIGHT", 31: "BASE_UP"}
    return "+".join(names.get(t, str(t)) for (t, _a0, _a1) in key)


# ────────────────────────── 主流程 ──────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="INJECT_VS_RU_B")
    ap.add_argument("--games", default="7,10,12")
    ap.add_argument("--ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    ap.add_argument("--max-dec", type=int, default=12, help="每局最多分析几个中局决策点")
    ap.add_argument("--r0", type=int, default=40, help="从第几回合开始取中局")
    ap.add_argument("--step", type=int, default=12, help="每 N 回合取一个决策点（1 = 全回合）")
    ap.add_argument("--whatif-n", type=int, default=96, help="what-if 每类采样次数")
    ap.add_argument("--no-whatif", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cpu")
    # 单变量消融（默认 = BASE）：用来**实测**某个开关在这条路径上到底生不生效
    ap.add_argument("--mc-cells", type=int, default=4, help="BASE=4；0 ⇒ 菜单不含 (类,格)")
    ap.add_argument("--tclass", type=float, default=0.5, help="BASE=0.5；改大看类温度是否生效")
    ap.add_argument("--pospins", default="mc_quiet", help="BASE=mc_quiet")
    ap.add_argument("--mc-nodown", type=int, default=0, help="1 ⇒ 经济类排序里排除类 16（降级）")
    ap.add_argument("--mc-no-light", type=int, default=0, help="1 ⇒ 闪电不进 MC 菜单")
    ap.add_argument("--pin-k", type=int, default=0,
                    help="`AZAI_PIN_K`：>1 ⇒ 整束允许落在 K 个不同类上（0 = 现役单类）")
    args = ap.parse_args()

    ladder = REPO / "match_results" / "ladder_logs" / args.tag
    if not ladder.is_dir():
        print(f"未找到 {ladder}")
        return 1
    tokens = [t.strip() for t in args.games.split(",") if t.strip()]

    import torch
    torch.set_num_threads(4)
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
    from my_ai.az_intent.value_batch import make_batched_value_fn
    from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy

    feat = FeatureExtractor(max_actions=96)
    ckpt = Path(args.ckpt)
    ckpt = ckpt if ckpt.is_absolute() else (REPO / ckpt)
    model, net_fn = make_net_fn_from_ckpt(str(ckpt), feat, value_tanh=True)
    _, _, value_model = load_three_models(str(ckpt))
    value_model.eval()
    mc_value_fn = make_batched_value_fn(value_model, feat, batch_size=32)
    print(f"[net] ckpt={ckpt.name}  device={args.device}  threads=4")

    # ⚠️ 与 `code/test_match/_inject_vs_runnerup.sh:48` 的 BASE **逐字对应**
    #    （只改 k/预算无关项：iterations 在 flat 下不用；rollouts 不是候选集的一部分）
    mcts = FlatRolloutGreedy(
        net_fn,
        rollouts_per_candidate=64,
        horizon=16,
        value_model=value_model,
        feature_extractor=feat,
        value_batch=32,
        eval_mode="net",
        max_total_rounds=49152,
        rollout_mode="random",
        rollout_act_prob=0.5,
        rollout_build_ratio=0.7,
        rollout_seed=53,
        rollout_random_side="opponent",
        rollout_prob_mode="decay",
        rollout_decay_c=1.0,
        rollout_decay_k=3.0,
        rollout_share_prefix=True,
        rollout_dedupe_weight=False,
        # ↓ 与 BASE 的 AZAI_* 一一对应
        iterations=256,          # AZAI_ITERS
        max_depth_rounds=4,      # AZAI_DEPTH
        k=48,                    # AZAI_K
        sample_mult=15,          # AZAI_SAMPLE_MULT
        t_class=args.tclass,     # AZAI_TCLASS
        t_pos=1.0,               # AZAI_TPOS
        seed=0,
        search_mode="pos-only",  # AZAI_MODE
        pos_pin=args.pospins,    # AZAI_POSPIN
        skip_single_candidate=True,   # AZAI_SKIP1（flat 下不生效，照配）
        mc_horizon=16,           # AZAI_MC_HORIZON
        mc_bidir=0,
        mc_every=1,              # AZAI_MC_EVERY
        mc_no_downgrade=int(args.mc_nodown),
        mc_m4=0,
        mc_root_only=1,          # AZAI_MC_ROOTONLY
        mc_top2=0,
        mc_topk=0,
        mc_cells=int(args.mc_cells),   # AZAI_MC_CELLS
        mc_upg=0,                # AZAI_MC_UPG 未设
        mc_margin=0.0,
        mc_score="net",          # AZAI_MC_SCORE
        mc_selfspend=0,
        mc_selfspend_mix=1.0,
        mc_net_samples=4,        # AZAI_MC_NET_SAMPLES
        mc_upgrade_bonus=0.0,
        mc_upg_targets=1,
        mc_no_light=bool(int(args.mc_no_light)),
        mc_value_fn=mc_value_fn,
        pin_k=int(args.pin_k),   # AZAI_PIN_K（默认 0 = 现役单类行为）
    )
    print(f"[cfg] k={mcts.k} sample_mult={mcts.sample_mult} "
          f"n_samples={mcts.k * mcts.sample_mult} mode={mcts.search_mode} "
          f"pos_pin={mcts.pos_pin} mc_cells={mcts.mc_cells} mc_score={mcts.mc_score}")

    orig_sb = install_patches()

    rows: list[dict] = []
    games_out: dict[str, dict] = {}
    t_all = time.perf_counter()

    for tok in tokens:
        side = our_side(tok)
        logp = ladder / f"{tok}.log"
        if not logp.is_file():
            print(f"[!] 缺 {logp}")
            continue
        ops, tws = parse_ladder(logp)
        if not ops:
            print(f"[!] {logp.name} 没有 --trace-ops 行")
            continue
        seed = int(re.sub(r"[^0-9]", "", tok) or "0")
        st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
        n_mismatch = 0
        n_analyzed = 0
        cand_hist: Counter = Counter()
        for rnd in sorted(ops):
            if st.terminal:
                break
            if rnd in tws:
                want = sorted((t[0], t[1], t[2], t[3], int(t[4])) for t in tws[rnd])
                got = sorted((int(t.tower_id), int(t.player), int(t.x), int(t.y),
                              int(t.tower_type)) for t in st.towers)
                if want != got:
                    n_mismatch += 1
            take = (rnd >= args.r0 and n_analyzed < args.max_dec
                    and (rnd - args.r0) % max(1, args.step) == 0)
            if take:
                row = analyze_decision(st, side, rnd, mcts, net_fn, feat,
                                       orig_sb, args, len(rows))
                if row is not None:
                    row["token"] = tok
                    row["side"] = side
                    # 保真度自检：对局日志里**我方这一回合真的发了什么**
                    row["log_ops"] = [list(map(int, o)) for o in ops[rnd][side]]
                    if row["c"]["n_cand"] == 1:
                        # n_cand==1 ⇒ 评估器没有选择余地 ⇒ 重放出的候选**必须**等于实战动作
                        row["replay_match"] = (tuple(tuple(int(x) for x in o)
                                                     for o in row["c"]["bundles"][0])
                                               == tuple(tuple(int(x) for x in o)
                                                        for o in row["log_ops"]))
                    rows.append(row)
                    n_analyzed += 1
                    cand_hist[row["c"]["n_cand"]] += 1
                    _d = row["d"]
                    print(f"  [{tok} r{rnd:>3}] coins={row['coins']:>3} 塔{row['n_towers']}"
                          f" 菜单={row['a']['n_menu_raw']}(类{row['a']['n_menu_classes']},"
                          f"格{row['a']['n_menu_cells']}) MC选={row['a']['chosen']} "
                          f"pin={row['a']['pinned']} tie={row['a'].get('mc_tie')} "
                          f"→ 每类不同bundle={row['b']['per_class']} "
                          f"⇒ n_cand={row['c']['n_cand']} "
                          f"whatif top1={_d.get('topK1')}(类{_d.get('top1_cls')}) "
                          f"top3={_d.get('topK3')} top5={_d.get('topK5')} "
                          f"all={_d.get('all')} 原子={_d.get('n_legal_atomic')}", flush=True)
            # 推进（与对局日志同序：P0 → P1 → advance）
            for pl in (0, 1):
                pl_ops = ops[rnd][pl]
                if pl_ops:
                    st.apply_operation_list(pl, [Operation(OperationType(int(t)), int(a0), int(a1))
                                                 for (t, a0, a1) in pl_ops])
            st.advance_round()
        games_out[tok] = {"rounds": len(ops), "tower_mismatch": n_mismatch,
                          "analyzed": n_analyzed,
                          "n_cand_hist": {str(k): v for k, v in cand_hist.items()}}
        print(f"[game {tok}] 重放回合={len(ops)} 塔态不符={n_mismatch} 分析决策点={n_analyzed}",
              flush=True)

    print(f"\n[time] 总耗时 {time.perf_counter() - t_all:.1f} s，决策点 {len(rows)}")
    summarize(rows, args)
    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"tag": args.tag, "games": games_out, "rows": rows,
                                 "whatif_n": args.whatif_n},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[json] -> {p}")
    return 0


def analyze_decision(st, side: int, rnd: int, mcts, net_fn, feat, orig_sb, args, idx: int):
    """在一个真实中局决策点上做 (a)(b)(c) 三段 + what-if。**只读**。"""
    pos_mask = make_position_masks(st, side, intent_decoding=True)
    cls_mask = make_class_mask(st, side, position_mask=pos_mask, intent_decoding=True)
    towers = st.towers_of(side)
    coins = int(st.coins[side])
    if int(cls_mask[23]) is False:      # 理论不可能（HOLD 恒真）
        return None

    # ── 走**真实代码路径**建候选集（= FlatRolloutGreedy.search() 的第一行）──
    from my_ai.az_intent.bundle_mcts import BundleNode
    root = BundleNode(state=st.clone(), player=side)
    mcts._mc_root = root
    REC.reset()
    REC.enabled = True
    t0 = time.perf_counter()
    try:
        mcts._expand(root, mcts.k)
    finally:
        REC.enabled = False
    dt = time.perf_counter() - t0

    bundles = list(root.bundles)
    n_cand = len(bundles)

    # ── (a) 类选择：MC 菜单里有几个**不同的类** ──
    a = {"n_menu_raw": 0, "n_menu_classes": 0, "n_menu_cells": 0,
         "menu": [], "pinned": None, "chosen": None, "mc_ran": bool(REC.mc)}
    if REC.mc:
        m0 = REC.mc[-1]
        menu = m0["cands"]
        a["n_menu_raw"] = len(menu)
        a["menu"] = menu
        classes = []
        cells = set()
        for c in menu:
            if isinstance(c, list):
                classes.append(int(c[0]))
                if len(c) >= 3:
                    cells.add((int(c[0]), int(c[1]), int(c[2])))
            else:
                classes.append(int(c))
        a["n_menu_classes"] = len(set(classes))
        a["n_menu_cells"] = len(cells)
        b = m0["best"]
        a["chosen"] = (int(b[0]) if isinstance(b, list) else b)
        a["mc_scores_n"] = len(m0["scores"])
        a["mc_tie"] = (len({round(v, 12) for v in m0["scores"].values()}) == 1
                       if m0["scores"] else False)
    # 真正钉住的类（从采样记录里读，最可靠）
    if REC.samples:
        cids = Counter(s[0] for s in REC.samples)
        top = cids.most_common(1)[0][0]
        a["pinned"] = int(top[0]) if len(set(top)) == 1 else f"mixed{list(top)}"
        a["pin_hist"] = {str(k): v for k, v in cids.items()}

    # ── (b) 位置枚举：每类能产出几个**不同** bundle（本决策的真实 720 次采样）──
    per_class: dict[int, int] = {}
    cls_keys: dict[int, set] = {}
    for cid, key in REC.samples:
        c = int(cid[0]) if cid else -1
        cls_keys.setdefault(c, set()).add(key)
    for c, ks in cls_keys.items():
        per_class[c] = len(ks)
    legal_cells = {int(c): int(pos_mask[c].sum()) for c in range(24) if pos_mask[c].any()}

    # ── (c) 去重后 ──
    samp_keys = set(k for _c, k in REC.samples)
    csec = {"n_cand": n_cand,
            "n_unique_raw": len(samp_keys),
            "n_samples": len(REC.samples),
            "collapse_from": len(samp_keys) - n_cand,
            "bundles": [list(map(list, b)) for b in bundles[:8]],
            "kinds": op_kind(tuple(bundles[0])) if bundles else "EMPTY",
            "expand_s": round(dt, 3)}
    # 被 "去重/k 截断" 丢掉的样例：`bundle_mcts.py:1049` 只累积计数，
    # `:1061` 按**出现频次**留前 k 个 ⇒ 剩下的（合法但稀有的）束被扔掉。这里把它们打出来。
    _cnt = Counter(key for _c, key in REC.samples)
    _kept = set(tuple(b) for b in bundles)
    _dropped = [(key, n) for key, n in _cnt.most_common() if key not in _kept]
    csec["dropped"] = [[list(map(list, key)), int(n)] for key, n in _dropped[:20]]
    csec["n_dropped"] = len(_dropped)

    # ── (d) what-if 估算（用同一个 net_out，新 rng，不碰 mcts.rng）──
    d: dict = {}
    if not args.no_whatif:
        net_out = net_fn(st, side)
        # 经济类排序（复刻 bundle_mcts.py:700-712，决定"若允许 K 个类"先加谁）
        am = np.asarray(net_out["action_map"], dtype=np.float32)
        rank = []
        for c in range(0, 17):
            if cls_mask[c] and pos_mask[c].any():
                v = float(np.where(pos_mask[c], am[c], -np.inf).max())
                if np.isfinite(v):
                    rank.append((v, int(c)))
        rank.sort(key=lambda x: -x[0])
        order = [c for _v, c in rank]
        all_legal = [int(c) for c in range(24) if cls_mask[c]]
        cache: dict[int, set] = {}

        def keys_of(c):
            if c not in cache:
                cache[c] = keys_for_class(orig_sb, net_out, st, side, c, args.whatif_n,
                                          1000 * idx + c, pos_mask, cls_mask)
            return cache[c]

        if order:
            for K in (1, 2, 3, 5):
                u = set()
                for c in order[:K]:
                    u |= keys_of(c)
                d[f"topK{K}"] = min(mcts.k, len(u))
                d[f"topK{K}_raw"] = len(u)
            d["top1"] = d.get("topK1")
            d["top1_cls"] = order[0]
            # ★ 忠实模拟 `AZAI_PIN_K=K`：类序列 = [MC 的赢家] + [经济类排序]（与补丁同序），
            #   **总采样预算不变**（720），平摊到 K 个类 ⇒ 每类 720//K 次 → 并集 → 按 k=48 截断
            N = mcts.k * mcts.sample_mult
            _pin0 = a.get("pinned")
            _base = [int(_pin0)] if isinstance(_pin0, int) else []
            _full = _base + [c for c in order if c not in _base]
            for K in (2, 3, 5):
                cls = _full[:K]
                if len(cls) < 2:
                    continue
                m = max(1, N // len(cls))
                u = set()
                for c in cls:
                    u |= keys_for_class(orig_sb, net_out, st, side, c, m, 2000 * idx + c,
                                        pos_mask, cls_mask)
                d[f"pinK{K}"] = min(mcts.k, len(u))
                d[f"pinK{K}_raw"] = len(u)
                d[f"pinK{K}_cls"] = cls
            # 全部合法类（0..16 + 17）：同样 720 平摊
            cls = [c for c in all_legal if c <= 17]
            if cls:
                m = max(1, N // len(cls))
                u = set()
                for c in cls:
                    u |= keys_for_class(orig_sb, net_out, st, side, c, m, 3000 * idx + c,
                                        pos_mask, cls_mask)
                d["pinAll"] = min(mcts.k, len(u))
                d["pinAll_raw"] = len(u)
                d["pinAll_K"] = len(cls)
        # 单类最优（= "位置枚举放开"的上界；证明位置不是瓶颈）
        if all_legal:
            per = {c: len(keys_of(c)) for c in all_legal if c <= 17}
            if per:
                d["single_class_best"] = max(per.values())
                d["single_class_best_cls"] = max(per, key=lambda c: per[c])
        # 全部合法塔类 0..16（+闪电）
        u = set()
        for c in all_legal:
            if c <= 17:
                u |= keys_of(c)
        d["all"] = min(mcts.k, len(u))
        d["all_raw"] = len(u)
        d["classes_used"] = len([c for c in all_legal if c <= 17])
        # 上界：引擎认定的**合法原子 op** 数（含升级，用 can_apply_dry 逐条问）
        n_atomic = 0
        for x in range(19):
            for y in range(19):
                if pos_mask[0][x, y]:
                    n_atomic += 1
        n_upg = 0
        for tw in towers:
            for tgt in TOWER_UPGRADE_TREE.get(tw.tower_type, ()):
                if coins < st.upgrade_tower_cost(tgt):
                    continue
                op = Operation(OperationType(UPG), int(tw.tower_id), int(tgt))
                try:
                    if st.can_apply_operation(side, op):
                        n_upg += 1
                except Exception:  # noqa: BLE001
                    pass
        d["n_legal_atomic"] = n_atomic + n_upg + len(towers)  # 建塔格 + 升级 + 降级
        d["n_atomic_build"] = n_atomic
        d["n_atomic_upg"] = n_upg
        d["n_atomic_down"] = len(towers)
        d["econ_order"] = order[:6]

    return {"round": int(rnd), "coins": coins, "n_towers": len(towers),
            "tower_types": [int(t.tower_type) for t in towers],
            "hp": [int(b.hp) for b in st.bases],
            "legal_cells": legal_cells,
            "a": a, "b": {"per_class": per_class}, "c": csec, "d": d}


def summarize(rows: list[dict], args) -> None:
    if not rows:
        print("[!] 没有样本")
        return
    n = len(rows)
    print("\n" + "=" * 96)
    print(f"§1 三段候选数（真实中局 n={n} 个决策点，配置 = _inject_vs_runnerup.sh:48 的 BASE）")
    print("-" * 96)
    h_a = Counter(r["a"]["n_menu_classes"] for r in rows)
    h_menu = Counter(r["a"]["n_menu_raw"] for r in rows)
    h_b = Counter(max(r["b"]["per_class"].values()) if r["b"]["per_class"] else 0 for r in rows)
    h_c = Counter(r["c"]["n_cand"] for r in rows)
    print(f"(a) MC 菜单里**不同类**的个数：{dict(sorted(h_a.items()))}"
          f"   （菜单原始长度：{dict(sorted(h_menu.items()))}）")
    print(f"    ⇒ 菜单中位数 = {sorted(h_menu.elements())[n // 2]} 项，但其中不同类中位数 = "
          f"{sorted(h_a.elements())[n // 2]} 个")
    print(f"(b) 被钉住的类**一次决策内**能解出的不同 bundle 数：{dict(sorted(h_b.items()))}")
    print(f"(c) 去重后送进 flat rollout 的候选数 n_cand：{dict(sorted(h_c.items()))}")
    print(f"    ⇒ n_cand==1 占比 {100.0 * h_c.get(1, 0) / n:.1f}%   中位数 = "
          f"{sorted(h_c.elements())[n // 2]}")
    pin = Counter(str(r["a"]["pinned"]) for r in rows)
    print(f"    被钉住的类（采样记录里 720 次全同）：{dict(pin.most_common())}")
    ch = Counter(str(r["a"]["chosen"]) for r in rows)
    print(f"    MC 仲裁的赢家（`mc_quiet_choose` 的 best）：{dict(ch.most_common())}")
    ties = sum(1 for r in rows if r["a"].get("mc_tie"))
    print(f"    MC 全部候选同分（完全平局，`max()` 取菜单第一个 = HOLD 23）的决策点："
          f"{ties}/{n} = {100.0 * ties / n:.1f}%")

    # n_cand==1 的三种路线（这是本文最关键的分解）
    print("\n    n_cand==1 的决策点按**瓶颈**分解：")
    routes = Counter()
    for r in rows:
        if r["c"]["n_cand"] != 1:
            continue
        cid = r["a"]["pinned"]
        if cid == 23:
            routes["类 23 = HOLD（MC 选了「不出招」⇒ 空束就只有 1 条）"] += 1
        elif isinstance(cid, int) and 0 <= cid <= 16:
            routes[f"类 {cid}（塔/经济类：该类只解出 1 个不同的 op）"] += 1
        else:
            routes[f"类 {cid}（超武/基地升级：只解出 1 个 op）"] += 1
    for k, v in routes.most_common():
        print(f"      {k}: {v}/{n} = {100.0 * v / n:.1f}%")

    print("\n" + "=" * 96)
    print("§1b n_cand 分布按『MC 是否钉了 HOLD』分组")
    print("-" * 96)
    for label, sel in (("pinned==23（HOLD）", lambda r: r["a"]["pinned"] == 23),
                       ("pinned!=23（真的出招）", lambda r: r["a"]["pinned"] != 23)):
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        hh = Counter(r["c"]["n_cand"] for r in sub)
        print(f"    {label}: {len(sub)}/{n} = {100.0 * len(sub) / n:.1f}% 决策点；"
              f"n_cand 分布 = {dict(sorted(hh.items()))}；"
              f"n_cand==1 占该组 {100.0 * hh.get(1, 0) / len(sub):.1f}%")

    print("\n" + "=" * 96)
    print("§1c 保真度自检：n_cand==1 ⇒ 评估器没有选择余地 ⇒ 重放候选必须 == 实战动作")
    print("-" * 96)
    chk = [r for r in rows if "replay_match" in r]
    ok = sum(1 for r in chk if r["replay_match"])
    print(f"    可自检的决策点 {len(chk)}（n_cand==1）；逐 op 完全一致 {ok} "
          f"= {100.0 * ok / max(1, len(chk)):.1f}%")
    bad = [r for r in chk if not r["replay_match"]][:5]
    for r in bad:
        print(f"      ✗ {r['token']} r{r['round']} 重放={r['c']['bundles'][0]} "
              f"实战={r['log_ops']}")

    print("\n" + "=" * 96)
    print("§1d (b) 段明细：**被钉住的类** → 该类一次决策能解出几个不同 bundle")
    print("-" * 96)
    by_cls: dict = {}
    for r in rows:
        by_cls.setdefault(str(r["a"]["pinned"]), []).append(r)
    print(f"    {'类':>4} {'决策点':>6} {'占比':>7} {'该类合法格(中位)':>16} "
          f"{'不同bundle(中位/最大)':>22} {'n_cand(中位)':>12}")
    for c, sub in sorted(by_cls.items(), key=lambda kv: -len(kv[1])):
        nb = sorted(r["c"]["n_unique_raw"] for r in sub)
        nc = sorted(r["c"]["n_cand"] for r in sub)
        ncell = sorted(int(r["legal_cells"].get(str(c), r["legal_cells"].get(int(c), 0)))
                       for r in sub)
        print(f"    {c:>4} {len(sub):>6} {100.0 * len(sub) / n:>6.1f}% "
              f"{ncell[len(ncell) // 2]:>16} "
              f"{nb[len(nb) // 2]:>10} / {nb[-1]:<9} {nc[len(nc) // 2]:>12}")

    print("\n" + "=" * 96)
    print("§2 塌缩归因：n_cand==1 的决策点（前 25 个），瓶颈在哪一段？")
    print("-" * 96)
    shown2 = 0
    for r in rows:
        if r["c"]["n_cand"] != 1 or shown2 >= 25:
            continue
        shown2 += 1
        cid = r["a"]["pinned"]
        ncell = r["legal_cells"].get(str(cid), r["legal_cells"].get(cid, 0))
        print(f"    r{r['round']:>3} 类{cid}  该类合法格={ncell}  塔={r['n_towers']}座  "
              f"采样={r['c']['n_samples']}  不同key={r['c']['n_unique_raw']}  "
              f"key={r['c']['bundles'][0] if r['c']['bundles'] else None}")

    print("\n" + "=" * 96)
    print("§3 被 k=48 截断丢掉的 bundle 样例（`bundle_mcts.py:1061` 按出现频次留前 k 个）")
    print("-" * 96)
    shown = 0
    for r in rows:
        if r["c"].get("n_dropped") and shown < 20:
            print(f"    r{r['round']:>3} 类{r['a']['pinned']}: 720 次采样解出 "
                  f"{r['c']['n_unique_raw']} 个不同 key，只留 {r['c']['n_cand']}；"
                  f"丢掉 {r['c']['n_dropped']} 个，样例：")
            for key, n in r["c"]["dropped"][:4]:
                print(f"        {key}   （出现 {n}/720 次）")
            shown += 1
    if shown == 0:
        print("    （没有：n_unique_raw == n_cand ⇒ **去重一步没丢任何东西**，"
              "是采样本身只解出这么几条）")

    if not args.no_whatif and any(r["d"] for r in rows):
        print("\n" + "=" * 96)
        print("§4 放开后每决策的有效候选数（估算；**总采样预算 720 不变**，平摊到 K 个类，"
              "取不同 key 的并集，再按 k=48 截断）")
        print("-" * 96)
        print(f"{'token':>6} {'rnd':>4} {'现役':>5} {'pinK2':>6} {'pinK3':>6} {'pinK5':>6} "
              f"{'全部合法类':>10} {'单类最优':>8} {'原子op上界':>10} {'(类数)':>6}")
        for r in rows:
            d = r["d"]
            print(f"{r['token']:>6} {r['round']:>4} {r['c']['n_cand']:>5} {d.get('pinK2', 0):>6} "
                  f"{d.get('pinK3', 0):>6} {d.get('pinK5', 0):>6} {d.get('pinAll', 0):>10} "
                  f"{d.get('single_class_best', 0):>8} {d.get('n_legal_atomic', 0):>10} "
                  f"{d.get('pinAll_K', 0):>6}")
        for k in ("pinK2", "pinK3", "pinK5", "pinAll", "single_class_best", "n_legal_atomic"):
            vals = [r["d"].get(k, 0) for r in rows if r["d"]]
            if not vals:
                continue
            vals_s = sorted(vals)
            print(f"    中位 {k:<18} = {vals_s[len(vals_s) // 2]:>4}   "
                  f"均值 {sum(vals) / len(vals):>6.2f}   max {vals_s[-1]}")


if __name__ == "__main__":
    raise SystemExit(main())
