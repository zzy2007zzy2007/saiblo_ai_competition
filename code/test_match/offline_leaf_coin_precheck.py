# -*- coding: utf-8 -*-
"""叶子金币项 α 的**离线预检**（跑对局扫描之前，便宜）。同时算 **linear** 与 **hinge90** 两版。

问题：`AZAI_LEAF_COIN_ALPHA=α` 把叶子分数加成 `V(leaf) + α·bonus(coins)` 后，
      unified 搜索各候选的 s2 会怎么变、top-1 会在 α 多大时从"当时的选择"变成什么？

两种 bonus（与 `flat_rollout_greedy._leaf_values` 逐字同义）：
  · linear  = coins/300                       （现值网同尺度线性项）
  · hinge90 = min(coins, 90)/300              （<90 斜率 α 激励攒钱；≥90 封顶 ⇒ 放闪电不被惩罚）

做法（复用 `dump_rollout_tree.py` 的重放机制，不重写 rollout）：
  1. 从 `training_history/collect_v2_s1/*.cand.jsonl` 抽三类"纯"局面各 N 个：
     上一决策 chosen = op12(升级) / op11(建塔) / op21(闪电)。
  2. 每个局面：`replay_state` 重建决策点状态 → 对**每个候选** bundle 建 child →
     用 `FlatRolloutGreedy._tree_rollout_value(_dump_tree=True)` 重跑 rollout（与采集同 rng），
     从 `_tree_dump["leaf_coins"]` 读每条 rollout 叶子的"己方金币"（raw coins）。
     ⚠️ 值网叶子分**不重算**（α=0 基线直接用 cand.jsonl 记录的 s2）；`_leaf_values` 被
     monkeypatch 成 0，只为省值网前向、只读金币。s2 与 bonus 项线性可加：
     `s2'(α) = s2 + α·mean(bonus_norm)` ⇒ 对 α 网格做**解析扫描**（等价逐 α 重跑）。
  3. 统计：每个 α 下 top-1 变没变、变成什么类别；以及每类局面的"翻转 α"（top-1 首次
     ≠ chosen 的最小 α）。linear / hinge90 各出一份。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/offline_leaf_coin_precheck.py \
      --n-per-cat 20 [--alphas 0,0.05,0.1,0.2,0.3,0.5,1.0,2.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
for p in [ROOT / "code", ROOT / "code" / "test_match", ROOT / "code" / "cpp_engine",
          ROOT / "Ant-Game", ROOT / "Ant-Game" / "SDK"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import dump_rollout_tree as d                                          # noqa: E402
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy     # noqa: E402

DATA = ROOT / "training_history" / "collect_v2_s1"
CFG = d.CFG

OP_NAMES = d.OP_NAMES
UPG, BUILD, LIGHT, DN = 12, 11, 21, 13


def cat(bundle):
    ts = sorted({int(o[0]) for o in (bundle or ())})
    return "HOLD" if not ts else "+".join(OP_NAMES.get(t, f"op{t}") for t in ts)


def load_records():
    recs = []
    for f in sorted((DATA).glob("*.cand.jsonl")):
        seed = int(f.name.replace("az_selfplay_seed", "").replace(".cand.jsonl", ""))
        for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines()):
            ln = ln.strip()
            if not ln:
                continue
            c = json.loads(ln)
            recs.append((seed, int(c["round"]), int(c["player"]), c, i))
    return recs


def pure_type(chosen):
    ts = sorted({int(o[0]) for o in (chosen or ())})
    return ts[0] if len(ts) == 1 else None


def sample(recs, op, n, rng):
    pool = [r for r in recs if pure_type(r[3].get("chosen") or []) == op]
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[i] for i in idx]


def build_searcher():
    searcher = FlatRolloutGreedy(lambda s, p: None, **CFG,
                                 value_model=object(), feature_extractor=object())
    # 省值网前向：本预检只需要叶子金币，不需要叶子值网分。
    searcher._leaf_values = lambda states, root_player: np.zeros(len(states), dtype=np.float64)
    return searcher


def leaf_coins_raw(searcher, st, player, bundle, decision_idx, idx, n_rollouts):
    """重跑该候选的 rollout 树，返回每条 rollout 叶子的己方金币（raw coins ndarray）。"""
    child = d.child_for(st, player, bundle)
    base = ((CFG["rollout_seed"] * 1000003 + decision_idx) * 100003 + idx * 101)
    rngs = [np.random.default_rng(base + k) for k in range(n_rollouts)]
    searcher._dump_tree = True
    searcher._tree_rollout_value(child, player, rngs)
    coins = searcher._tree_dump.get("leaf_coins") or []
    return np.asarray(coins, dtype=np.float64)


def bonus_norm(coins, mode):
    """mean over rollouts of the bonus term（单位 = 每 coins/300 的 V 增量）。"""
    if coins.size == 0:
        return float("nan")
    c = np.minimum(coins, 90.0) if mode == "hinge90" else coins
    return float(np.mean(c)) / 300.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=20)
    ap.add_argument("--alphas", default="0,0.01,0.02,0.037,0.05,0.07,0.1,0.15,0.2,0.3,0.5,1.0,2.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)
    MODES = ["linear", "hinge90"]

    recs = load_records()
    cats = {"升级(op12)": UPG, "建塔(op11)": BUILD, "闪电(op21)": LIGHT}
    situations = {}
    for name, op in cats.items():
        situations[name] = sample(recs, op, args.n_per_cat, rng)
        print(f"[pre] {name}: 抽到 {len(situations[name])} 个纯局面")

    searcher = build_searcher()
    n_rollouts = int(min(CFG["rollouts_per_candidate"],
                         max(1, CFG["max_total_rounds"] // max(1, 48 * CFG["horizon"]))))

    # 每类局面逐条记录：{name, seed, round, player, chosen_idx, s2, cn[mode]}
    rows = []
    for name, sits in situations.items():
        for (seed, rnd, player, cand_rec, line_idx) in sits:
            decision_idx = line_idx + 1
            cand = cand_rec["cand"]
            s2 = np.asarray(cand_rec["s2"], dtype=np.float64)
            chosen_idx = int(cand_rec["chosen_idx"])
            st = d.replay_state(seed, rnd, player)
            cns = {m: np.zeros(len(cand), dtype=np.float64) for m in MODES}
            for i, bundle in enumerate(cand):
                raw = leaf_coins_raw(searcher, st, player, bundle,
                                     decision_idx, i, n_rollouts)
                for m in MODES:
                    cns[m][i] = bonus_norm(raw, m)
            rows.append(dict(name=name, seed=seed, rnd=rnd, player=player,
                             chosen_idx=chosen_idx, s2=s2, cns=cns, cand=cand,
                             chosen_s2=float(s2[chosen_idx])))
            print(f"[pre] {name} seed{seed} r{rnd} p{player}: n_cand={len(cand)} "
                  f"chosen={cat(cand[chosen_idx])} s2={s2[chosen_idx]:+.4f} "
                  f"lin={cns['linear'][chosen_idx]:.3f} hinge={cns['hinge90'][chosen_idx]:.3f}",
                  flush=True)

    # ── 汇总表：各 α / 各 mode 下 top-1 ≠ chosen 的占比（%）──
    print("\n" + "=" * 92)
    print("=== 汇总：各 α 下 top-1 ≠ chosen 的占比（%）；linear vs hinge90 ===")
    print("=" * 92)
    for mode in MODES:
        print(f"\n── mode={mode} ──")
        hdr = "  α      " + "".join(f"{name:>14}" for name in cats)
        print(hdr)
        for a in alphas:
            line = f"  {a:<7.3g} "
            for name in cats:
                rs = [r for r in rows if r["name"] == name]
                nflip = sum(1 for r in rs
                            if int(np.nanargmax(r["s2"] + a * r["cns"][mode])) != r["chosen_idx"])
                line += f"{100.0*nflip/max(1,len(rs)):>13.0f}%"
            print(line)

    # ── 逐类翻转 α（中位）+ 翻转去向（关键看"闪电局面"在 hinge90 下是否不再被压）──
    print("\n" + "=" * 92)
    print("=== 逐类翻转 α（中位）与翻转去向 ===")
    print("=" * 92)
    for mode in MODES:
        print(f"\n── mode={mode} ──")
        for name in cats:
            rs = [r for r in rows if r["name"] == name]
            if not rs:
                continue
            flip_alphas = []
            flips = []
            for r in rs:
                fa = float("inf")
                new_top = None
                for a in alphas:
                    s2n = r["s2"] + a * r["cns"][mode]
                    top = int(np.nanargmax(s2n))
                    if top != r["chosen_idx"]:
                        fa = a
                        new_top = top
                        break
                if np.isfinite(fa):
                    flip_alphas.append(fa)
                    flips.append(cat(r["cand"][new_top]))
            nflip = len(flip_alphas)
            med = f"{np.median(flip_alphas):.3f}" if flip_alphas else "  不翻"
            fc = Counter(flips)
            print(f"  {name:<12} 翻转 {nflip}/{len(rs)}  翻转α中位={med:>8}  去向={dict(fc)}")

    # ── 闪电局面的"被压"对比（linear vs hinge90）：在 α=0.3/1.0 时闪电仍 top-1 的占比 ──
    print("\n" + "=" * 92)
    print("=== 闪电局面：加大 α 后闪电仍是 top-1 的占比（linear vs hinge90）===")
    print("=" * 92)
    lr = [r for r in rows if r["name"] == "闪电(op21)"]
    print(f"  {'α':>6}  {'linear 闪电保持率':>20}  {'hinge90 闪电保持率':>20}")
    for a in [0.0, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0]:
        lin_keep = sum(1 for r in lr
                       if int(np.nanargmax(r["s2"] + a * r["cns"]["linear"])) == r["chosen_idx"])
        hin_keep = sum(1 for r in lr
                       if int(np.nanargmax(r["s2"] + a * r["cns"]["hinge90"])) == r["chosen_idx"])
        print(f"  {a:<6.2g}  {lin_keep}/{len(lr)} ({100.0*lin_keep/len(lr):.0f}%)"
              f"{'':>6}  {hin_keep}/{len(lr)} ({100.0*hin_keep/len(lr):.0f}%)")


if __name__ == "__main__":
    main()
