# -*- coding: utf-8 -*-
"""叶子金币项 α 的**离线预检**（跑对局扫描之前，便宜）。

问题：`AZAI_LEAF_COIN_ALPHA=α` 把叶子分数加成 `V(leaf) + α·(my_coins/300)` 后，
      unified 搜索各候选的 s2 会怎么变、top-1 会在 α 多大时从"当时的选择"变成什么？

做法（复用 `dump_rollout_tree.py` 的重放机制，不重写 rollout）：
  1. 从 `training_history/collect_v2_s1/*.cand.jsonl` 抽三类"纯"局面各 N 个：
     上一决策 chosen = op12(升级) / op11(建塔) / op21(闪电)。
  2. 每个局面：`replay_state` 重建决策点状态 → 对**每个候选** bundle 建 child →
     用 `FlatRolloutGreedy._tree_rollout_value(_dump_tree=True)` 重跑 rollout（与采集同 rng），
     从新加的 `_tree_dump["leaf_coins"]` 读每条 rollout 叶子的"己方金币"，算
     `mean(coins)/300`。
     ⚠️ 值网叶子分**不重算**（α=0 基线直接用 cand.jsonl 记录的 s2）；`_leaf_values` 被
     monkeypatch 成 0，只为省值网前向、只读金币。s2 与金币项是线性可加的：
     `s2'(α) = s2 + α·mean(coins)/300` ⇒ 对 α 网格做**解析扫描**（等价于逐 α 重跑）。
  3. 统计：每个 α 下 top-1 变没变、变成什么类别；以及每类局面的"翻转 α"（top-1 首次
     ≠ chosen 的最小 α）。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/offline_leaf_coin_precheck.py \
      --n-per-cat 20 [--alphas 0,0.01,0.02,0.037,0.05,0.07,0.1,0.15,0.2,0.3,0.5,1.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
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
    """遍历全部 cand.jsonl，返回 [(seed, round, player, cand_rec, line_idx)]。"""
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


def leaf_coins_norm(searcher, st, player, bundle, decision_idx, idx, n_rollouts):
    child = d.child_for(st, player, bundle)
    base = ((CFG["rollout_seed"] * 1000003 + decision_idx) * 100003 + idx * 101)
    rngs = [np.random.default_rng(base + k) for k in range(n_rollouts)]
    searcher._dump_tree = True
    searcher._tree_rollout_value(child, player, rngs)
    coins = searcher._tree_dump.get("leaf_coins") or []
    if not coins:
        return float("nan")
    return float(np.mean(coins)) / 300.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=20)
    ap.add_argument("--alphas", default="0,0.01,0.02,0.037,0.05,0.07,0.1,0.15,0.2,0.3,0.5,1.0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    recs = load_records()
    cats = {"升级(op12)": UPG, "建塔(op11)": BUILD, "闪电(op21)": LIGHT}
    situations = {}
    for name, op in cats.items():
        situations[name] = sample(recs, op, args.n_per_cat, rng)
        print(f"[pre] {name}: 抽到 {len(situations[name])} 个纯局面")

    searcher = build_searcher()
    n_rollouts = int(min(CFG["rollouts_per_candidate"],
                         max(1, CFG["max_total_rounds"] // max(1, 48 * CFG["horizon"]))))

    # 每类局面逐条记录：{name, seed, round, player, chosen_idx, s2, coins_norm[], top1_by_alpha}
    rows = []
    for name, sits in situations.items():
        for (seed, rnd, player, cand_rec, line_idx) in sits:
            decision_idx = line_idx + 1
            cand = cand_rec["cand"]
            s2 = np.asarray(cand_rec["s2"], dtype=np.float64)
            chosen_idx = int(cand_rec["chosen_idx"])
            st = d.replay_state(seed, rnd, player)
            cn = np.zeros(len(cand), dtype=np.float64)
            for i, bundle in enumerate(cand):
                cn[i] = leaf_coins_norm(searcher, st, player, bundle,
                                        decision_idx, i, n_rollouts)
            rows.append(dict(name=name, seed=seed, rnd=rnd, player=player,
                             chosen_idx=chosen_idx, s2=s2, cn=cn,
                             cand=cand, chosen_s2=float(s2[chosen_idx]),
                             chosen_cn=float(cn[chosen_idx])))
            print(f"[pre] {name} seed{seed} r{rnd} p{player}: n_cand={len(cand)} "
                  f"chosen={cat(cand[chosen_idx])} s2={s2[chosen_idx]:+.4f} "
                  f"coins_norm(chosen)={cn[chosen_idx]:.3f}", flush=True)

    # ── 逐类：翻转率 + 翻转后 top-1 类别分布 + 翻转 α ──
    print("\n" + "=" * 90)
    print("=== 结果：加大 α 后 top-1 变化（top-1 首次 ≠ chosen 记为翻转）===")
    print("=" * 90)
    for name in cats:
        rs = [r for r in rows if r["name"] == name]
        if not rs:
            continue
        print(f"\n── {name}（n={len(rs)}）──")
        # 每局 flip_alpha：使 top-1 != chosen 的最小 α（无则 inf）
        flip_alphas = []
        for r in rs:
            fa = float("inf")
            new_top_at_fa = None
            for a in alphas:
                s2n = r["s2"] + a * r["cn"]
                top = int(np.nanargmax(s2n))
                if top != r["chosen_idx"]:
                    fa = a
                    new_top_at_fa = top
                    break
            r["flip_alpha"] = fa
            r["flip_to"] = cat(r["cand"][new_top_at_fa]) if new_top_at_fa is not None else None
            if np.isfinite(fa):
                flip_alphas.append(fa)
        print(f"  翻转局数（α∈[{alphas[0]},{alphas[-1]}] 内 top-1 变）: "
              f"{len(flip_alphas)}/{len(rs)}")
        if flip_alphas:
            print(f"  翻转 α：中位={np.median(flip_alphas):.3f}  最小={min(flip_alphas):.3f}  "
                  f"最大={max(flip_alphas):.3f}")
        # 翻转后 top-1 类别
        flips = [r for r in rs if np.isfinite(r["flip_alpha"])]
        fc = Counter(r["flip_to"] for r in flips)
        print(f"  翻转后 top-1 变成: {dict(fc)}")
        # chosen 的 s2 与 coins_norm，以及"最赚币候选"（HOLD/降级）的差
        chosen_cn = [r["chosen_cn"] for r in rs]
        print(f"  chosen 的 mean(coins)/300：中位={np.median(chosen_cn):.3f} "
              f"[{min(chosen_cn):.3f},{max(chosen_cn):.3f}]")
        # 每局明细（翻转 α / 翻转去向 / chosen 类别 / 新 top-1 的 coins_norm）
        print(f"  {'seed':>8} {'r':>4} {'p':>2} {'chosen':<8} {'s2':>7} {'coins/300':>9} "
              f"{'翻转α':>7} {'翻转→':<10}")
        for r in sorted(rs, key=lambda x: (x["flip_alpha"] if np.isfinite(x["flip_alpha"]) else 1e9)):
            fa = f"{r['flip_alpha']:.3f}" if np.isfinite(r["flip_alpha"]) else "  不翻"
            print(f"  {r['seed']:>8} {r['rnd']:>4} {r['player']:>2} "
                  f"{cat(r['cand'][r['chosen_idx']]):<8} {r['chosen_s2']:>+7.4f} "
                  f"{r['chosen_cn']:>9.3f} {fa:>7} {str(r['flip_to']):<10}")

    # ── 汇总表：每个 α 的翻转率（三分类分别）──
    print("\n" + "=" * 90)
    print("=== 汇总：各 α 下 top-1 ≠ chosen 的占比（%）===")
    print("=" * 90)
    hdr = "  α      " + "".join(f"{name:>14}" for name in cats)
    print(hdr)
    for a in alphas:
        line = f"  {a:<7.3g} "
        for name in cats:
            rs = [r for r in rows if r["name"] == name]
            nflip = sum(1 for r in rs if int(np.nanargmax(r["s2"] + a * r["cn"])) != r["chosen_idx"])
            line += f"{100.0*nflip/max(1,len(rs)):>13.0f}%"
        print(line)


if __name__ == "__main__":
    main()
