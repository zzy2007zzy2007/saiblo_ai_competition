"""**实验**：`k=48 / N=64 / horizon=16 / share_prefix=1` 下四种"随机性来源"的
叶子多样性 vs 搜索耗时（2026-10-27）。

四组（父任务指定）：
```
A. 现状         动作随机 C=1.0,k=3.0        （rollout_mode=random, 不消费 RNG）
B. 仅 RNG 消费  C=0.3,k=1.0                （rollout_mode=idle,  消费 RNG）
C. 仅 RNG 消费  C=1.0,k=3.0                （rollout_mode=idle,  消费 RNG）← 重点
D. 两者叠加     都 C=1.0,k=3.0             （rollout_mode=random, 消费 RNG）
```

每个 (组, 局面, 重复) 报：
- **去重叶子数**（每候选 N=64 条轨迹里不同轨迹签名的个数，48 个候选求和/均值）
- **树节点数**（每候选新建节点数；硬上界 `1 + N·H = 1025`）
- **耗时**：search 总耗时 + 其中 rollout 阶段耗时
- 实际消费的 RNG 次数

另外用 `--ablate-key` 再跑一遍 B/C 的"缓存键不带 rng_state"版本（**诊断用**，
会错误合并 rng_state 不同的分支），量化"把 rng_state 加进键"在弱随机化下的最坏
节点涨幅。

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
bash code/run_logged.sh expl_rng_perturb $P code/test_match/expl_rng_perturb.py
```
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ["AZAI_ROLLOUT_SAMPLES"] = "1"      # 让 last_tree_stats 被记录
os.environ.setdefault("AZAI_TORCH_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game",
           REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402

CONFIGS = [
    ("A 动作随机", dict(rollout_mode="random", rollout_rng_perturb=False,
                        rollout_prob_mode="decay", rollout_decay_c=1.0,
                        rollout_decay_k=3.0)),
    ("B 仅RNG 弱", dict(rollout_mode="idle", rollout_rng_perturb=True,
                        rollout_prob_mode="decay", rollout_decay_c=0.3,
                        rollout_decay_k=1.0)),
    ("C 仅RNG 强", dict(rollout_mode="idle", rollout_rng_perturb=True,
                        rollout_prob_mode="decay", rollout_decay_c=1.0,
                        rollout_decay_k=3.0)),
    ("D 叠加", dict(rollout_mode="random", rollout_rng_perturb=True,
                    rollout_prob_mode="decay", rollout_decay_c=1.0,
                    rollout_decay_k=3.0)),
    # 追加（2026-10-27）：decay(1.0,3) 的"有效随机回合数"只有 ~3.5 次，所以 C 组的
    # 消费模式熵不够（实测只有 39.8/64 条不同轨迹）。这里补两种**均匀**调度，
    # 用来回答"想拿到接近 A 的多样性，最便宜要多少时间"。
    ("E RNG const.5", dict(rollout_mode="idle", rollout_rng_perturb=True,
                           rollout_prob_mode="const", rollout_act_prob=0.5)),
    ("F RNG const1.0", dict(rollout_mode="idle", rollout_rng_perturb=True,
                            rollout_prob_mode="const", rollout_act_prob=1.0)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--states", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--advance", type=int, default=60)
    ap.add_argument("--reps", type=int, nargs="+", default=[53, 61, 71])
    ap.add_argument("--ablate-key", action="store_true")
    ap.add_argument("--only", default="",
                    help="只跑名字里含这些子串的组（逗号分隔，如 'A 动作,E RNG'）")
    ap.add_argument("--ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    ap.add_argument("--out", default="code/cpp_engine/build_probe/rng_impl/expl_rng.json")
    args = ap.parse_args()

    # ⚠️ 必须在第一次建局之前 enable，否则 native_game 已经被 import 成现役 pyd
    from my_ai.az_intent import rng_engine
    ok = rng_engine.enable()
    print(f"[expl] rng_engine.enable -> {ok}  ({rng_engine.describe()})", flush=True)

    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
    from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from SDK.utils.features import FeatureExtractor

    feat = FeatureExtractor(max_actions=96)
    ckpt = str(REPO / args.ckpt)
    _, net_fn = make_net_fn_from_ckpt(ckpt, feat, value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models(ckpt)
    vm.eval()
    print(f"[expl] ckpt={ckpt} k={args.k} N={args.n} H={args.horizon} "
          f"状态={args.states}@{args.advance}回合 重复={args.reps}", flush=True)

    states = []
    for sd in args.states:
        st = GameStateFacade.initial(seed=sd, cold_handle_rule_illegal=True)
        for _ in range(args.advance):
            st.advance_round()
        states.append((sd, st.clone()))
    print(f"[expl] 局面就绪：{[(sd, s.bases[0].hp, s.bases[1].hp, len(s.ants)) for sd, s in states]}",
          flush=True)

    rows = []
    CAP = 1 + args.n * args.horizon

    def run_one(tag, cfg, sd, snap, rep, extra_wrap=None):
        kw = dict(rollouts_per_candidate=args.n, horizon=args.horizon,
                  max_total_rounds=10 ** 9, value_model=vm,
                  feature_extractor=feat, value_batch=256,
                  rollout_random_side="opponent", rollout_act_prob=0.3,
                  rollout_share_prefix=True, rollout_seed=int(rep))
        kw.update(cfg)                     # 组配置覆盖默认（如 const 模式的 act_prob）
        s = FlatRolloutGreedy(net_fn, k=args.k, **kw)
        per_cand = []
        orig = s._tree_rollout_value

        def wrapped(child, player, rngs):
            v = orig(child, player, rngs)
            st_ = getattr(s, "last_tree_stats", None)
            if st_:
                per_cand.append(st_)
            return v

        s._tree_rollout_value = wrapped
        t0 = time.perf_counter()
        res = s.search(snap.clone(), 0)
        dt = time.perf_counter() - t0
        nodes = [x[0] for x in per_cand]
        leaves = [x[1] for x in per_cand]
        uniq = [x[2] for x in per_cand]
        row = dict(
            tag=tag, seed=sd, rep=int(rep), cand=int(len(res.bundles)),
            nodes_sum=int(sum(nodes)), nodes_mean=float(statistics.mean(nodes)) if nodes else 0.0,
            nodes_max=int(max(nodes)) if nodes else 0, cap=CAP,
            leaves_mean=float(statistics.mean(leaves)) if leaves else 0.0,
            uniq_mean=float(statistics.mean(uniq)) if uniq else 0.0,
            uniq_sum=int(sum(uniq)),
            total_s=dt, rollout_s=float(s.last_timing.get("rollout_s", 0.0)),
            expand_s=float(s.last_timing.get("expand_s", 0.0)),
            rng_consumed=int(getattr(s, "last_rng_consumed", 0)),
            rounds=int(s.last_rounds),
            chosen=int(res.chosen_index),
        )
        print(f"[expl] {tag:10s} seed={sd} rep={rep} | 候选={row['cand']} "
              f"节点Σ={row['nodes_sum']:6d} (均{row['nodes_mean']:7.1f} / 上界{row['cap']} "
              f"= {100.0 * row['nodes_mean'] / row['cap']:5.1f}%) "
              f"去重叶子均={row['uniq_mean']:5.1f}/64 "
              f"耗时={row['total_s']:6.3f}s (rollout {row['rollout_s']:6.3f}s) "
              f"rng={row['rng_consumed']:6d}", flush=True)
        return row

    def _want(tag: str) -> bool:
        if not args.only:
            return True
        return any(x.strip() in tag for x in args.only.split(","))

    for tag, cfg in CONFIGS:
        if not _want(tag):
            continue
        for sd, snap in states:
            for rep in args.reps:
                rows.append(run_one(tag, cfg, sd, snap, rep))

    if args.ablate_key:
        for tag in ("B 仅RNG 弱", "C 仅RNG 强"):
            cfg = dict(dict(CONFIGS)[tag])
            cfg_none = dict(cfg, rollout_rng_key="none")
            for sd, snap in states:
                a = run_one(tag + " [键含rng]", cfg, sd, snap, args.reps[0])
                b = run_one(tag + " [键不含]", cfg_none, sd, snap, args.reps[0])
                a["variant"], b["variant"] = "key=state", "key=none"
                rows += [a, b]
                print(f"[expl] ↑ 键 ablate seed={sd}: 节点Σ {b['nodes_sum']} -> {a['nodes_sum']} "
                      f"(+{100.0 * (a['nodes_sum'] - b['nodes_sum']) / max(1, b['nodes_sum']):.1f}%)",
                      flush=True)

    # ── 汇总表 ────────────────────────────────────────────────────────────
    print("\n[expl] ==== 汇总（每格 = 3 局面 × 3 重复的均值；节点/叶子都是**每候选**口径）====")
    hdr = (f"{'组':12s}{'候选':>5s}{'节点/候选':>11s}{'占上界':>8s}"
           f"{'去重叶子/64':>12s}{'总耗时s':>9s}{'rollout s':>10s}"
           f"{'ms/千节点':>10s}{'rng消费':>9s}")
    print(hdr, flush=True)
    summary = {}
    for tag, _cfg in CONFIGS:
        if not _want(tag):
            continue
        rs = [r for r in rows if r["tag"] == tag]
        if not rs:
            continue
        m = dict(
            nodes=statistics.mean(r["nodes_mean"] for r in rs),
            ratio=statistics.mean(r["nodes_mean"] / r["cap"] for r in rs),
            uniq=statistics.mean(r["uniq_mean"] for r in rs),
            total=statistics.mean(r["total_s"] for r in rs),
            rollout=statistics.mean(r["rollout_s"] for r in rs),
            ms_per_knode=statistics.mean(
                1000.0 * r["rollout_s"] / max(1.0, r["nodes_sum"]) for r in rs),
            rng=statistics.mean(r["rng_consumed"] for r in rs),
            cand=statistics.mean(r["cand"] for r in rs),
        )
        summary[tag] = m
        print(f"{tag:12s}{m['cand']:5.0f}{m['nodes']:11.1f}{100 * m['ratio']:7.1f}%"
              f"{m['uniq']:12.1f}{m['total']:9.3f}{m['rollout']:10.3f}"
              f"{m['ms_per_knode']:10.1f}{m['rng']:9.0f}", flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "summary": summary,
                               "params": vars(args)}, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n[expl] 已写入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
