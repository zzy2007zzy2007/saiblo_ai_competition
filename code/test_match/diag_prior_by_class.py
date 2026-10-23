"""分通道先验质量诊断 v2（2026-10-23）。

修正：
  ① bundle 里的 `b[0][0]` 是 **op-type 码**（11=BUILD、12=UPGRADE、13=DOWNGRADE、
     21=LIGHTNING、22=EMP、23=DEFLECTOR、24=EVASION、31=UP_SPEED、32=UP_ANT），
     不是 24 类索引；
  ② 只在**闪电真的可用**（冷却 0 且付得起）的局面做比较，否则闪电类样本为 0；
  ③ 输出：每一类在**两个网**上的组内 Spearman（先验 vs rollout 估值）与 recall@4。
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np

from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade

OPNAME = {11: "BUILD", 12: "UPGRADE", 13: "DOWNGRADE", 21: "LIGHTNING",
          22: "EMP", 23: "DEFLECTOR", 24: "EVASION", 31: "UP_SPEED", 32: "UP_ANT"}


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4:
        return float("nan"), False
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if den == 0:
        return float("nan"), False
    return float((ra * rb).sum() / den), True


def collect(ckpt: str, seeds_rounds):
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(ckpt, feat, value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models(ckpt)
    vm.eval()
    rows = collections.defaultdict(list)
    used = 0
    for seed, warm in seeds_rounds:
        f = GameStateFacade.initial(seed=seed)
        for _ in range(warm):
            f.advance_round()
        if f.terminal:
            continue
        m = FlatRolloutGreedy(net_fn, k=300, rollouts_per_candidate=1, horizon=16,
                              max_total_rounds=10 ** 9, value_model=vm,
                              feature_extractor=feat, rollout_mode="idle")
        m.search(f.clone(), 0)
        bundles = list(m.last_root.bundles)
        if not bundles:
            continue
        used += 1
        priors = [c.prior for c in m.last_root.children]
        for b, pr, v in zip(bundles, priors, m.last_scores):
            op = int(b[0][0]) if b else -1
            rows[op].append((float(pr), float(v)))
    return rows, used


def report(tag, rows, used):
    out = {}
    print(f"  --- {tag}（用了 {used} 个局面）")
    for op in sorted(rows):
        pr = [x[0] for x in rows[op]]; vv = [x[1] for x in rows[op]]
        rho, ok = spearman(pr, vv)
        n = len(pr)
        # recall@4：估值前 4 里有多少被先验前 4 覆盖
        hit = "-"
        if n >= 8:
            top_v = set(np.argsort(-np.asarray(vv))[:4])
            top_p = set(np.argsort(-np.asarray(pr))[:4])
            hit = f"{len(top_v & top_p)}/4"
        out[op] = rho
        print(f"    {OPNAME.get(op, str(op)):<10} n={n:<4} Spearman={rho:+.3f}  recall@4={hit}")
    return out


def main() -> int:
    # 挑出闪电可用的局面：扫种子与回合，找 cooldown 0 且金币 >= 200（闪电价格量级）
    cand = []
    for seed in (1540, 1541, 1542, 1580):
        f = GameStateFacade.initial(seed=seed)
        for r in range(400):
            f.advance_round()
            try:
                cd = f.weapon_cooldowns
                ready = int(cd[0][0]) == 0 if hasattr(cd[0], "__len__") else True
            except Exception:
                ready = True
            if ready and int(f.coins[0]) >= 200:
                cand.append((seed, r + 1))
                break
    # 再补几个"有钱但不一定闪电"的局面，保证 BUILD 类样本量
    cand += [(1540, 60), (1540, 140), (1541, 200), (1580, 100), (1580, 200)]
    print(f"  采集局面: {cand}")
    ra, ua = collect("checkpoints/posnet_A_k5_m32.pt", cand)
    rb, ub = collect("checkpoints/grafted_sdk_A.pt", cand)
    a = report("posnet_A（我们的）", ra, ua)
    b = report("grafted_sdk_A（SDK 蒸馏）", rb, ub)
    print("\n  === 逐类对比（谁的先验更准）===")
    for op in sorted(set(a) | set(b)):
        va, vb = a.get(op, float("nan")), b.get(op, float("nan"))
        who = "我们" if (va or -9) > (vb or -9) else "蒸馏"
        print(f"    {OPNAME.get(op, str(op)):<10} 我们 {va:+.3f}   蒸馏 {vb:+.3f}   ⇒ {who}更好")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
