"""诊断：**如果让"我们自己的 1-ply 价值先验"来选类**，它会选什么？

动机：候选 C（在合法类之间重排序）进不了部署路径；候选 D（按 **logits** 取可执行类）灾难
（0/16，建塔 152 座/局、闪电塌到 1.9 次）。于是问：**按"价值"而不是"logits"选类**会怎样？
本脚本只用**已有 npz** 回答（不跑对局、不要引擎）：

  * 每一行 = (决策, head, 类) 的候选格与逐格 `adv`（`adv` 由 `valnet_A2` 算，= 1-ply 价值差）；
  * `cnt` = 该类在该决策下**可执行的候选格数** ⇒ `cnt>0` 就是"这一类此刻能真的做出来"；
  * 对每个 (决策, head)：在**可执行**的类里取 `max(adv)` 最大的那个 → 这就是"价值选类"；
  * 报告它的类分布，并与"策略 logits 选类"（先前诊断：`argmax` 基本上是闪电 17）对比。

类的语义（`code/my_ai/decoder.py:34-71`）：**0-15** 建塔/升到某塔型、**16** 降级、
**17-20** 超武（17=闪电）、**21-22** 基地升级、**23** HOLD。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/diag_value_class_choice.py \
      --data training_history/vprior/vp_A_k5_m32.npz --max-decisions 6000
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

NAME = {17: "LIGHTNING", 18: "EMP", 19: "DEFLECTOR", 20: "EVASION",
        16: "downgrade", 21: "base_gen", 22: "base_ant", 23: "HOLD"}


def label(c: int) -> str:
    return NAME.get(c, f"tower_c{c}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "training_history" / "vprior" / "vp_A_k5_m32.npz"))
    ap.add_argument("--max-decisions", type=int, default=6000)
    a = ap.parse_args()

    z = np.load(a.data, allow_pickle=True)
    stats, player = z["stats"], z["player"]
    head, cls, adv, cnt = z["head"], z["cls"], z["adv"], z["cnt"]
    key = np.concatenate([stats.astype(np.float32), player[:, None].astype(np.float32)], axis=1)
    same = np.all(key[1:] == key[:-1], axis=1)
    bounds = [0] + (np.nonzero(~same)[0] + 1).tolist() + [len(player)]
    n_dec = len(bounds) - 1
    step = max(1, n_dec // a.max_decisions)

    hist = collections.Counter()
    hist_if_light_unavail = collections.Counter()
    n_heads = 0
    light_available = 0
    chosen_light = 0
    pair = collections.Counter()

    for di in range(0, n_dec, step):
        lo, hi = bounds[di], bounds[di + 1]
        by_head = collections.defaultdict(list)
        for j in range(lo, hi):
            by_head[int(head[j])].append(j)
        for h, rows in by_head.items():
            playable = [j for j in rows if int(cnt[j]) > 0]
            if not playable:
                continue
            n_heads += 1
            amax = {j: float(np.max(adv[j])) for j in playable}
            best = max(amax, key=amax.get)
            c_best = int(cls[best])
            hist[label(c_best)] += 1
            has_light = any(int(cls[j]) == 17 for j in playable)
            if has_light:
                light_available += 1
                if c_best == 17:
                    chosen_light += 1
            else:
                hist_if_light_unavail[label(c_best)] += 1
            # "价值选类" vs "HOLD（什么都不做）"：价值的绝对水平
            pair["价值最优的 adv>0" if amax[best] > 0 else "价值最优的 adv<=0"] += 1

    print(f"数据: {Path(a.data).name}  决策={n_dec}  可比 (决策, head) 对={n_heads}")
    print(f"其中『闪电此刻可执行』的占 {light_available/max(n_heads,1):.1%}；"
          f"这些里价值仍选闪电的占 {chosen_light/max(light_available,1):.1%}")
    print("\n【价值选类】的类分布（前 12）：")
    for k, v in hist.most_common(12):
        print(f"    {k:14s} {v:6d}  {v/max(n_heads,1):6.1%}")
    print("\n【闪电不可执行时】价值选什么（前 12）：")
    tot = sum(hist_if_light_unavail.values())
    for k, v in hist_if_light_unavail.most_common(12):
        print(f"    {k:14s} {v:6d}  {v/max(tot,1):6.1%}")
    print("\n价值最优项的 adv 正负：", dict(pair))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
