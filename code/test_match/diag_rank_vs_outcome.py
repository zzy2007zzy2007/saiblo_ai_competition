"""放大版诊断：**"位置先验"与"价值网"对格子的排序，跟真实结果代理相不相关？**
（回答用户 2026-10-06 的命题："位置网的高分位置与真正的好位置几乎无关 ⇒ 对搜索没有增强"）

做法（离线、单进程，不跑对局）：
  1. 从既有日志重放一局；在若干决策点上，取**机制真正会用的那个类**
     （闪电可执行 ⇒ 17；否则 ⇒ `action_map` 最热的经济类，与 `pos_pin=mc_quiet` 一致）；
  2. 该类的合法格里采样（先验 top-8 + 其余随机 8 格，共 ≈16 格）；对每格记三个数：
       * `map`   = 位置头该通道的值（**先验**）
       * `val`   = 把这一手打下去后，**价值网**对该局面的输出（**搜索的叶评价**）
       * `out`   = **安静 rollout** 的终局血量差（**结果代理**；有偏但低方差）
  3. 逐决策算 Spearman(map, out) 与 Spearman(val, out)；再算"map 的 top-1 / val 的 top-1
     在 out 里的百分位"（50% = 与随机无异）；最后**按"闪电(17) / 经济类(0-16)"分组汇总**。

判读（**先写死**）：
  * 若某一路（map 或 val）的 Spearman 均值 |ρ|<0.10、且 top-1 百分位≈50% ⇒ **那一路对搜索没有增强**；
  * ρ ≥ 0.30 且 top-1 百分位 ≥ 65% ⇒ 那一路是有信息的。
  * ⚠️ 结果代理是**有偏**的安静 rollout（不是真值）⇒ 结论只对"该代理"成立；要换成真值需跑真实对局。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/diag_rank_vs_outcome.py \
      --tag M6cROOT_128 --seed 7 --decisions 40 --cells 16
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from SDK.utils.features import FeatureExtractor                      # noqa: E402
from SDK.backend.model import Operation                              # noqa: E402
from SDK.utils.constants import OperationType                        # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade        # noqa: E402
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt        # noqa: E402
from my_ai.az_intent.mc_quiet import quiet_value                     # noqa: E402
from my_ai.decoder import decode_head, make_class_mask, make_position_masks  # noqa: E402

OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def read_ops(tag: str, name: str):
    txt = (REPO / "match_results" / "ladder_logs" / tag / name).read_text(encoding="utf-8", errors="replace")
    out: dict[int, dict[int, list]] = {}
    for m in OPLINE.finditer(txt):
        out.setdefault(int(m.group(1)), {})[int(m.group(2)[1])] = [
            (int(o.group(2)), int(o.group(3)), int(o.group(4))) for o in OP.finditer(m.group(5))]
    return out


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="M6cROOT_128")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--decisions", type=int, default=40)
    ap.add_argument("--cells", type=int, default=16)
    ap.add_argument("--rollout-rounds", type=int, default=256)
    ap.add_argument("--ckpt", default="training_history/vprior/posnet_A_k5_m32.pt")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rounds = read_ops(args.tag, f"{args.seed}.log")
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)

    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    rec = []          # (kind, rho_map, rho_val, pct_map, pct_val, n)
    picked = 0
    for r in range(0, min(max(rounds) + 1, 512)):
        for pl in (0, 1):
            if pl == 0 and picked < args.decisions and not st.terminal:
                out = net_fn(st, pl)
                am = np.asarray(out["action_map"], dtype=np.float32)
                hl = np.asarray(out["head_logits"][0], dtype=np.float32)
                pm = make_position_masks(st, pl, intent_decoding=True)
                cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
                # 与 mc_quiet 一致：闪电"真的放得出来"（含金币）才选 17
                cls_id, kind = None, None
                can_light = False
                if cm[17] and pm[17].any() and int(st.coins[pl]) >= 90:
                    op17 = decode_head(hl, am, np.array(cm, copy=True), np.array(pm, copy=True),
                                       st, pl, class_id=17, temperature=0.0,
                                       pos_temperature=0.0, intent_decoding=True)
                    can_light = op17 is not None
                if can_light:
                    cls_id, kind = 17, "light"
                else:
                    best_v, best_c = -np.inf, None
                    for c in range(0, 17):
                        if cm[c] and pm[c].any():
                            v = float(np.where(pm[c], am[c], -np.inf).max())
                            if v > best_v:
                                best_v, best_c = v, c
                    cls_id, kind = best_c, "econ"
                if cls_id is not None and pm[cls_id].any():
                    ys, xs = np.where(pm[cls_id])
                    vals = am[cls_id][ys, xs]
                    order = np.argsort(-vals)
                    top = list(order[:8])
                    rest = [i for i in range(len(order)) if i not in set(top)]
                    rng.shuffle(rest)
                    sel = top + rest[: max(0, args.cells - len(top))]
                    maps, vnets, outs = [], [], []
                    for i in sel:
                        y, x = int(ys[i]), int(xs[i])
                        am2 = np.full_like(am, -1e9)
                        am2[cls_id, y, x] = float(am[cls_id, y, x])
                        op = decode_head(hl, am2, np.array(cm, copy=True), np.array(pm, copy=True),
                                         st, pl, class_id=int(cls_id), temperature=0.0,
                                         pos_temperature=0.0, intent_decoding=True)
                        if op is None:
                            continue
                        s2 = st.clone()
                        s2.apply_operation_list(pl, [Operation(OperationType(int(op.op_type)),
                                                               int(op.arg0), int(op.arg1))])
                        maps.append(float(am[cls_id, y, x]))
                        vnets.append(float(net_fn(s2, pl)["value"]))
                        outs.append(float(quiet_value(s2, pl, max_rounds=args.rollout_rounds)))
                    if len(outs) >= 8:
                        rm, rv = spearman(maps, outs), spearman(vnets, outs)
                        o = np.asarray(outs)
                        pm_pct = float((o < o[int(np.argmax(maps))]).mean() * 100)
                        pv_pct = float((o < o[int(np.argmax(vnets))]).mean() * 100)
                        rec.append((kind, rm, rv, pm_pct, pv_pct, len(outs)))
                        print(f"[r={r} cls={cls_id}({kind}) n={len(outs)}] ρ_map={rm:+.3f} ρ_val={rv:+.3f} "
                              f"map-top1 百分位={pm_pct:.0f}% val-top1 百分位={pv_pct:.0f}%", flush=True)
                        picked += 1
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in ops])
            if pl == 1:
                st.advance_round()
        if picked >= args.decisions or st.terminal:
            break

    print("\n=== 汇总（结果代理 = 安静 rollout；随机基准：ρ=0、top-1 百分位=50%）===")
    for kind, label in (("light", "闪电(17)"), ("econ", "经济类(0-16)")):
        rows = [x for x in rec if x[0] == kind]
        if not rows:
            print(f"{label}: 无样本"); continue
        rm = np.array([x[1] for x in rows]); rv = np.array([x[2] for x in rows])
        pm_ = np.array([x[3] for x in rows]); pv_ = np.array([x[4] for x in rows])
        print(f"{label}: n={len(rows)} 决策点（每点 {rows[0][5]} 格）")
        print(f"    ρ(先验, 结果): 均值 {rm.mean():+.3f} 中位 {np.median(rm):+.3f} |ρ|≥0.3 占比 {100*np.mean(np.abs(rm)>=0.3):.0f}%")
        print(f"    ρ(价值网, 结果): 均值 {rv.mean():+.3f} 中位 {np.median(rv):+.3f} |ρ|≥0.3 占比 {100*np.mean(np.abs(rv)>=0.3):.0f}%")
        print(f"    先验 top-1 格的结果百分位: 均值 {pm_.mean():.0f}% ；价值网 top-1: 均值 {pv_.mean():.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
