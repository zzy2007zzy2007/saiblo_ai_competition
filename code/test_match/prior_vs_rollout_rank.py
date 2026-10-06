"""直接检验用户的命题：**位置网给的高分格子，与"真正好"的格子相不相关？**
（= 它对搜索有没有增强作用）

做法（离线，不跑对局）：
  1. 从既有日志把 seed 的整局操作重放到某个局面（得到真实的中局状态：有塔、有金币）；
  2. 在若干"决策点"上取**我们机制真正会用的类**（闪电可执行 ⇒ 17；否则 ⇒ 经济类里 map 最热的那个，
     与 `pos_pin=mc_quiet` 的候选构造一致）；
  3. 对该类的**合法格**采样（map 的 top-8 + 其余随机 N 个，共 ≤24 格），对每格：
       * `map_val`  = 位置头该通道在该格的值（**先验**）
       * `score`    = **安静 rollout** 的终局血量差（**我们机制实际用的评价**）
     做法：把该通道的 action_map 只留这一格（其余 −inf）⇒ `decode_head` 必然解出这一格 ⇒ apply ⇒ 推进到终局；
  4. 统计**先验值 与 评分的 Spearman 相关**（逐决策 + 汇总），以及
     "先验 top-1 格在评分里的百分位"（50% = 与随机无异）。

判读（**先写死**）：
  * 相关 ≈ 0（|ρ|<0.1）或 top-1 百分位 ≈50% ⇒ **用户命题成立**：先验对搜索无增强（该把位置也交给模拟/或换先验）；
  * ρ ≥ 0.3 且 top-1 百分位 ≥ 70% ⇒ 先验是有信息的（用户命题不成立）。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/prior_vs_rollout_rank.py \
      --tag M6cROOT_128 --seed 7 --decisions 12 --cells 24
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
    f = REPO / "match_results" / "ladder_logs" / tag / name
    txt = f.read_text(encoding="utf-8", errors="replace")
    out: dict[int, dict[int, list]] = {}
    for m in OPLINE.finditer(txt):
        rnd, pl = int(m.group(1)), int(m.group(2)[1])
        out.setdefault(rnd, {})[pl] = [(int(o.group(2)), int(o.group(3)), int(o.group(4)))
                                       for o in OP.finditer(m.group(5))]
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="M6cROOT_128")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--decisions", type=int, default=12, help="采样多少个决策点")
    ap.add_argument("--cells", type=int, default=24, help="每个决策采样多少格")
    ap.add_argument("--stop-round", type=int, default=9999)
    ap.add_argument("--ckpt", default="training_history/vprior/posnet_A_k5_m32.pt")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rounds = read_ops(args.tag, f"{args.seed}.log")
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)

    # 重放，并在采样到的决策点上做对照
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    picked = 0
    rhos, pcts = [], []
    maxr = min(max(rounds) + 1, args.stop_round)
    for r in range(max_r := maxr):
        for pl in (0, 1):
            if st.terminal:
                break
            if pl == 0 and picked < args.decisions:
                out = net_fn(st, pl)
                am = np.asarray(out["action_map"], dtype=np.float32)
                hl = np.asarray(out["head_logits"][0], dtype=np.float32)
                pm = make_position_masks(st, pl, intent_decoding=True)
                cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
                # 与 mc_quiet 的候选构造一致：闪电真能放 ⇒ 17；否则取 map 最热的经济类
                cls_id = None
                if cm[17] and pm[17].any():
                    cls_id = 17
                else:
                    best_v, best_c = -np.inf, None
                    for c in range(0, 17):
                        if cm[c] and pm[c].any():
                            v = float(np.where(pm[c], am[c], -np.inf).max())
                            if v > best_v:
                                best_v, best_c = v, c
                    cls_id = best_c
                if cls_id is not None and pm[cls_id].any():
                    ys, xs = np.where(pm[cls_id])
                    vals = am[cls_id][ys, xs]
                    order = np.argsort(-vals)
                    top = list(order[:8])
                    rest = [i for i in range(len(order)) if i not in set(top)]
                    rng.shuffle(rest)
                    sel = top + rest[:max(0, args.cells - len(top))]
                    map_vals, scores = [], []
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
                        if pl == 1:
                            s2.advance_round()
                        sc = quiet_value(s2, pl, max_rounds=256)
                        map_vals.append(float(am[cls_id, y, x]))
                        scores.append(float(sc))
                    if len(scores) >= 6:
                        a, b = np.asarray(map_vals), np.asarray(scores)
                        rho = spearman(a, b)
                        # 先验 top-1 在评分里的百分位
                        bi = int(np.argmax(a))
                        pct = float((b < b[bi]).mean() * 100.0)
                        if np.isfinite(rho):
                            rhos.append(rho); pcts.append(pct)
                        print(f"[r={r} p{pl} cls={cls_id}] n={len(scores)} ρ={rho:+.3f} "
                              f"先验top1评分百分位={pct:.0f}%  评分范围=[{b.min():+.3f},{b.max():+.3f}]",
                              flush=True)
                        picked += 1
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in ops])
            if pl == 1:
                st.advance_round()
        if picked >= args.decisions:
            break

    if rhos:
        print("\n=== 汇总 ===")
        print(f"决策点 n={len(rhos)}  先验值 vs 安静-rollout 评分的 Spearman: "
              f"均值 {np.mean(rhos):+.3f}  中位 {np.median(rhos):+.3f}  "
              f"|ρ|>=0.3 的占比 {100*float(np.mean(np.abs(rhos) >= 0.3)):.0f}%")
        print(f"先验 top-1 格的评分百分位: 均值 {np.mean(pcts):.0f}%  中位 {np.median(pcts):.0f}%  "
              f"（随机基准 = 50%；>50% 才说明先验挑得比随机好）")
    else:
        print("没采到有效决策点（试试 --tag/--seed 换一批日志）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
