"""检验用户的命题：**价值网能分辨"闪电打哪儿"，但分辨不了"塔建哪儿"**。

做法（离线，不跑对局）：
  1. 从既有日志重放到某个中局状态（有塔/有金币）；
  2. **对照 A（塔）**：取一个可建的塔类，枚举其合法格，对每格把动作打下去，读价值网对该局面的输出；
  3. **对照 B（闪电）**：对闪电通道做同样的事；
  4. 报：两种情况下"位置引起的价值差异"（max−min、std）与**价值量级**（同一局面 value 的绝对值尺度）。

判读（先写死）：
  * 若 B 的 spread 明显大于 A（例如 ≥2×）⇒ **用户命题成立**（价值网看得到闪电位置、看不清塔位置）
    ⇒ 搜索的 value 项无法给"塔建哪儿"排序 ⇒ 塔位置在机制里≈随机。
  * 若两者同量级 ⇒ 命题不成立（价值网对两类动作的位置都同样钝/同样灵）。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/diag_value_tower_positions.py \
      --tag M6cROOT_128 --seed 7 --cells 8
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
from my_ai.az_intent.mc_quiet import quiet_value                          # noqa: E402
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
    import numpy as _np
    a = _np.asarray(a, dtype=float); b = _np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")
    ra = _np.argsort(_np.argsort(a)).astype(float); rb = _np.argsort(_np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = _np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def spread_for_class(net_fn, st, pl, am, hl, cm, pm, cls_id, cells, rng, with_rollout=False):
    """把 cls_id 的动作分别打到 cells 指定的合法格上，读价值网输出 → (values, ops)"""
    ys, xs = np.where(pm[cls_id])
    if len(ys) == 0:
        return [], []
    vals = am[cls_id][ys, xs]
    order = np.argsort(-vals)            # 按先验最热排序取前几格（模拟"机制会考虑的那些格"）
    sel = order[:cells]
    out = []
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
        s2.apply_operation_list(pl, [Operation(OperationType(int(op.op_type)), int(op.arg0), int(op.arg1))])
        v = float(net_fn(s2, pl)["value"])
        sc = float(quiet_value(s2, pl, max_rounds=256)) if with_rollout else float("nan")
        out.append((v, (y, x), (int(op.op_type), int(op.arg0), int(op.arg1)), sc))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="M6cROOT_128")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--round", type=int, default=-1, help="-1 = 自动找一个闪电可执行的回合；否则用给定回合")
    ap.add_argument("--cells", type=int, default=8)
    ap.add_argument("--ckpt", default="training_history/vprior/posnet_A_k5_m32.pt")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rounds = read_ops(args.tag, f"{args.seed}.log")
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    pl = 0
    target = args.round if args.round >= 0 else 10 ** 9
    for r in range(0, min(max(rounds) + 1, 512)):
        for p in (0, 1):
            ops = rounds.get(r, {}).get(p, [])
            if ops:
                st.apply_operation_list(p, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in ops])
            if p == 1:
                st.advance_round()
        if r + 1 >= target:
            break
        if args.round < 0 and r >= 20:
            # 自动模式：遇到"闪电可执行 + 有可建塔类"的回合就停下
            _pm = make_position_masks(st, pl, intent_decoding=True)
            _cm = make_class_mask(st, pl, position_mask=_pm, intent_decoding=True)
            # ⚠️ `intent_decoding` 的类掩码**只查冷却、不查金币** ⇒ 必须自己加"买得起"这一条
            _afford = int(st.coins[pl]) >= 90
            if _afford and _cm[17] and _pm[17].any() and any(_cm[c] and _pm[c].any() for c in range(0, 16)):
                break
        if st.terminal:
            break
    print(f"[状态] round≈{getattr(st,'round_index','?')} hp={[b.hp for b in st.bases]} coins={st.coins} "
          f"terminal={st.terminal}")

    out = net_fn(st, pl)
    am = np.asarray(out["action_map"], dtype=np.float32)
    hl = np.asarray(out["head_logits"][0], dtype=np.float32)
    pm = make_position_masks(st, pl, intent_decoding=True)
    cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
    v_base = float(out["value"])
    print(f"[基准] 原局面 value = {v_base:+.4f}")

    # ---- 对照 B：闪电（先做，因为它是"好的"那个）----
    if cm[17] and pm[17].any():
        rows = spread_for_class(net_fn, st, pl, am, hl, cm, pm, 17, args.cells, rng, with_rollout=True)
        vs = np.array([x[0] for x in rows])
        if vs.size == 0:
            print(f"\n[B 闪电] 合法格 {int(pm[17].sum())}，但**没有一格能解出 op**（多半是买不起）⇒ 跳过")
            b_spread = b_std = float("nan")
        else:
            print(f"\n[B 闪电] 合法格 {int(pm[17].sum())}，取前 {len(vs)} 格：")
            for v, (y, x), op, sc in rows:
                print(f"    ({y:>2},{x:>2}) value={v:+.4f}  Δ={v - v_base:+.4f}  rollout={sc:+.4f}  op={op}")
            print(f"  ⇒ spread(max−min)={vs.max() - vs.min():.4f}  std={vs.std():.4f}")
            b_spread, b_std = float(vs.max() - vs.min()), float(vs.std())
            print(f"  ⇒ Spearman(价值网, 安静rollout) = {spearman(vs, [r[3] for r in rows]):+.3f}")
    else:
        print("\n[B 闪电] 此刻不可执行，跳过")
        b_spread = b_std = float("nan")

    # ---- 对照 A：塔（选"合法格最多"的可建类）----
    best_c, best_n = None, 0
    for c in range(0, 16):
        if cm[c] and int(pm[c].sum()) > best_n:
            best_c, best_n = c, int(pm[c].sum())
    if best_c is not None:
        rows = spread_for_class(net_fn, st, pl, am, hl, cm, pm, best_c, args.cells, rng, with_rollout=True)
        vs = np.array([x[0] for x in rows])
        if vs.size == 0:
            print(f"\n[A 塔 cls={best_c}] 没有一格能解出 op ⇒ 跳过")
            a_spread = a_std = float("nan")
            vs = None
        else:
            print(f"\n[A 塔 cls={best_c}] 合法格 {best_n}，取前 {len(vs)} 格：")
            for v, (y, x), op, sc in rows:
                print(f"    ({y:>2},{x:>2}) value={v:+.4f}  Δ={v - v_base:+.4f}  rollout={sc:+.4f}  op={op}")
            print(f"  ⇒ spread(max−min)={vs.max() - vs.min():.4f}  std={vs.std():.4f}")
            a_spread, a_std = float(vs.max() - vs.min()), float(vs.std())
            print(f"  ⇒ Spearman(价值网, 安静rollout) = {spearman(vs, [r[3] for r in rows]):+.3f}")
    else:
        print("\n[A 塔] 此刻没有可建类，跳过")
        a_spread = a_std = float("nan")

    print("\n=== 判读 ===")
    print(f"闪电位置 spread={b_spread:.4f}（std {b_std:.4f}）；塔位置 spread={a_spread:.4f}（std {a_std:.4f}）")
    if np.isfinite(a_spread) and np.isfinite(b_spread) and b_spread > 0:
        print(f"比值（闪电/塔）= {b_spread / max(a_spread, 1e-9):.1f}× ⇒ "
              + ("**塔位置确实被价值网压平**（用户命题成立）" if b_spread / max(a_spread, 1e-9) >= 2
                 else "两者同量级（命题不成立）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
