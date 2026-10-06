"""查清**类号 → 解码出的 op** 的对应（尤其：哪个类会在"格子上已有我方塔"时给出升级 op 12）。

动机（2026-10-07 第 3 轮）：我们 196 局里 `UPGRADE_TOWER = 0`，而对手 9.4/局。
已查明：升级 op 的正确参数是 `(tower_id, target_type)`（SDK 官方样例），我第一版误用 `(x,y)` ⇒ 引擎静默拒绝。
本脚本回答"**怎么用现有 class+position 通路表达升级**"：对每个类，分别在
  ① 一个**空格**、② 一座**我方已有塔的格子** 上尖峰解码，看 op 类型。
用法:
  python -u code/test_match/probe_class_to_op.py --tag SMOKE_lvboth --seed 9 --round 120
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

from SDK.utils.features import FeatureExtractor                       # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade         # noqa: E402
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt         # noqa: E402
from my_ai.decoder import decode_head, make_class_mask, make_position_masks  # noqa: E402

OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="SMOKE_lvboth")
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--round", type=int, default=120)
    ap.add_argument("--ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    args = ap.parse_args()

    txt = (REPO / "match_results" / "ladder_logs" / args.tag / f"{args.seed}.log").read_text(
        encoding="utf-8", errors="replace")
    rounds: dict[int, dict[int, list]] = {}
    for m in OPLINE.finditer(txt):
        rounds.setdefault(int(m.group(1)), {})[int(m.group(2)[1])] = [
            (int(o.group(2)), int(o.group(3)), int(o.group(4))) for o in OP.finditer(m.group(5))]

    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    for r in range(0, args.round):
        for p in (0, 1):
            for (t, a0, a1) in rounds.get(r, {}).get(p, []):
                from SDK.backend.model import Operation
                from SDK.utils.constants import OperationType
                st.apply_operation_list(p, [Operation(OperationType(t), a0, a1)])
            if p == 1:
                st.advance_round()
        if st.terminal:
            break

    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)
    out = net_fn(st, 0)
    am = np.asarray(out["action_map"], dtype=np.float32)
    hl = np.asarray(out["head_logits"][0], dtype=np.float32)
    pm = make_position_masks(st, 0, intent_decoding=True)
    cm = make_class_mask(st, 0, position_mask=pm, intent_decoding=True)
    tws = st.towers_of(0)
    print(f"[状态] round={getattr(st, 'round_index', '?')} coins={st.coins} 我方塔={[(t.tower_id,t.x,t.y,str(t.tower_type),t.level) for t in tws]}")
    empty = None
    for c in range(18):
        if cm[c] and pm[c].any():
            ys, xs = np.where(pm[c])
            for i in range(len(ys)):
                if st.tower_at(int(xs[i]), int(ys[i])) is None:
                    empty = (int(xs[i]), int(ys[i])); break
        if empty:
            break
    print(f"  取一个空格做『建塔』对照: {empty}")

    # 尖峰解码：类 c × {空格, 我方塔格}
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    def spike(cid, x, y):
        am2 = np.full_like(am, -1e9)
        am2[cid, y, x] = float(am[cid, y, x])
        return decode_head(hl, am2, np.array(cm, copy=True), np.array(pm, copy=True),
                           st, 0, class_id=int(cid), temperature=0.0,
                           pos_temperature=0.0, intent_decoding=True)

    print("\n类号 | 空格上的 op | 我方塔格上的 op")
    for c in range(0, 18):
        a = spike(c, *empty) if empty and pm[c].any() else None
        b = None
        if tws:
            w = tws[0]
            if pm[c].any():
                b = spike(c, int(w.x), int(w.y))
        fmt = lambda o: f"({int(o.op_type)},{o.arg0},{o.arg1})" if o is not None else "—"
        mark = ""
        if b is not None and int(b.op_type) == 12:
            mark = "   <== **升级**"
        if b is not None and int(b.op_type) == 13:
            mark = "   (降级)"
        print(f"  {c:>2} | {fmt(a):>16} | {fmt(b):>16}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
