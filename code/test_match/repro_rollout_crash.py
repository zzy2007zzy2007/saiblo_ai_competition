"""复现"贪心 rollout 原生崩溃"（M6 / `pos_pin=mc_light`）并定位到**具体哪一个 op**。

做法（不需要跑对局）：
  1. 从 `match_results/ladder_logs/<tag>/11.log` 把 **seed 11 的整局操作序列**读出来；
  2. 用 `GameStateFacade` **逐回合重放**到第 55 回合（= 崩溃发生处）：
     每回合先 apply p0 的 op 列表、再 apply p1 的、然后 advance_round；
  3. 从这里跑**贪心 rollout**（`mc_class.greedy_step` + apply），
     **每次 apply 之前先把 (round, player, op, 该 op 在 SDK 掩码下是否合法) 落盘并 flush**；
  4. ⚠️ 用 `python -X faulthandler` 跑：原生访问违例时 faulthandler 会打印 Python 调用栈，
     从而知道是 `apply_operation_list` 崩的还是别处。

用法:
  python -X faulthandler -u code/test_match/repro_rollout_crash.py [--tag M6_retry8] [--seed 11]
         [--stop-round 55] [--max-rounds 200] [--out training_history/vprior/crash_trace.log]
"""
from __future__ import annotations

import argparse
import json
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
from my_ai.az_intent.mc_class import greedy_step                     # noqa: E402
from my_ai.decoder import make_class_mask, make_position_masks       # noqa: E402

OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def read_ops(tag: str, name: str):
    """→ {round: {player: [(op_type, arg0, arg1), ...]}}（保持日志顺序）"""
    f = REPO / "match_results" / "ladder_logs" / tag / name
    if not f.exists():
        raise SystemExit(f"缺日志: {f}")
    txt = f.read_text(encoding="utf-8", errors="replace")
    out: dict[int, dict[int, list]] = {}
    for m in OPLINE.finditer(txt):
        rnd = int(m.group(1))
        pl = int(m.group(2)[1])
        ops = [(int(o.group(2)), int(o.group(3)), int(o.group(4))) for o in OP.finditer(m.group(5))]
        out.setdefault(rnd, {})[pl] = ops
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="M6_retry8")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--stop-round", type=int, default=55)
    ap.add_argument("--max-rounds", type=int, default=200)
    ap.add_argument("--ckpt", default="training_history/vprior/posnet_A_k5_m32.pt")
    ap.add_argument("--out", default="training_history/vprior/crash_trace.log")
    args = ap.parse_args()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", encoding="utf-8")

    def log(msg: str) -> None:
        fh.write(msg + "\n")
        fh.flush()          # ⚠️ 关键：崩之前必须落盘

    name = f"{args.seed}.log"
    rounds = read_ops(args.tag, name)
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    log(f"[replay] seed={args.seed} tag={args.tag} 日志里的回合数={len(rounds)} 目标=重放到第 {args.stop_round} 回合前")

    # ---------- 1) 重放 ----------
    for r in range(0, args.stop_round):
        per = rounds.get(r, {})
        for pl in (0, 1):
            ops = per.get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in ops])
                log(f"[replay] r={r} p{pl} apply {ops}")
            if pl == 1:
                st.advance_round()
        if st.terminal:
            log(f"[replay] !! 在第 {r} 回合就终局了，无法到达 {args.stop_round}")
            break
    log(f"[replay] 完成：round_index={getattr(st, 'round_index', '?')} "
        f"hp={[b.hp for b in st.bases]} coins={st.coins} terminal={st.terminal}")

    # ---------- 2) 贪心 rollout ----------
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)
    log("[rollout] 开始（贪心：类=类头 argmax，位置=位置图 argmax；每步 apply 前落盘）")

    r = 0
    while (not st.terminal) and r < args.max_rounds:
        for pl in (0, 1):
            if st.terminal:
                break
            t = greedy_step(net_fn, st, pl)
            if t is None:
                log(f"[rollout] r={r} p{pl} greedy_step=None（不出手）")
            else:
                # 该 op 在 SDK 掩码下合法吗？（帮助区分"非法 op 喂进原生" vs 别的）
                legal = None
                try:
                    pm = make_position_masks(st, pl, intent_decoding=True)
                    cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
                    legal = {"n_legal_pos": int(pm.sum()), "n_legal_cls": int(cm.sum())}
                except Exception as e:  # noqa: BLE001
                    legal = {"mask_err": str(e)}
                log(f"[rollout] r={r} p{pl} ABOUT-TO-APPLY op={t} masks={legal}")
                st.apply_operation_list(pl, [Operation(OperationType(int(t[0])), int(t[1]), int(t[2]))])
                log(f"[rollout] r={r} p{pl} applied ok")
            if pl == 1:
                st.advance_round()
        r += 1
    log(f"[rollout] 正常结束：rounds={r} terminal={st.terminal} hp={[b.hp for b in st.bases]}")
    fh.close()
    print(f"[repro] 未崩溃；轨迹见 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
