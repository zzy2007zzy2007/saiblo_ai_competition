"""复现 M6（`pos_pin=mc_light`）在 seed 11 第 55 回合的原生崩溃 —— **跑真实的 `BundleMCTS.search`**。

与 `repro_rollout_crash.py` 的区别：那个只跑了我手写的 rollout（不崩）；
这里**完整复刻 bridge 的调用链**：重放局面 → `BundleMCTS(... mc_light ...).search(facade, 0)`，
并在 search 内部的关键点落盘（round / 分支 / 候选 / apply），配合 `python -X faulthandler` 抓原生栈。

用法:
  python -X faulthandler -u code/test_match/repro_search_crash.py --seed 11 --stop-round 55 \
      --pos-pin mc_light --mc-every 4 --mc-horizon 100 --out training_history/vprior/crash_trace2.log
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

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
from my_ai.az_intent.bundle_mcts import BundleMCTS                   # noqa: E402

OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def read_ops(tag: str, name: str):
    f = REPO / "match_results" / "ladder_logs" / tag / name
    txt = f.read_text(encoding="utf-8", errors="replace")
    out: dict[int, dict[int, list]] = {}
    for m in OPLINE.finditer(txt):
        rnd, pl = int(m.group(1)), int(m.group(2)[1])
        ops = [(int(o.group(2)), int(o.group(3)), int(o.group(4))) for o in OP.finditer(m.group(5))]
        out.setdefault(rnd, {})[pl] = ops
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="M6_retry8")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--stop-round", type=int, default=55)
    ap.add_argument("--iters", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--sm", type=int, default=15)
    ap.add_argument("--pos-pin", default="mc_light")
    ap.add_argument("--mc-every", type=int, default=4)
    ap.add_argument("--mc-horizon", type=int, default=100)
    ap.add_argument("--ckpt", default="training_history/vprior/posnet_A_k5_m32.pt")
    ap.add_argument("--out", default="training_history/vprior/crash_trace2.log")
    args = ap.parse_args()

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w", encoding="utf-8")

    def log(msg: str) -> None:
        fh.write(msg + "\n")
        fh.flush()

    # ---- 重放 ----
    rounds = read_ops(args.tag, f"{args.seed}.log")
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    for r in range(0, args.stop_round):
        per = rounds.get(r, {})
        for pl in (0, 1):
            ops = per.get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1) for (t, a0, a1) in ops])
            if pl == 1:
                st.advance_round()
    log(f"[repro] 重放完成 round_index={getattr(st,'round_index','?')} hp={[b.hp for b in st.bases]} "
        f"coins={st.coins} terminal={st.terminal}  （期望 hp=[46,47] coins=[95,63]）")

    # ---- 真实搜索 ----
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)
    mcts = BundleMCTS(
        net_fn,
        iterations=args.iters, max_depth_rounds=args.depth, k=args.k, sample_mult=args.sm,
        t_class=0.5, t_pos=1.0, c_puct=1.25, seed=args.seed,
        search_mode="pos-only", pos_pin=args.pos_pin, skip_single_candidate=True,
        class_pin_random_prob=0.0, reserve_coins=90,
        mc_horizon=args.mc_horizon, mc_every=args.mc_every,
    )
    log(f"[repro] 开始 search（pos_pin={args.pos_pin} iters={args.iters} depth={args.depth} "
        f"k={args.k} sm={args.sm} mc_every={args.mc_every} mc_horizon={args.mc_horizon}）")
    res = mcts.search(st, 0, temperature=1e-6)
    log(f"[repro] search 正常返回：chosen_bundle={res.chosen_bundle} "
        f"root_value={getattr(res, 'root_value', None)} mc_taken={getattr(mcts,'mc_taken',None)} "
        f"mc_fallback={getattr(mcts,'mc_fallback',None)}")
    fh.close()
    print(f"[repro] 未崩溃；轨迹见 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
