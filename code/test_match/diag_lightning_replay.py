"""用**回放**重建两侧金币，验证「升级吃钱 ⇒ 闪电放不起」 vs 「值网主动少放」。

为什么回放：配对模式下同一 seed 的两局并发，`ai{0,1}_..._seed{S}.stderr.log` **会互相覆盖**
⇒ 桥接日志不能用来区分两侧（实测两侧统计完全相同，就是这个原因）。
回放是确定性的（今天已验证 6/6 复现），且能拿到**每一回合双方的准确金币**。

闪电规则：cost=90、cooldown=35（SDK SUPER_WEAPON_STATS）。

输出（A=值网+升级@h64 / B=quiet@H256+不升级）：
  1. 就绪回合数、其中施放数、施放率
  2. 就绪回合金币中位 / 买得起(≥90)比例
  3. **买得起且就绪**的回合里，施放率（这才是"想不想放"的干净指标）
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

from SDK.backend.model import Operation                     # noqa: E402
from SDK.utils.constants import OperationType               # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

COOLDOWN = 35
COST = 90
AB = REPO / "training_history/ab_challenger_vs_champion"
pat = re.compile(r".*\[round (\d+)\] (p[01])\([^)]*\)[^\n]*?-> \[(.*?)\]\s*$")
op_pat = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def load_rounds(log: Path) -> dict[int, dict[str, list[tuple[int, int, int]]]]:
    out: dict[int, dict[str, list]] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line)
        if not m:
            continue
        rnd, seat, body = int(m.group(1)), m.group(2), m.group(3)
        ops = [(int(o.group(2)), int(o.group(3)), int(o.group(4))) for o in op_pat.finditer(body)]
        out.setdefault(rnd, {})[seat] = ops
    return out


def trace(seed: int, a_seat: str):
    """回放一局，返回每回合双方的 (coins, 是否放闪电)。"""
    rounds = load_rounds(AB / f"seed{seed}_{a_seat}.log")
    if not rounds:
        return None
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    rec = []                                # [(round, coinsA, coinsB, castA, castB)]
    max_r = max(rounds)
    for r in range(max_r + 1):
        casts = {"p0": False, "p1": False}
        for seat in ("p0", "p1"):
            ops = rounds.get(r, {}).get(seat, [])
            if not ops:
                continue
            try:
                st.apply_operation_list(int(seat[1]),
                                        [Operation(OperationType(t), a0, a1) for t, a0, a1 in ops])
            except Exception:
                continue
            if any(t == int(OperationType.USE_LIGHTNING_STORM) for t, _a, _b in ops):
                casts[seat] = True
        try:
            st.advance_round()
        except Exception:
            break
        c0, c1 = int(st.coins[0]), int(st.coins[1])
        a_coins = c0 if a_seat == "p0" else c1
        b_coins = c1 if a_seat == "p0" else c0
        rec.append((r, a_coins, b_coins, casts["p0"] if a_seat == "p0" else casts["p1"],
                    casts["p1"] if a_seat == "p0" else casts["p0"]))
        if getattr(st, "terminal", False):
            break
    return rec


agg = {k: {"ready": 0, "cast": 0, "ready_coins": [], "afford_ready": 0, "afford_cast": 0}
       for k in ("A", "B")}
games = 0
for seed in range(960, 976):
    for a_seat in ("p0", "p1"):
        rec = trace(seed, a_seat)
        if not rec:
            continue
        games += 1
        res = {}
        for key, ci, ki in (("A", 1, 3), ("B", 2, 4)):
            last = -10**9
            ready = cast = aff_ready = aff_cast = 0
            coins_at_ready = []
            for rnd, ca, cb, ka, kb in rec:
                coins = ca if key == "A" else cb
                cast_now = ka if key == "A" else kb
                if rnd - last < COOLDOWN:
                    if cast_now:
                        last = rnd
                    continue
                ready += 1
                coins_at_ready.append(coins)
                if cast_now:
                    cast += 1
                    last = rnd
                if coins >= COST:
                    aff_ready += 1
                    if cast_now:
                        aff_cast += 1
            res[key] = (ready, cast, coins_at_ready, aff_ready, aff_cast)
        for key in ("A", "B"):
            ready, cast, cs, aff_ready, aff_cast = res[key]
            s = agg[key]
            s["ready"] += ready
            s["cast"] += cast
            s["ready_coins"].extend(cs)
            s["afford_ready"] += aff_ready
            s["afford_cast"] += aff_cast

print(f"回放了 {games} 局（闪电 cost={COST} / cooldown={COOLDOWN}）\n")
hdr = f"{'':<16}{'就绪':>7}{'施放':>6}{'施放率':>8}{'买得起且就绪':>13}{'其中施放':>9}{'条件施放率':>11}{'就绪时金币中位':>15}"
print(hdr)
for k, label in (("A", "A 值网+升级"), ("B", "B quiet+不升级")):
    s = agg[k]
    if not s["ready"]:
        print(f"{label:<16}（无数据）")
        continue
    rate = 100.0 * s["cast"] / s["ready"]
    cond = 100.0 * s["afford_cast"] / max(1, s["afford_ready"])
    med = statistics.median(s["ready_coins"]) if s["ready_coins"] else float("nan")
    print(f"{label:<16}{s['ready']:>7}{s['cast']:>6}{rate:>7.1f}%{s['afford_ready']:>13}"
          f"{s['afford_cast']:>9}{cond:>10.1f}%{med:>15.0f}")
