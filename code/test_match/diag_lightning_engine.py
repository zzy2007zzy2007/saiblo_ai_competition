"""闪电"能不能放 / 放没放"的**引擎口径**统计（修正版）。

上一版的两个问题（用户指出冷却）：
  1. 就绪状态是**推断**的（"上次施放 +35"），没读引擎真实冷却；
  2. 成本假设恒为 90，没考虑按使用次数涨价（`weapon_cost(weapon_type)`）。
本版全部走引擎：
  · 就绪：`state.weapon_cooldowns[player, LIGHTNING_STORM] == 0`
  · 价格：`state.weapon_cost(LIGHTNING_STORM)`（每次读当前值，可能随使用次数变）
  · 可放：就绪 **且** 金币 ≥ 当前价格
再配合回放得到的"这一回合实际放了没有"，得到：
  · 就绪率、可放率、**可放时的施放率**（这才是"想不想放"的干净指标）
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

from SDK.backend.model import Operation                          # noqa: E402
from SDK.utils.constants import OperationType, SuperWeaponType   # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade    # noqa: E402

LIGHT = int(SuperWeaponType.LIGHTNING_STORM)
OP_LIGHT = int(OperationType.USE_LIGHTNING_STORM)
AB = REPO / "training_history/ab_challenger_vs_champion"
pat = re.compile(r".*\[round (\d+)\] (p[01])\([^)]*\)[^\n]*?-> \[(.*?)\]\s*$")
op_pat = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def load_rounds(log: Path):
    out: dict[int, dict[str, list]] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.match(line)
        if not m:
            continue
        out.setdefault(int(m.group(1)), {})[m.group(2)] = [
            (int(o.group(2)), int(o.group(3)), int(o.group(4))) for o in op_pat.finditer(m.group(3))]
    return out


def trace(seed: int, a_seat: str):
    rounds = load_rounds(AB / f"seed{seed}_{a_seat}.log")
    if not rounds:
        return None
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    rec = []           # [(round, {side: (coins, cd_ready, cost, cast)})]
    for r in range(max(rounds) + 1):
        # ⚠️ 必须在**执行本回合 op 之前**采样：决策时看到的是上回合结算后的金币/冷却。
        #    上一版先 apply 再读 ⇒ 放闪电的那回合金币已被扣 90 ⇒ 永远显示钱不够，
        #    于是可放时施放率恒为 0（2026-10-22 自查发现）。
        info = {}
        try:
            cost = int(st.weapon_cost(LIGHT))
        except Exception:
            cost = -1
        for seat in ("p0", "p1"):
            p = int(seat[1])
            ops = rounds.get(r, {}).get(seat, [])
            cd = int(st.weapon_cooldowns[p, LIGHT])
            info[seat] = (int(st.coins[p]), cd == 0, cost,
                          any(t == OP_LIGHT for t, _a, _b in ops))
        for seat in ("p0", "p1"):
            ops = rounds.get(r, {}).get(seat, [])
            if ops:
                try:
                    st.apply_operation_list(int(seat[1]),
                                            [Operation(OperationType(t), a0, a1) for t, a0, a1 in ops])
                except Exception:
                    pass
        try:
            st.advance_round()
        except Exception:
            break
        rec.append((r, info))
        if getattr(st, "terminal", False):
            break
    return rec


agg = {k: {"rounds": 0, "ready": 0, "afford": 0, "cast": 0, "cast_when_afford": 0,
           "coins_at_afford": [], "costs": []} for k in ("A", "B")}
games = 0
for seed in range(960, 976):
    for a_seat in ("p0", "p1"):
        rec = trace(seed, a_seat)
        if not rec:
            continue
        games += 1
        b_seat = "p1" if a_seat == "p0" else "p0"
        for key, seat in (("A", a_seat), ("B", b_seat)):
            s = agg[key]
            for _r, info in rec:
                coins, ready, cost, cast = info[seat]
                s["rounds"] += 1
                s["costs"].append(cost)
                if ready:
                    s["ready"] += 1
                if cast:
                    s["cast"] += 1
                if ready and cost >= 0 and coins >= cost:
                    s["afford"] += 1
                    s["coins_at_afford"].append(coins)
                    if cast:
                        s["cast_when_afford"] += 1

print(f"回放 {games} 局（引擎口径：冷却读 weapon_cooldowns，价格读 weapon_cost）\n")
print(f"{'':<15}{'总回合':>8}{'就绪':>7}{'可放(就绪+钱够)':>16}{'其中施放':>9}{'可放时施放率':>13}{'施放/局':>9}{'可放/局':>9}")
for k, label in (("A", "A 值网+升级"), ("B", "B quiet+不升级")):
    s = agg[k]
    conv = 100.0 * s["cast_when_afford"] / max(1, s["afford"])
    print(f"{label:<15}{s['rounds']:>8}{s['ready']:>7}{s['afford']:>16}{s['cast_when_afford']:>9}"
          f"{conv:>12.1f}%{s['cast']/max(1,games):>9.1f}{s['afford']/max(1,games):>9.1f}")
print()
for k in ("A", "B"):
    s = agg[k]
    if s["coins_at_afford"]:
        cs = sorted(s["coins_at_afford"])
        print(f"  {k} 可放回合的金币: p25={cs[len(cs)//4]} p50={cs[len(cs)//2]} p75={cs[len(cs)*3//4]}"
              f"   价格中位={statistics.median(s['costs']):.0f}")
