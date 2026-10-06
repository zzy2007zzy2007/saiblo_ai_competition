"""前提检查：**`Operation(UPGRADE_TOWER, x, y)` 到底能不能真的升级我们的塔？**

动机（2026-10-07 第 2 轮）：`AZAI_MC_UPG=2` 把"显式 `Operation(12,x,y)`"放进 MC 候选后
**仍然 0 次升级**。两种可能：① 升级合法、只是**分数输给 HOLD**（⇒ 平局/更差，属策略问题）；
② 升级 op 根本**不合法/无效果**（⇒ 那个臂等于没做，整条"升级"方向的前提就不成立）。
本脚本离线判它为哪一种：**重放到有塔的中局 → 对某座塔 apply `UPGRADE_TOWER` → 看 level/hp/金币 是否变化**。
"""
from __future__ import annotations

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

from SDK.backend.model import Operation                      # noqa: E402
from SDK.utils.constants import OperationType                # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "SMOKE_upg2b"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    txt = (REPO / "match_results" / "ladder_logs" / tag / f"{seed}.log").read_text(encoding="utf-8", errors="replace")
    rounds: dict[int, dict[int, list]] = {}
    for m in OPLINE.finditer(txt):
        rounds.setdefault(int(m.group(1)), {})[int(m.group(2)[1])] = [
            (int(o.group(2)), int(o.group(3)), int(o.group(4))) for o in OP.finditer(m.group(5))]

    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    for r in range(0, 260):
        for p in (0, 1):
            for (t, a0, a1) in rounds.get(r, {}).get(p, []):
                st.apply_operation_list(p, [Operation(OperationType(t), a0, a1)])
            if p == 1:
                st.advance_round()
        if st.terminal:
            break
        tws = st.towers_of(0)
        if len(tws) >= 3:
            break

    tws = st.towers_of(0)
    print(f"[状态] round≈{getattr(st, 'round_index', '?')}  我方塔 {len(tws)} 座  金币={st.coins}")
    if not tws:
        print("!! 没有我方塔，换个 seed/tag"); return 1
    for w in tws[:8]:
        print(f"   塔 id={w.tower_id} ({w.x},{w.y}) type={w.tower_type} level={w.level} hp={w.hp}")

    t0 = tws[0]
    before = [(w.tower_id, w.level, w.hp, w.tower_type) for w in st.towers_of(0)]
    coins_before = list(st.coins)
    st2 = st.clone()
    st2.apply_operation_list(0, [Operation(OperationType.UPGRADE_TOWER, int(t0.x), int(t0.y))])
    after = [(w.tower_id, w.level, w.hp, w.tower_type) for w in st2.towers_of(0)]
    print(f"\n[试验] 对塔 ({t0.x},{t0.y}) apply UPGRADE_TOWER")
    print(f"   金币 {coins_before} -> {list(st2.coins)}")
    changed = [f"id {b[0]}: level {b[1]}->{a[1]} hp {b[2]}->{a[2]} type {b[3]}->{a[3]}"
               for b, a in zip(before, after) if b != a]
    if changed:
        print("   ⇒ **升级生效** ✓：" + "; ".join(changed))
    else:
        print("   ⇒ **没有任何变化** ⇒ 该 op 在这个状态下是**空操作/非法**（这就是 UPG 臂 0 次升级的原因之一）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
