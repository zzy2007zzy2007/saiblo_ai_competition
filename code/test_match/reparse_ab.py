"""从已完成的 A/B 日志重新解析结果（对局不必重跑）。

用法：python code/test_match/reparse_ab.py training_history/ab_flat_m6croot_pilot
"""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("_tmp_ladder", REPO / "_tmp_ladder.py")
_ladder = importlib.util.module_from_spec(_spec)
sys.modules["_tmp_ladder"] = _ladder
_spec.loader.exec_module(_ladder)

out_dir = REPO / (sys.argv[1] if len(sys.argv) > 1 else "training_history/ab_flat_m6croot_pilot")
rows = []
for log in sorted(out_dir.glob("seed*.log")):
    seed = int(log.stem.replace("seed", ""))
    parsed = _ladder.read_game(log, str(seed))
    flat_is_p0 = (seed % 2 == 0)
    winner_side = parsed.get("winner_side")
    if isinstance(winner_side, str) and winner_side.strip().lower() in ("p0", "p1"):
        winner_side = int(winner_side.strip().lower()[1])
    elif winner_side is not None:
        winner_side = int(winner_side)
    flat_side = 0 if flat_is_p0 else 1
    score = 0.5 if winner_side is None else (1.0 if winner_side == flat_side else 0.0)
    rows.append((seed, flat_is_p0, score, parsed.get("rounds"), parsed.get("hp"),
                 parsed.get("terminal"), parsed.get("illegal")))
    print(f"seed {seed:>3} flat={'P0' if flat_is_p0 else 'P1'} "
          f"score={score} winner={winner_side} rounds={parsed.get('rounds')} "
          f"hp={parsed.get('hp')} terminal={parsed.get('terminal')} illegal={parsed.get('illegal')}")

if rows:
    scores = [r[2] for r in rows]
    wins = sum(1 for x in scores if x == 1.0)
    draws = sum(1 for x in scores if x == 0.5)
    losses = sum(1 for x in scores if x == 0.0)
    print(f"\n=== flat vs M6c-ROOT：{len(scores)} 局  "
          f"flat 胜率 {sum(scores)/len(scores):.1%}  ({wins}W/{draws}D/{losses}L) ===")
