"""从 ladder_logs 里解析**进行中**的臂的部分战绩（不必等整臂跑完）。

用法：python code/test_match/partial_arm.py A3_flat_every5
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

tag = sys.argv[1] if len(sys.argv) > 1 else "A3_flat_every5"
log_dir = REPO / "match_results/ladder_logs" / tag
done = []
for log in sorted(log_dir.glob("*.log")):
    parsed = _ladder.read_game(log, log.stem)
    if parsed.get("winner_side") is None:
        continue
    # ⚠️ ladder 的 "7r" 会把 ai0 换到 P1 ⇒ 必须用 p0_label 判断哪一侧是我们的臂
    p0 = (parsed.get("p0_label") or "").strip()
    ours_is_p0 = (p0 == "az_bridge_ai")
    winner = str(parsed["winner_side"]).lower()
    won = (winner == "p0") if ours_is_p0 else (winner == "p1")
    done.append((log.stem, "win" if won else "loss", parsed.get("rounds")))

if not done:
    print(f"[{tag}] 还没有完成的对局")
    raise SystemExit(0)

wins = sum(1 for _, w, _ in done if w == "win")
losses = sum(1 for _, w, _ in done if w == "loss")
rounds = [r for _, _, r in done if r]
print(f"[{tag}] 已完成 {len(done)} 局：我们 {wins}W-{losses}L = {wins/len(done):.3f}")
if rounds:
    print(f"  平均回合 {sum(int(r) for r in rounds)/len(rounds):.0f}")
