"""统计某一臂的**终局血量差**（我方 - 对手）分布：均值 / 中位 / 极值 / 分位。

用法：python code/test_match/hp_margin.py A4_flat_every1
"""
import importlib.util
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("_tmp_ladder", REPO / "_tmp_ladder.py")
_ladder = importlib.util.module_from_spec(_spec)
sys.modules["_tmp_ladder"] = _ladder
_spec.loader.exec_module(_ladder)

tag = sys.argv[1]
log_dir = REPO / "match_results/ladder_logs" / tag
rows = []
for log in sorted(log_dir.glob("*.log")):
    parsed = _ladder.read_game(log, log.stem)
    if parsed.get("hp") is None or parsed.get("winner_side") is None:
        continue
    hp = parsed["hp"]
    p0 = (parsed.get("p0_label") or "").strip()
    ours_is_p0 = (p0 == "az_bridge_ai")
    ours, theirs = (hp[0], hp[1]) if ours_is_p0 else (hp[1], hp[0])
    margin = ours - theirs
    rows.append((log.stem, ours, theirs, margin, parsed.get("rounds")))

if not rows:
    print(f"[{tag}] 没有可用对局")
    raise SystemExit(0)

margins = sorted(r[3] for r in rows)
n = len(margins)


def pct(p):
    return margins[min(n - 1, int(p * n))]


wins = [r for r in rows if r[3] > 0]
losses = [r for r in rows if r[3] < 0]
draws = [r for r in rows if r[3] == 0]

print(f"[{tag}] {n} 局  胜/平/负 = {len(wins)}/{len(draws)}/{len(losses)}")
print(f"\n终局血量差（我方 - 对手）：")
print(f"  均值 {statistics.mean(margins):+7.2f}")
print(f"  中位 {statistics.median(margins):+7.2f}")
print(f"  p5 / p25 / p75 / p95 = {pct(0.05):+d} / {pct(0.25):+d} / {pct(0.75):+d} / {pct(0.95):+d}")
print(f"  最小 {margins[0]:+d}   最大 {margins[-1]:+d}")
print(f"  平均 |差| = {statistics.mean([abs(m) for m in margins]):.2f}")
if wins:
    wm = [r[3] for r in wins]
    print(f"\n  仅胜局（{len(wins)} 局）：均值 {statistics.mean(wm):+.2f}  中位 {statistics.median(wm):+.1f}  "
          f"最小 {min(wm):+d}  最大 {max(wm):+d}")
print(f"\n  平均回合 {statistics.mean([int(r[4]) for r in rows if r[4]]):.0f}")
print(f"\n  逐局（我方血-对手血）：")
for name, ours, theirs, margin, rounds in rows:
    print(f"    {name:>5}  我 {ours:>3} : {theirs:>3} 对手   差 {margin:+4d}   回合 {rounds}")
