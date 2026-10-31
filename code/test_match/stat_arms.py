"""统计若干 ladder tag 的胜负/血量（2026-10-29）。"""
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OURS = "az_bridge_ai"


def stats(tag: str) -> None:
    d = REPO / "match_results" / "ladder_logs" / tag
    if not d.exists():
        print(f"  [{tag}] 目录不存在")
        return
    rows = []
    for f in sorted(d.glob("*.log")):
        if f.stem == "_run":
            continue
        t = f.read_text(encoding="utf-8", errors="replace")
        v = re.search(r"verdict=(\w+)", t)
        hp = re.search(r"base_hp=(\d+),(\d+)", t)
        p0 = re.search(r"\[round \d+\] p0\((\w+)\)", t)
        ws = re.search(r"winner_side=(\w+)", t)
        if not v:
            continue
        rows.append({
            "tok": f.stem,
            "verdict": v.group(1),
            "h0": int(hp.group(1)) if hp else None,
            "h1": int(hp.group(2)) if hp else None,
            "ours_p0": bool(p0 and p0.group(1) == OURS),
            "ws": ws.group(1) if ws else None,
        })
    done = [r for r in rows if r["verdict"] == "engine" and r["h0"] is not None]
    W = L = 0
    hps = []
    for r in done:
        my, th = (r["h0"], r["h1"]) if r["ours_p0"] else (r["h1"], r["h0"])
        hps.append(my - th)
        if r["ws"] and ((r["ws"] == "p0") == r["ours_p0"]):
            W += 1
        else:
            L += 1
    line = f"  [{tag}] engine {len(done)}/{len(rows)}"
    if W + L:
        line += f"  {W}W-{L}L = {100 * W / (W + L):.1f}%"
    if hps:
        line += f"  血量差 {statistics.mean(hps):+.1f}"
    print(line)
    other = [r["tok"] for r in rows if r["verdict"] != "engine"]
    if other:
        print(f"     非 engine: {other}")


if __name__ == "__main__":
    tags = sys.argv[1:] or ["US_VS_CHAMPION_BASE", "PIN_K2_VS_RU", "INJECT_VS_RU_B", "INJECT_VS_RU_A"]
    for t in tags:
        stats(t)
