"""从爬取的比赛 replay 里统计**头部选手之间的胜负矩阵**（比赛真实参照系）。

数据：F:\\download\\旧版本对战\\{top4,top8}_replays\\<match_id>.json
      + 同名 matches_info.json 提供 match_id → 双方选手（顺序即 p0/p1）
replay 最后一个元素的 round_state.winner = 0/1。

注意（用户提醒）：F:\\download 里混有**提交前的天梯数据**，与最终比赛无关。
这里只用 top4/top8 两份（按比赛期间的对局筛出来的），并在输出里带上对局时间以便判断。
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

BASE = Path(r"F:\download\旧版本对战")
INFO_FILES = [Path(r"F:\download\top4_matches_info.json"),
              Path(r"F:\download\top8_matches_info.json")]

meta: dict[str, dict] = {}
for path in INFO_FILES:
    if not path.exists():
        continue
    for item in json.load(path.open(encoding="utf-8")):
        meta[item["match_id"]] = item

print(f"[元数据] {len(meta)} 场")

records = []
missing = 0
for sub in ("top4_replays", "top8_replays"):
    folder = BASE / sub
    if not folder.exists():
        continue
    for replay_path in sorted(folder.glob("*.json")):
        item = meta.get(replay_path.stem)
        if item is None:
            missing += 1
            continue
        try:
            data = json.load(replay_path.open(encoding="utf-8", errors="replace"))
        except Exception:
            missing += 1
            continue
        winner = None
        if isinstance(data, list) and data and isinstance(data[-1], dict):
            winner = data[-1].get("round_state", {}).get("winner")
        players = item.get("players") or []
        if winner not in (0, 1) or len(players) < 2:
            continue
        p0, p1 = players[0], players[1]
        records.append({
            "match_id": replay_path.stem,
            "time": item.get("time"),
            "p0": p0.get("name"), "p0_ai": p0.get("ai_name") or p0.get("rank"),
            "p0_ver": p0.get("version"),
            "p1": p1.get("name"), "p1_ai": p1.get("ai_name") or p1.get("rank"),
            "p1_ver": p1.get("version"),
            "winner": int(winner),
        })

print(f"[replay] 有效 {len(records)} 场，跳过 {missing}")
if records:
    times = sorted(r["time"] for r in records if r["time"])
    print(f"[时间范围] {times[0]}  ~  {times[-1]}")

# ── 每个选手的净战绩（只统计这份子集内部的对局）──
win = collections.Counter()
loss = collections.Counter()
for r in records:
    w = r["p0"] if r["winner"] == 0 else r["p1"]
    l = r["p1"] if r["winner"] == 0 else r["p0"]
    win[w] += 1
    loss[l] += 1

names = sorted(set(list(win) + list(loss)), key=lambda n: -(win[n] + loss[n]))
print(f"\n=== 选手战绩（{len(names)} 人）===")
print(f"{'选手':<18}{'胜':>4}{'负':>4}{'胜率':>8}")
for n in names:
    g = win[n] + loss[n]
    print(f"{n:<18}{win[n]:>4}{loss[n]:>4}{win[n]/g if g else 0:>8.2f}")

# ── 前 8 名之间的对阵矩阵 ──
top = names[:8]
print(f"\n=== 前 8 名互相战绩（行=该选手，列=对手，W/L）===")
head = "".join(f"{n[:7]:>10}" for n in top)
print(f"{'':<16}{head}")
for a in top:
    cells = []
    for b in top:
        if a == b:
            cells.append(f"{'-':>10}")
            continue
        w = sum(1 for r in records
                if (r["winner"] == 0 and r["p0"] == a and r["p1"] == b)
                or (r["winner"] == 1 and r["p1"] == a and r["p0"] == b))
        l = sum(1 for r in records
                if (r["winner"] == 1 and r["p0"] == a and r["p1"] == b)
                or (r["winner"] == 0 and r["p1"] == a and r["p0"] == b))
        cells.append(f"{f'{w}-{l}':>10}")
    print(f"{a[:15]:<16}" + "".join(cells))
