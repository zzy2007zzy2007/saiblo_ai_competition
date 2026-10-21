"""评估：能否用爬取数据重建一张（不完整的）真实比赛胜负图。

数据现状（已核实）：
  · rank_history.json = 397 条**排行榜快照**（rank/points/user/ai_name/version），**没有逐局记录**
  · F:\\download\\<玩家>\\*.json = replay 本体（小文件是截断的 1–2 回合版，取不到 winner）
  · F:\\download\\matches_result\\matches_info.json = 430 场，含 match_id / 时间 / 双方玩家

本脚本：统计这 430 场里有多少能在磁盘上找到**完整 replay**（能读到最后状态 winner），
以及覆盖到多少名选手 —— 用来判断"重建排名"到底可行不可行。
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

DL = Path(r"F:\download")
INFO = DL / "matches_result" / "matches_info.json"

info = json.load(INFO.open(encoding="utf-8"))
print(f"[元数据] {len(info)} 场")

# 建立 match_id → 可能的 replay 路径（玩家目录 / replay 目录 / top4,top8）
candidates = collections.defaultdict(list)
for sub in ("旧版本对战/replay", "旧版本对战/top4_replays", "旧版本对战/top8_replays",
            "旧版本对战/FalseCat_", "旧版本对战/Happigy", "旧版本对战/kskvnkadvnj",
            "旧版本对战/PeterWu", "旧版本对战/XHX_P", "旧版本对战/Xishiqing",
            "旧版本对战/zyfkid", "旧版本对战/zzy25", "旧版本对战/gary318",
            "旧版本对战/licx", "旧版本对战/yyzsanyi", "旧版本对战/stepp", "yyzsanyi"):
    folder = DL / sub
    if not folder.exists():
        continue
    for f in folder.glob("*.json"):
        candidates[f.stem].append(f)

found = small = 0
winners = []
players = collections.Counter()
dates = collections.Counter()
for item in info:
    mid = str(item.get("match_id"))
    files = candidates.get(mid) or []
    if not files:
        continue
    path = max(files, key=lambda p: p.stat().st_size)
    size = path.stat().st_size
    if size < 300_000:          # 截断版，只有 1–2 回合，取不到最终 winner
        small += 1
        continue
    try:
        data = json.load(path.open(encoding="utf-8", errors="replace"))
    except Exception:
        continue
    if not isinstance(data, list) or not data:
        continue
    winner = data[-1].get("round_state", {}).get("winner")
    if winner not in (0, 1):
        continue
    found += 1
    ps = item.get("players") or []
    if len(ps) < 2:
        continue
    names = [ps[0].get("name"), ps[1].get("name")]
    winners.append((mid, names[0], names[1], int(winner), item.get("time")))
    players[names[0]] += 1
    players[names[1]] += 1
    dates[(item.get("time") or "")[:7]] += 1

print(f"[可读完整 replay] {found} 场（另有 {small} 场只有截断版）")
print(f"[覆盖选手] {len(players)} 人；按月份分布: {dict(sorted(dates.items()))}")
if players:
    print("[出场最多]:", players.most_common(8))
if winners:
    print("\n样例 5 场：")
    for w in winners[:5]:
        print(f"  {w[4]}  {w[1]} vs {w[2]} → 胜者 {w[1] if w[3]==0 else w[2]}")
