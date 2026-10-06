"""诊断：**败局的"签名"** —— 我们输掉的那些局，和赢的局在哪些可直接观测的量上不同？

动机（2026-10-07，第 2 轮）：两条组件线（类仲裁评价器、位置网权重）都已证明"换掉一半决策、强度不动"
⇒ 需要知道**胜负到底由什么区分**，否则下一轮还是瞎试。

做法（纯离线，读既有日志）：对给定 tag + seed 集，逐局解析
  * `RESULT` 行：winner / rounds / terminal / base_hp / coins（引擎判词）
  * `原文='...'` 行：双方逐手操作 ⇒ 统计每方 建塔 / 升塔 / 闪电 / 总操作 / 出招回合
然后按"我方胜 / 我方负"分组比均值中位数，并报：
  * 终局血量形态（我们赢时对手 HP？我们输时我们 HP？"险胜/惨败"占比）
  * 手数、出招回合数、建塔数、闪电数的差异（带 Cliff's delta，避免只看均值）
  * 先手/后手分开（避免位置混淆）

用法:
  python -u code/test_match/diag_loss_signature.py --tags M6cROOT_128,M6cROOT_rand128
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

RESULT = re.compile(r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) winner_side=(\S+) verdict=(\S+)"
                    r" engine_winner=(\S+) rounds=(\d+) terminal=(\S+) base_hp=([-\d,]+) coins=([-\d,]+)")
ROUNDLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OPNAME = re.compile(r"OperationType\.(\w+): (\d+)>")


def parse_game(path: Path) -> dict | None:
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = RESULT.search(txt)
    if not m:
        return None
    seed = int(m.group(1))
    us_first = (m.group(2) == "az_bridge_ai")          # 我方是否执先手
    winner, verdict, rounds = m.group(4), m.group(6), int(m.group(8))
    hp = [int(x) for x in m.group(10).split(",")]
    coins = [int(x) for x in m.group(11).split(",")]
    ops = {0: {"build": 0, "upgrade": 0, "light": 0, "all": 0, "act_rounds": 0},
           1: {"build": 0, "upgrade": 0, "light": 0, "all": 0, "act_rounds": 0}}
    seen_round = {0: set(), 1: set()}
    for mm in ROUNDLINE.finditer(txt):
        rnd, side = int(mm.group(1)), int(mm.group(2)[1])
        body = mm.group(5)
        names = OPNAME.findall(body)
        ops[side]["all"] += len(names)
        for nm, _ in names:
            if nm == "BUILD_TOWER":
                ops[side]["build"] += 1
            elif nm in ("UPGRADE_TOWER", "DOWNGRADE_TOWER"):
                ops[side]["upgrade"] += 1
            elif nm == "USE_LIGHTNING_STORM":
                ops[side]["light"] += 1
        seen_round[side].add(rnd)
    for s in (0, 1):
        ops[s]["act_rounds"] = len(seen_round[s])
    us, them = (0, 1) if us_first else (1, 0)
    return {"seed": seed, "rev": path.stem.endswith("r"), "us_first": us_first, "winner": winner,
            "verdict": verdict, "rounds": rounds, "hp_us": hp[us], "hp_them": hp[them],
            "coins_us": coins[us], "coins_them": coins[them],
            "us": ops[us], "them": ops[them], "win": (winner == "az_bridge_ai")}


def cliff_delta(a, b) -> float:
    """Cliff's delta（-1..1）：a 比 b 大的概率 − 小的概率。0.147/0.33/0.474 = 小/中/大效应。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="M6cROOT_128,M6cROOT_rand128")
    ap.add_argument("--roots", default="7-70,1000-9927")
    args = ap.parse_args()
    tags = args.tags.split(",")

    games = []
    for tag in tags:
        d = REPO / "match_results" / "ladder_logs" / tag
        if not d.is_dir():
            print(f"!! 缺 tag {tag}"); continue
        for f in sorted(d.glob("*.log")):
            g = parse_game(f)
            if g:
                g["tag"] = tag
                games.append(g)
    if not games:
        print("没解析到对局"); return 1

    wins = [g for g in games if g["win"]]
    loss = [g for g in games if not g["win"]]
    print(f"=== 汇总：{len(games)} 局（胜 {len(wins)} / 负 {len(loss)}）= "
          f"{100*len(wins)/len(games):.1f}% ===\n")

    print("【胜局/负局的形态】")
    for label, grp in (("胜", wins), ("负", loss)):
        hp_us = np.array([g["hp_us"] for g in grp]); hp_th = np.array([g["hp_them"] for g in grp])
        print(f"  {label}：我方终局 HP 中位 {np.median(hp_us):.0f}（min {hp_us.min()}）| "
              f"对手终局 HP 中位 {np.median(hp_th):.0f}（max {hp_th.max()}）| "
              f"手数中位 {np.median([g['rounds'] for g in grp]):.0f}")
    close = [g for g in loss if g["hp_them"] <= 10]
    print(f"  ⇒ 负局里对手只剩 <=10 HP 的（惜败）= {len(close)}/{len(loss)} = {100*len(close)/max(1,len(loss)):.0f}%；"
          f"对手 HP >=30 的（被压着打）= {100*sum(1 for g in loss if g['hp_them'] >= 30)/max(1,len(loss)):.0f}%")

    print("\n【哪些量在胜负之间有区分度】（Cliff's delta：正=胜局更大；|d|≥0.33 中效应、≥0.474 大效应）")
    feats = [("手数", lambda g: g["rounds"]),
             ("我方出招回合", lambda g: g["us"]["act_rounds"]),
             ("对手出招回合", lambda g: g["them"]["act_rounds"]),
             ("我方建塔", lambda g: g["us"]["build"]),
             ("对手建塔", lambda g: g["them"]["build"]),
             ("我方升级/降级", lambda g: g["us"]["upgrade"]),
             ("对手升级/降级", lambda g: g["them"]["upgrade"]),
             ("我方闪电", lambda g: g["us"]["light"]),
             ("对手闪电", lambda g: g["them"]["light"]),
             ("我方总操作", lambda g: g["us"]["all"]),
             ("对手总操作", lambda g: g["them"]["all"])]
    rows = []
    for name, fn in feats:
        a = [fn(g) for g in wins]; b = [fn(g) for g in loss]
        rows.append((abs(cliff_delta(a, b)), cliff_delta(a, b), name, np.median(a), np.median(b)))
    for _, d, name, ma, mb in sorted(rows, reverse=True):
        print(f"  {name:14s} 胜中位 {ma:6.1f} | 负中位 {mb:6.1f} | Cliff's d = {d:+.3f}")

    print("\n【先手/后手分开看胜率】")
    for first in (True, False):
        sub = [g for g in games if g["us_first"] == first]
        w = sum(1 for g in sub if g["win"])
        print(f"  我方{'先手' if first else '后手'}：{w}/{len(sub)} = {100*w/max(1,len(sub)):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

