#!/usr/bin/env python
"""`AZAI_ALWAYS_HOLD` A/B 消融的读数（2026-10-29）—— 父任务六项指标里的 1/2/3/4/6。

## 口径（写死在代码里，事后不改）

| 指标 | 怎么来 |
|---|---|
| 胜负/血量差（配对） | `analyze_inject_vs_runnerup.read_arm` + `paired()`（同 token） |
| **HOLD 率** | 每局 `rounds`（RESULT 里的引擎回合数）− `act[我方]`（有 ≥1 个 op 的回合数）⇒ HOLD 回合 |
| **降级 churn** | `DOWNGRADE_TOWER` 的 op 数 / 我方总 op 数（`hist` 直方图） |
| **MC 仲裁 vs 评估器** | 臂 A 的 `[ahold]` 行：`added`（候选里本来没有 HOLD 的回合）/ `chose_after_add`（这些回合里评估器仍选 HOLD） |
| 健康度 | 每局 `verdict`、裁判侧 `illegal`、我方 `illegal_total` |

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
$P -u code/test_match/_ahold_report.py --out match_results/us_vs_external/_ahold_report.md
```
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "test_match", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUTROOT = "match_results/us_vs_external"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _ahold_counters(stderr_dir: Path) -> dict:
    """从臂 A 的 stderr 里取**每局最后一条** `[ahold]` 行，再跨局相加。

    ⚠️ 必须"每局只取最后一条"：`[ahold] 累计:` 每 50 决策打一次，是**累计值**，
    把多条加起来会得到几万这种离谱数字（第一版就踩了：hold=36262 > dec=500）。
    """
    tot: dict = {}
    pct: dict = {}
    n_games = 0
    for f in sorted(stderr_dir.glob("ai?_az_bridge_ai_seed*.stderr.log")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        last = None
        for ln in txt.splitlines():
            if "[ahold]" in ln and "dec=" in ln:
                last = ln
        if not last:
            continue
        n_games += 1
        d: dict = {}
        for k, v in re.findall(r"([a-z_]+)=([-\d.]+)", last):
            d[k] = float(v)
        m = re.search(r"chose_after_add=\d+\(([\d.]+)% of added\)", last)
        if m:
            pct["chose_after_add_pct"] = float(m.group(1))
        m = re.search(r"kept=\d+\(([\d.]+)%\)", last)
        if m:
            pct["kept_pct"] = float(m.group(1))
        for k in ("dec", "hold", "added", "chose_after_add", "kept", "moved"):
            if k in d:
                tot[k] = tot.get(k, 0.0) + d[k]
    out = dict(tot)
    out["n_games"] = n_games
    if tot.get("added"):
        out["chose_after_add_pct"] = 100.0 * tot.get("chose_after_add", 0.0) / tot["added"]
    if tot.get("kept", 0) + tot.get("moved", 0):
        out["kept_pct"] = 100.0 * tot.get("kept", 0.0) / (tot["kept"] + tot.get("moved", 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-a", default="ALWAYSHOLD_A")
    ap.add_argument("--tag-b", default="ALWAYSHOLD_B")
    ap.add_argument("--out", default=f"{OUTROOT}/_ahold_report.md")
    ap.add_argument("--json", default=f"{OUTROOT}/_ahold_report.json")
    args = ap.parse_args()

    av = _load("_analyze_inj", REPO / "code" / "test_match" / "analyze_inject_vs_runnerup.py")
    arms = {}
    for tag in (args.tag_a, args.tag_b):
        logd = REPO / "match_results" / "ladder_logs" / tag
        if not logd.is_dir():
            print(f"[skip] 没有 {logd}")
            continue
        arm = av.read_arm(logd, REPO / OUTROOT / tag / "stderr")
        arm["summary"] = av.summarize(arm)
        arms[tag] = arm

    lines: list[str] = []

    def P(s: str = "") -> None:
        print(s)
        lines.append(s)

    def _hold_stats(arm: dict) -> dict:
        """HOLD 率 + op 构成（只用**引擎记的**回合数与 ladder 的 `--trace-ops`）。"""
        rows = []
        for g in arm["games"]:
            if g.get("seat") is None or not g.get("rounds"):
                continue
            rnd = int(g["rounds"])
            # ⚠️ `read_arm` 已经把"我方"的计数摊平成 `our_act / our_ops / our_hist`
            #    （原始 dict 的 `act`/`ops`/`hist` 不再挂在 rec 上）。
            act = int(g.get("our_act") or 0)
            ops = int(g.get("our_ops") or 0)
            hist = g.get("our_hist") or {}
            down = int(hist.get("DOWNGRADE_TOWER", 0))
            rows.append({"token": g["token"], "rounds": rnd, "act": act, "hold": rnd - act,
                         "ops": ops, "down": down,
                         "down_share": (down / ops) if ops else None,
                         "build": int(hist.get("BUILD_TOWER", 0)),
                         "upg": int(hist.get("UPGRADE_TOWER", 0)),
                         "light": int(hist.get("USE_LIGHTNING_STORM", 0))})
        tot_r = sum(r["rounds"] for r in rows)
        tot_h = sum(r["hold"] for r in rows)
        tot_o = sum(r["ops"] for r in rows)
        tot_d = sum(r["down"] for r in rows)
        return {"rows": rows, "rounds": tot_r, "hold": tot_h, "ops": tot_o, "down": tot_d,
                "hold_rate": (tot_h / tot_r) if tot_r else None,
                "down_share": (tot_d / tot_o) if tot_o else None,
                "upg": sum(r["upg"] for r in rows),
                "build": sum(r["build"] for r in rows),
                "light": sum(r["light"] for r in rows)}

    P("\n# `AZAI_ALWAYS_HOLD` A/B 消融读数（自动生成）\n")
    P("## 1. 两臂汇总\n")
    P("| 臂 | 局数(valid) | 胜-负 | 胜率 | Wilson 95%CI | 血量差 | t 95%CI | verdict 非engine | 裁判illegal |")
    P("|---|---|---|---|---|---|---|---|---|")
    hs: dict = {}
    for tag in (args.tag_a, args.tag_b):
        if tag not in arms:
            continue
        s = arms[tag]["summary"]
        wr = s["winrate"] or 0.0
        wl, wh = s["winrate_ci"]
        wl = wl if wl is not None else 0.0
        wh = wh if wh is not None else 0.0
        hp = s["hp_diff"]
        bad = {k: v for k, v in (s.get("verdicts") or {}).items() if k not in ("engine", None)}
        hs[tag] = _hold_stats(arms[tag])
        P(f"| {tag} | {s['n_scored']} | {s['wins']}-{s['losses']} | **{100.0 * wr:.1f}%** | "
          f"[{100.0 * wl:.1f}%, {100.0 * wh:.1f}%] | {hp.get('mean', 0):+.2f} | "
          f"[{hp.get('lo') or 0:+.2f}, {hp.get('hi') or 0:+.2f}] | {bad or '无'} | "
          f"{s.get('illegal_judge', 0)} |")

    P("\n## 2. HOLD 率与 op 构成（指标 2/3）\n")
    P("| 臂 | 回合数 | HOLD 回合 | **HOLD 率** | 总 op | 降级 op | **降级占比** | 建塔 | 升级 | 闪电 |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for tag in (args.tag_a, args.tag_b):
        h = hs.get(tag)
        if not h or not h["rounds"]:
            continue
        P(f"| {tag} | {h['rounds']} | {h['hold']} | **{100.0 * h['hold_rate']:.1f}%** | {h['ops']} | "
          f"{h['down']} | **{100.0 * h['down_share']:.1f}%** | {h['build']} | {h['upg']} | {h['light']} |")
    P("\n> 基线（`docs/inject_vs_runnerup_20261028.md`）：HOLD **78%**、降级占 op **46.0%**、升级 0 次。")

    P("\n## 3. MC 仲裁 vs 强评估器：动/不动（指标 4，仅臂 A 有这项仪表）\n")
    cnt = _ahold_counters(REPO / OUTROOT / args.tag_a / "stderr")
    if cnt:
        P("| 量 | 读数 | 含义 |")
        P("|---|---|---|")
        P(f"| `dec` | {int(cnt.get('dec', 0))} | 决策数 |")
        P(f"| `hold` | {int(cnt.get('hold', 0))} | 最终发出空 op 的回合（臂 A 的 HOLD 率 = "
          f"{100.0 * cnt.get('hold', 0) / max(1, cnt.get('dec', 1)):.1f}%） |")
        P(f"| `added` | {int(cnt.get('added', 0))}（{100.0 * cnt.get('added', 0) / max(1, cnt.get('dec', 1)):.1f}%） "
          f"| **候选里本来没有 HOLD** 的回合（≈类仲裁把动/不动钉在非 HOLD 类） |")
        P(f"| `chose_after_add` | {int(cnt.get('chose_after_add', 0))}"
          f"（{cnt.get('chose_after_add_pct', 0):.1f}% of added） | ★这些回合里评估器**仍然选了 HOLD** "
          f"⇒『被架构剥夺的决策』被拿回来的比例 |")
        kept = int(cnt.get("kept", 0))
        moved = int(cnt.get("moved", 0))
        P(f"| `present_*` | kept={kept} / moved={moved} | HOLD 本来就在候选里的回合：评估器选 HOLD / 选动 |")
        P("\n> 读法：`kept` 占绝对多数 ⇒ **一旦 HOLD 在候选集里，评估器几乎总是选它**；"
          "`added` 那部分就是它平时**想看却看不到**的决策。")
    else:
        P("（臂 A 的 stderr 里没有 `[ahold]` 行 —— 检查 `AZAI_ALWAYS_HOLD` 是否真的传进去了）")

    P("\n## 4. 配对（A − B，同 token）\n")
    if args.tag_a in arms and args.tag_b in arms:
        pr = av.paired(arms[args.tag_a], arms[args.tag_b])
        tab = pr["table"]
        hp = pr["d_hp"]
        P(f"- 配对数 = {pr['n_pairs']}；两臂都赢 {tab['both_win']}、**只 A 赢 {tab['A_only']}**、"
          f"**只 B 赢 {tab['B_only']}**、都输 {tab['both_lose']}")
        P(f"- McNemar 精确双侧 **p = {pr['mcnemar_p']:.4f}**；胜率差 = "
          f"{100.0 * (pr['delta_winrate'] or 0):+.1f}pp")
        P(f"- 配对血量差（A − B）= **{hp.get('mean', 0):+.2f}** "
          f"[{(hp.get('lo') or 0):+.2f}, {(hp.get('hi') or 0):+.2f}]（n={hp.get('n', 0)}）")
        P("\n逐 token：\n")
        P("| token | 座位 | A 胜 | B 胜 | A Δhp | B Δhp | Δhp(A−B) |")
        P("|---|---|---|---|---|---|---|")
        for r in pr["rows"]:
            P(f"| {r['token']} | P{r['seat']} | {r['score_A']:.1f} | {r['score_B']:.1f} | "
              f"{r['hp_A']:+d} | {r['hp_B']:+d} | {r['d_hp']:+d} |")
    else:
        P("（两臂还没都跑完）")

    report = "\n".join(lines) + "\n"
    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(report, encoding="utf-8")
    print(f"\n[out] {outp}")
    js = REPO / args.json
    js.write_text(json.dumps({"hold": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                                      for k, v in hs.items()},
                              "ahold_counters": cnt},
                             ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[out] {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
