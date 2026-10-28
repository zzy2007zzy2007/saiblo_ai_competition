"""**只读诊断**：为什么我们自己的候选生成器几乎从不产生 `UPGRADE_TOWER`(op 12)？

三层证据，全部来自**已落盘的日志**（不跑对局、不改 `code/my_ai/**`）：

  §1 出手构成（每方分开）：`training_history/progress/<tag>/az_progress_seed*.txt`
      每行 `round=N HP0 HP1 G0 G1 P0=[...] P1=[...]`（G=**本回合操作执行前**的金币，
      见 `code/test_match/progress.py:101-117` 与 `match_via_sdk.py:715-727`）
  §2 金币/塔等级时间序列：同一份 trace（金币）+ `ladder_logs/<tag>/<token>.log` 里的
      `[round N] towers=[(id, player, x, y, tower_type, hp)]`（`match_via_sdk.py:770-771`）。
      ⚠️ `tower_type` 是 **TowerType 真值**（BASIC=0 / HEAVY=1 / QUICK=2 / MORTAR=3 /
      PRODUCER=4 / 三级=11,12,13,21,22,23,31,32,33,41,42,43），**不是** decoder 的 0-15 类号。
  §3 「合法升级机会」：用 §2 的金币+塔等级**离线复算**（不碰引擎）：
      每座一级塔有 4 个升级目标（HEAVY/QUICK/MORTAR/PRODUCER，各 60 金），
      每座二级塔有 3 个升级目标（三级，各 200 金），费用口径见
      `game_state_facade.py:304-310` + `Ant-Game/SDK/utils/constants.py`（L2=60 / L3=200）。
      「有升级机会」= 那一刻至少存在一个 `金币 >= 费用` 的**合法单步升级**。

用法：
    python code/test_match/probe_why_no_upgrade.py --tag INJECT_VS_RU_B
    python code/test_match/probe_why_no_upgrade.py --tag VS_RUNNERUP_upg1fix3   # 开了 MC_UPG 的臂
    python code/test_match/probe_why_no_upgrade.py --tag VS_RUNNERUP_cells4 --json out.json

数据源自动选择（不需要开关）：
  * 有 `training_history/progress/<tag>/` ⇒ 用 trace（含**逐回合金币**）；
  * 没有 ⇒ 退回解析 `ladder_logs/<tag>/*.log` 的 `--trace-ops` 行（**无金币**，
    此时 §2 的金币层与 §3 的"金币够"判定会打印"未找到"/按"有可升级的塔"计）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO, REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from SDK.utils.constants import (  # noqa: E402
    LEVEL2_TOWER_UPGRADE_COST,
    LEVEL3_TOWER_UPGRADE_COST,
    TOWER_UPGRADE_TREE,
    OperationType,
    TowerType,
)

UPG = int(OperationType.UPGRADE_TOWER)      # 12
BUILD = int(OperationType.BUILD_TOWER)      # 11
DOWN = int(OperationType.DOWNGRADE_TOWER)   # 13

# TowerType → 等级（1 / 2 / 3）。按官方升级树定义：
#   一级 = BASIC；二级 = HEAVY/QUICK/MORTAR/PRODUCER；三级 = 其余（11,12,13,21,22,23,31,32,33,41,42,43）
LEVEL1 = {int(TowerType.BASIC)}
LEVEL2 = {int(TowerType.HEAVY), int(TowerType.QUICK), int(TowerType.MORTAR),
          int(TowerType.PRODUCER)}
TREE_INT = {int(k): [int(v) for v in vs] for k, vs in TOWER_UPGRADE_TREE.items()}


def tower_level(ttype: int) -> int:
    if ttype in LEVEL1:
        return 1
    if ttype in LEVEL2:
        return 2
    return 3


def upgrade_cost(target: int) -> int:
    """与 `game_state_facade.upgrade_tower_cost` 同口径：目标 <10（= 二级）60，否则 200。"""
    return LEVEL2_TOWER_UPGRADE_COST if int(target) < 10 else LEVEL3_TOWER_UPGRADE_COST


# ── op 名字表（与 progress.fmt_op 同口径，用于把 trace 里的标签打回 op 类型）──────
_LABEL_RE = re.compile(r"([A-Z_]+)(?:\(([^)]*)\))?")
_NAME_TO_TYPE = {
    "BUILD": BUILD,
    "UPGRADE": UPG,
    "DOWNGRADE": DOWN,
    "LIGHTNING": int(OperationType.USE_LIGHTNING_STORM),
    "EMP": int(OperationType.USE_EMP_BLASTER),
    "DEFLECTOR": int(OperationType.USE_DEFLECTOR),
    "EVASION": int(OperationType.USE_EMERGENCY_EVASION),
    "UP_SPEED": int(OperationType.UPGRADE_GENERATION_SPEED),
    "UP_ANT": int(OperationType.UPGRADE_GENERATED_ANT),
}
_TYPE_NAME = {v: k for k, v in _NAME_TO_TYPE.items()}


def parse_ops(s: str) -> list[tuple[int, int, int]]:
    """`[BUILD(4,16), UPGRADE(id=3,tgt=1), DOWNGRADE(id=0)]` → [(11,4,16), (12,3,1), (13,0,0)]。"""
    out: list[tuple[int, int, int]] = []
    s = s.strip()
    if not s or s in ("[", "[]"):
        return out
    for m in _LABEL_RE.finditer(s):
        name, args = m.group(1), m.group(2)
        if name not in _NAME_TO_TYPE:
            if name.isdigit():      # 裸数字（空列表里的残留）
                continue
            continue
        t = _NAME_TO_TYPE[name]
        a0 = a1 = 0
        if args:
            if "=" in args:          # UPGRADE(id=3,tgt=1) / DOWNGRADE(id=0)
                kv = dict(p.split("=") for p in args.split(",") if "=" in p)
                a0 = int(kv.get("id", 0))
                a1 = int(kv.get("tgt", 0))
            else:
                parts = [p for p in args.split(",") if p.strip() != ""]
                if len(parts) >= 2:
                    a0, a1 = int(parts[0]), int(parts[1])
                elif len(parts) == 1:
                    a0 = int(parts[0])
        out.append((t, a0, a1))
    return out


PROG_RE = re.compile(
    r"^round=(\d+)\s+HP0=(-?\d+)\s+HP1=(-?\d+)\s+G0=(-?\d+)\s+G1=(-?\d+)\s+(.*)$")
P0_RE = re.compile(r"P0=(\[.*?\])\s+P1=(\[.*?\])")
TOWERS_RE = re.compile(r"^\s*\[round (\d+)\] towers=\[(.*)\]\s*$")
# ladder log 的 op 行（`match_via_sdk.py:711-713` 的 --trace-ops 输出）
OPLINE_RE = re.compile(r"\[round (\d+)\] (p[01])\(\S+?\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OPOBJ_RE = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def parse_ladder_ops(path: Path) -> dict[int, list[list[tuple[int, int, int]]]]:
    """ladder log → {round: [[p0 ops], [p1 ops]]}（`--trace-ops` 的 Operation(...) 文本）。

    与 `probe_upgrade_op.py:28-39` 同一个正则口径。
    """
    out: dict[int, list[list[tuple[int, int, int]]]] = {}
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for m in OPLINE_RE.finditer(txt):
        rnd = int(m.group(1))
        side = 0 if m.group(2) == "p0" else 1
        ops = [(int(o.group(2)), int(o.group(3)), int(o.group(4)))
               for o in OPOBJ_RE.finditer(m.group(4))]
        row = out.setdefault(rnd, [[], []])
        row[side] = ops
    return out


def parse_progress(path: Path) -> list[dict]:
    """→ [{round, hp:[h0,h1], g:[g0,g1], ops:[[...],[...]]}]"""
    rows: list[dict] = []
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in txt.splitlines():
        m = PROG_RE.match(line)
        if not m:
            continue
        mm = P0_RE.search(m.group(6))
        ops0 = parse_ops(mm.group(1)) if mm else []
        ops1 = parse_ops(mm.group(2)) if mm else []
        rows.append({"round": int(m.group(1)),
                     "hp": [int(m.group(2)), int(m.group(3))],
                     "g": [int(m.group(4)), int(m.group(5))],
                     "ops": [ops0, ops1]})
    return rows


def parse_tower_lines(path: Path) -> dict[int, list[tuple[int, int, int, int, int]]]:
    """ladder log → {round: [(tower_id, player, x, y, tower_type, hp), ...]}"""
    out: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in txt.splitlines():
        m = TOWERS_RE.match(line)
        if not m:
            continue
        rnd = int(m.group(1))
        rows: list[tuple[int, int, int, int, int, int]] = []
        for t in re.finditer(r"\(([^()]*)\)", m.group(2)):
            parts = [p.strip() for p in t.group(1).split(",")]
            if len(parts) >= 6:
                rows.append(tuple(int(float(p)) for p in parts[:6]))  # type: ignore[arg-type]
        out.setdefault(rnd, []).extend(rows)
    return out


def our_side_from_token(token: str) -> int:
    """`7` ⇒ 我们执 P0；`7r` ⇒ 我们执 P1（见 docs/inject_vs_runnerup_20261028.md §1.1）。"""
    return 1 if token.strip().endswith("r") else 0


def _pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _q(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return float(s[i])


def analyze(tag: str, ladder_dir: Path | None, prog_dir: Path | None,
            max_games: int = 0) -> dict:
    res: dict = {"tag": tag, "games": {}}

    if ladder_dir is None or not ladder_dir.is_dir():
        print(f"[!] 未找到 ladder 日志目录：{ladder_dir}")
        ladder_dir = None
    if prog_dir is None or not prog_dir.is_dir():
        print(f"[!] 未找到 progress 目录：{prog_dir}（§1/§2 金币层将不可用）")
        prog_dir = None

    tokens: list[str] = []
    if prog_dir is not None:
        for f in sorted(prog_dir.glob("az_progress_seed*.txt")):
            m = re.match(r"az_progress_seed(\d+r?)\.txt$", f.name)
            if m:
                tokens.append(m.group(1))
    elif ladder_dir is not None:
        for f in sorted(ladder_dir.glob("*.log")):
            if f.name.startswith("_"):
                continue
            tokens.append(f.stem)
    if max_games:
        tokens = tokens[:max_games]
    if not tokens:
        print("未找到任何对局日志")
        return res

    # ── 汇总容器 ───────────────────────────────────────────────────────────
    op_cnt_ours: dict[int, int] = {}
    op_cnt_opp: dict[int, int] = {}
    op_cnt_ours_by_game: dict[str, dict[int, int]] = {}
    coins_all: list[int] = []
    coins_when_tower: list[int] = []
    lvl_cnt = {1: 0, 2: 0, 3: 0}
    rounds_with_tower = 0
    rounds_with_lvl1 = 0
    rounds_with_lvl2 = 0
    rounds_total = 0
    opp_lvl_cnt = {1: 0, 2: 0, 3: 0}
    opp_rounds_with_lvl1 = 0
    opp_rounds_with_lvl2 = 0
    # 「升级机会」统计
    opportunity_rounds = 0          # 至少一个"金币够"的单步升级
    opportunity_ops = 0             # 所有"金币够"的 (塔,目标) 组合数
    opportunity_rounds_lvl1 = 0     # 只由一级塔提供的机会
    opportunity_rounds_lvl2 = 0     # 只由二级塔提供的机会
    upgrade_taken = 0
    coins_hist: dict[str, int] = {}
    no_towers_rounds = 0
    tower_missing_rounds = 0

    for tok in tokens:
        side = our_side_from_token(tok)
        opp = 1 - side
        g: dict = {"side": side, "rounds": 0, "ops_ours": {}, "ops_opp": {},
                   "upgrade_ops": 0, "coins": [], "opportunity_rounds": 0}
        rows: list[dict] = []
        if prog_dir is not None:
            rows = parse_progress(prog_dir / f"az_progress_seed{tok}.txt")
        tw: dict[int, list] = {}
        lops: dict[int, list] = {}
        if ladder_dir is not None:
            tw = parse_tower_lines(ladder_dir / f"{tok}.log")
            if not rows:
                lops = parse_ladder_ops(ladder_dir / f"{tok}.log")
                rows = [{"round": r, "hp": [None, None], "g": [None, None], "ops": lops[r]}
                        for r in sorted(lops)]
        g["has_progress"] = bool(prog_dir is not None and lops == {})
        g["has_towers"] = bool(tw)
        g["has_coins"] = bool(prog_dir is not None and lops == {})
        g["rounds"] = len(rows)

        for r in rows:
            rounds_total += 1
            gc = None if r["g"][side] is None else int(r["g"][side])
            if gc is not None:
                coins_all.append(gc)
                coins_hist[gc] = coins_hist.get(gc, 0) + 1
            for (t, a0, a1) in r["ops"][side]:
                op_cnt_ours[t] = op_cnt_ours.get(t, 0) + 1
                g["ops_ours"][t] = g["ops_ours"].get(t, 0) + 1
                if t == UPG:
                    upgrade_taken += 1
                    g["upgrade_ops"] += 1
            for (t, a0, a1) in r["ops"][opp]:
                op_cnt_opp[t] = op_cnt_opp.get(t, 0) + 1
                g["ops_opp"][t] = g["ops_opp"].get(t, 0) + 1
            op_cnt_ours_by_game.setdefault(tok, {})
            # ── 塔态（同一时刻：`[round R] towers=` 是本回合操作**前**的快照） ──
            mine = [t for t in tw.get(int(r["round"]), []) if t[1] == side]
            theirs = [t for t in tw.get(int(r["round"]), []) if t[1] == opp]
            if not tw:
                tower_missing_rounds += 1
                continue
            for t in mine:
                lv = tower_level(t[4])
                lvl_cnt[lv] = lvl_cnt.get(lv, 0) + 1
            for t in theirs:
                lv = tower_level(t[4])
                opp_lvl_cnt[lv] = opp_lvl_cnt.get(lv, 0) + 1
            n1 = sum(1 for t in mine if tower_level(t[4]) == 1)
            n2 = sum(1 for t in mine if tower_level(t[4]) == 2)
            if mine:
                rounds_with_tower += 1
                if gc is not None:
                    coins_when_tower.append(gc)
            else:
                no_towers_rounds += 1
            if n1:
                rounds_with_lvl1 += 1
            if n2:
                rounds_with_lvl2 += 1
            if sum(1 for t in theirs if tower_level(t[4]) == 1):
                opp_rounds_with_lvl1 += 1
            if sum(1 for t in theirs if tower_level(t[4]) == 2):
                opp_rounds_with_lvl2 += 1
            # ── 合法单步升级机会（金币够 + 该塔在该目标上有直系升级）──────────
            #   金币未知（ladder-only 模式）⇒ 只统计"有可升级的塔"，不判金币
            n_op = 0
            n_op_l1 = 0
            n_op_l2 = 0
            for t in mine:
                tt = int(t[4])
                for tgt in TREE_INT.get(tt, ()):
                    if gc is None or gc >= upgrade_cost(tgt):
                        n_op += 1
                        if tower_level(tt) == 1:
                            n_op_l1 += 1
                        else:
                            n_op_l2 += 1
            if gc is not None:
                opportunity_ops += n_op
            if n_op > 0:
                opportunity_rounds += 1
                g["opportunity_rounds"] += 1
                if n_op_l1:
                    opportunity_rounds_lvl1 += 1
                if n_op_l2:
                    opportunity_rounds_lvl2 += 1
            if gc is not None:
                g["coins"].append(gc)
        res["games"][tok] = g

    # ── 输出 ───────────────────────────────────────────────────────────────
    n_op_ours = sum(op_cnt_ours.values())
    n_op_opp = sum(op_cnt_opp.values())
    print("=" * 100)
    print(f"§1 出手构成（我方 vs 对手），tag={tag}  对局={len(tokens)}  我方决策回合={rounds_total}")
    print("-" * 100)
    print(f"{'op':<26}{'我方 n':>10}{'我方 %':>10}{'对手 n':>10}{'对手 %':>10}")
    for t in sorted(set(op_cnt_ours) | set(op_cnt_opp)):
        nm = f"{_TYPE_NAME.get(t, 'OP' + str(t))}({t})"
        print(f"{nm:<26}{op_cnt_ours.get(t, 0):>10}{_pct(op_cnt_ours.get(t, 0) / max(1, n_op_ours)):>10}"
              f"{op_cnt_opp.get(t, 0):>10}{_pct(op_cnt_opp.get(t, 0) / max(1, n_op_opp)):>10}")
    print(f"{'合计':<26}{n_op_ours:>10}{'100.0%':>10}{n_op_opp:>10}{'100.0%':>10}")
    print(f"\n  ★ 我方 UPGRADE_TOWER(12)：{op_cnt_ours.get(UPG, 0)} 次 / {rounds_total} 回合"
          f" = {_pct(op_cnt_ours.get(UPG, 0) / max(1, rounds_total))} 的回合"
          f"（占我方 op {_pct(op_cnt_ours.get(UPG, 0) / max(1, n_op_ours))}）")
    print(f"  ★ 对手 UPGRADE_TOWER(12)：{op_cnt_opp.get(UPG, 0)} 次"
          f"（占对手 op {_pct(op_cnt_opp.get(UPG, 0) / max(1, n_op_opp))}）")

    print("\n" + "=" * 100)
    print("§2 我方金币 / 塔等级")
    print("-" * 100)
    if coins_all:
        print(f"金币（决策时刻，n={len(coins_all)}）："
              f"p10={_q(coins_all, .10):.0f} p25={_q(coins_all, .25):.0f} "
              f"p50={_q(coins_all, .50):.0f} p75={_q(coins_all, .75):.0f} "
              f"p90={_q(coins_all, .90):.0f} max={max(coins_all)} mean={sum(coins_all)/len(coins_all):.1f}")
        for th in (60, 200):
            k = sum(1 for c in coins_all if c >= th)
            print(f"  P(金币 >= {th:>3}) = {_pct(k / len(coins_all))}   ({k}/{len(coins_all)})")
    else:
        print("金币：未找到（无 progress trace）")
    if rounds_total and not tower_missing_rounds:
        tot_tw = sum(lvl_cnt.values())
        print(f"我方塔·等级分布（塔·回合计数，n={tot_tw}）："
              f"一级(BASIC)={lvl_cnt[1]} ({_pct(lvl_cnt[1]/max(1,tot_tw))})  "
              f"二级={lvl_cnt[2]} ({_pct(lvl_cnt[2]/max(1,tot_tw))})  "
              f"三级={lvl_cnt[3]} ({_pct(lvl_cnt[3]/max(1,tot_tw))})")
        print(f"  有至少 1 座一级塔的回合：{rounds_with_lvl1}/{rounds_total} = {_pct(rounds_with_lvl1/max(1,rounds_total))}")
        print(f"  有至少 1 座二级塔的回合：{rounds_with_lvl2}/{rounds_total} = {_pct(rounds_with_lvl2/max(1,rounds_total))}")
        print(f"  一座塔都没有的回合：{no_towers_rounds}/{rounds_total} = {_pct(no_towers_rounds/max(1,rounds_total))}")
        tots = sum(opp_lvl_cnt.values())
        print(f"对手塔·等级分布（n={tots}）：一级={opp_lvl_cnt[1]} 二级={opp_lvl_cnt[2]} 三级={opp_lvl_cnt[3]}")
        print(f"  对手有 1 座一级塔的回合占比={_pct(opp_rounds_with_lvl1/max(1,rounds_total))}，"
              f"有 2 座二级塔的回合占比={_pct(opp_rounds_with_lvl2/max(1,rounds_total))}")
    else:
        print("塔等级：未找到（无 ladder `towers=` 行）")

    print("\n" + "=" * 100)
    print("§3 「合法升级机会」 vs 「真的升了」")
    print("-" * 100)
    print(f"有 >=1 个『金币够的单步升级』的回合：{opportunity_rounds}/{rounds_total} "
          f"= {_pct(opportunity_rounds / max(1, rounds_total))}")
    print(f"  其中只靠一级塔（BASIC→二级，60 金）：{opportunity_rounds_lvl1}")
    print(f"  其中含二级塔（→三级，200 金）      ：{opportunity_rounds_lvl2}")
    print(f"累计『金币够的单步升级』组合数（塔×目标）：{opportunity_ops}")
    print(f"实际发出的 UPGRADE_TOWER：{upgrade_taken} 次 "
          f"⇒ 机会转化率 {_pct(upgrade_taken / max(1, opportunity_ops))}")

    res.update({
        "rounds_total": rounds_total,
        "op_ours": op_cnt_ours, "op_opp": op_cnt_opp,
        "n_op_ours": n_op_ours, "n_op_opp": n_op_opp,
        "upgrade_ops_ours": op_cnt_ours.get(UPG, 0),
        "upgrade_ops_opp": op_cnt_opp.get(UPG, 0),
        "coins_p50": _q(coins_all, .50), "coins_p90": _q(coins_all, .90),
        "coins_n": len(coins_all),
        "p_ge60": (sum(1 for c in coins_all if c >= 60) / len(coins_all)) if coins_all else None,
        "p_ge200": (sum(1 for c in coins_all if c >= 200) / len(coins_all)) if coins_all else None,
        "tower_levels_ours": lvl_cnt, "tower_levels_opp": opp_lvl_cnt,
        "rounds_with_lvl1": rounds_with_lvl1, "rounds_with_lvl2": rounds_with_lvl2,
        "opportunity_rounds": opportunity_rounds,
        "opportunity_rounds_lvl1": opportunity_rounds_lvl1,
        "opportunity_rounds_lvl2": opportunity_rounds_lvl2,
        "opportunity_ops": opportunity_ops,
        "tokens": tokens,
    })
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ladder", default="")
    ap.add_argument("--progress", default="")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    tag = args.tag
    ladder = Path(args.ladder) if args.ladder else REPO / "match_results" / "ladder_logs" / tag
    prog = Path(args.progress) if args.progress else REPO / "training_history" / "progress" / tag
    res = analyze(tag, ladder, prog, max_games=args.max_games)

    # 逐局 UPGRADE 计数（最有用的一张表）
    print("\n" + "=" * 100)
    print("§4 逐局 UPGRADE_TOWER 计数（我方 / 回合数 / 我方塔等级构成）")
    print("-" * 100)
    print(f"{'token':<8}{'座位':<6}{'回合':>6}{'我方UPG':>9}{'对手UPG':>9}"
          f"{'机会回合':>10}")
    for tok, g in res["games"].items():
        side = "P1" if g["side"] == 1 else "P0"
        print(f"{tok:<8}{side:<6}{g['rounds']:>6}{g['upgrade_ops']:>9}"
              f"{g['ops_opp'].get(UPG, 0):>9}{g['opportunity_rounds']:>10}")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
