"""读数：`AZAI_UPG_BUNDLES` 诊断臂 —— **升级候选的选中率** + 预算稀释实测（2026-10-28）

报告：`docs/upgrade_bundles_20261028.md` §7（指标 1）与 §8（预算）。

数据源 = AI 自己的 stderr（`[upg]` 前缀，`code/test_match/az_bridge_ai.py`）：
  * `[upg] round=R 加入 k 条升级候选 targets=[(12,tid,tgt), …]`  —— **每决策一条**（分母）
  * `[upg] sel round=R hit=h mine=m extra=e n_cand=N n_rollouts=K chosen=...` —— 有候选的回合才有一条
  * `[upg] 统计: …`  —— 每 50 决策 + 收尾各一条（用来交叉核对，不作为主口径）
  * `[upg] ⚠️ …`      —— 异常（应当 0 条）

口径（预注册在报告 §3.1）：
  * **选中率 = Σhit / Σoffered**（"塞进去的每一条被评估器选中的概率"；每决策 ≤1 条被选中）
  * 另一个等价读法：Σhit / 有候选的决策数（"有候选的回合里，有多少回合真的走了升级"）
  * **有升级候选的决策点占比 = 有候选的决策数 / 全部决策数**
  * 预算：解析 `n_cand/n_rollouts` 直方图（`mine=0` 的 round 不计入 sel 行，但预算来自
    `mcts.last_rollouts`，已按"每决策"独立统计在 `[upg] 统计` 的 `cand/rollouts=` 里）

用法：
  python code/test_match/analyze_upg_bundles.py --stderr match_results/inject_vs_ru/C/stderr
  python code/test_match/analyze_upg_bundles.py --glob "match_results/ai?_az_bridge_ai_seed*.stderr.log"
  python code/test_match/analyze_upg_bundles.py --stderr <dir> --json out.json
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")   # Windows 下重定向时默认 cp936 ⇒ 中文会炸
    except Exception:  # noqa: BLE001
        pass

RE_ADD = re.compile(r"\[upg\] round=(\d+) 加入 (\d+) 条升级候选 targets=\[(.*?)\]")
RE_SEL = re.compile(r"\[upg\] sel round=(\d+) hit=(\d+) mine=(\d+) extra=(\d+) "
                    r"n_cand=(\d+) n_rollouts=(\d+) chosen=(.*)$")
RE_STAT = re.compile(r"\[upg\] 统计: (.*)$")
RE_TGT = re.compile(r"\(12,(\d+),(\d+)\)")
RE_ENABLE = re.compile(r"\[upg\] 升级候选已启用: 每回合最多 (\d+) 条")


def parse_file(path: Path) -> dict:
    """一个 stderr 文件 = 一局（我方一个座位）。"""
    g = {"file": path.name, "decisions": 0, "with_cand": 0, "offered": 0, "sel": 0,
         "tgt_all": collections.Counter(), "tgt_sel": collections.Counter(),
         "budget": collections.Counter(), "kinds": collections.Counter(),
         "warn": 0, "enable": None, "stat_line": None, "n_max": None,
         "stat": {}, "budget_stat": collections.Counter(),
         "cand_rounds_unsel": 0, "no_cand": 0}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = RE_ENABLE.search(ln)
        if m:
            g["enable"] = int(m.group(1))
            g["n_max"] = int(m.group(1))
            continue
        m = RE_ADD.search(ln)
        if m:
            k = int(m.group(2))
            g["decisions"] += 1
            if k > 0:
                g["with_cand"] += 1
                g["offered"] += k
                g["kinds"][k] += 1
                for tid, tgt in RE_TGT.findall(m.group(3)):
                    g["tgt_all"][int(tgt)] += 1
            else:
                g["no_cand"] += 1
            continue
        m = RE_SEL.search(ln)
        if m:
            hit = int(m.group(2))
            g["sel"] += hit
            g["budget"][f"{int(m.group(5))}/{int(m.group(6))}"] += 1
            ch = m.group(7)
            for tid, tgt in RE_TGT.findall(ch):
                g["tgt_sel"][int(tgt)] += 1
            if not hit:
                g["cand_rounds_unsel"] += 1
            continue
        if "[upg] ⚠️" in ln:
            g["warn"] += 1
            continue
        m = RE_STAT.search(ln)
        if m:
            g["stat_line"] = m.group(1)      # 保留最后一条（最全）
            mm = re.search(r"cand/rollouts=\{([^}]*)\}", m.group(1))
            if mm:
                g["budget_stat"] = collections.Counter(
                    {k: int(v) for k, v in (x.split(":") for x in mm.group(1).split(",") if x)})
            continue
    # 交叉核对：per-decision 累计 vs `统计` 行（后者是**周期快照**，每 50 决策一条 + 收尾一条；
    #   bridge 会硬杀 ⇒ 最后一局的后半段可能没有收尾快照 ⇒ 快照值会**小于等于**累计值，这是正常的）
    g["ok_cross"] = None
    g["stat"] = {}
    if g["stat_line"]:
        try:
            d = dict(re.findall(r"(\w+)=([\w./%()]+)", g["stat_line"]))

            def _i(k: str) -> int:                  # `sel=56(12.2% of offered)` ⇒ 取括号前的整数
                return int(str(d[k]).split("(")[0])

            g["stat"] = {"offered": _i("offered"), "sel": _i("sel"),
                         "sent_op12": _i("sent_op12"), "decisions": _i("decisions"),
                         "rounds_with_cand": _i("rounds_with_cand")}
            if g["stat"]["offered"] > g["offered"]:
                g["ok_cross"] = False            # 快照比逐决策还多 ⇒ 解析有问题
            elif g["stat"]["offered"] == g["offered"]:
                g["ok_cross"] = True             # 收尾快照在（完整核对）
            else:
                g["ok_cross"] = None             # 只有周期快照（部分核对）
        except Exception:  # noqa: BLE001
            g["ok_cross"] = None
    return g


def fmt_hist(c: collections.Counter, top: int = 12) -> str:
    items = sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))[:top]
    return "{" + ", ".join(f"{k}:{v}" for k, v in items) + "}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stderr", default="", help="目录（摸 *.stderr.log）")
    ap.add_argument("--glob", default="", help="glob（相对 repo 根或绝对）")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    files: list[Path] = []
    if a.stderr:
        files = sorted(Path(a.stderr).glob("*.stderr.log"))
    if a.glob:
        files = sorted(REPO.glob(a.glob))
    files = [f for f in files if "_preupg" not in f.name]   # 零影响对照的"改动前"副本不算
    files = [f for f in files if "[upg]" in f.read_text(encoding="utf-8", errors="replace")
             or "升级候选已启用" in f.read_text(encoding="utf-8", errors="replace")]
    if not files:
        print("[an] !! 没找到 stderr 文件（给 --stderr <dir> 或 --glob <pat>）", file=sys.stderr)
        return 2

    games = [parse_file(f) for f in files]
    tot = collections.Counter()
    tg_all, tg_sel, budget, kinds, budget_stat = (collections.Counter() for _ in range(5))
    for g in games:
        tot["decisions"] += g["decisions"]; tot["with_cand"] += g["with_cand"]
        tot["offered"] += g["offered"]; tot["sel"] += g["sel"]
        tot["warn"] += g["warn"]; tot["no_cand"] += g["no_cand"]
        tot["sent_op12"] += int(g["stat"].get("sent_op12", 0))
        tg_all.update(g["tgt_all"]); tg_sel.update(g["tgt_sel"])
        budget.update(g["budget"]); kinds.update(g["kinds"])
        budget_stat.update(g["budget_stat"])

    print("=" * 108)
    print(f"### 升级候选选中率（{len(games)} 局；源 = AI stderr 的 `[upg]` 行）")
    print("=" * 108)
    print(f"{'file':<42}{'n_max':>6}{'决策':>7}{'有候选':>8}{'占比':>8}"
          f"{'塞入':>7}{'选中':>6}{'选中/条':>9}{'选中/回合':>10}{'⚠️':>5}{'交叉':>5}")
    for g in games:
        wc = g["with_cand"]; dec = g["decisions"]
        print(f"{g['file'][:41]:<42}{str(g['n_max'] or '-'):>6}{dec:>7}{wc:>8}"
              f"{(100.0 * wc / max(1, dec)):>7.1f}%{g['offered']:>7}{g['sel']:>6}"
              f"{(100.0 * g['sel'] / max(1, g['offered'])):>8.1f}%"
              f"{(100.0 * g['sel'] / max(1, wc)):>9.1f}%{g['warn']:>5}"
              f"{('OK' if g['ok_cross'] else ('--' if g['ok_cross'] is None else 'X')):>5}")

    dec, wc, off, sel = tot["decisions"], tot["with_cand"], tot["offered"], tot["sel"]
    print("-" * 108)
    print(f"{'合计':<42}{'':>6}{dec:>7}{wc:>8}{(100.0 * wc / max(1, dec)):>7.1f}%"
          f"{off:>7}{sel:>6}{(100.0 * sel / max(1, off)):>8.1f}%"
          f"{(100.0 * sel / max(1, wc)):>9.1f}%{tot['warn']:>5}")
    print()
    print(f"**指标 1（核心）**：升级候选被选中率 = {sel}/{off} = "
          f"**{100.0 * sel / max(1, off):.1f}%**（每塞进 1 条升级候选，被 flat rollout 选中的概率）")
    print(f"  另一读法：有升级候选的决策点里 {sel}/{wc} = {100.0 * sel / max(1, wc):.1f}% 真的走了升级")
    print(f"  **有升级候选的决策点占比 = {wc}/{dec} = {100.0 * wc / max(1, dec):.1f}%**")
    print(f"  每回合塞入条数直方图 {fmt_hist(kinds)}；⚠️ 异常行 {tot['warn']} 条")
    print(f"  出口 `sent_op12` 合计（`统计` 行，可能只覆盖到最后一个快照）= {tot['sent_op12']}")
    print()
    print(f"我方塔型目标：塞进去 {dict(sorted(tg_all.items()))}")
    print(f"              被选中 {dict(sorted(tg_sel.items()))}"
          f"  （1=HEAVY 2=QUICK 3=MORTAR 4=PRODUCER 11=HEAVY_PLUS …）")
    print()
    print("预算稀释（实测 `n_cand/n_rollouts` 直方图）：")
    print("  ① 全部决策（源 = `[upg] 统计` 的周期快照；覆盖每个决策，含『本回合没塞升级』的）：")
    if budget_stat:
        for k, v in sorted(budget_stat.items(), key=lambda kv: -kv[1]):
            print(f"    {k:>10}  ×{v}")
        _tot = sum(budget_stat.values())
        _full = budget_stat.get("48/64", 0) + budget_stat.get("49/63", 0)
        _dl = sum(v for k, v in budget_stat.items()
                  if int(k.split("/")[1]) < 64)
        print(f"  ⇒ n_rollouts **被压到 64 以下**的决策占比 = {_dl}/{_tot} = "
              f"{100.0 * _dl / max(1, _tot):.2f}%")
    else:
        print("    （本批没有 `统计` 行 ⇒ 只能看 ②）")
    print("  ② 有升级候选的决策（源 = `[upg] sel` 行）：")
    for k, v in sorted(budget.items(), key=lambda kv: -kv[1]):
        print(f"    {k:>10}  ×{v}")

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "files": [g["file"] for g in games],
            "totals": dict(tot),
            "sel_rate_per_offered": sel / max(1, off),
            "sel_rate_per_round": sel / max(1, wc),
            "cand_round_share": wc / max(1, dec),
            "targets_offered": dict(tg_all), "targets_selected": dict(tg_sel),
            "budget": dict(budget), "kinds": dict(kinds),
            "per_game": [{k: (dict(v) if isinstance(v, collections.Counter) else v)
                          for k, v in g.items()} for g in games],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[an] JSON 已写 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
