"""策略轴扫描：用 fast_arena 的进程内对局，把"游戏层面的策略旋钮"逐个数出来。

每个臂 = 一个 RulePolicy 参数字典；参照臂 = 默认 `ref`（3 塔 / 升到 Sniper / 只用闪电 /
不迁塔 / 不升基地）。每臂 N 个 seed（座位镜像），指标 = 终局血量差 margin（A 视角）。

三阶段：
  axes   一次一个旋钮（塔数 / 目标塔型 / 升级策略 / 武器组合 / 基地升级 / 迁塔 / 阈值）
  cells  第一个塔建在哪个格（33 个可建格点各一臂）→ 塔位经验价值表
  greedy 贪心建塔顺序：第 k 个位置在 33 个格里穷举，选 margin 最高的
  final  大 N 复核（best vs ref / best vs 另一个参照，防过拟合）

用法：
  python code/test_match/explore_strategy_sweep.py axes --seeds 128 --workers 6
  python code/test_match/explore_strategy_sweep.py cells --seeds 96 --workers 6
  python code/test_match/explore_strategy_sweep.py greedy --seeds 96 --workers 6 --k 3
  python code/test_match/explore_strategy_sweep.py final --seeds 256 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT_DIR = "match_results/explore_20261102/strategy"
BASE = dict(n_towers=2, target="sniper", upg="always", weapons=("light",),
            base_upg=(), relocate=False, build_reserve=100, upg_reserve=0,
            light_min_ants=3, light_max_dist=99, light_mode="cluster",
            slot_mode="fixed", max_ops=3)


def spec(**kw) -> str:
    d = dict(BASE)
    d.update(kw)
    return json.dumps(d, sort_keys=True)


REF = spec()


def _job(args):
    a, b, seed = args
    from test_match.fast_arena import play_game
    return play_game(a, b, seed)


def run_arms(arms: list[tuple[str, str, str]], seeds: int, workers: int,
             start_seed: int = 2000) -> list[dict]:
    """arms: [(name, spec_a, spec_b), ...]；串行臂、并行 seed。"""
    jobs, index = [], []
    for name, sa, sb in arms:
        for s in range(seeds):
            jobs.append((sa, sb, start_seed + s))
            index.append(name)
    t0 = time.perf_counter()
    if workers <= 1:
        outs = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(_job, jobs, chunksize=8))
    dt = time.perf_counter() - t0
    print(f"[run] {len(jobs)} games in {dt:.1f}s ({len(jobs)/dt:.1f} games/s)", flush=True)
    by = {}
    for nm, o in zip(index, outs):
        by.setdefault(nm, []).append(o)
    rows = []
    for name, sa, sb in arms:
        gs = by[name]
        mg = [g.margin_a for g in gs]
        rows.append(dict(
            name=name, n=len(gs), margin=statistics.mean(mg),
            se=statistics.pstdev(mg) / max(len(mg) ** 0.5, 1),
            win=sum(1 for x in mg if x > 0), loss=sum(1 for x in mg if x < 0),
            tie=sum(1 for x in mg if x == 0),
            rounds=statistics.mean(g.rounds for g in gs),
            towers=statistics.mean(g.towers[0] + g.towers[1] for g in gs),
            coins=statistics.mean(sum(g.coins) for g in gs),
            light=statistics.mean(g.opmix_a.get(21, 0) for g in gs),
            build=statistics.mean(g.opmix_a.get(11, 0) for g in gs),
            a=sa, b=sb,
        ))
    return rows


def print_rows(rows, sort=True):
    if sort:
        rows = sorted(rows, key=lambda r: -r["margin"])
    print(f"{'arm':28s} {'n':>4s} {'margin':>8s} {'se':>6s} {'win%':>6s} {'rounds':>7s} "
          f"{'tw':>5s} {'coin':>6s} {'light':>6s} {'build':>6s}")
    for r in rows:
        print(f"{r['name']:28s} {r['n']:4d} {r['margin']:8.2f} {r['se']:6.2f} "
              f"{r['win']/r['n']*100:5.1f}% {r['rounds']:7.1f} {r['towers']:5.2f} {r['coins']:6.1f} "
              f"{r.get('light', 0):6.2f} {r.get('build', 0):6.2f}")


def save_rows(rows, fn):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, fn), "w", encoding="utf-8") as f:
        f.write("name\tn\tmargin\tse\twin\tloss\ttie\trounds\ttowers\tcoins\tspec_a\tspec_b\n")
        for r in rows:
            f.write(f"{r['name']}\t{r['n']}\t{r['margin']:.4f}\t{r['se']:.4f}\t{r['win']}\t{r['loss']}\t"
                    f"{r['tie']}\t{r['rounds']:.1f}\t{r['towers']:.2f}\t{r['coins']:.1f}\t"
                    f"{json.dumps(r['a'])}\t{json.dumps(r['b'])}\n")
    print(f"[ok] {OUT_DIR}/{fn}")


# ───────────────────────────── 三阶段 ─────────────────────────────

def stage_axes(seeds, workers):
    arms = [("ref", REF, REF)]
    for n in (0, 1, 3, 4, 5):
        arms.append((f"n_towers={n}", spec(n_towers=n), REF))
    for tg in ("basic", "heavy", "heavy+", "quick", "quick+", "double",
               "mortar", "mortar+", "pulse", "missile", "siege", "medic"):
        arms.append((f"target={tg}", spec(target=tg), REF))
    for u in ("never", "lvl2"):
        arms.append((f"upg={u}", spec(upg=u), REF))
    for w in ((), ("light", "defl"), ("light", "evas"), ("light", "defl", "evas"),
              ("light", "emp"), ("defl",), ("evas",), ("emp",)):
        arms.append((f"weapons={'+'.join(w) or 'none'}", spec(weapons=w), REF))
    for b in (("speed",), ("hp",), ("speed", "hp")):
        arms.append((f"base_upg={'+'.join(b)}", spec(base_upg=b), REF))
    arms.append(("relocate", spec(relocate=True), REF))
    for k in (1, 2, 4, 5, 8):
        arms.append((f"light_min_ants={k}", spec(light_min_ants=k), REF))
    for lm in ("nearbase", "towers"):
        arms.append((f"light_mode={lm}", spec(light_mode=lm), REF))
        arms.append((f"light_mode={lm}+k1", spec(light_mode=lm, light_min_ants=1), REF))
    for dm in (6, 10, 14):
        arms.append((f"light_max_dist={dm}", spec(light_max_dist=dm), REF))
    for br in (0, 50, 200, 300):
        arms.append((f"build_reserve={br}", spec(build_reserve=br), REF))
    for ur in (60, 120, 200):
        arms.append((f"upg_reserve={ur}", spec(upg_reserve=ur), REF))
    rows = run_arms(arms, seeds, workers)
    print_rows(rows)
    save_rows(rows, f"axes_n{seeds}.tsv")


def _cell_indices():
    """两个座位**都能建塔**的 STRATEGIC_BUILD_ORDER 下标（0/1 是基地格和基地旁格，不可建）。"""
    from test_match.fast_arena import _new_state, _op, default_slots, OP_BUILD
    st = _new_state(12345)
    ok = []
    for i in range(len(default_slots(0))):
        good = True
        for p in (0, 1):
            x, y = default_slots(p)[i]
            if not st.can_apply_operation(p, _op(OP_BUILD, x, y), []):
                good = False
                break
        if good:
            ok.append(i)
    return ok


def stage_cells(seeds, workers):
    inds = _cell_indices()
    n = len(default_len())
    arms = [("default_order", REF, REF)]
    for k in inds:
        rank = [k] + [i for i in range(n) if i != k]
        arms.append((f"first={k}", spec(slots=rank), REF))
    rows = run_arms(arms, seeds, workers)
    print_rows(rows)
    save_rows(rows, f"cells_n{seeds}.tsv")


def default_len():
    from test_match.fast_arena import default_slots
    return default_slots(0)


def stage_greedy(seeds, workers, k=3, resume_from=None):
    inds = _cell_indices()
    n = len(default_len())
    chosen: list[int] = []
    if resume_from and os.path.exists(resume_from):
        prev = sorted(_read_rows(resume_from), key=lambda r: -r["margin"])
        for r in prev:
            if r["name"].startswith("first="):
                i = int(r["name"][6:])
                if i not in chosen:
                    chosen.append(i)
    all_rows = []
    for step in range(k):
        arms = []
        for i in inds:
            if i in chosen:
                continue
            rank = chosen + [i] + [j for j in range(n) if j not in chosen and j != i]
            arms.append((f"pos{step}={i}", spec(slots=rank), REF))
        rows = run_arms(arms, seeds, workers)
        rows.sort(key=lambda r: -r["margin"])
        print(f"--- 第 {step} 个位置 ---")
        for r in rows[:8]:
            print(f"  {r['name']:16s} margin={r['margin']:+7.2f} ±{r['se']:.2f}")
        best = rows[0]
        i = int(best["name"].split("=")[1])
        chosen.append(i)
        print(f"  ⇒ 选第 {step} 位：idx={i}  margin={best['margin']:+.2f} ±{best['se']:.2f}")
        all_rows.extend(rows)
    print(f"[greedy] 建塔顺序 idx = {chosen}")
    rank = chosen + [j for j in range(n) if j not in chosen]
    rows = run_arms([("greedy_order", spec(slots=rank), REF),
                     ("default_order", REF, REF)], max(seeds, 128), workers)
    print_rows(rows)
    save_rows(all_rows, f"greedy_steps_n{seeds}.tsv")
    save_rows(rows, "greedy_final.tsv")
    with open(os.path.join(OUT_DIR, "greedy_order.json"), "w", encoding="utf-8") as f:
        json.dump(rank, f)


def _read_rows(fn):
    rows = []
    with open(fn, encoding="utf-8") as f:
        head = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(name=parts[0], n=int(parts[1]), margin=float(parts[2]),
                             se=float(parts[3]), win=int(parts[4]), loss=int(parts[5]),
                             tie=int(parts[6]), rounds=float(parts[7]),
                             towers=float(parts[8]), coins=float(parts[9]),
                             a=parts[10], b=parts[11]))
    return rows


def stage_grid(seeds, workers):
    """把最强的几根轴组合起来，看是否叠加 + 找联合最优。"""
    arms = [("ref", REF, REF)]
    for br in (200, 300, 400):
        for up in ("never", "always"):
            for nt in (1, 2, 4):
                arms.append((f"br{br}_up{up}_n{nt}",
                             spec(build_reserve=br, upg=up, n_towers=nt), REF))
    # 再加几个"富 + 不升级 + 不建塔"的极端臂
    arms.append(("br400_never_n0", spec(build_reserve=400, upg="never", n_towers=0), REF))
    arms.append(("br300_never_n2_k2", spec(build_reserve=300, upg="never", n_towers=2,
                                           light_min_ants=2), REF))
    arms.append(("br300_never_n2_k4", spec(build_reserve=300, upg="never", n_towers=2,
                                           light_min_ants=4), REF))
    rows = run_arms(arms, seeds, workers)
    print_rows(rows)
    save_rows(rows, f"grid_n{seeds}.tsv")
    best = rows[0]
    print(f"[grid] best = {best['name']} margin={best['margin']:+.2f} ±{best['se']:.2f}")
    with open(os.path.join(OUT_DIR, "best_policy.json"), "w", encoding="utf-8") as f:
        json.dump(json.loads(best["a"]), f, sort_keys=True)


def stage_final(seeds, workers):
    fn = os.path.join(OUT_DIR, "best_policy.json")
    if os.path.exists(fn):
        best = json.load(open(fn, encoding="utf-8"))
        best_spec = json.dumps(best, sort_keys=True)
        print(f"[final] best = {best_spec}")
    else:
        best_spec = REF
        print("[final] ! no best_policy.json, 用 REF")
    go = os.path.join(OUT_DIR, "greedy_order.json")
    if os.path.exists(go):
        rank = json.load(open(go, encoding="utf-8"))
        d = json.loads(best_spec)
        d["slots"] = rank
        best_spec = json.dumps(d, sort_keys=True)
        print(f"[final] + greedy 建塔顺序 idx={rank[:5]}...")
    arms = [
        ("best_vs_ref", best_spec, REF),
        ("best_vs_basic", best_spec, spec(target="basic", upg="never", weapons=())),
        ("best_vs_lightonly", best_spec, spec(n_towers=0, upg="never", weapons=("light",),
                                             build_reserve=0)),
    ]
    rows = run_arms(arms, seeds, workers)
    print_rows(rows, sort=False)
    save_rows(rows, f"final_n{seeds}.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["axes", "cells", "greedy", "grid", "final"])
    ap.add_argument("--seeds", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    if a.stage == "axes":
        stage_axes(a.seeds, a.workers)
    elif a.stage == "cells":
        stage_cells(a.seeds, a.workers)
    elif a.stage == "greedy":
        stage_greedy(a.seeds, a.workers, k=a.k)
    elif a.stage == "grid":
        stage_grid(a.seeds, a.workers)
    elif a.stage == "final":
        stage_final(a.seeds, a.workers)


if __name__ == "__main__":
    main()
