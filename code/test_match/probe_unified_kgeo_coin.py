#!/usr/bin/env python
"""只读诊断②：**统一搜索自己下的对局**里，现算的 `kgeo tau=50` 标签有没有"金币≥90 ⇒ 值更高"。

与 `probe_kgeo_coin_threshold.py` 的区别（那条跑的是 `lowq_f256` = M6c 老数据）：
  这里用的是**统一候选搜索（unified cand）自己打的 match 对局**的 progress trace
  （`training_history/progress/<TAG>/az_progress_seed*.txt`，格式见 `code/test_match/progress.py:5-14`）：
  ```
  round=r HP0=.. HP1=.. G0=.. G1=.. P0=[...] P1=[...]
  terminal winner=.. hp=[..] rounds=..
  ```
  `HP/G` 是**本回合操作执行前（= 决策时刻）**的双方值 ⇒ 正好是"局面"。

这些对局**没有现成的 kgeo 标签**，所以这里按 `az_train.add_weighted_labels`（`az_train.py:119-129`）
与 `value_warmup._exp_weighted_labels`（`value_warmup.py:193-211`）的定义**现算**：
      gamma = exp(-1/tau)
      d_t   = 统一 P0 视角的瞬时血差（= HP0 - HP1）
      kgeo  : raw_t = Σ_{k≥0} k·γ^k·d_{t+k} / Σ_{k≥0} k·γ^k   （w_0=0；末帧退化回 d_t）
      rel   : label_t = clip((raw_t - d_t)/HP_SCALE, ±1)，再按该样本的 player 翻符号
  ⚠️ 时间刻度差异：本 trace **每回合一个样本**，而训练数据是"每回合 2 个决策点"。
     所以这里的 tau 是"回合"单位；`tau=50` ≈ 训练数据里的 `tau=100`（帧单位）。
     为了让两者可比，脚本同时报 `tau=25`（≈ 训练数据的 tau=50）。

座位约定（**这条会决定结论**）：本项目的 match 驱动用 `seed` / `seedr` 标记配对座位，
     `_us_vs_external.sh:223-231`、`_pin_k2_vs_runnerup.sh:161-169`：
       token == base（无 `r`）⇒ 我们坐 **P0**；token 带 `r` ⇒ 我们坐 **P1**。
     本脚本按此解析；已用 UNIFIED_AB_A 的 seed7 / seed7r 实测核对（无 r 那局 P0 第 0 回合就建 2 座塔）。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/probe_unified_kgeo_coin.py
  ... --dirs UNIFIED_SPLIT_VERBOSE UNIFIED_AB_A       # 指定 tag 子目录
  ... --all-unified                                    # 加总所有 UNIFIED_* 目录（鲁棒性）
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PROG = REPO / "training_history/progress"
HP_SCALE = 20.0
BINS = [(0.0, 45.0, "<45"), (45.0, 60.0, "45-59"), (60.0, 90.0, "60-89"), (90.0, 1e9, ">=90")]
LINE = re.compile(r"round=(\d+) HP0=(-?\d+) HP1=(-?\d+) G0=(-?\d+) G1=(-?\d+)")
FNAME = re.compile(r"^az_progress_seed(\d+)(r?)(?:_([A-Za-z0-9]+))?\.txt$")


def parse(path: Path) -> dict | None:
    """-> {rounds, hp0, hp1, g0, g1, our_side, winner, final_hp, n_round}"""
    m = FNAME.match(path.name)
    if not m:
        return None
    side = 1 if m.group(2) == "r" else 0
    rows = []
    winner = None
    final_hp = None
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        mm = LINE.match(ln)
        if mm:
            rows.append([int(x) for x in mm.groups()])
        elif ln.startswith("terminal"):
            w = re.search(r"winner=(-?\d+)", ln)
            h = re.search(r"hp=\[(\d+), (\d+)\]", ln)
            winner = int(w.group(1)) if w else None
            final_hp = (int(h.group(1)), int(h.group(2))) if h else None
    if len(rows) < 20:
        return None
    a = np.asarray(rows, dtype=np.int64)
    return {"rnd": a[:, 0], "hp0": a[:, 1], "hp1": a[:, 2], "g0": a[:, 3], "g1": a[:, 4],
            "side": side, "winner": winner, "final_hp": final_hp, "file": path.name,
            "tag": path.parent.name}


def kgeo_labels(d_p0: np.ndarray, player: int, tau: float) -> np.ndarray:
    """与 `az_train.add_weighted_labels`（label_weight='kgeo', label_mode='rel'）逐步一致。"""
    n = len(d_p0)
    d = d_p0 if player == 0 else -d_p0        # 统一到 P0 视角（与训练器一致）
    g = float(np.exp(-1.0 / tau))
    out = np.empty(n, dtype=np.float64)
    A = B = WA = WB = 0.0
    for t in range(n - 1, -1, -1):
        B = g * (B + A)
        WB = g * (WB + WA)
        A = d[t] + g * A
        WA = 1.0 + g * WA
        raw = (B / WB) if WB > 1e-9 else d[t]
        lab = float(np.clip((raw - d[t]) / HP_SCALE, -1.0, 1.0))
        out[t] = lab if player == 0 else -lab
    return out


def _ols(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((r ** 2).sum()) / sst if sst > 1e-12 else float("nan")
    return b, r2


def _demean(v, key):
    cnt = np.bincount(key).astype(np.float64)
    cnt[cnt == 0] = 1.0
    return v - (np.bincount(key, weights=v) / cnt)[key]


def summarize(games: list[dict], tau_main: float, min_rounds: int, use_opp: bool = False) -> dict:
    """把多局拼成 (sample, game_id) 两个数组，返回标签/金币等。

    ``use_opp=True`` 时把**对手**当作"player"（同一批对局、另一个视角），用来对照
    "positive Δ 是我们这边特有的，还是任何一方都有"。
    """
    Y, C, R, GID, SIDE, OPPC, HP = [], [], [], [], [], [], []
    for gi, gm in enumerate(games):
        n = len(gm["rnd"])
        if n < min_rounds:
            continue
        side = gm["side"] if not use_opp else 1 - gm["side"]
        d_p0 = gm["hp0"] - gm["hp1"]
        lab = kgeo_labels(d_p0.astype(np.float64), side, tau_main)
        coins = (gm["g0"] if side == 0 else gm["g1"]).astype(np.float64)
        oppc = (gm["g1"] if side == 0 else gm["g0"]).astype(np.float64)
        hpv = (gm["hp0"] if side == 0 else gm["hp1"]).astype(np.float64) - \
              (gm["hp1"] if side == 0 else gm["hp0"]).astype(np.float64)
        Y.append(lab); C.append(coins); R.append(gm["rnd"].astype(np.float64)); HP.append(hpv)
        GID.append(np.full(n, gi)); SIDE.append(np.full(n, side)); OPPC.append(oppc)
    if not Y:
        return {}
    return {"y": np.concatenate(Y), "coins": np.concatenate(C), "rnd": np.concatenate(R),
            "gid": np.concatenate(GID), "side": np.concatenate(SIDE),
            "oppc": np.concatenate(OPPC), "hp": np.concatenate(HP)}


def _cluster_boot(y, coins, rnd, gid, reps: int = 2000, seed: int = 0):
    """按**局**重采样的 bootstrap（同局样本强自相关，不能按样本重采）。

    返回 (Δ(>=90 − 60-89) 的 5–95 分位, slope(within, coins) 的 5–95 分位)。
    """
    rng = np.random.default_rng(seed)
    ids = np.unique(gid)
    d_lo = d_hi = s_lo = s_hi = float("nan")
    ds, ss = [], []
    for _ in range(reps):
        pick = rng.choice(ids, size=len(ids), replace=True)
        yy, cc, rr, gid2 = [], [], [], []
        for k, gi in enumerate(pick):
            m = gid == gi
            yy.append(y[m]); cc.append(coins[m]); rr.append(rnd[m])
            gid2.append(np.full(int(m.sum()), k))
        yy = np.concatenate(yy); cc = np.concatenate(cc); rr = np.concatenate(rr)
        gid2 = np.concatenate(gid2)
        yw = _demean(yy, gid2); cw = _demean(cc, gid2)
        m90 = cc >= 90
        m68 = (cc >= 60) & (cc < 90)
        if m90.sum() > 2 and m68.sum() > 2:
            ds.append(float(yw[m90].mean() - yw[m68].mean()))
        if len(np.unique(gid2)) > 1:
            b, _ = _ols(yw, np.column_stack([cw]))
            ss.append(float(b[1]))
    if ds:
        d_lo, d_hi = float(np.percentile(ds, 5)), float(np.percentile(ds, 95))
    if ss:
        s_lo, s_hi = float(np.percentile(ss, 5)), float(np.percentile(ss, 95))
    return (d_lo, d_hi), (s_lo, s_hi)


def report(name: str, games: list[dict], tau: float, min_rounds: int) -> None:
    S = summarize(games, tau, min_rounds)
    if not S:
        print(f"\n=== {name}: 无可用对局")
        return
    y, coins, rnd, gid = S["y"], S["coins"], S["rnd"], S["gid"]
    n_games = len(np.unique(gid))
    print(f"\n=== {name}   tau={tau:g} 回合（≈训练数据 tau={tau*2:g} 帧）")
    print(f"    对局 {n_games} 局 / {len(y):,} 个回合样本（每局 ≥{min_rounds} 回合）")
    print(f"    kgeo 标签: mean={y.mean():+.4f}  std={y.std():.4f}  range=[{y.min():+.2f},{y.max():+.2f}]")
    print(f"    我方金币: p5/p50/p95 = {np.percentile(coins,[5,50,95])}  均值 {coins.mean():.1f}"
          f"   对手金币均值 {S['oppc'].mean():.1f}")
    print(f"    ★ 我方 coins>=90 的回合占比 = {float((coins>=90).mean()):.3%}"
          f"   (>=45: {float((coins>=45).mean()):.1%})")

    yw = _demean(y, gid)
    cw = _demean(coins, gid)
    rw = _demean(rnd, gid)
    print(f"\n    分桶（pooled / within 局）：")
    print(f"    {'桶':<8} {'回合数':>8} {'占比':>7} {'pooled':>10} {'within':>10}")
    means = {}
    for lo, hi, nm in BINS:
        m = (coins >= lo) & (coins < hi)
        if m.sum() == 0:
            continue
        means[nm] = float(yw[m].mean())
        print(f"    {nm:<8} {int(m.sum()):>8,} {m.mean():>7.3f} {y[m].mean():>+10.5f} {yw[m].mean():>+10.5f}")
    if ">=90" in means and "60-89" in means:
        d = means[">=90"] - means["60-89"]
        print(f"    Δ(within, >=90 − 60-89) = {d:+.5f}   （within 标签 std={yw.std():.4f} ⇒ {d/yw.std():+.1%} 个 std）")
    if ">=90" in means and "45-59" in means:
        print(f"    Δ(within, >=90 − 45-59) = {means['>=90'] - means['45-59']:+.5f}")

    print("\n    回归（within 局）：")
    hw = _demean(S["hp"], gid)
    for nm, X in (("coins", [cw]), ("coins + round", [cw, rw]), ("coins + round + hp", [cw, rw, hw])):
        b, r2 = _ols(yw, np.column_stack(X))
        print(f"      {nm:<19} slope={b[1]:+.6f}/币 (×100={b[1]*100:+.4f})  R2={r2:.4f}")

    # 按局 bootstrap：给"仅提示"一个定量尺度
    dci, sci = _cluster_boot(y, coins, rnd, gid)
    n90 = int((coins >= 90).sum())
    print(f"\n    按局 bootstrap（2000 次）：Δ(>=90−60-89) 90%CI = [{dci[0]:+.5f}, {dci[1]:+.5f}]"
          f"   slope 90%CI = [{sci[0]:+.6f}, {sci[1]:+.6f}]/币")
    print(f"    >=90 的回合总数 = {n90}（分布在 {len(np.unique(gid[coins>=90]))}/{n_games} 局）"
          f" ⇒ {'样本太少，方向不可判定' if n90 < 60 else '样本量可做粗判'}")

    # 时间趋势控制：再扣掉"回合十分位"（金币是随回合累积的慢变量）
    dec = np.searchsorted(np.percentile(rnd, [10, 20, 30, 40, 50, 60, 70, 80, 90]), rnd)
    k2 = gid * 10 + dec
    y2 = yw - _demean(yw, k2)
    c2 = cw - _demean(cw, k2)
    print("\n    双向固定效应（局 + 回合十分位）：")
    print(f"    {'桶':<8} {'n':>7} {'均值':>10}")
    m90 = coins >= 90
    m68 = (coins >= 60) & (coins < 90)
    for lo, hi, nm in BINS:
        m = (coins >= lo) & (coins < hi)
        if m.sum():
            print(f"    {nm:<8} {int(m.sum()):>7,} {y2[m].mean():>+10.5f}")
    if m90.sum() > 2 and m68.sum() > 2:
        d2 = float(y2[m90].mean() - y2[m68].mean())
        b, r2 = _ols(y2, np.column_stack([c2]))
        print(f"    Δ(>=90 − 60-89) = {d2:+.5f}（{d2/y2.std():+.1%} 个 std）"
              f"   slope={b[1]:+.6f}/币 (×100={b[1]*100:+.4f})  R2={r2:.4f}")

    print("\n    逐局明细（我方 coins>=90 占比 / >=90 桶与 60-89 桶的 within 均值差）：")
    for gi in np.unique(gid):
        m = gid == gi
        r90 = float((coins[m] >= 90).mean())
        m90 = m & (coins >= 90)
        m68 = m & (coins >= 60) & (coins < 90)
        d = (yw[m90].mean() - yw[m68].mean()) if (m90.sum() > 3 and m68.sum() > 3) else float("nan")
        print(f"      game{gi:<3} n={int(m.sum()):>4}  coins.p50={np.median(coins[m]):>6.1f}"
              f"  >=90={r90:>6.1%}  Δ(>=90−60-89)={d:+.5f}")


def contrast(name: str, games: list[dict], tau: float, min_rounds: int) -> None:
    """同一批对局、两个视角（我方 / 对手）的紧凑对照。"""
    print(f"\n=== 视角对照：{name}   tau={tau:g} 回合")
    print(f"    {'视角':<6} {'>=90 占比':>9} {'coins.p50':>10} {'Δ(>=90−60-89)':>15} {'slope/100币':>12} {'n(>=90)':>8}")
    for tag, use_opp in (("我方", False), ("对手", True)):
        S = summarize(games, tau, min_rounds, use_opp)
        if not S:
            continue
        y, coins, gid = S["y"], S["coins"], S["gid"]
        yw = _demean(y, gid)
        cw = _demean(coins, gid)
        m90 = coins >= 90
        m68 = (coins >= 60) & (coins < 90)
        d = float(yw[m90].mean() - yw[m68].mean()) if (m90.sum() > 2 and m68.sum() > 2) else float("nan")
        b, _ = _ols(yw, np.column_stack([cw]))
        print(f"    {tag:<6} {float(m90.mean()):>9.2%} {np.median(coins):>10.1f} {d:>+15.5f} "
              f"{b[1]*100:>+12.4f} {int(m90.sum()):>8,}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["UNIFIED_SPLIT_VERBOSE", "UNIFIED_AB_A"])
    ap.add_argument("--all-unified", action="store_true",
                    help="把 training_history/progress/ 下所有 UNIFIED_* 目录加总（鲁棒性）")
    ap.add_argument("--tau", type=float, nargs="+", default=[50.0])
    ap.add_argument("--tau-alt", type=float, default=25.0, help="时间刻度对齐后的 tau（≈训练数据 tau=50 帧）")
    ap.add_argument("--min-rounds", type=int, default=50)
    a = ap.parse_args()

    print(f"progress 根目录: {PROG}")
    for dname in a.dirs:
        d = PROG / dname
        if not d.is_dir():
            print(f"  !! 缺目录 {d}")
            continue
        games = [g for f in sorted(d.glob("az_progress_seed*.txt")) if (g := parse(f))]
        for tau in a.tau + ([a.tau_alt] if a.tau_alt else []):
            report(f"{dname}", games, tau, a.min_rounds)

    if a.all_unified:
        games = []
        for d in sorted(PROG.glob("UNIFIED_*")):
            for f in sorted(d.glob("az_progress_seed*.txt")):
                g = parse(f)
                if g:
                    g["tag"] = d.name
                    games.append(g)
        for tau in a.tau + ([a.tau_alt] if a.tau_alt else []):
            report("ALL UNIFIED_* （加总，鲁棒性）", games, tau, a.min_rounds)
        for tau in a.tau + ([a.tau_alt] if a.tau_alt else []):
            contrast("ALL UNIFIED_* （加总）", games, tau, a.min_rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
