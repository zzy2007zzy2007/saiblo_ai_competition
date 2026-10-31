#!/usr/bin/env python
"""在**已采集的 npz**（`training_history/collect_v2_s1/*.npz`）上，离线检验"逐帧标签 ~ 金币"的信号强度。

背景（`docs/value_net_label_pipeline_survey_20261031.md` §12/§13）：
  · `lowq_f256`（M6c 老数据）上 `kgeo50` 的「≥90 币 vs 60-89 币」桶差是 **−9.2% 个标签 std**（无信号）；
  · 统一搜索自己的 match trace 上同一口径是 **+55.8%**（有信号，但 n=275 事件）；
  ⇒ 现在换了**正式采集数据**（部署 BASE 配置、完整 board+stats），要重算同一张表。

做什么：
  1. 每局从 `stats[:,1]`（hp_delta，player 视角）取逐帧血差，按 **`az_train.py:119-129`** 的 kgeo/geo/rel/abs
     公式现算标签（tau = 10/25/50/100；kgeo 是 `w_k=k·γ^k`、w_0=0）。**与训练器实现逐位对齐**（脚本自带断言）。
  2. 对金币（`stats[:,22]×300`）做 **within-局** 回归：斜率/100 币、R²；再报"≥90 vs 60-89"的桶差（按标签 std 归一）。
  3. 另加"控制回合+血差"的斜率、以及**按局 bootstrap** 的置信区间（同局样本强自相关，不能按样本重采）。

兼容两种输入：
  · `--npz-dir <dir>`：本次采集（每局一个 npz，含 `stats/player`）
  · `--memmap-dir <dir>`：`lowq_f*`（`stats.npy` + `game_id.npy`），用同一套代码重算历史读数做对照

用法：
  python code/test_match/probe_collected_kgeo_tau.py --npz-dir training_history/collect_v2_s1
  python code/test_match/probe_collected_kgeo_tau.py --memmap-dir training_history/distill_top2/lowq_f256 --tag lowq_f256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HP_SCALE = 20.0
BINS = [(0.0, 45.0, "<45"), (45.0, 60.0, "45-59"), (60.0, 90.0, "60-89"), (90.0, 1e9, ">=90")]


# ────────────────────────────── 标签（与训练器逐位一致） ──────────────────────────────
def labels_for_game(stats: np.ndarray, player: np.ndarray, *, mode: str, weight: str,
                    tau: float, label_scale: float = 1.0, mix_alpha: float = 0.5) -> np.ndarray:
    """`az_train.add_weighted_labels` 的向量化等价实现（`label_mode`/`label_weight` 语义相同）。

    · d = 统一 P0 视角的瞬时血差（player==0 取 +hp_delta，否则取 −hp_delta）
    · geo : raw_t = Σ_{k≥0} γ^k d_{t+k} / Σ γ^k         （峰值在 k=0）
    · kgeo: raw_t = Σ_{k≥0} k γ^k d_{t+k} / Σ k γ^k     （w_0 = 0；末帧退化回 d_t）
    · abs : lab = raw；rel : lab = raw − d_t；mix : α·d_t + (1−α)·raw
    · label = clip(lab / HP_SCALE, ±1) * label_scale，再按 player 翻符号
    """
    hp = np.asarray(stats[:, 1], dtype=np.float64)          # hp_delta（player 视角）
    pl = np.asarray(player, dtype=np.int64)
    d = np.where(pl == 0, hp, -hp)                          # 统一 P0 视角
    g = float(np.exp(-1.0 / tau))
    n = d.shape[0]
    out = np.empty(n, dtype=np.float64)
    if weight == "kgeo":
        A = B = WA = WB = 0.0
        for t in range(n - 1, -1, -1):
            B = g * (B + A)
            WB = g * (WB + WA)
            A = d[t] + g * A
            WA = 1.0 + g * WA
            raw = (B / WB) if WB > 1e-9 else d[t]
            out[t] = raw
    else:
        suffix = wsum = 0.0
        for t in range(n - 1, -1, -1):
            suffix = d[t] + g * suffix
            wsum = 1.0 + g * wsum
            out[t] = suffix / wsum
    if mode == "abs":
        lab = out
    elif mode == "mix":
        lab = mix_alpha * d + (1.0 - mix_alpha) * out
    else:  # "rel"
        lab = out - d
    lab = np.clip(lab / HP_SCALE, -1.0, 1.0) * label_scale
    return np.where(pl == 0, lab, -lab).astype(np.float32)


# ────────────────────────────── 数据加载 ──────────────────────────────
def load_npz_dir(d: Path, limit: int = 0):
    """-> list[(name, stats, player, coins, round)]；坏文件跳过（采集中可能撞到半截文件）。"""
    files = sorted(d.glob("*.npz"))
    if limit:
        files = files[:limit]
    games, bad = [], 0
    for f in files:
        try:
            z = np.load(f)
            stats = np.asarray(z["stats"], dtype=np.float32)
            player = np.asarray(z["player"], dtype=np.int64)
            if stats.ndim != 2 or stats.shape[0] < 8 or player.shape[0] != stats.shape[0]:
                bad += 1
                continue
            coins = stats[:, 22].astype(np.float64) * 300.0     # stats[22] = coins[player]/300
            rounds = stats[:, 0].astype(np.float64) * 512.0     # stats[0] = round_ratio
            games.append((f.name, stats, player, coins, rounds))
        except Exception:
            bad += 1
    return games, bad


def load_memmap_dir(d: Path, limit: int = 0):
    stats = np.load(d / "stats.npy", mmap_mode="r")
    gid = np.load(d / "game_id.npy").astype(np.int64)
    p0 = (np.asarray(stats[:, 36], dtype=np.float32) > 0.5).astype(np.int64)
    ng = int(gid.max()) + 1
    games = []
    for g in range(ng):
        idx = np.where(gid == g)[0]
        if limit and len(games) >= limit:
            break
        st = np.asarray(stats[idx], dtype=np.float32)
        games.append((f"g{g:05d}", st, p0[idx],
                      st[:, 22].astype(np.float64) * 300.0,
                      st[:, 0].astype(np.float64) * 512.0))
    return games, 0


# ────────────────────────────── 累计统计（可跨局相加 ⇒ bootstrap 很便宜） ──────────────────────────────
def accumulate(games, lab_key: str, tau: float) -> dict:
    """对某一 (mode, weight, tau) 的标签，累计"within-局"回归所需的二阶矩 + 分桶和。

    within 局 = 每局各自去均值（标签/金币/回合/血差都减本局均值），再把各局的
    Σx²/Σxy/… 相加 ⇒ 等价于"带局固定效应"的 pooled OLS（共同斜率）。
    """
    acc = {"n": 0.0, "syy": 0.0, "XtX": np.zeros((3, 3)), "Xty": np.zeros(3),
           "bucket_n": {nm: 0.0 for _, _, nm in BINS},
           "bucket_sy": {nm: 0.0 for _, _, nm in BINS},
           "per_game": []}
    for (_name, stats, player, coins, rounds) in games:
        mode, weight = lab_key.split(":")
        y = labels_for_game(stats, player, mode=mode, weight=weight, tau=tau)
        hp = np.asarray(stats[:, 1], dtype=np.float64)
        y = y.astype(np.float64) - y.mean()
        c = coins - coins.mean()
        r = rounds - rounds.mean()
        h = hp - hp.mean()
        X = np.column_stack([c, r, h])
        pg = {"n": float(len(y)), "syy": float(y @ y),
              "XtX": X.T @ X, "Xty": X.T @ y,
              "bucket_n": {}, "bucket_sy": {}}
        for lo, hi, nm in BINS:
            m = (coins >= lo) & (coins < hi)
            pg["bucket_n"][nm] = float(m.sum())
            pg["bucket_sy"][nm] = float(y[m].sum())
        acc["per_game"].append(pg)
    return acc


def pooled(acc: dict, mask=None) -> dict:
    """把若干局（mask=下标数组）的累计量相加并算指标。"""
    pgs = acc["per_game"]
    idx = range(len(pgs)) if mask is None else mask
    n = syy = 0.0
    XtX = np.zeros((3, 3)); Xty = np.zeros(3)
    bn = {nm: 0.0 for _, _, nm in BINS}; bsy = {nm: 0.0 for _, _, nm in BINS}
    for i in idx:
        pg = pgs[i]
        n += pg["n"]; syy += pg["syy"]; XtX += pg["XtX"]; Xty += pg["Xty"]
        for nm in bn:
            bn[nm] += pg["bucket_n"].get(nm, 0.0)
            bsy[nm] += pg["bucket_sy"].get(nm, 0.0)
    if n <= 0 or syy <= 0:
        return {}
    slope = Xty[0] / XtX[0, 0] if XtX[0, 0] > 0 else float("nan")
    r2 = (Xty[0] ** 2) / (XtX[0, 0] * syy) if XtX[0, 0] > 0 else float("nan")
    try:
        b = np.linalg.solve(XtX + 1e-9 * np.eye(3), Xty)
        slope_ctl = float(b[0])
        r2_ctl = float(Xty @ b / syy)
    except np.linalg.LinAlgError:
        slope_ctl = r2_ctl = float("nan")
    sd = float(np.sqrt(syy / n))
    m90, m68 = bn[">=90"], bn["60-89"]
    d_abs = (bsy[">=90"] / m90 - bsy["60-89"] / m68) if (m90 > 0 and m68 > 0) else float("nan")
    return {"n": n, "sd": sd, "slope100": slope * 100.0, "r2": r2,
            "slope100_ctl": slope_ctl * 100.0, "r2_ctl": r2_ctl,
            "delta": d_abs, "delta_pct": (d_abs / sd * 100.0) if sd > 0 else float("nan"),
            "mean_90": (bsy[">=90"] / m90) if m90 else float("nan"),
            "mean_68": (bsy["60-89"] / m68) if m68 else float("nan"),
            "n90": m90, "n68": m68,
            "bins": {nm: (bsy[nm] / bn[nm] if bn[nm] else float("nan"), bn[nm])
                     for _, _, nm in BINS}}


def boot_ci(acc: dict, reps: int = 500, seed: int = 0) -> dict:
    """按**局**重采样（同局样本强自相关 ⇒ 只能按局重采）。返回 delta/slope 的 5–95 分位。"""
    rng = np.random.default_rng(seed)
    ng = len(acc["per_game"])
    if ng < 3:
        return {}
    ds, ss = [], []
    for _ in range(reps):
        pick = rng.integers(0, ng, size=ng)
        p = pooled(acc, mask=pick)
        if p:
            ds.append(p["delta_pct"]); ss.append(p["slope100"])
    if not ds:
        return {}
    return {"delta_lo": float(np.percentile(ds, 5)), "delta_hi": float(np.percentile(ds, 95)),
            "slope_lo": float(np.percentile(ss, 5)), "slope_hi": float(np.percentile(ss, 95))}


# ────────────────────────────── 主流程 ──────────────────────────────
def selfcheck(games) -> None:
    """与训练器实现逐位对齐的断言（只用第 1 局，快）。"""
    try:
        import sys
        for p in (REPO / "Ant-Game", REPO / "code"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from my_ai.az_intent.value_warmup import _exp_weighted_labels
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [selfcheck] 跳过（导入失败：{type(exc).__name__}: {exc}）")
        return
    _name, stats, player, _c, _r = games[0]
    for mode, weight in (("rel", "kgeo"), ("rel", "geo"), ("abs", "kgeo"), ("mix", "geo")):
        for tau in (10.0, 50.0):
            mine = labels_for_game(stats, player, mode=mode, weight=weight, tau=tau)
            ref = _exp_weighted_labels(stats.astype(np.float16), player, tau=tau,
                                      label_scale=1.0, label_mode=mode, label_weight=weight)
            diff = float(np.abs(mine.astype(np.float64) - ref.astype(np.float64)).max())
            print(f"  [selfcheck] {mode}+{weight} tau={tau:g}: max|mine-ref| = {diff:.2e}"
                  + ("  ✓" if diff < 2e-3 else "  ✗ 不一致！"))
            if diff >= 2e-3:
                raise SystemExit("自检失败：标签实现与训练器不一致")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="")
    ap.add_argument("--memmap-dir", default="")
    ap.add_argument("--tag", default="collect_v2_s1")
    ap.add_argument("--taus", type=float, nargs="+", default=[10.0, 25.0, 50.0, 100.0])
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 局（0 = 全部）")
    ap.add_argument("--boot", type=int, default=500, help="按局 bootstrap 次数（0 = 不做）")
    a = ap.parse_args()

    if a.npz_dir:
        d = Path(a.npz_dir)
        d = d if d.is_absolute() else REPO / d
        games, bad = load_npz_dir(d, a.limit)
    elif a.memmap_dir:
        d = Path(a.memmap_dir)
        d = d if d.is_absolute() else REPO / d
        games, bad = load_memmap_dir(d, a.limit)
    else:
        raise SystemExit("给 --npz-dir 或 --memmap-dir")
    if not games:
        raise SystemExit("没有可用对局")
    n_dec = sum(len(g[3]) for g in games)
    print(f"=== {a.tag}  局数={len(games)}  决策点={n_dec:,}  坏文件={bad}")
    print(f"    金币 p50 = {np.median(np.concatenate([g[3] for g in games])):.1f}  "
          f"coins>=90 帧占比 = {np.mean(np.concatenate([g[3] >= 90 for g in games])):.3%}")
    selfcheck(games)

    variants = [(m, w, t) for (m, w) in (("rel", "kgeo"),) for t in a.taus]
    variants += [("rel", "geo", 50.0), ("abs", "geo", 50.0), ("abs", "kgeo", 50.0),
                 ("rel", "kgeo", 50.0)]
    seen, ordered = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v); ordered.append(v)

    print(f"\n{'标签':<18} {'n':>7} {'标签std':>8} {'slope/100币':>11} {'R2':>8} "
          f"{'+round+hp':>10} {'Δ(>=90−60-89)':>13} {'占std':>7} {'n(>=90)':>8}")
    rows = []
    for (mode, weight, tau) in ordered:
        key = f"{mode}:{weight}"
        acc = accumulate(games, key, tau)
        p = pooled(acc)
        ci = boot_ci(acc, reps=a.boot) if a.boot else {}
        rows.append({"variant": f"{mode}+{weight}", "tau": tau, **p, **ci})
        print(f"{mode + '+' + weight + f' tau={tau:g}':<18} {int(p['n']):>7,} {p['sd']:>8.4f} "
              f"{p['slope100']:>+11.4f} {p['r2']:>8.4f} {p['slope100_ctl']:>+10.4f} "
              f"{p['delta']:>+13.5f} {p['delta_pct']:>+6.1f}% {int(p['n90']):>8,}"
              + (f"   CI[{ci['delta_lo']:+.0f}%,{ci['delta_hi']:+.0f}%]" if ci else ""))

    # 明细：每个变体的分桶均值
    print("\n--- 分桶均值（within 局；括号里是帧数）")
    for r in rows:
        if r["variant"] != "rel+kgeo" and r["tau"] != 50.0:
            continue
        bins = r["bins"]
        print(f"  {r['variant']+' tau='+format(r['tau'],'g'):<18} " + "  ".join(
            f"{nm}={bins[nm][0]:+.5f}({int(bins[nm][1])})" for _, _, nm in BINS))

    # 拟合最好的 tau
    kg = [r for r in rows if r["variant"] == "rel+kgeo"]
    if kg:
        best = max(kg, key=lambda r: abs(r["delta_pct"]))
        print(f"\n⇒ kgeo 里 |Δ/std| 最大的 tau = {best['tau']:g} "
              f"（Δ={best['delta']:+.5f}, 占 std {best['delta_pct']:+.1f}%, "
              f"slope={best['slope100']:+.4f}/100币）")
        print("⇒ kgeo 各 tau 的 |Δ/std|：" + "  ".join(
            f"tau{int(r['tau'])}={r['delta_pct']:+.1f}%" for r in kg))

    out = REPO / "match_results" / f"probe_collected_kgeo_tau_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tag": a.tag, "games": len(games), "decisions": n_dec,
                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[json] {out}（match_results/ 在 .gitignore:30，不入库）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
