#!/usr/bin/env python
"""只读诊断：`kgeo tau=50` 逐帧标签里到底有没有"金币≥90 ⇒ 值更高"这个阈值。

问题背景（`docs/value_net_label_pipeline_survey_20261031.md` §12）：
  现役值网用 terminal 标签（局内常数）；逐帧候选是 `abs/rel × geo/kgeo`，它们**只由未来 HP 轨迹**算出来，
  没有任何一项显式含金币/冷却。于是有一个便宜的问题：
  **"金币≥90（= 闪电价格）"这件事，是不是已经间接编码进 kgeo50 标签里了？**
  - 若是（≥90 桶明显更高 / 90 处有跳变）⇒ 标签已含阈值，**多采闪电样本**就够；
  - 若否（只单调或无关）⇒ 光换标签不够，**得换数据或加显式金币项**。

做法：纯 numpy，读 `lowq_f256`（1.28M 帧，10 个标签齐）的 `stats.npy` + `label_kgeo_tau50.npy`
      + `game_id.npy`，在 `lowq_f64`（320k 帧）上做一致性复核。不训练、不跑对局、不碰模型。

读出字段（`Ant-Game/SDK/utils/features.py`）：
  stats[22] = coins[player]/300 → 金币 = ×300
  stats[26] = 我方闪电剩余冷却/35 → 冷却(轮) = ×35，0 = 已好
  stats[1]  = hp_delta（我方 − 敌方）
  stats[0]  = round_ratio（×MAX_ROUND）
  stats[36] = (player==0)

口径说明（**两个都报**，别只看一个）：
  · `pooled`  ：所有帧混在一起回归 —— 会被"局级混杂"污染（赢的局可能同时金币多、血差好）；
  · `within`  ：按 (局, player) 去均值后再回归（固定效应）—— 回答"**同一局同一方**，多 1 币时代未来优势是否更好"，
                这才是"标签有没有教金币价值"的正确问法。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/probe_kgeo_coin_threshold.py
  ... --data training_history/distill_top2/lowq_f256 --tag f256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT = "training_history/distill_top2/lowq_f256"
BINS = [(0.0, 45.0, "<45"), (45.0, 60.0, "45-59"), (60.0, 90.0, "60-89"), (90.0, 1e9, ">=90")]
MAX_ROUND = 512.0


def _load(d: Path, label_file: str):
    stats = np.load(d / "stats.npy", mmap_mode="r")
    gid = np.load(d / "game_id.npy").astype(np.int64)
    y = np.load(d / label_file).astype(np.float64)
    col = lambda i: np.asarray(stats[:, i], dtype=np.float64)  # noqa: E731
    coins = col(22) * 300.0
    cd = col(26) * 35.0
    hp = col(1)
    rnd = col(0) * MAX_ROUND
    p = (col(36) > 0.5).astype(np.int64)
    return coins, cd, hp, rnd, p, gid, y


def _demean(v: np.ndarray, key: np.ndarray) -> np.ndarray:
    """按 key 分组去均值（组内均值从 v 里减掉）。"""
    cnt = np.bincount(key).astype(np.float64)
    cnt[cnt == 0] = 1.0            # 空组：避免 0 除（其下标不会出现在 key 里）
    return v - (np.bincount(key, weights=v) / cnt)[key]


def _ols(y: np.ndarray, X: np.ndarray):
    """返回 (coef, r2, n)。X 只放自变量（常数项内部加）。"""
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((r ** 2).sum()) / sst if sst > 1e-12 else float("nan")
    return b, r2, len(y)


def _reg_block(name: str, y, coins, rnd, hp, extra=()):
    print(f"\n  [{name}]  n={len(y):,}")
    print(f"    {'模型':<28} {'slope/1币':>11} {'×100币':>9} {'R2':>8}")
    specs = [("coins", np.column_stack([coins])),
             ("coins + round", np.column_stack([coins, rnd])),
             ("coins + round + hp", np.column_stack([coins, rnd, hp]))]
    for label, X in specs:
        b, r2, n = _ols(y, X)
        print(f"    {label:<28} {b[1]:+11.6f} {b[1]*100:+9.4f} {r2:8.4f}")
    return specs


def _ols_infer(y: np.ndarray, X: np.ndarray):
    """返回 (b, se, r2, n)：普通最小二乘 + 同方差标准误（只用于看 t 的量级）。"""
    A = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    dof = max(len(y) - A.shape[1], 1)
    s2 = float((r ** 2).sum()) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((r ** 2).sum()) / sst if sst > 1e-12 else float("nan")
    return b, se, r2, len(y)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT)
    ap.add_argument("--tag", default="f256")
    ap.add_argument("--label", default="label_kgeo_tau50.npy")
    ap.add_argument("--ref-labels", nargs="*", default=["label.npy", "label_geo_tau50.npy"])
    a = ap.parse_args()
    d = Path(a.data) if Path(a.data).is_absolute() else REPO / a.data
    coins, cd, hp, rnd, p, gid, y = _load(d, a.label)
    key = gid * 2 + p
    print(f"\n=== {a.tag}  data={d}  label={a.label}")
    print(f"    N={len(y):,}  games={int(gid.max())+1}  (game,player)={len(np.unique(key)):,}")
    print(f"    kgeo50: mean={y.mean():+.4f} std={y.std():.4f}")
    print(f"    coins : p5/p50/p95={np.percentile(coins,[5,50,95])}  frac>=90={float((coins>=90).mean()):.3f}")
    print(f"    cd    : frac==0={float((cd==0).mean()):.3f}  p50={float(np.median(cd)):.1f} 轮")

    # ---------- 1) 回归 ----------
    print("\n--- 1) 线性回归 kgeo50 ~ coins（pooled vs within game×player）")
    _reg_block("pooled", y, coins, rnd, hp)
    yw = _demean(y, key)
    cw, rw, hw = _demean(coins, key), _demean(rnd, key), _demean(hp, key)
    _reg_block("within (game,player)", yw, cw, rw, hw)

    # 断点回归（90 币处 jump + 两侧斜率）
    print("\n--- 1b) 分段（断点 = 90 币）：jump + 左右斜率（within）")
    z = cw - 90.0
    X = np.column_stack([z, (z > 0).astype(np.float64), z * (z > 0)])
    b, r2, _ = _ols(yw, X)
    print(f"    slope(<90)={b[1]:+.6f}/币  jump@90={b[2]:+.5f}  Δslope={b[3]:+.6f}  R2={r2:.4f}")

    # 阈值的正式检验：hinge + step，控制 coins/round，报 t
    print("\n--- 1c) 阈值检验（within）：y ~ coins + round + (coins-90)+ + 1[coins>=90]")
    hinge = np.maximum(cw - 90.0, 0.0)
    step = (cw >= 90.0).astype(np.float64)
    X = np.column_stack([cw, rw, hinge, step])
    b, se, r2, n = _ols_infer(yw, X)
    for nm, i in (("coins", 1), ("round", 2), ("(coins-90)+", 3), ("1[coins>=90]", 4)):
        t = b[i] / se[i] if se[i] > 0 else float("nan")
        print(f"    {nm:<15} coef={b[i]:+.6f}  se={se[i]:.6f}  t={t:+.2f}")
    print(f"    R2={r2:.4f}")

    # ---------- 2) 分桶 ----------
    print("\n--- 2) 金币分桶：kgeo50 均值（核心判据）")
    print(f"    {'桶':<8} {'帧数':>10} {'占比':>7} {'pooled均值':>11} {'pooled_std':>10} {'within均值':>11} {'se':>8}")
    means = {}
    for lo, hi, name in BINS:
        m = (coins >= lo) & (coins < hi)
        n = int(m.sum())
        if n == 0:
            continue
        mean_w = float(yw[m].mean())
        se = float(yw[m].std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        means[name] = (n, float(y[m].mean()), mean_w, se)
        print(f"    {name:<8} {n:>10,} {n/len(y):>7.3f} {y[m].mean():>+11.5f} {y[m].std():>10.5f} "
              f"{mean_w:>+11.5f} {se:>8.5f}")
    if ">=90" in means and "60-89" in means:
        d_hi = means[">=90"][2] - means["60-89"][2]
        d_mid = means[">=90"][2] - means["45-59"][2]
        print(f"\n    Δ(within)  >=90 − 60-89 = {d_hi:+.5f}   (标签 std={yw.std():.4f} ⇒ {d_hi/yw.std():+.1%} 个 std)")
        print(f"    Δ(within)  >=90 − 45-59 = {d_mid:+.5f}")

    # 90 附近 ±10 币的局部对比（更锐）
    print("\n--- 2b) 90 币附近的局部对比（within）")
    for lo, hi in ((70, 80), (80, 90), (90, 100), (100, 110)):
        m = (coins >= lo) & (coins < hi)
        if m.sum() > 200:
            print(f"    [{lo},{hi}) n={int(m.sum()):>7,}  mean={yw[m].mean():+.5f}")

    # 冷却分层的条件斜率（这才是 P0-3 判据在**标签**上的对应物）
    print("\n--- 2c) 按冷却分层的条件斜率（within，控制 round）：cd==0 vs cd>0")
    for name, m in (("cd==0", cd == 0), ("cd>0", cd > 0), ("cd>=20轮", cd >= 20)):
        if m.sum() < 500:
            continue
        b, r2, n = _ols(yw[m], np.column_stack([cw[m], rw[m]]))
        print(f"    {name:<9} n={n:>9,}  slope={b[1]:+.6f}/币 (×100={b[1]*100:+.4f})  R2={r2:.4f}")
    print("\n    分桶均值（within，按冷却分组）：")
    print(f"    {'cd组':<9}" + "".join(f"{nm:>12}" for _, _, nm in BINS) + f"{'n':>10}")
    for name, m in (("cd==0", cd == 0), ("cd>0", cd > 0)):
        row = f"    {name:<9}"
        for lo, hi, nm in BINS:
            mm = m & (coins >= lo) & (coins < hi)
            row += f"{(yw[mm].mean() if mm.sum() > 100 else float('nan')):>+12.5f}"
        row += f"{int(m.sum()):>10,}"
        print(row)
    # 双向固定效应：再扣掉"回合十分位"的均值（去掉局内时间趋势），重做桶均值与条件斜率
    print("\n--- 2d) 双向固定效应（(局,player) + 回合十分位）：桶均值与条件斜率")
    dec = np.minimum((rnd / (MAX_ROUND / 10.0)).astype(np.int64), 9)
    k2 = key * 10 + dec
    y2 = yw - _demean(yw, k2)      # 先扣 (局,player) 再扣回合十分位 ⇒ 两次去均值足够近似
    c2 = cw - _demean(cw, k2)
    print(f"    {'桶':<8} {'n':>9} {'均值':>10}   {'cd==0':>10} {'cd>0':>10}")
    for lo, hi, name in BINS:
        m = (coins >= lo) & (coins < hi)
        if m.sum() < 200:
            continue
        print(f"    {name:<8} {int(m.sum()):>9,} {y2[m].mean():>+10.5f}   "
              f"{y2[m & (cd == 0)].mean():>+10.5f} {y2[m & (cd > 0)].mean():>+10.5f}")
    for name, m in (("cd==0", cd == 0), ("cd>0", cd > 0)):
        b, r2, n = _ols(y2[m], np.column_stack([c2[m]]))
        print(f"    slope({name:<6}) = {b[1]:+.6f}/币  (×100={b[1]*100:+.4f})  n={n:,}")
    if (cd == 0).any() and (cd > 0).any():
        b0, _, _ = _ols(y2[cd == 0], np.column_stack([c2[cd == 0]]))
        b1, _, _ = _ols(y2[cd > 0], np.column_stack([c2[cd > 0]]))
        print(f"    条件斜率比 |cd==0| / |cd>0| = {abs(b0[1]) / max(abs(b1[1]), 1e-12):.2f}"
              f"   （符号：{'+' if b0[1] > 0 else '-'} / {'+' if b1[1] > 0 else '-'}）")

    print("\n--- 3) >=90 币桶内按冷却分（within）")
    m90 = coins >= 90
    for name, m2 in (("cd==0", m90 & (cd == 0)), ("cd>0", m90 & (cd > 0)),
                     ("cd>=10轮", m90 & (cd >= 10)), ("cd>=20轮", m90 & (cd >= 20))):
        if m2.sum() > 100:
            print(f"    {name:<10} n={int(m2.sum()):>8,}  mean={yw[m2].mean():+.5f}  "
                  f"pooled={y[m2].mean():+.5f}")
    # 交互项：coins>=90 × cd==0
    print("\n--- 3b) 交互回归（within）：50 币以上的币量 × 冷却已好")
    z50 = cw - 50.0
    good = (cd == 0).astype(np.float64)
    X = np.column_stack([z50, good, z50 * good])
    b, r2, _ = _ols(yw, X)
    print(f"    slope(coins)={b[1]:+.6f}/币  cd==0 主效应={b[2]:+.5f}  交互={b[3]:+.6f}  R2={r2:.4f}")

    # ---------- 4) 参考标签对照 ----------
    for ref in a.ref_labels:
        rp = d / ref
        if not rp.exists():
            continue
        yr = np.load(rp).astype(np.float64)
        yrw = _demean(yr, key)
        print(f"\n--- 4) 参考标签 {ref}: std={yr.std():.4f}")
        for lo, hi, name in BINS:
            m = (coins >= lo) & (coins < hi)
            if m.sum() == 0:
                continue
            print(f"    {name:<8} pooled={yr[m].mean():+.5f}  within={yrw[m].mean():+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
