"""用 **autograd** 直接测「价值网对各输入的敏感度」——不需要改任何状态。

历史：本脚本原来只在 `GameStateFacade.initial()` 上**只 `advance_round()`、从不 apply 任何 op**
的轨迹上取样（旧版 `:42-59`），于是量出的 `∂V/∂(我币/300) = +0.0274` 全部来自
**无塔、金币只涨不花**的轨迹（`docs/value_net_coin_pricing_hypothesis_20261030.md` §B.2）。
本版补上 P0-1：**同样的 autograd，在有塔的真实局面上再量一遍**，并给出比值判据。

两个取样源：
  · `no_tower` —— 原口径：`GameStateFacade.initial()` + 只 `advance_round()`（从不 apply op）
  · `tower`    —— 按 `match_results/ladder_logs/<tag>/<seed>.log` 逐 op `apply_operation_list`
                  复现的真实对局（`UNIFIED_SPLIT_VERBOSE/8.log` 第 0 轮就建了 2 座塔）

为什么用 autograd 而不是"改 coins 再前向"：
  · `state.coins[0] = x` 在 facade 上写的是**缓存列表**，`_cache.clear()` 会退回引擎真值
    ⇒ 干预无效 ⇒ 曾测出 `∂V/∂coins ≡ 0` 的假象（`docs/final_report_20261010.md:102-103`）。
  · 把观测当 `requires_grad=True` 的张量，前向一次、反传一次，`∂V/∂输入` 就是解析梯度。

判据（`docs/value_net_coin_pricing_hypothesis_20261030.md` §E-P0-1 写死）：
  有塔局面的金币梯度 vs 无塔的 +0.0274，若差 ≥2×（或符号翻转）⇒ **+0.0274 不可外推**。

用法：
  D:\\anaconda3\\envs\\pytorch-gpu\\python.exe code/test_match/probe_value_grad.py
  ... --tag UNIFIED_SPLIT_VERBOSE --seed 8 --every 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from SDK.utils.features import FeatureExtractor                      # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models           # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade        # noqa: E402

# 复用 P0-2 里的取样函数（同一套复现/口径，避免两份代码漂移）
from probe_value_coin_curve import (COIN_IDX, HP_SCALE, COIN_SCALE,  # noqa: E402
                                    LIGHTNING_CD_IDX, collect_logs, collect_trajectory)

CKPT = REPO / "checkpoints/posnet_A_k5_m32.pt"
BASELINE = 0.0274          # 报告里那个"无塔梯度"，本脚本要复核它


def grad_one_state(value_model, board: np.ndarray, stats: np.ndarray):
    """对一个局面做一次前向+反传，返回 (∂V/∂stats(42), |∂V/∂board| 逐通道(28), V)。"""
    st_t = torch.tensor(stats.astype(np.float32)).unsqueeze(0)
    bd_t = torch.tensor(board.astype(np.float32)).unsqueeze(0)
    st_t.requires_grad_(True)
    bd_t.requires_grad_(True)
    out = value_model(bd_t, st_t)
    v = (out["value"] if isinstance(out, dict) else out).sum()
    gs, gb = torch.autograd.grad(v, [st_t, bd_t])
    return (gs.detach().numpy().ravel(), np.abs(gb.detach().numpy()[0]).mean(axis=(1, 2)),
            float(v.detach()))


def group_stats(name: str, grp: list, value_model) -> dict:
    """对一组局面算梯度，打印该组的读数。"""
    if not grp:
        print(f"\n[{name}] 无样本"); return {}
    G, B, V = [], [], []
    for s in grp:
        g, b, v = grad_one_state(value_model, s["board"], s["stats"])
        G.append(g); B.append(b); V.append(v)
    gs, gb, vv = np.stack(G), np.stack(B), np.array(V)
    mag = np.abs(gs).mean(axis=0)
    g_co = gs[:, COIN_IDX]
    g_en = gs[:, COIN_IDX + 1]

    print(f"\n{'=' * 82}")
    print(f"[{name}] {len(grp)} 个局面 · V 均值 {vv.mean():+.4f}（std {vv.std():.4f}）")
    print(f"{'=' * 82}")
    print(f"  ∂V/∂(我币/300)：均值 {g_co.mean():+.6f}  中位 {np.median(g_co):+.6f}"
          f"  std {g_co.std():.6f}  min {g_co.min():+.6f}  max {g_co.max():+.6f}")
    print(f"                  为正 {int((g_co > 0).sum())}/{len(g_co)}"
          f"；p10 {np.quantile(g_co, 0.1):+.6f} / p90 {np.quantile(g_co, 0.9):+.6f}")
    print(f"  ∂V/∂(敌币/300)：均值 {g_en.mean():+.6f}  中位 {np.median(g_en):+.6f}"
          f"  为负 {int((g_en < 0).sum())}/{len(g_en)}")
    print(f"  换算（均值）：300 币 ≈ {g_co.mean() * HP_SCALE:+.3f} HP；"
          f"每 100 币 ≈ {(g_co.mean() / 3) * HP_SCALE:+.4f} HP；"
          f"45 币（第 3 座塔）≈ {(45 / COIN_SCALE) * g_co.mean() * HP_SCALE:+.4f} HP")
    print(f"  |∂V/∂x| 在 42 维 stats 里：金币维排名 "
          f"{int((mag > mag[COIN_IDX]).sum()) + 1}/{len(mag)}"
          f"（|grad|={mag[COIN_IDX]:.6f}，全体中位 {np.median(mag):.6f}，最大 {mag.max():.6f}）")
    print(f"  最敏感的 6 维："
          + ", ".join(f"dim{i}({mag[i]:.5f}{'·金币' if i == COIN_IDX else ''})"
                      for i in np.argsort(-mag)[:6]))
    bm = gb.mean(axis=0)
    print(f"  board 塔相关通道平均 |∂V/∂x|："
          + "  ".join(f"ch{i}={bm[i]:.5f}" for i in (4, 5, 6, 7, 8, 9)))
    print(f"  board 全通道 top-6："
          + ", ".join(f"ch{i}={bm[i]:.5f}" for i in np.argsort(-bm)[:6]))
    return {"n": len(grp), "g_coin": g_co, "g_enemy": g_en, "gs": gs, "mag": mag,
            "board_mag": bm, "V": vv}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--tag", default="UNIFIED_SPLIT_VERBOSE:8,UNIFIED_AB_A:10,UNIFIED_AB_A:11",
                    help="有塔组的 ladder 日志，逗号分隔，可带 seed（tag:seed）；自动去重")
    ap.add_argument("--seed", type=int, default=8, help="--tag 里没写 seed 时用它")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--traj-every", type=int, default=25)
    ap.add_argument("--traj-seeds", default="7,11")
    args = ap.parse_args()

    feat = FeatureExtractor(max_actions=96)
    _cls, _pos, value_model = load_three_models(args.ckpt)
    value_model.eval()

    st0 = GameStateFacade.initial(seed=0, cold_handle_rule_illegal=True)
    n_stats = int(feat.encode_stats(st0, 0).size)
    print(f"[值网] {args.ckpt} 的 value_state；stats 维度 = {n_stats}")
    print(f"[下标核对] Ant-Game/SDK/utils/features.py:227-228 ⇒ "
          f"stats[{COIN_IDX}] = 我币/300、stats[{COIN_IDX + 1}] = 敌币/300"
          f"（summarize 出 22 维，extras 从 22 起）；"
          f"stats[{LIGHTNING_CD_IDX}] = 我方法闪电剩余冷却/35（features.py:231-232）")

    traj_seeds = tuple(int(x) for x in args.traj_seeds.split(",") if x.strip())
    no_t = collect_trajectory(feat, seeds=traj_seeds, every=args.traj_every)
    log_s = collect_logs(args.tag, args.seed, feat, every=args.every)
    with_t = [s for s in log_s if s["n_my_tower"] > 0]
    no_t_log = [s for s in log_s if s["n_my_tower"] == 0]
    for nm, grp in (("无塔轨迹（只 advance_round）", no_t),
                    ("日志复现有塔局面", with_t)):
        if not grp:
            print(f"[取样] {nm}：0 个"); continue
        c = [s["coins"] for s in grp]
        per_tag = {}
        for s in grp:
            per_tag[s.get("tag", "?")] = per_tag.get(s.get("tag", "?"), 0) + 1
        print(f"[取样] {nm}：{len(grp)} 个局面"
              f"（{', '.join(f'{k}×{v}' for k, v in sorted(per_tag.items()))}）；"
              f"金币中位 {int(np.median(c))}"
              f"（min {min(c)} / max {max(c)}）；"
              f"我塔数 {sorted({s['n_my_tower'] for s in grp})}；"
              f"cd>0 的 {sum(1 for s in grp if s['cd_light'] > 0)}/{len(grp)}")

    r_no = group_stats("A. 无塔（原口径，复核 +0.0274）", no_t, value_model)
    r_tw = group_stats("B. 有塔（日志复现的真实局面）", with_t, value_model)

    if not (r_no and r_tw):
        print("\n⚠️ 缺一组样本，无法比较"); return 1

    # ---- 判据 -------------------------------------------------------------- #
    print(f"\n{'=' * 82}")
    print("C. 判据：有塔 vs 无塔的金币梯度")
    print(f"{'=' * 82}")
    g0 = r_no["g_coin"].mean()
    g1 = r_tw["g_coin"].mean()
    print(f"  无塔（本脚本复现）：均值 {g0:+.6f}  中位 {np.median(r_no['g_coin']):+.6f}"
          f"   vs 报告基线 {BASELINE:+.6f}"
          f"  ⇒ 偏差 {abs(g0 - BASELINE) / BASELINE:+.1%}")
    print(f"  有塔（本脚本新测）：均值 {g1:+.6f}  中位 {np.median(r_tw['g_coin']):+.6f}")
    for lbl, ratio in (("均值比 |有塔|/|无塔|", abs(g1) / abs(g0)),
                       ("中位比 |有塔|/|无塔|",
                        abs(float(np.median(r_tw["g_coin"]))) / abs(float(np.median(r_no["g_coin"])))),
                       ("基线比 |有塔均值|/|+0.0274|", abs(g1) / BASELINE)):
        print(f"  {lbl} = {ratio:.2f}×")
    sign_flip = (g1 > 0) != (g0 > 0)
    ratio_mean = abs(g1) / abs(g0)
    ratio_med = (abs(float(np.median(r_tw["g_coin"])))
                 / abs(float(np.median(r_no["g_coin"]))))
    print(f"\n  判据（报告 §E-P0-1 写死）：差 ≥2× 或符号翻转 ⇒ 「+0.0274 不可外推」成立")
    print(f"    · 均值口径：{ratio_mean:.2f}× ⇒ "
          f"{'成立' if ratio_mean >= 2 else '不成立'}"
          f"{'（且符号翻转）' if sign_flip else ''}")
    print(f"    · 中位口径：{ratio_med:.2f}× ⇒ {'成立' if ratio_med >= 2 else '不成立'}")
    print(f"    · 离散度（这条比比值本身更硬）："
          f"无塔 std {r_no['g_coin'].std():.6f} vs 有塔 std {r_tw['g_coin'].std():.6f}"
          f" ⇒ 比值 {r_tw['g_coin'].std() / max(r_no['g_coin'].std(), 1e-12):.2f}×")
    print(f"      变异系数 CV：无塔 {r_no['g_coin'].std() / abs(g0):.2f}"
          f" vs 有塔 {r_tw['g_coin'].std() / abs(g1):.2f}")
    print(f"      有塔组内 min {r_tw['g_coin'].min():+.6f} / max {r_tw['g_coin'].max():+.6f}"
          f"（跨度 {r_tw['g_coin'].max() - r_tw['g_coin'].min():.6f}）"
          f"；为负 {int((r_tw['g_coin'] < 0).sum())}/{r_tw['n']}"
          f"（无塔组为负 {int((r_no['g_coin'] < 0).sum())}/{r_no['n']}）")

    # ---- 分层：局面已经定了没有（|V| 大 = 胜负已分）------------------------- #
    print(f"\n  分层：按 |V| 切「均势局面」vs「胜负已分局面」（值网的极端区域）")
    print(f"    {'组':<6}{'层':<10}{'n':>3}{'∂V/∂我币 均值':>16}{'中位':>12}{'std':>10}")
    for nm, res in (("无塔", r_no), ("有塔", r_tw)):
        for lbl, mask in (("|V|<0.5", np.abs(res["V"]) < 0.5),
                          ("|V|>=0.5", np.abs(res["V"]) >= 0.5)):
            g = res["g_coin"][mask]
            if len(g) == 0:
                print(f"    {nm:<6}{lbl:<10}{0:>3}{'—':>16}"); continue
            print(f"    {nm:<6}{lbl:<10}{len(g):>3}{g.mean():>+16.6f}"
                  f"{np.median(g):>+12.6f}{g.std():>10.6f}")
    m0 = np.abs(r_no["V"]) < 0.5
    m1 = np.abs(r_tw["V"]) < 0.5
    if m0.sum() and m1.sum():
        a, b = r_no["g_coin"][m0].mean(), r_tw["g_coin"][m1].mean()
        print(f"    ⇒ 只在「均势局面」里比：无塔 {a:+.6f} vs 有塔 {b:+.6f}"
              f" ⇒ 比值 {abs(b) / abs(a):.2f}×"
              f"（这才是「要不要花这笔钱」真正发生的那些局面）")

    # ---- 分层：按「谁领先」切（V<0 = 我在输，V>0 = 我在赢）------------------- #
    print(f"\n  分层：按 V 的正负切「我在输 / 打平 / 我在赢」（V = (我血−敌血)/20）")
    print(f"    {'层（V）':<18}{'n':>4}{'∂V/∂我币 均值':>16}{'中位':>12}{'std':>10}"
          f"{'为负个数':>9}{'每 100 币 HP':>13}")
    buckets = [("V ≤ −0.5（大劣）", lambda v: v <= -0.5),
               ("−0.5 < V < −0.1", lambda v: -0.5 < v < -0.1),
               ("|V| ≤ 0.1（均势）", lambda v: abs(v) <= 0.1),
               ("0.1 < V ≤ 0.5", lambda v: 0.1 < v <= 0.5),
               ("V > 0.5（大优）", lambda v: v > 0.5)]
    for nm, res in (("无塔", r_no), ("有塔", r_tw)):
        print(f"    —— {nm} ——")
        for lbl, fn in buckets:
            mask = np.array([fn(v) for v in res["V"]])
            g = res["g_coin"][mask]
            if len(g) == 0:
                print(f"    {lbl:<18}{0:>4}{'—':>16}"); continue
            print(f"    {lbl:<18}{len(g):>4}{g.mean():>+16.6f}{np.median(g):>+12.6f}"
                  f"{g.std():>10.6f}{int((g < 0).sum()):>9}{(g.mean() / 3) * HP_SCALE:>13.4f}")
    # Pearson 相关：梯度 vs V
    for nm, res in (("无塔", r_no), ("有塔", r_tw)):
        if len(res["V"]) > 2:
            r = float(np.corrcoef(res["V"], res["g_coin"])[0, 1])
            print(f"    corr(∂V/∂我币, V) [{nm}] = {r:+.3f}"
                  f"（负 ⇒ '越在输，钱越值钱'）")

    # 逐局面对照
    print(f"\n  逐局面梯度（无塔 {r_no['n']} 个 | 有塔 {r_tw['n']} 个）")
    print(f"    {'组':<6}{'来源':<20}{'round':>6}{'coins':>6}{'我塔':>4}{'cd':>5}"
          f"{'∂V/∂我币':>12}{'V':>9}")
    for nm, grp, res in (("无塔", no_t, r_no), ("有塔", with_t, r_tw)):
        order = np.argsort(-res["g_coin"])
        for i in order:
            src = f"{grp[i].get('tag', '?')}:{grp[i].get('seed', '?')}"
            print(f"    {nm:<6}{src:<20}{grp[i]['round']:>6}{grp[i]['coins']:>6}"
                  f"{grp[i]['n_my_tower']:>4}{grp[i]['cd_light']:>5.0f}"
                  f"{res['g_coin'][i]:>+12.6f}{res['V'][i]:>+9.4f}")
    print(f"\n  （附注：日志里另有 {len(no_t_log)} 个还没建塔的局面，未计入「有塔」组）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
