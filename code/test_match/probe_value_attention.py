"""价值网**输入敏感性 / 注意力**探针：这个值网到底在看什么。

对应 `docs/value_net_attention_20261031.md`。
延续 P0-1（`probe_value_grad.py`）的机制：同一个 checkpoint、同一种输入构造、同一套
"按 ladder 日志逐 op `apply_operation_list` 复现真实局面"的取样，把 autograd 从"只看金币"
扩展到**全部 42 维 stats + 全部 28 个 board 通道**。

## 三个读数（单位不同，**不能混着比**）

1. **`|∂V/∂x|` 均值**（batch 平均）—— 网络的局部斜率，"这个输入每 +1.0，V 动多少"。
   · stats：per +1.0 该特征（注意很多特征**未归一化**，+1.0 可能微不足道，见 std 列）
   · board：per **单格** +1.0（对 0/1 通道就是"这一格从无到有"）——这是最能对得上决策的单位
2. **实测 1σ 干预 `|ΔV|`**（有限差分，不靠泰勒外推）——
   "把这个输入按它在数据里的标准差推一下，V 实际动多少"。
   · stats：整维 ±σ
   · board：**单格** ±σ（取该通道在数据里最活跃的 8 个格子；整通道 361 格一起推没意义）
3. **决策级干预表**（`ΔV` 全部是实测有限差分，单位统一为 HP）——
   把"多 45/90 币""闪电刚冷却好""多一座塔""多 20 HP 血差"这些**具体决策**各推一次，
   直接比"谁让 V 动得多"。**这张表才是跨特征可比的。**

为什么必须给实测而不是 `|grad|×std`：
  在 `safe_coin` 上 `|grad|×std` 会算出 **2.81 V**（= 56 HP），而 V 自己的 std 只有 **0.63 V**
  ⇒ 线性外推在这些维上完全失效（值网强非线性）。实测有限差分没有这个问题。

用法：
  D:\\anaconda3\\envs\\pytorch-gpu\\python.exe code/test_match/probe_value_attention.py
  ... --tag "TAG:SEED,..." --every 5
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

from probe_value_coin_curve import (HP_SCALE, collect_logs,          # noqa: E402
                                    collect_trajectory)

CKPT = REPO / "checkpoints/posnet_A_k5_m32.pt"

# --------------------------------------------------------------------------- #
# 语义表：严格按 Ant-Game/SDK/utils/features.py 的顺序抄
#   stats[0..21]  = summarize() 的 named dict（features.py:134-157）
#   stats[22..41] = extras（features.py:225-257，extras 从 22 起）
#   board 28 通道 = features.py:161-213
# --------------------------------------------------------------------------- #
STATS_NAME = [
    (0,  "round_ratio",                    "回合进度 /MAX_ROUND"),
    (1,  "hp_delta",                       "血量差 (我−敌)，**未归一化，单位 HP**"),
    (2,  "coin_ratio",                     "金币比 我/敌，**未归一化**"),
    (3,  "safe_coin",                      "超出安全线的金币，**未归一化，单位币**"),
    (4,  "frontline_advantage",            "前线优势 (敌前线距−我前线距)"),
    (5,  "enemy_front_distance",           "敌前锋离我基地距离"),
    (6,  "my_front_distance",              "我前锋离敌基地距离"),
    (7,  "enemy_progress",                 "敌蚁推进度均值"),
    (8,  "my_progress",                    "我蚁推进度均值"),
    (9,  "tower_count",                    "我方塔数"),
    (10, "enemy_tower_count",              "敌方塔数"),
    (11, "tower_level_sum",                "我方塔等级和"),
    (12, "enemy_tower_level_sum",          "敌方塔等级和"),
    (13, "kill_delta",                     "击杀差 (敌死−我死)"),
    (14, "old_delta",                      "老死差 (敌−我)（facade 里是 stub，恒 0）"),
    (15, "tower_spread",                   "塔分布分（负=挤在一起）"),
    (16, "slot_fill_ratio",                "高地格填充率 塔数/格子数"),
    (17, "generation_level",               "出兵等级（越高越快）"),
    (18, "ant_level",                      "蚂蚁血量等级"),
    (19, "hostile_distance",               "最近的敌蚁距离"),
    (20, "base_arc_coverage",              "基地扇区覆盖率（依赖塔位）"),
    (21, "tower_spacing",                  "塔间距分（负=太近）"),
    (22, "coins[me]/300",                  "**我方金币 /300**"),
    (23, "coins[enemy]/300",               "敌方金币 /300"),
    (24, "base_hp[me]/50",                 "我方基地血量 /50"),
    (25, "base_hp[enemy]/50",              "敌方基地血量 /50"),
    (26, "cd_light[me]/35",                "**我方闪电剩余冷却 /35（0=已好）**"),
    (27, "cd_emp[me]/45",                  "我方 EMP 冷却 /45"),
    (28, "cd_deflect[me]/25",              "我方偏转 冷却 /25"),
    (29, "cd_evasion[me]/25",              "我方闪避 冷却 /25"),
    (30, "cd_light[enemy]/35",             "敌方闪电冷却 /35"),
    (31, "cd_emp[enemy]/45",               "敌方 EMP 冷却 /45"),
    (32, "cd_deflect[enemy]/25",           "敌方偏转 冷却 /25"),
    (33, "cd_evasion[enemy]/25",           "敌方闪避 冷却 /25"),
    (34, "superweapon_uses[me]/16",        "我方超武使用次数 /16"),
    (35, "superweapon_uses[enemy]/16",     "敌方超武使用次数 /16"),
    (36, "is_player0",                     "player==0（本数据集恒 1）"),
    (37, "is_player1",                     "player==1（本数据集恒 0）"),
    (38, "to_play_0",                      "context.to_play==0"),
    (39, "to_play_1",                      "context.to_play==1"),
    (40, "settles_after_action",           "本动作后结算"),
    (41, "opponent_already_acted",         "对手已行动"),
]

BOARD_NAME = [
    (0,  "terrain: PATH",                  "路径"),
    (1,  "terrain: P0_HIGHLAND",           "P0 高地（player=0 时 1.0，否则 0.5）"),
    (2,  "terrain: P1_HIGHLAND",           "P1 高地"),
    (3,  "terrain: BARRIER",               "障碍"),
    (4,  "tower[me]: 位置",                "**我方塔所在格 = 1.0**"),
    (5,  "tower[enemy]: 位置",             "敌方塔所在格 = 1.0"),
    (6,  "tower: level/2",                 "塔等级 /2（不分归属）"),
    (7,  "tower: attack_range/6",          "塔射程 /6"),
    (8,  "tower: cooldown/6",              "塔当前开火冷却 /6"),
    (9,  "tower: damage/50",               "塔伤害 /50"),
    (10, "ant[me]: hp/max_hp",             "我方蚁血比（同格可叠加）"),
    (11, "ant[enemy]: hp/max_hp",          "敌方蚁血比"),
    (12, "ant: level/2",                   "蚁等级 /2"),
    (13, "ant: age/AGE_LIMIT",             "蚁年龄 /上限"),
    (14, "pheromone[me]",                  "我方信息素"),
    (15, "pheromone[enemy]",               "敌方信息素"),
    (16, "ant: frozen",                    "被冰冻的蚁"),
    (17, "ant: behavior=RANDOM",           "行为=随机"),
    (18, "ant: behavior=BEWITCHED",        "行为=被魅惑"),
    (19, "ant: behavior=CONTROL_FREE",     "行为=脱离控制"),
    (20, "effect: LIGHTNING[me]",          "我方闪电风暴效果覆盖"),
    (21, "effect: LIGHTNING[enemy]",       "敌方闪电风暴效果覆盖"),
    (22, "effect: EMP[me]",                "我方 EMP 效果"),
    (23, "effect: EMP[enemy]",             "敌方 EMP 效果"),
    (24, "effect: DEFLECTOR[me]",          "我方偏转效果"),
    (25, "effect: DEFLECTOR[enemy]",       "敌方偏转效果"),
    (26, "effect: EVASION[me]",            "我方闪避效果"),
    (27, "effect: EVASION[enemy]",         "敌方闪避效果"),
]

STATS_GROUP = {
    "血量/比分": [1, 13, 14, 24, 25],
    "金币": [2, 3, 22, 23],
    "塔（stats 聚合）": [9, 10, 11, 12, 15, 16, 20, 21],
    "超武冷却": [26, 27, 28, 29, 30, 31, 32, 33],
    "超武使用次数": [34, 35],
    "前线/推进": [4, 5, 6, 7, 8, 19],
    "基地/兵力等级": [17, 18],
    "回合进度": [0],
    "上下文/常数": [36, 37, 38, 39, 40, 41],
}
BOARD_GROUP = {
    "地形": [0, 1, 2, 3],
    "塔": [4, 5, 6, 7, 8, 9],
    "蚁": [10, 11, 12, 13, 16, 17, 18, 19],
    "信息素": [14, 15],
    "超武效果": [20, 21, 22, 23, 24, 25, 26, 27],
}


# --------------------------------------------------------------------------- #
# 前向 / 反向
# --------------------------------------------------------------------------- #
def _fwd(stats: np.ndarray, board: np.ndarray, value_model, chunk: int = 256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(stats), chunk):
            st = torch.from_numpy(np.ascontiguousarray(stats[i:i + chunk])).float()
            bd = torch.from_numpy(np.ascontiguousarray(board[i:i + chunk])).float()
            o = value_model(bd, st)
            v = o["value"] if isinstance(o, dict) else o
            out.append(v.numpy().ravel())
    return np.concatenate(out)


def grads_for(grp: list, value_model, chunk: int = 64):
    """返回 (G_stats(n,42), G_board(n,28) —— 通道内 |grad| 的空间均值, V(n))。"""
    Gs, Gb, Vs = [], [], []
    for i in range(0, len(grp), chunk):
        part = grp[i:i + chunk]
        with torch.enable_grad():
            st = torch.tensor(np.stack([s["stats"] for s in part]).copy())
            bd = torch.tensor(np.stack([s["board"] for s in part]).copy())
            st.requires_grad_(True)
            bd.requires_grad_(True)
            out = value_model(bd, st)
            val = out["value"] if isinstance(out, dict) else out
            gs, gb = torch.autograd.grad(val.sum(), [st, bd])
        Gs.append(gs.detach().numpy())
        Gb.append(np.abs(gb.detach().numpy()).mean(axis=(2, 3)))
        Vs.append(val.detach().numpy().ravel())
    return np.concatenate(Gs), np.concatenate(Gb), np.concatenate(Vs)


def actual_1sigma_stats(grp: list, value_model):
    """42 维 stats：整维 ±1σ 的**实测** |ΔV| 均值。"""
    S = np.stack([s["stats"] for s in grp])
    B = np.stack([s["board"] for s in grp])
    V0 = _fwd(S, B, value_model)
    sd = S.std(axis=0)
    act = np.zeros(42)
    for i in range(42):
        if sd[i] <= 1e-12:
            continue
        for sgn in (+1.0, -1.0):
            S2 = S.copy()
            S2[:, i] += sgn * sd[i]
            act[i] += np.abs(_fwd(S2, B, value_model) - V0).mean() / 2.0
    return act


def actual_1sigma_board(grp: list, value_model, k_cells: int = 8):
    """28 个 board 通道：**单格** ±1σ 的实测 |ΔV| 均值（取每通道在数据里最活跃的 k 个格子）。

    整通道 361 格一起推 +σ 是个巨大的全局干预（会把 V 推出分布），所以按单格来。
    """
    S = np.stack([s["stats"] for s in grp])
    B = np.stack([s["board"] for s in grp])
    n = len(grp)
    V0 = _fwd(S, B, value_model)
    act = np.zeros(28)
    cell_std = B.reshape(n, 28, -1).std(axis=(0, 2))
    for c in range(28):
        sd = cell_std[c]
        if sd <= 1e-12:
            continue
        flat = B[:, c].reshape(n, -1)
        # 该通道在数据里波动最大的格子
        cells = np.argsort(-flat.std(axis=0))[:k_cells]
        acc, cnt = 0.0, 0
        for cell in cells:
            xs = np.arange(n)
            for sgn in (+1.0, -1.0):
                B2 = B.copy()
                B2[xs, c] += 0.0
                bf = B2[:, c].reshape(n, -1)
                bf[:, cell] += sgn * sd
                acc += np.abs(_fwd(S, B2, value_model) - V0).mean()
                cnt += 1
        act[c] = acc / max(cnt, 1)
    return act


def decision_table(grp: list, value_model):
    """**决策级**干预：全部是实测有限差分，输出统一为 HP（V × 20）。"""
    S = np.stack([s["stats"] for s in grp])
    B = np.stack([s["board"] for s in grp])
    n = len(grp)
    V0 = _fwd(S, B, value_model)
    rows = []

    def rec(name, V2, idx=None, extra=""):
        d = V2 - V0 if idx is None else V2[idx] - V0[idx]
        rows.append((name, float(d.mean()), float(np.median(d)), int(len(d)), extra))

    # 1) 金币 +45 / +90 / +300（stats[22] 是 coins/300）
    for coins in (45, 90, 300):
        S2 = S.copy()
        S2[:, 22] += coins / 300.0
        rec(f"我方金币 +{coins}", _fwd(S2, B, value_model))
    # 2) 敌方金币 +90
    S2 = S.copy(); S2[:, 23] += 90 / 300.0
    rec("敌方金币 +90", _fwd(S2, B, value_model))
    # 3) 闪电冷却 → 0（只在该维 >0 的局面里统计）
    m = S[:, 26] > 1e-9
    if m.sum():
        S2 = S.copy(); S2[:, 26] = 0.0
        rec("我方闪电冷却 → 立即好", _fwd(S2, B, value_model), idx=np.where(m)[0])
    # 4) 一棵塔：level_sum +1 且 tower_count +1
    S2 = S.copy(); S2[:, 9] += 1.0; S2[:, 11] += 1.0
    rec("塔数+1 且 塔等级和+1（stats 侧）", _fwd(S2, B, value_model))
    # 5) 血差 +20 HP（stats[1] 是未归一化 HP）
    S2 = S.copy(); S2[:, 1] += 20.0
    rec("血差 +20 HP（stats[1]）", _fwd(S2, B, value_model))
    # 6) 敌基地血 −20（= 我多打掉 20 血，走 /50 的归一化维）
    S2 = S.copy(); S2[:, 25] -= 20.0 / 50.0
    rec("敌基地血 −20（stats[25]）", _fwd(S2, B, value_model))
    # 7) **真的放一座塔**：在空的己方高地格里写满 5 个塔通道
    tower_cells = (B[:, 4] > 0.5)
    fp = {}
    for c in (6, 7, 8, 9):
        vals = B[:, c][tower_cells]
        fp[c] = float(np.median(vals)) if vals.size else 0.0
    empty = (B[:, 1] > 0.9) & (B[:, 4] < 0.5)      # player=0 的高地且没塔
    Srows, Brows, owner = [], [], []
    for i in range(n):
        cells = np.where(empty[i].ravel())[0]
        if cells.size == 0:
            continue
        pick = cells[np.linspace(0, cells.size - 1, min(12, cells.size)).astype(int)]
        for cell in pick:
            b2 = B[i].copy()
            b2[4].ravel()[cell] = 1.0
            for c, v in fp.items():
                b2[c].ravel()[cell] = v
            Brows.append(b2)
            Srows.append(S[i])
            owner.append(i)
    best = None
    if Brows:
        V1 = _fwd(np.stack(Srows), np.stack(Brows), value_model)
        owner = np.array(owner)
        d = V1 - V0[owner]
        rows.append((f"放一座 1 级塔·**平均格子**（指纹 level={fp.get(6, 0) * 2:.0f},"
                     f" range={fp.get(7, 0) * 6:.0f}, dmg={fp.get(9, 0) * 50:.0f}）",
                     float(d.mean()), float(np.median(d)), len(d), "每局面 12 个空格"))
        # 每局面挑最好的格子 ⇒ "如果位置挑对了，建塔值多少"
        best = np.zeros(n)
        seen = np.zeros(n, dtype=bool)
        for i in range(n):
            m = owner == i
            if m.any():
                best[i] = d[m].max()
                seen[i] = True
        rows.append(("放一座 1 级塔·**每局面最好的格子**",
                     float(best[seen].mean()), float(np.median(best[seen])),
                     int(seen.sum()), "同一批格子里取 max"))
        rows.append(("放一座 1 级塔·**每局面最差的格子**",
                     float(np.array([d[owner == i].min() for i in range(n) if (owner == i).any()]).mean()),
                     float(np.median([d[owner == i].min() for i in range(n) if (owner == i).any()])),
                     int(seen.sum()), "同一批格子里取 min"))
    # 8) 我方基地血 −20
    S2 = S.copy(); S2[:, 24] -= 20.0 / 50.0
    rec("我基地血 −20（stats[24]）", _fwd(S2, B, value_model))

    # 9) **一致性检查**：同一件事（我方金币 +90）走三条不同的编码
    S2 = S.copy(); S2[:, 3] += 90.0                       # 只动 safe_coin（原始币数）
    rec("我方金币 +90 · 只走 stats[3] safe_coin", _fwd(S2, B, value_model))
    S2 = S.copy(); S2[:, 22] += 90 / 300.0; S2[:, 3] += 90.0
    rec("我方金币 +90 · stats[22]+stats[3] 自洽", _fwd(S2, B, value_model))
    # 我方血量 −20（= 血差 −20）走两条路
    S2 = S.copy(); S2[:, 1] -= 20.0
    rec("血差 −20 HP · 只走 stats[1]", _fwd(S2, B, value_model))
    S2 = S.copy(); S2[:, 24] -= 20.0 / 50.0
    rec("我方基地血 −20 · 只走 stats[24]（= 等价的血差 −20）", _fwd(S2, B, value_model))
    print(f"\n  [决策级干预] 全部实测有限差分（不是泰勒外推），ΔV 换算成 HP（×20）")
    print(f"    {'干预':<46}{'ΔV 均值':>11}{'×20[HP]':>10}{'ΔV 中位':>11}{'n':>7}  备注")
    for name, mu, med, cnt, extra in rows:
        print(f"    {name:<46}{mu:>+11.5f}{mu * HP_SCALE:>+10.4f}{med:>+11.5f}{cnt:>7}  {extra}")
    return rows


# --------------------------------------------------------------------------- #
# 打印
# --------------------------------------------------------------------------- #
def rank_stats(grp, G, act, title, topn=42):
    S = np.stack([s["stats"] for s in grp])
    mag = np.abs(G).mean(axis=0)
    std = S.std(axis=0)
    order = np.argsort(-mag)
    tot = float(act.sum())
    print(f"\n  {title}")
    print(f"    {'排名':>4} {'idx':>4} {'特征':<24} {'|∂V/∂x|均值':>12} {'std':>10}"
          f" {'实测1σ|ΔV|':>11} {'×20[HP]':>9} {'占比':>7}  语义")
    for r, i in enumerate(order[:topn], 1):
        print(f"    {r:>4} {i:>4} {STATS_NAME[i][1]:<24} {mag[i]:>12.6f} {std[i]:>10.4f}"
              f" {act[i]:>11.6f} {act[i] * HP_SCALE:>9.4f}"
              f" {100 * act[i] / tot if tot else 0:>6.1f}%  {STATS_NAME[i][2]}")
    order_a = np.argsort(-act)
    print(f"    —— 按**实测 1σ 干预**重排（top 12）——")
    for r, i in enumerate(order_a[:12], 1):
        print(f"    {r:>4} {i:>4} {STATS_NAME[i][1]:<24} {mag[i]:>12.6f} {std[i]:>10.4f}"
              f" {act[i]:>11.6f} {act[i] * HP_SCALE:>9.4f}"
              f" {100 * act[i] / tot if tot else 0:>6.1f}%  {STATS_NAME[i][2]}")
    return mag, std, act


def rank_board(grp, Gb, act, title, topn=28):
    B = np.stack([s["board"] for s in grp])
    mag = Gb.mean(axis=0)
    std = B.reshape(len(grp), 28, -1).std(axis=(0, 2))
    order = np.argsort(-mag)
    tot = float(act.sum())
    print(f"\n  {title}")
    print(f"    {'排名':>4} {'ch':>3} {'通道':<26} {'单格|∂V/∂x|':>12} {'格子std':>10}"
          f" {'单格1σ|ΔV|':>11} {'×20[HP]':>9} {'占比':>7}  语义")
    for r, i in enumerate(order[:topn], 1):
        print(f"    {r:>4} {i:>3} {BOARD_NAME[i][1]:<26} {mag[i]:>12.6f} {std[i]:>10.4f}"
              f" {act[i]:>11.6f} {act[i] * HP_SCALE:>9.4f}"
              f" {100 * act[i] / tot if tot else 0:>6.1f}%  {BOARD_NAME[i][2]}")
    order_a = np.argsort(-act)
    print(f"    —— 按**单格实测 1σ 干预**重排（top 10）——")
    for r, i in enumerate(order_a[:10], 1):
        print(f"    {r:>4} {i:>3} {BOARD_NAME[i][1]:<26} {mag[i]:>12.6f} {std[i]:>10.4f}"
              f" {act[i]:>11.6f} {act[i] * HP_SCALE:>9.4f}"
              f" {100 * act[i] / tot if tot else 0:>6.1f}%  {BOARD_NAME[i][2]}")
    return mag, std, act


def group_summary(mag_s, act_s, mag_b, act_b):
    ts, tb_ = float(act_s.sum()), float(act_b.sum())
    print(f"\n  [stats 分组] 占比 = 组内 1σ 实测 |ΔV| 之和 / **stats 内**之和（不与 board 混算）")
    print(f"    {'组':<20}{'维数':>4}{'Σ|grad|':>11}{'Σ实测1σ|ΔV|':>14}{'组内占比':>9}")
    for name, idxs in STATS_GROUP.items():
        print(f"    {name:<20}{len(idxs):>4}{mag_s[idxs].sum():>11.5f}"
              f"{act_s[idxs].sum():>14.6f}{100 * act_s[idxs].sum() / ts:>8.2f}%")
    print(f"\n  [board 分组] 占比 = 组内 / **board 内**之和（不与 stats 混算）")
    print(f"    {'组':<20}{'通道数':>5}{'Σ单格|grad|':>13}{'Σ单格1σ|ΔV|':>14}{'组内占比':>9}")
    for name, idxs in BOARD_GROUP.items():
        print(f"    {name:<20}{len(idxs):>5}{mag_b[idxs].sum():>13.5f}"
              f"{act_b[idxs].sum():>14.6f}{100 * act_b[idxs].sum() / tb_:>8.2f}%")
    print(f"\n  ⚠️ 两组的「占比」不能互相比较：stats 的干预是「整维 ±σ」，"
          f"board 的是「单格 ±σ」，而 board 有 361 倍多的输入位。"
          f"跨特征可比的只有 §D 的决策级干预表。")


def linear_probe(grp, V):
    S = np.stack([s["stats"] for s in grp])
    keep = [i for i in range(S.shape[1]) if S[:, i].std() > 1e-9]
    X = S[:, keep]
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = V - V.mean()
    lam = 1.0
    w = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    pred = X @ w
    ss = float((y ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else float("nan")
    print(f"\n  [线性探针] V ~ 42 维 stats（标准化 + 岭回归 λ={lam}，保留 {len(keep)} 个非常数维）")
    print(f"    R² = {r2:.4f}（{len(grp)} 个局面；V 的 std = {V.std():.4f} = {V.std() * HP_SCALE:.2f} HP）")
    print(f"    {'排名':>4} {'idx':>4} {'特征':<24} {'标准化系数':>12}  语义")
    for r, j in enumerate(np.argsort(-np.abs(w)), 1):
        i = keep[j]
        print(f"    {r:>4} {i:>4} {STATS_NAME[i][1]:<24} {w[j]:>+12.4f}  {STATS_NAME[i][2]}")
    return r2


def by_score_layer(grp, G, V, topn=8):
    print(f"\n  [按比分分层] 每层 top-{topn}（按 |∂V/∂x| 均值）")
    layers = [("大劣 V≤−0.5", V <= -0.5),
              ("均势 |V|≤0.1", np.abs(V) <= 0.1),
              ("领先 V≥0.1", V >= 0.1)]
    for lbl, m in layers:
        n = int(m.sum())
        print(f"    ── {lbl}（n={n}）")
        if n == 0:
            print("      （无样本）"); continue
        mag = np.abs(G[m]).mean(axis=0)
        for i in np.argsort(-mag)[:topn]:
            print(f"       {STATS_NAME[i][1]:<24} |grad|={mag[i]:.6f}"
                  f"（22 金币={mag[22]:.6f}、26 冷却={mag[26]:.6f}、9 塔数={mag[9]:.6f}）"
                  if i == np.argsort(-mag)[0] else
                  f"       {STATS_NAME[i][1]:<24} |grad|={mag[i]:.6f}")
    print(f"    ── 关键特征在三层里的 |∂V/∂x|")
    print(f"      {'特征':<24}" + "".join(f"{lbl.split(' ')[0]:>14}" for lbl, _ in layers))
    for i in (1, 24, 25, 22, 23, 26, 9, 11):
        row = "".join(f"{np.abs(G[m, i]).mean():>14.5f}" if m.sum() else f"{'—':>14}"
                      for _, m in layers)
        print(f"      {STATS_NAME[i][1]:<24}{row}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--tag", default=",".join([
        "UNIFIED_SPLIT_VERBOSE:8", "UNIFIED_CAND_FULL:8",
        "UNIFIED_AB_A:10", "UNIFIED_AB_A:11",
        "UNIFIED_AB_A:10r", "UNIFIED_AB_A:11r"]),
        help="有塔局面的 ladder 日志，逗号分隔，可带 seed；自动去重")
    ap.add_argument("--seed", default="8")
    ap.add_argument("--every", type=int, default=5, help="每几轮取一个局面")
    ap.add_argument("--no-traj", action="store_true", help="不并入无塔轨迹样本（对照用）")
    ap.add_argument("--traj-every", type=int, default=10)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    feat = FeatureExtractor(max_actions=96)
    _cls, _pos, value_model = load_three_models(args.ckpt)
    value_model.eval()
    st0 = GameStateFacade.initial(seed=0, cold_handle_rule_illegal=True)
    print(f"[值网] {args.ckpt} 的 value_state；stats 维度 = {int(feat.encode_stats(st0, 0).size)}")
    print("[下标] 严格按 Ant-Game/SDK/utils/features.py："
          "stats[0..21]=summarize()（:134-157）、stats[22..41]=extras（:225-257）、"
          "board 28 通道（:161-213）")

    log_s = collect_logs(args.tag, args.seed, feat, every=args.every)
    traj = [] if args.no_traj else collect_trajectory(feat, every=args.traj_every)
    if not log_s:
        print("⚠️ 没采到日志样本"); return 1
    with_t = [s for s in log_s if s["n_my_tower"] > 0]
    print(f"\n[样本] 日志复现 {len(log_s)} 个局面（有己方塔 {len(with_t)}）"
          f" + 无塔轨迹 {len(traj)} 个 = {len(log_s) + len(traj)}")
    for nm, g in (("日志（有塔）", with_t), ("日志（全部）", log_s), ("+无塔轨迹", log_s + traj)):
        if not g:
            continue
        coins = np.array([s["coins"] for s in g])
        cds = np.array([s["cd_light"] for s in g])
        nt = np.array([s["n_my_tower"] for s in g])
        rnd = np.array([s["round"] for s in g])
        dist = {int(k): int(v) for k, v in zip(*np.unique(nt, return_counts=True))}
        print(f"  {nm}（n={len(g)}）：回合 {rnd.min()}–{rnd.max()}（中位 {int(np.median(rnd))}）；"
              f"金币 中位 {int(np.median(coins))}（{coins.min()}–{coins.max()}）；"
              f"塔数 {dist}；闪电 cd>0 的 {int((cds > 0).sum())}")

    main_grp = with_t
    print(f"\n[主分析组] 有塔真实对局 {len(main_grp)} 个局面"
          f"（无塔轨迹只作 §F 的对照：它金币最高到 "
          f"{max([s['coins'] for s in traj]) if traj else 0}，超出 stats[22] 常见范围）")

    G, Gb, V = grads_for(main_grp, value_model)
    print(f"\n[前向] V 均值 {V.mean():+.4f}、std {V.std():.4f}、"
          f"范围 {V.min():+.4f}..{V.max():+.4f}（跨度 {(V.max() - V.min()) * HP_SCALE:.2f} HP）")
    print(f"[实测 1σ 干预] 42 维 stats（整维 ±σ）+ 28 通道（单格 ±σ，每通道 8 格）"
          f"× {len(main_grp)} 个局面，batch 前向…")
    act_s = actual_1sigma_stats(main_grp, value_model)
    act_b = actual_1sigma_board(main_grp, value_model)
    lin_max = float(np.max(np.abs(G).mean(axis=0) * np.stack([s["stats"] for s in main_grp]).std(axis=0)))
    print(f"        完成。对照：`|grad|×std` 线性外推最高会给出 {lin_max:.2f} V，"
          f"而 V 自己的 std 只有 {V.std():.2f} V ⇒ 外推在这些维上完全失效，必须用实测。")

    print("\n" + "=" * 122)
    print("A. stats(42) 注意力排行榜 —— 按 |∂V/∂x| 均值降序")
    print("=" * 122)
    mag_s, std_s, _ = rank_stats(main_grp, G, act_s, "stats[0..41]")

    print("\n" + "=" * 122)
    print("B. board(28 通道) 注意力排行榜 —— 按**单格** |∂V/∂x| 空间均值降序")
    print("=" * 122)
    mag_b, std_b, _ = rank_board(main_grp, Gb, act_b, "board ch0..27")

    print("\n" + "=" * 122)
    print("C. 分组汇总")
    print("=" * 122)
    group_summary(mag_s, act_s, mag_b, act_b)

    print("\n" + "=" * 122)
    print("D. 决策级干预：跨特征唯一可比的表（全部实测有限差分，单位 HP）")
    print("=" * 122)
    drows = decision_table(main_grp, value_model)

    print("\n" + "=" * 122)
    print("E. 线性探针 + 按比分分层")
    print("=" * 122)
    linear_probe(main_grp, V)
    by_score_layer(main_grp, G, V)

    if traj:
        print("\n" + "=" * 122)
        print(f"F. 对照：并上 {len(traj)} 个无塔轨迹局面后，|grad| 排序稳不稳")
        print("=" * 122)
        G2, Gb2, _ = grads_for(log_s + traj, value_model)
        mag_s2 = np.abs(G2).mean(axis=0)
        mag_b2 = Gb2.mean(axis=0)
        o1 = list(np.argsort(-mag_s)[:8])
        o2 = list(np.argsort(-mag_s2)[:8])
        print(f"    有塔组 top8：{[STATS_NAME[i][1] for i in o1]}")
        print(f"    全部   top8：{[STATS_NAME[i][1] for i in o2]}")
        rc = float(np.corrcoef(np.argsort(np.argsort(-mag_s)),
                               np.argsort(np.argsort(-mag_s2)))[0, 1])
        print(f"    42 维 Spearman 排序相关 = {rc:.3f}")
        print(f"    金币 stats[22]：|grad| 有塔 {mag_s[22]:.5f} vs 全部 {mag_s2[22]:.5f}")
        print(f"    board top5 有塔：{[BOARD_NAME[i][1] for i in np.argsort(-mag_b)[:5]]}")
        print(f"    board top5 全部：{[BOARD_NAME[i][1] for i in np.argsort(-mag_b2)[:5]]}")

    # ---- G. 判据 ---------------------------------------------------------- #
    print("\n" + "=" * 122)
    print("G. 判据")
    print("=" * 122)
    print("  1) stats 按 |grad| 最敏感 6 维：" + ", ".join(
        f"{STATS_NAME[i][1]}({mag_s[i]:.5f})" for i in np.argsort(-mag_s)[:6]))
    print("     按实测 1σ 干预最敏感 6 维：" + ", ".join(
        f"{STATS_NAME[i][1]}({act_s[i]:.4f}V={act_s[i] * HP_SCALE:.2f}HP)"
        for i in np.argsort(-act_s)[:6]))
    print(" 2) board 按单格 |grad| 最敏感 5 通道：" + ", ".join(
        f"{BOARD_NAME[i][1]}({mag_b[i]:.5f})" for i in np.argsort(-mag_b)[:5]))
    print("     按单格实测 1σ 干预最敏感 5 通道：" + ", ".join(
        f"{BOARD_NAME[i][1]}({act_b[i]:.4f}V={act_b[i] * HP_SCALE:.2f}HP)"
        for i in np.argsort(-act_b)[:5]))
    print(f" 3) 血量差 stats[1]：|grad|={mag_s[1]:.6f}（42 维里排 "
          f"{int((mag_s > mag_s[1]).sum()) + 1}），但实测 1σ |ΔV| = {act_s[1]:.4f} V"
          f"（{act_s[1] * HP_SCALE:.2f} HP，占 stats 内 {100 * act_s[1] / act_s.sum():.1f}%）")
    for nm, i in (("金币 stats[22]", 22), ("敌金币 stats[23]", 23),
                  ("闪电冷却 stats[26]", 26), ("塔数 stats[9]", 9),
                  ("塔等级和 stats[11]", 11), ("塔等级和(敌) stats[12]", 12)):
        print(f"    {nm}：|grad|={mag_s[i]:.6f}（排 {int((mag_s > mag_s[i]).sum()) + 1}/42），"
              f"实测 1σ |ΔV|={act_s[i]:.5f} V = {act_s[i] * HP_SCALE:.4f} HP"
              f"（stats 内 {100 * act_s[i] / act_s.sum():.1f}%）")
    tb = [4, 5, 6, 7, 8, 9]
    print(f" 4) board 塔通道 ch4-9：单格 |grad| " + ", ".join(f"ch{i}={mag_b[i]:.5f}" for i in tb))
    print(f"    单格实测 1σ |ΔV| " + ", ".join(f"ch{i}={act_b[i]:.5f}" for i in tb)
          + f"（board 内占比 {100 * act_b[tb].sum() / act_b.sum():.1f}%）")
    print(f" 5) 恒定维（std=0，|grad| 非零但实际无影响）：stats 有 "
          f"{int((np.stack([s['stats'] for s in main_grp]).std(axis=0) <= 1e-9).sum())}/42，"
          f"board 有 {int((std_b <= 1e-9).sum())}/28")
    np.savez(REPO / "match_results" / "probe_attention.npz",
             mag_stats=mag_s, std_stats=std_s, act_stats=act_s,
             mag_board=mag_b, std_board=std_b, act_board=act_b, V=V)
    print(f"\n  [落盘] match_results/probe_attention.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
