"""诊断：**类轴上"价值先验"与"策略头"分歧有多大？**（候选 C 的前提检验）

背景：A1 配置（`pos-only`）把**类钉在策略 argmax**，实测我方**建塔 = 0**（几乎只出闪电=类 17）。
候选 C 的想法是"用**我们自己的** 1-ply 价值先验给类头造训练目标"。**在写训练代码之前**，
先花几分钟用**已有数据**回答：价值先验真的会选别的类吗？如果它和策略头几乎总是一致，
那么 C 从一开始就没有可学的东西。

做法（**不需要引擎、不需要跑对局**）：
  1. 读 `vp_A_k5_m32.npz`（每行 = (决策, head, 类) 的候选格 + 逐格 `adv`）；
  2. 按"相邻行 board/player 相同"把行**分组回决策**（实测 12.1 行/决策）；
  3. 对每个决策做**一次**策略网前向 → 三个 head 的 24 类 logits；
  4. 在**同一组被评估的类**上比较两件事（都是**条件**分布 ⇒ 不受"只评估了 5 个类"影响）：
       * 价值先验：把每类的 `max(adv)` 在类间 z-score → softmax → top-1；
       * 策略头  ：把这些类的 logits 做 softmax → top-1；
  5. 统计一致率、双方 top-1 的类直方图、以及"价值想改、策略没改"的比例。

类轴语义（`code/my_ai/decoder.py:34-71`）：**0–15** = 建塔/升级到某塔型（0 = 建 BASIC）、
**16** = 降级、**17–20** = 超武（17 = 闪电）、**21–22** = 基地升级、**23** = HOLD。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/diag_class_axis_vs_value.py \
      [--npz <path>] [--ckpt <three-net.pt>] [--max-decisions 3000] [--batch 256]
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Ant-Game"))
sys.path.insert(0, str(REPO / "code"))

DEFAULT_NPZ = REPO / "training_history" / "vprior" / "vp_A_k5_m32.npz"
DEFAULT_CKPT = REPO / "training_history" / "vprior" / "posnet_A_k5_m32.pt"

CLASS_NAME = {}
for _c in range(16):
    CLASS_NAME[_c] = f"tower_c{_c}"
CLASS_NAME[16] = "downgrade"
CLASS_NAME.update({17: "LIGHTNING", 18: "EMP", 19: "DEFLECTOR", 20: "EVASION",
                   21: "base_gen", 22: "base_ant", 23: "HOLD"})


def group_decisions(stats, player, head, cls, adv, cnt) -> list[dict]:
    """把行分组回决策：相邻行 (stats, player) 相同视为同一决策。

    ⚠️ 用 `stats`（42 维）+ `player` 当分组键，**不用 `board`**：board 是 350k×28×19×19，
    逐行比较要 3.5G 次比较（又慢又费内存），而 `stats` 只有 29 MB。
    用 stats 的代价是"统计量完全相同但局面不同"的两个决策**可能被并成一组**（极罕见），
    影响只是让该组多几行、统计上无关紧要。
    """
    key = np.concatenate([stats.astype(np.float32), player[:, None].astype(np.float32)], axis=1)
    same = np.all(key[1:] == key[:-1], axis=1)
    n = len(player)
    bounds = [0] + (np.nonzero(~same)[0] + 1).tolist() + [n]
    out = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        rows = [{"head": int(head[j]), "cls": int(cls[j]), "adv": adv[j], "cnt": int(cnt[j])}
                for j in range(a, b)]
        out.append({"idx": a, "rows": rows})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--max-decisions", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--min-classes", type=int, default=2,
                    help="某个 head 至少要有几个被评估的类才计入（默认 2）")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt

    z = np.load(args.npz, allow_pickle=True)
    # ⚠️ 一次性读进内存：npz 的每次 z[k] 访问都会**重新解压整个数组**（实测把内存打爆）
    board_a = z["board"]; stats_a = z["stats"]; player_a = z["player"]
    head_a = z["head"]; cls_a = z["cls"]; adv_a = z["adv"]; cnt_a = z["cnt"]
    decs = group_decisions(stats_a, player_a, head_a, cls_a, adv_a, cnt_a)
    print(f"npz={Path(args.npz).name}  行={len(player_a)}  决策={len(decs)}  "
          f"(平均 {len(player_a)/max(len(decs),1):.1f} 行/决策)")
    step = max(1, len(decs) // args.max_decisions)
    picked = decs[::step]
    print(f"抽样 {len(picked)} 个决策（每 {step} 个取 1）\n")

    feat = FeatureExtractor(max_actions=96)
    model, _net_fn = make_net_fn_from_ckpt(args.ckpt, feat)

    # ---- 组装 (board, stats, player) 批次 ----
    boards, statss, players = [], [], []
    for d in picked:
        i = d["idx"]
        boards.append(board_a[i]); statss.append(stats_a[i]); players.append(player_a[i])
    boards = np.stack(boards).astype(np.float32)
    statss = np.stack(statss).astype(np.float32)

    logits_all = []
    model.eval()   # ⚠️ 必须 eval：类头带 dropout(0.5)，train 模式下 logits 是随机的
    with torch.no_grad():
        for s in range(0, len(boards), args.batch):
            b = torch.from_numpy(boards[s:s + args.batch])
            st = torch.from_numpy(statss[s:s + args.batch])
            o = model(b, st)
            if not isinstance(o, dict):
                raise RuntimeError(f"模型返回 {type(o)}，预期 dict（键形如 head1_logits）")
            # 网络输出键是 head1_logits/head2_logits/head3_logits（1-based），
            # 而 npz 里的 `head` 是 0/1/2 ⇒ 这里按 h+1 取
            arr = np.stack([np.asarray(o[f"head{h+1}_logits"], dtype=np.float64) for h in range(3)],
                           axis=1)                                        # (B,3,C)
            logits_all.append(arr)
    logits_all = np.concatenate(logits_all, axis=0)          # (N,3,C)
    print(f"策略网前向完成：logits shape={logits_all.shape}\n")

    n_pair = 0
    agree = 0
    val_top_hist = collections.Counter()
    pol_top_hist = collections.Counter()
    val_nonlight_but_pol_light = 0
    adv_gain = []            # 价值最优类 vs 策略所选类的 adv 差（z 后）
    per_head = collections.defaultdict(lambda: [0, 0])

    for di, d in enumerate(picked):
        by_head = collections.defaultdict(list)
        for r in d["rows"]:
            by_head[r["head"]].append(r)
        for h, rows in by_head.items():
            if h >= 3 or len(rows) < args.min_classes:
                continue
            cls = np.array([r["cls"] for r in rows])
            amax = np.array([float(np.max(r["adv"])) for r in rows])   # 该类里最好的格
            zz = (amax - amax.mean()) / (amax.std() + 1e-9)
            v_top = int(np.argmax(zz))
            p_logits = logits_all[di, h, cls]
            pol_top = int(np.argmax(p_logits))
            n_pair += 1
            per_head[h][1] += 1
            val_top_hist[CLASS_NAME.get(int(cls[v_top]), str(int(cls[v_top])))] += 1
            pol_top_hist[CLASS_NAME.get(int(cls[pol_top]), str(int(cls[pol_top])))] += 1
            if v_top == pol_top:
                agree += 1
                per_head[h][0] += 1
            else:
                adv_gain.append(float(zz[v_top] - zz[pol_top]))
                if cls[pol_top] == 17 and cls[v_top] != 17:
                    val_nonlight_but_pol_light += 1

    print(f"可比 (决策, head) 对 = {n_pair}")
    print(f"价值 top-1 == 策略 top-1 的比例 = {agree / max(n_pair,1):.1%}")
    print(f"（其中 head0/1/2 各自：{ {h: f'{v[0]}/{v[1]}' for h, v in sorted(per_head.items())} }）")
    if adv_gain:
        ag = np.array(adv_gain)
        qs = np.percentile(ag, [10, 25, 50, 75, 90])
        print(f"分歧时价值最优比策略所选好的 z 幅度："
              f"p10={qs[0]:.2f} p25={qs[1]:.2f} 中位={qs[2]:.2f} p75={qs[3]:.2f} p90={qs[4]:.2f}")
        for thr in (0.25, 0.5, 1.0, 2.0):
            print(f"    其中 z 幅度 > {thr:>4}: {int((ag > thr).sum()):5d}  "
                  f"({(ag > thr).mean():5.1%} of 分歧, {(ag > thr).sum()/max(n_pair,1):5.1%} of 全部可比对)")
        print("    ⚠️ 中位数接近 0 说明**大量分歧其实是并列**（价值先验在这些类上分不出高下）")
    print(f"\n分歧且『策略选闪电(17)、价值想选别的』的次数 = {val_nonlight_but_pol_light}"
          f"（占全部分歧 {val_nonlight_but_pol_light / max(n_pair - agree,1):.1%}）")
    print("\n价值 top-1 的类直方图（前 10）：")
    for k, v in val_top_hist.most_common(10):
        print(f"    {k:14s} {v:6d}  {v/max(n_pair,1):6.1%}")
    print("策略 top-1 的类直方图（前 10）：")
    for k, v in pol_top_hist.most_common(10):
        print(f"    {k:14s} {v:6d}  {v/max(n_pair,1):6.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
