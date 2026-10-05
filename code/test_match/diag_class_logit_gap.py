"""诊断：**类头的 logits 到底"锁"得有多死？**（决定"类轴还能不能被微调打开"）

背景（2026-10-05 实测）：类头对**全 24 类**的 argmax 在 9,663/9,663 个 (决策, head) 对上恒为
类 17（闪电）；候选 C（在被评估的 5 个合法类之间重排序）训练后**部署行为逐字节不变**。
本脚本直接量"锁"的强度：**top1 与 top2 的 logit 差**，以及 **top1 与"最好的可执行类"的差**。

判读（先写死）：
  * 若 `top1 - top2` 的中位数 ≫ 1（比如 > 5）⇒ **微调几乎不可能翻转 argmax**
    （那需要把另一个类的 logit 抬上来或把闪电压下去，量级远超训练能提供的梯度）
    ⇒ 类轴在"冻结骨干 + 小步微调"的范式下**不可达**，要动只能**换参数化/重训**。
  * 若差距很小（< 1）⇒ 理论上微调可翻转，之前的零结果就更可能是"目标/损失设计"的问题。

"可执行的类"用 npz 里的 `cnt>0`（该类此刻真有可执行候选格）判定 —— 这是采集时就记下来的。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/diag_class_logit_gap.py \
      --data training_history/vprior/vp_A_k5_m32.npz --ckpt training_history/vprior/posnet_A_k5_m32.pt \
      --max-decisions 2000
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "training_history" / "vprior" / "vp_A_k5_m32.npz"))
    ap.add_argument("--ckpt", default=str(REPO / "training_history" / "vprior" / "posnet_A_k5_m32.pt"))
    ap.add_argument("--max-decisions", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt

    z = np.load(a.data, allow_pickle=True)
    # ⚠️ 一次性读进内存：npz 每次 z[k] 都会**重新解压整个数组**（3.5 GB）⇒ 逐行访问会打爆内存
    board_a, stats_a = z["board"], z["stats"]
    player = z["player"]
    head, cls, cnt = z["head"], z["cls"], z["cnt"]
    key = np.concatenate([stats_a.astype(np.float32), player[:, None].astype(np.float32)], axis=1)
    same = np.all(key[1:] == key[:-1], axis=1)
    bounds = [0] + (np.nonzero(~same)[0] + 1).tolist() + [len(player)]
    n_dec = len(bounds) - 1
    step = max(1, n_dec // a.max_decisions)

    decs = []
    for d in range(0, n_dec, step):
        lo, hi = bounds[d], bounds[d + 1]
        by_head = collections.defaultdict(list)
        for j in range(lo, hi):
            by_head[int(head[j])].append(j)
        decs.append((lo, by_head))

    feat = FeatureExtractor(max_actions=96)
    model, _ = make_net_fn_from_ckpt(a.ckpt, feat)
    model.eval()

    boards = np.stack([board_a[lo] for lo, _ in decs]).astype(np.float32)
    statss = np.stack([stats_a[lo] for lo, _ in decs]).astype(np.float32)

    logits = np.zeros((len(decs), 3, 24), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(decs), a.batch):
            o = model(torch.from_numpy(boards[s:s + a.batch]),
                      torch.from_numpy(statss[s:s + a.batch]))
            for h in range(3):
                logits[s:s + a.batch, h] = np.asarray(o[f"head{h+1}_logits"], dtype=np.float64)

    gaps12, gaps_exe, top1_hist = [], [], collections.Counter()
    exe_same_as_top1 = 0
    n_head = 0
    for di, (_lo, by_head) in enumerate(decs):
        for h, rows in by_head.items():
            if h >= 3:
                continue
            lg = logits[di, h]
            order = np.argsort(-lg)
            top1 = int(order[0])
            top1_hist[top1] += 1
            gaps12.append(float(lg[order[0]] - lg[order[1]]))
            exe = [int(cls[j]) for j in rows if int(cnt[j]) > 0]
            if exe:
                best_exe = max(exe, key=lambda c: lg[c])
                gaps_exe.append(float(lg[top1] - lg[best_exe]))
                exe_same_as_top1 += int(best_exe == top1)
            n_head += 1

    def q(xs, name):
        if not xs:
            print(f"{name}: n=0")
            return
        arr = np.asarray(xs)
        print(f"{name}: n={len(arr)} 中位={np.median(arr):.2f} "
              f"p10={np.percentile(arr,10):.2f} p90={np.percentile(arr,90):.2f} "
              f"min={arr.min():.2f} max={arr.max():.2f}")

    print(f"数据={Path(a.data).name}  决策={len(decs)}  可比 (决策,head) 对={n_head}")
    print(f"top-1 类分布（前 6）: {top1_hist.most_common(6)}")
    q(gaps12, "top1 - top2 的 logit 差")
    q(gaps_exe, "top1 - 最好的可执行类 的 logit 差")
    if gaps_exe:
        print(f"其中 top1 本身就是最好的可执行类的比例 = {exe_same_as_top1/max(len(gaps_exe),1):.1%}")
    # ★ 关键：这些 logits **随局面变不变**？（若几乎不变 ⇒ 类头≈常数函数、没有状态信息）
    lg = logits.reshape(-1, 24)
    pcs = lg.std(axis=0)
    pcm = lg.mean(axis=0)
    print(f"\n各类 logit 的跨局面标准差: 中位={np.median(pcs):.4f} min={pcs.min():.4f} max={pcs.max():.4f}")
    print(f"各类 logit 的均值:        中位={np.median(pcm):.4f} min={pcm.min():.4f} max={pcm.max():.4f}")
    g = np.asarray(gaps12)
    verdict = ("几乎恒定 ⇒ 类头基本是常数函数、**没有状态信息**" if g.std() < 0.05
               else "随局面变化")
    print(f"gap(top1-top2) 的跨局面标准差 = {g.std():.4f}（中位 {np.median(g):.4f}）⇒ {verdict}")
    print("\n判读：中位差 ≫ 1 ⇒ 微调翻不动 argmax；**差很小但恒定** ⇒ 能翻、但类头本身没有状态信息")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
