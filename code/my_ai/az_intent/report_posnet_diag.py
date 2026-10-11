"""位置网的两个诊断（A 一跑就要挂上去看的）：

  ① 通道覆盖：位置目标的质量落在哪些动作类上（当前是 **1/24**，只有闪电）。
     —— 判据来自 docs/az_posnet_random_class_plan.md §6-1。

  ② **argmax 与搜索选择的一致率**（本次新增，最关键）：
     在"有得选"的回合上，搜索选中的位置 vs 策略在该通道上 argmax 的位置。
     raw 解码只读 argmax、搜索从整张图采样，所以**训练可能改了细结构却没改 argmax**
     —— 那 raw 镜像就必然不动。这个数就是为了区分
     "蒸馏没进 argmax" 与 "进了也没用"（见与用户的讨论）。

用法：
    python code/my_ai/az_intent/report_posnet_diag.py \
        --position-data <紧凑数据集或原始 pkl 目录> \
        --ref-raw <提供"搜索选中格子"的原始 pkl 目录> \
        --ckpts <ckpt1> <ckpt2> ...
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
for p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def load_pkls(path: Path, limit: int | None = None) -> list[dict]:
    fs = sorted(path.rglob("az_selfplay_seed*.pkl"))
    out: list[dict] = []
    for f in (fs[:limit] if limit else fs):
        with open(f, "rb") as fh:
            out.extend(pickle.load(fh)["samples"])
    return out


def search_choice_per_head(s: dict, num_heads: int = 3) -> list[tuple | None]:
    """从 (visit, intent_counts) 还原搜索"选中"的每头意图 = 访问量最大的候选里
    采样次数最多的那个 intent。"""
    out: list[tuple | None] = [None] * num_heads
    ic = s.get("intent_counts")
    vis = s.get("visit")
    if not ic or vis is None:
        return out
    j = int(np.argmax(np.asarray(vis, dtype=float)))
    for h in range(num_heads):
        d = ic[j][h] if j < len(ic) and h < len(ic[j]) else {}
        if d:
            out[h] = max(d.items(), key=lambda kv: kv[1])[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--position-data", required=True,
                    help="提供'有位置目标的样本'的数据集（紧凑或原始 pkl 目录）")
    ap.add_argument("--ref-raw", default=None,
                    help="提供搜索选择/掩码的原始 pkl 目录；默认同 --position-data")
    ap.add_argument("--ckpts", nargs="+", required=True, help="待比较的 checkpoint（第一个当作基线）")
    ap.add_argument("--limit", type=int, default=None, help="只读前 N 个 pkl")
    ap.add_argument("--num-heads", type=int, default=3)
    args = ap.parse_args()

    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import load_three_models
    from my_ai.az_intent.az_train import _marginalize

    ref_dir = Path(args.ref_raw or args.position_data)
    samples = load_pkls(ref_dir, args.limit)
    pos = [s for s in samples
           if any(x >= 0 for h in range(args.num_heads) for (_c, x, _y) in _marginalize(s, h))]
    print(f"[diag] 参考局面：{len(samples)} 样本 → 其中带位置目标的 {len(pos)} 个"
          f"（来自 {ref_dir}）", flush=True)
    if not pos:
        sys.exit("没有带位置目标的样本")

    # ① 通道覆盖
    cls_mass: Counter = Counter()
    for s in pos:
        for h in range(args.num_heads):
            for (c, x, _y), m in _marginalize(s, h).items():
                if x >= 0:
                    cls_mass[c] += m
    tot = sum(cls_mass.values()) or 1.0
    print("[diag] ① 通道覆盖（位置目标的质量按动作类）:")
    for c, m in cls_mass.most_common(8):
        print(f"        class {c:2d}: {m/tot:6.1%}")
    print(f"        涉及类数 = {len(cls_mass)} / 24", flush=True)

    # ② argmax 与搜索选择的一致率
    feat = FeatureExtractor(max_actions=96)
    boards = np.stack([s["board"] for s in pos])
    stats = np.stack([s["stats"] for s in pos])
    choices = [search_choice_per_head(s, args.num_heads) for s in pos]
    masks = [s["position_mask"] for s in pos]

    print(f"\n[diag] ② argmax vs 搜索选择（{len(pos)} 个出招回合，逐头统计）")
    for ck in args.ckpts:
        _, pos_model, _ = load_three_models(ck)
        pos_model.eval()
        hit = tot_n = 0
        per_head = Counter()
        for i in range(0, len(pos), 64):
            b = torch.from_numpy(boards[i:i + 64]).float()
            st = torch.from_numpy(stats[i:i + 64]).float()
            with torch.no_grad():
                am = pos_model(b, st)["action_map"].numpy()
            for k in range(am.shape[0]):
                idx = i + k
                for h in range(args.num_heads):
                    ch = choices[idx][h]
                    if ch is None:
                        continue
                    c, x, y = ch
                    if x < 0 or c >= am.shape[1]:
                        continue
                    legal = np.asarray(masks[idx][c], dtype=bool)
                    if not legal.any():
                        continue
                    m = np.where(legal, am[k, c], -np.inf)
                    ax, ay = np.unravel_index(int(np.argmax(m)), m.shape)
                    tot_n += 1
                    if (ax, ay) == (x, y):
                        hit += 1
                        per_head[h] += 1
        print(f"        {Path(ck).name:22s} 一致率 = {hit/max(tot_n,1):6.2%}  (n={tot_n})", flush=True)


if __name__ == "__main__":
    main()
