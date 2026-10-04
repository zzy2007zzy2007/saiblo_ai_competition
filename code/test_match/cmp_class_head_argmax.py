"""旧类头 vs 新类头（候选 C）：**部署行为到底变了没有？**

为什么必须先测这个：A1 配置是 `pos-only` + `pos_pin=argmax` ⇒ 部署时**类 = 策略头 argmax**
（不是采样）。所以"训练改了多少**概率质量**"不重要，**argmax 变没变**才决定棋力读数有没有意义。
本脚本在**同一批决策**上比较两个 ckpt 的类头 argmax（只在被评估的类集合上比，避免类掩码问题）。

用法:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/cmp_class_head_argmax.py \
      --a training_history/vprior/posnet_A_k5_m32.pt \
      --b training_history/vprior/posnet_C_class.pt \
      --data training_history/vprior/vp_A_k5_m32.npz --max-decisions 3000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

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
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-decisions", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()

    torch.set_num_threads(1)
    from my_ai.az_intent.az_selfplay import load_three_models
    from my_ai.az_intent.train_class_prior import build_dataset

    board, stats, dec_idx, cls_pad, _tgt, mask = build_dataset(Path(a.data), a.max_decisions)
    print(f"[data] 决策={len(dec_idx)}", flush=True)

    ids = np.arange(len(dec_idx))
    xs = torch.from_numpy(board[dec_idx[ids]].astype(np.float32))
    ss = torch.from_numpy(stats[dec_idx[ids]].astype(np.float32))
    cls_t = torch.from_numpy(cls_pad)

    def argmaxes(ckpt: str) -> tuple[np.ndarray, np.ndarray]:
        """返回 (在被评估类集合上的 argmax, **全 24 类**的 argmax)。

        ⚠️ 后者才是**部署时真正用的**：`bundle_mcts.py:311` 的 `pos_pin=argmax` 取的是
        `int(np.argmax(hl))`（**不套类掩码**）⇒ 若这个 argmax 不可执行，整回合就退化成 HOLD。
        """
        cm, _pm, _vm = load_three_models(ckpt)
        cm.eval()
        out = np.full((len(ids), 3), -1, dtype=np.int64)
        full = np.full((len(ids), 3), -1, dtype=np.int64)
        with torch.no_grad():
            for s in range(0, len(ids), a.batch):
                o = cm(xs[s:s + a.batch], ss[s:s + a.batch])
                logit = torch.stack([o[f"head{h+1}_logits"] for h in range(3)], dim=1)  # (B,3,C)
                full[s:s + a.batch] = logit.argmax(dim=-1).numpy()
                c = cls_t[s:s + a.batch]
                safe = c.clamp(min=0)
                g = torch.gather(logit, 2, safe)
                g = torch.where(c >= 0, g, torch.full_like(g, float("-inf")))
                out[s:s + a.batch] = g.argmax(dim=-1).numpy()
        return out, full

    A, A_full = argmaxes(a.a)
    B, B_full = argmaxes(a.b)
    m = mask.astype(bool)
    same_all = (A == B)
    print(f"\n=== 旧类头 vs 新类头（只在被评估类集合上比 argmax）===")
    print(f"全部可比 (决策, head) 对 = {int(m.sum())}")
    print(f"argmax 不变的比例 = {same_all[m].mean():.2%}")
    for h in range(3):
        sel = m[:, h]
        if sel.sum():
            print(f"  head{h}: n={int(sel.sum())}  不变 {same_all[sel, h].mean():.2%}")
    full_same = (A_full == B_full)
    print(f"\n=== **全 24 类**的 argmax（= 部署真正用的那个）===")
    print(f"全部 (决策, head) 对 = {full_same.size}")
    print(f"argmax 不变的比例 = {full_same.mean():.2%}")
    for h in range(3):
        print(f"  head{h}: 不变 {full_same[:, h].mean():.2%}")
    import collections as _c
    hist_old = _c.Counter(A_full[:, :].ravel().tolist())
    hist_new = _c.Counter(B_full[:, :].ravel().tolist())
    print("旧全 argmax 类分布（前 8）:", hist_old.most_common(8))
    print("新全 argmax 类分布（前 8）:", hist_new.most_common(8))
    # 换成"类 id"再比一次（不同类就算变）
    clsA = np.take_along_axis(cls_pad, np.clip(A, 0, None)[:, :, None], axis=2)[:, :, 0]
    clsB = np.take_along_axis(cls_pad, np.clip(B, 0, None)[:, :, None], axis=2)[:, :, 0]
    chg = (clsA != clsB) & m
    print(f"\n选中的**类 id** 变化的 (决策, head) 对 = {int(chg.sum())}（占 {chg.sum()/max(int(m.sum()),1):.2%}）")
    import collections
    pair_hist = collections.Counter(zip(clsA[chg].tolist(), clsB[chg].tolist()))
    for (x, y), n in pair_hist.most_common(10):
        print(f"    {x:>3} -> {y:>3} : {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
