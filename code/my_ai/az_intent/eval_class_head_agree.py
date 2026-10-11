"""闸门 ①（预注册 §9）：类头对"**实际执行类**"的一致率 —— 新头必须高于 incumbent 头。

口径：
- 数据与切分**完全复刻** `train_class_outcome.py`：`sorted(glob("az_selfplay_seed*.pkl"))`、
  样本按文件内顺序拼接、`n_val = int(N*0.1)`、`tr=[0,N-n)`、`va=[N-n,N)`；
- 每个样本有 `chosen_cls[3]`（每头实际执行的类；**255 = 该头无操作 ⇒ 跳过**）与 `class_mask[24]`；
- 一致率 = argmax_{c: class_mask[c]} head_logits[h][c] == chosen_cls[h]，只统计 chosen_cls[h] != 255 的样本；
- 同时报 HOLD（类 23）与"非 HOLD 动作"两个子集，避免只看总体被 HOLD 主导。

用法：
  python -u code/my_ai/az_intent/eval_class_head_agree.py \
    --data data/posroll_c_labeled --incumbent checkpoints/grafted_sdk_A.pt \
    --candidates checkpoints/grafted_sdk_A_clsC_e3.pt checkpoints/grafted_sdk_A_clsC.pt
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HOLD = 23


def load_rows(data_dir: Path):
    rows = []
    for f in sorted(data_dir.glob("az_selfplay_seed*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        for s in d["samples"]:
            if "chosen_cls" in s and "class_mask" in s:
                rows.append(s)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidates", nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--which", default="val", choices=["val", "train", "all"])
    ap.add_argument("--split", default="random", choices=["block", "random"],
                    help="random=随机行切（默认，避免块切成局面尾部切片）；block=最后 10% 行（历史口径）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    from my_ai.az_intent.az_selfplay import load_three_models

    dev = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                       else ("cpu" if args.device == "auto" else args.device))
    rows = load_rows(Path(args.data))
    N = len(rows)
    n_val = max(1, int(N * 0.1))
    if args.split == "random":
        _perm = np.random.default_rng(args.seed).permutation(N)
        _va = np.sort(_perm[:n_val]); _tr = np.sort(_perm[n_val:])
        if args.which == "val":
            idx = _va
        elif args.which == "train":
            idx = _tr
        else:
            idx = np.arange(N)
        print(f"[gate1] 切分=random（n_val={n_val}）", flush=True)
    elif args.which == "val":
        idx = np.arange(N - n_val, N)
    elif args.which == "train":
        idx = np.arange(0, N - n_val)
    else:
        idx = np.arange(N)
    print(f"[gate1] 样本 {N}（{args.which} 子集 {len(idx)} 条），device={dev}", flush=True)

    board = torch.from_numpy(np.stack([np.asarray(rows[i]["board"], dtype=np.float32) for i in idx]))
    stats = torch.from_numpy(np.stack([np.asarray(rows[i]["stats"], dtype=np.float32) for i in idx]))
    chosen = np.stack([np.asarray(rows[i]["chosen_cls"], dtype=np.int64) for i in idx])      # (n,3)
    mask = np.stack([np.asarray(rows[i]["class_mask"], dtype=bool) for i in idx])            # (n,24)

    def evaluate(ckpt: str) -> dict:
        cm, _pm, _vm = load_three_models(ckpt)
        cm.eval().to(dev)
        hits = np.zeros(3, dtype=np.int64); tot = np.zeros(3, dtype=np.int64)
        hits_hold = np.zeros(3, dtype=np.int64); tot_hold = np.zeros(3, dtype=np.int64)
        hits_act = np.zeros(3, dtype=np.int64); tot_act = np.zeros(3, dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(idx), args.batch_size):
                b = board[i:i + args.batch_size].to(dev)
                s = stats[i:i + args.batch_size].to(dev)
                out = cm(b, s)
                for h in range(3):
                    # 类头子网的输出键是 head1_logits/head2_logits/head3_logits（1-based，(B,24)）
                    lg = out[f"head{h+1}_logits"].float().cpu().numpy()
                    m = mask[i:i + args.batch_size]
                    lg = np.where(m, lg, -1e9)
                    pred = lg.argmax(1)
                    want = chosen[i:i + args.batch_size, h]
                    ok = (want != 255) & (want < 24)
                    if not ok.any():
                        continue
                    hit = (pred[ok] == want[ok])
                    tot[h] += ok.sum(); hits[h] += hit.sum()
                    is_hold = want[ok] == HOLD
                    tot_hold[h] += is_hold.sum(); hits_hold[h] += (hit & is_hold).sum()
                    tot_act[h] += (~is_hold).sum(); hits_act[h] += (hit & ~is_hold).sum()
        return dict(hits=hits, tot=tot, hits_hold=hits_hold, tot_hold=tot_hold,
                    hits_act=hits_act, tot_act=tot_act)

    def show(name: str, r: dict) -> None:
        allh = r["hits"].sum() / max(r["tot"].sum(), 1)
        hold = r["hits_hold"].sum() / max(r["tot_hold"].sum(), 1)
        act = r["hits_act"].sum() / max(r["tot_act"].sum(), 1)
        per = " ".join(f"h{h}={r['hits'][h]}/{r['tot'][h]}={r['hits'][h]/max(r['tot'][h],1)*100:.1f}%"
                       for h in range(3))
        print(f"   {name:34s} 总体={allh*100:6.2f}%  HOLD={hold*100:6.2f}%  非HOLD动作={act*100:6.2f}%  [{per}]", flush=True)
        return allh

    base = evaluate(args.incumbent)
    print("[gate1] ===== 一致率（argmax 掩码内 vs 实际执行类）=====")
    a0 = show(Path(args.incumbent).name, base)
    results = {}
    for c in args.candidates:
        results[c] = show(Path(c).name, evaluate(c))
    for c, a in results.items():
        d = (a - a0) * 100
        verdict = "✅ 达标（> incumbent）" if d > 0 else "❌ 不达标（未超过 incumbent）"
        print(f"[gate1] {Path(c).name}: 总体一致率 {a*100:.2f}% vs incumbent {a0*100:.2f}% ⇒ {d:+.2f}pp ⇒ {verdict}")
        if d < 2.0:
            print("          ⚠️ 预注册 §9 要求『高于 incumbent』；实践上 <2pp 视为『学不到』")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
