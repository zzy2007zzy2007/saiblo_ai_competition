"""给已打好标签的类头数据**加上价值基线**：`adv = value_target − V(s)`
（预注册 `docs/plan_20261015_explore_awr_class.md` §1.1 修正：把"每局常数 advantage"的 AWR
换成**REINFORCE 价值基线**，从而在同局内产生对比信号）。

`value_target` 已按 `player` 取过符号（胜为正）；这里用价值网对该 `player` 的估值做基线。

用法:
  python -u code/test_match/add_value_baseline.py --src <ingested_dir> --dst <out_dir> \
      --ckpt checkpoints/posnet_A_k5_m32.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from my_ai.az_intent.az_selfplay import load_three_models  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    dev = torch.device(args.device if args.device != "auto" else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    _, _, value_model = load_three_models(args.ckpt)
    value_model = value_model.to(dev).eval()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("az_selfplay_seed*.pkl"))
    print(f"[baseline] {len(files)} 个 pkl；ckpt={args.ckpt}；device={dev}", flush=True)
    tot, n_rows = 0.0, 0
    advs = []
    for f in files:
        d = pickle.load(open(f, "rb"))
        samples = d.get("samples", [])
        if not samples:
            continue
        b = torch.from_numpy(np.stack([np.asarray(s["board"], dtype=np.float32) for s in samples])).to(dev)
        st = torch.from_numpy(np.stack([np.asarray(s["stats"], dtype=np.float32) for s in samples])).to(dev)
        vals = np.zeros(len(samples), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(samples), args.batch):
                out = value_model(b[i:i + args.batch], st[i:i + args.batch])
                vals[i:i + args.batch] = out["value"].detach().cpu().numpy().reshape(-1)
        for s, v in zip(samples, vals):
            s["v_base"] = float(v)
            s["adv"] = float(s["value_target"]) - float(v)      # 优势 = 结果 − 价值基线
            advs.append(s["adv"])
        tot += float(np.mean(vals)) * len(samples)
        n_rows += len(samples)
        with open(dst / f.name, "wb") as fh:
            pickle.dump(d, fh)
    a = np.asarray(advs, dtype=np.float32)
    print(f"[baseline] {n_rows} 条样本；V 均值 {tot / max(n_rows,1):+.3f}；"
          f"优势 adv：均值 {a.mean():+.4f}、std {a.std():.4f}、"
          f"|adv|>0.05 的比例 {(np.abs(a) > 0.05).mean():.1%} ⇒ {dst}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
