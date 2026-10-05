"""离线预测「部署会怎么选类」：在**已有数据的状态**上，比较三种决策各自会选什么类：
  ① `argmax`（= A1 的实际机制，类头常数 ⇒ 恒闪电/HOLD）
  ② `masked`（合法类 argmax）
  ③ `q`（Q 头在合法类上 argmax）
看 ③ 里"花钱类"的比例 —— 这直接预测部署会不会 churn（省掉 20 s/回合 的对局成本）。

用法:
  python -u code/my_ai/az_intent/probe_q_deployment.py \
      --data training_history/vprior/data_sp_reserve60_exec \
      --q training_history/vprior/qhead_M3.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--ckpt", default="training_history/vprior/posnet_A_k5_m32.pt")
    ap.add_argument("--limit-games", type=int, default=20)
    a = ap.parse_args()

    src = Path(a.data)
    if not src.is_absolute():
        src = REPO / src
    files = sorted(src.glob("az_selfplay_seed*.pkl"))[:a.limit_games]
    rows = []
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        rows.extend(d["samples"] if isinstance(d, dict) else d)
    idx = np.arange(0, len(rows), max(1, len(rows) // 4000))
    b = np.stack([np.asarray(rows[i]["board"], dtype=np.float32) for i in idx])
    s = np.stack([np.asarray(rows[i]["stats"], dtype=np.float32) for i in idx])
    cm = np.stack([np.asarray(rows[i]["class_mask"], dtype=bool) for i in idx])
    print(f"局数={len(files)} 用 {len(idx)} 个决策")

    from my_ai.az_intent.az_selfplay import load_three_models
    from my_ai.az_intent.q_head import QHead

    ck = str((REPO / a.ckpt) if not Path(a.ckpt).is_absolute() else a.ckpt)
    class_model, _p, _v = load_three_models(ck)
    class_model.eval()
    embs = np.zeros((len(idx), 128), dtype=np.float32)
    heads = None
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            o = class_model(torch.from_numpy(b[i:i + 512]), torch.from_numpy(s[i:i + 512]))
            embs[i:i + 512] = o["state_emb"].cpu().numpy()
            hl = torch.stack([o[f"head{h+1}_logits"] for h in range(3)], dim=1).cpu().numpy()
            heads = hl if heads is None else np.concatenate([heads, hl], axis=0)

    qf = str((REPO / a.q) if not Path(a.q).is_absolute() else a.q)
    q = QHead.load(qf)
    with torch.no_grad():
        qv = q.score_all(torch.from_numpy(embs)).cpu().numpy()       # (M,24)

    names = {23: "HOLD", 17: "LIGHTNING"}
    def nm(c):
        if c in names:
            return names[c]
        if 0 <= c <= 15:
            return "tower"
        if c == 16:
            return "downgrade"
        if 18 <= c <= 20:
            return "super"
        return f"c{c}"

    def report(tag, cls_pick):
        cnt = Counter(nm(int(c)) for c in cls_pick)
        tot = len(cls_pick)
        print(f"  {tag}: " + ", ".join(f"{k}={v/tot:.1%}" for k, v in cnt.most_common(5)))

    # ① argmax（不套掩码）
    report("argmax(全24类)", heads[:, 0, :].argmax(axis=1))
    # ② masked：合法类里取最大 logit
    masked_logit = np.where(cm, heads[:, 0, :], -np.inf)
    report("masked(合法类)", masked_logit.argmax(axis=1))
    # ③ q：合法类里取最大 Q
    masked_q = np.where(cm, qv, -np.inf)
    report("q(合法类取max Q)", masked_q.argmax(axis=1))
    # 附加：Q 在"闪电可执行"的局面上是否仍选闪电
    can_light = cm[:, 17]
    if can_light.any():
        pick = masked_q[can_light].argmax(axis=1)
        report("q|闪电可执行", pick)
    if (~can_light).any():
        pick = masked_q[~can_light].argmax(axis=1)
        report("q|闪电不可执行", pick)
    print("  判读：③ 里 tower/downgrade 占比很高 => 部署会 churn（D 型）；闪电占比高 => 与 A1 相近。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
