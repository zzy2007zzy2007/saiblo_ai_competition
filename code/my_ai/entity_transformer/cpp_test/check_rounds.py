"""粗略统计已采集对局的回合数（读 .ents.jsonl 的 max round + 行数）。"""
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else r"F:\2026智能体大赛_新2\training_history\lowq_e2"
n_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200
files = sorted(f for f in os.listdir(d) if f.endswith(".ents.jsonl"))[:n_limit]
rounds = []
rows = []
for f in files:
    mx = -1
    n = 0
    with open(os.path.join(d, f), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            mx = max(mx, int(r.get("round", -1)))
            n += 1
    rounds.append(mx)
    rows.append(n)

if not rounds:
    print("no games")
    raise SystemExit(0)
import numpy as np
a = np.asarray(rounds, dtype=float)
print(f"sampled games: {len(a)}")
print(f"max-round(=局回合数): mean={a.mean():.1f} median={np.median(a):.0f} min={a.min():.0f} max={a.max():.0f}")
print(f"分位: p10={np.percentile(a,10):.0f} p25={np.percentile(a,25):.0f} p75={np.percentile(a,75):.0f} p90={np.percentile(a,90):.0f}")
print(f"样本/局: mean={np.mean(rows):.0f}  (≈2×回合，符合每回合两方各一条)")
print(f"回合数分布桶: 512={int((a>=511).sum())}, 400-511={int(((a>=400)&(a<511)).sum())}, "
      f"300-399={int(((a>=300)&(a<400)).sum())}, 200-299={int(((a>=200)&(a<300)).sum())}, <200={int((a<200).sum())}")
