#!/usr/bin/env python3
"""计算最新训练中各 loss 分量的占比"""
import json, sys, os

log_dir = sys.argv[1] if len(sys.argv) > 1 else None
if log_dir is None:
    base = "/root/autodl-tmp/AntWar/ppo_v1/outputs"
    dirs = sorted([d for d in os.listdir(base) if d.startswith("20")], reverse=True)
    log_dir = os.path.join(base, dirs[0])

path = os.path.join(log_dir, "training/training_metrics_history.jsonl")
with open(path) as f:
    lines = f.readlines()
latest = json.loads(lines[-1])

ep = latest.get('episode', '?')
loss = latest.get('avg_loss', 0)
pl = latest.get('policy_loss', 0)
vl = latest.get('value_loss', 0)
ec = latest.get('entropy_coef', 0.2)
ent = latest.get('entropy', 0)
atl = latest.get('aux_tower_loss', 0)
agl = latest.get('aux_gold_loss', 0)

# 系数（与 yaml 配置保持一致）
vf_coef = 0.25
atc = 0.1
agc = 0.01

components = [
    ("policy_loss (raw)",           pl,   1.0),
    ("value_loss * vf_coef(0.25)",  vl,   vf_coef),
    ("entropy * ent_coef(%.4f)" % ec, ent, ec),
    ("aux_tower_loss * atc(0.1)",   atl,  atc),
    ("aux_gold_loss * agc(0.01)",   agl,  agc),
]

print("=" * 80)
print("  Ep %s  Loss 占比分析" % ep)
print("  JSONL: %s" % path)
print("=" * 80)
print("%-40s %10s  %8s  %12s  %8s" % ("Component", "Raw", "Weight", "Weighted", "%"))
print("-" * 80)
total_w = 0
for name, raw, w in components:
    weighted = raw * w
    total_w += weighted
    pct = weighted / loss * 100 if loss else 0
    print("%-40s %10.4f  %8.4f  %12.4f  %7.1f%%" % (name, raw, w, weighted, pct))
print("-" * 80)
print("%-40s %10s  %8s  %12.4f  100.0%%" % ("Total Loss (avg_loss)", "", "", loss))

print()
print("--- 字段原始值 ---")
for k in ['episode','avg_reward','avg_loss','policy_loss','value_loss','entropy','entropy_coef','aux_tower_loss','aux_gold_loss']:
    print("  %s = %s" % (k, latest.get(k, 'N/A')))
