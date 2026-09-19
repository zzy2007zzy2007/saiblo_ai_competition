#!/usr/bin/env python3
import json, math

with open("ppo/outputs/20260525_172045/training/weight_stats.jsonl") as f:
    weights = [json.loads(l) for l in f]

all_params = list(weights[-1]["weights"].keys())
layers = {}
for name in sorted(all_params):
    layer = ".".join(name.split(".")[:3])
    layers.setdefault(layer, []).append(name)

print(f"总参数数: {len(all_params)}")
print()

for wr in weights:
    ep = wr["episode"]
    wd = wr["weights"]
    print(f"=== Ep {ep} ===")
    for layer, params in sorted(layers.items()):
        nan_count = sum(1 for p in params 
                        if any(math.isnan(wd[p].get(k, 0)) for k in ["mean","std","min","max"]))
        total = len(params)
        if nan_count > 0 or ep == 40:
            sample = ", ".join(p.split(".")[-1] for p in params[:2])
            print(f"  {layer} ({total}): {nan_count} NaN  示例: {sample}")
    print()
