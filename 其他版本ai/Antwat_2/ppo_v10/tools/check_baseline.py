#!/usr/bin/env python3
import json, sys, os

if len(sys.argv) < 2:
    print("Usage: check_baseline.py <generations_dir>")
    sys.exit(1)

gen_dir = sys.argv[1]
for gen in (1, 2):
    fp = os.path.join(gen_dir, f"gen_{gen:04d}", "evaluation.json")
    if not os.path.exists(fp):
        print(f"Gen {gen}: file not found")
        continue
    d = json.load(open(fp))
    print(f"\n=== Gen {gen} ===")
    rows = []
    for seed_id, agents in d.items():
        wins = losses = draws = ill = 0
        for aname, agg in agents.items():
            if not isinstance(agg, dict):
                continue
            w = agg.get("agent1_wins", 0)
            l = agg.get("agent2_wins", 0)
            dr = agg.get("draws", 0)
            wins += w
            losses += l
            draws += dr
            ill += agg.get("total_illegal", 0)
        total = wins + losses + draws
        wr = wins / total if total else 0
        rows.append((wr, seed_id, wins, losses, total, ill))
    for wr, sid, w, l, t, ill in sorted(rows, reverse=True):
        print(f"  {sid:22s}  wr={wr*100:5.1f}%  w={w:2d}  l={l:2d}  games={t:2d}  illegal={ill}")
