"""Analyze pool family structure at different cos thresholds."""
import json, os, torch, numpy as np
from collections import defaultdict

with open('outputs/20260613_004120/league/pool.json') as f:
    pool = json.load(f)
opp_ids = pool['opponents']
checkpoints = pool['checkpoint_paths']

genomes = {}
gen_info = {}
for oid in opp_ids:
    cp = checkpoints.get(oid, '')
    if not cp or not os.path.exists(cp):
        continue
    ckpt = torch.load(cp, map_location='cpu', weights_only=True)
    sd = ckpt['policy_state_dict']
    w = np.concatenate([p.cpu().numpy().flatten() for _, p in sd.items()]).astype(np.float32)
    genomes[oid] = w
    gen_info[oid] = {'gen': ckpt.get('generation', '?'), 'elo': ckpt.get('elo_rating', 0)}

sid = lambda x: x.replace('gen_', 'G').replace('_ind_', '_')
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

# Pre-compute all pairwise cos
pairs = []
ids_list = sorted(genomes.keys(), key=lambda x: gen_info[x]['gen'])
for i, oi in enumerate(ids_list):
    for j, oj in enumerate(ids_list):
        if i >= j:
            continue
        cos = np.dot(genomes[oi], genomes[oj]) / (np.linalg.norm(genomes[oi]) * np.linalg.norm(genomes[oj]))
        pairs.append((oi, oj, cos))

print('  Thr  Fams  MaxSz  Singles  Details')
print('=' * 90)

for th in thresholds:
    edges = defaultdict(set)
    for oi, oj, cos in pairs:
        if cos > th:
            edges[oi].add(oj)
            edges[oj].add(oi)

    visited = set()
    families = []
    for oid in ids_list:
        if oid in visited:
            continue
        if oid not in edges:
            families.append([oid])
            visited.add(oid)
            continue
        stack = [oid]
        fam = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            fam.append(cur)
            for nb in edges[cur]:
                if nb not in visited:
                    stack.append(nb)
        families.append(fam)

    families.sort(key=lambda x: -len(x))
    max_sz = len(families[0]) if families else 0
    singles = sum(1 for f in families if len(f) == 1)

    detail_parts = []
    for i, fam in enumerate(families):
        names = ','.join(sid(x) for x in fam)
        detail_parts.append('F' + str(i+1) + '(' + str(len(fam)) + '):[' + names + ']')
    details = ' | '.join(detail_parts)

    print(th, ' ' * 3, len(families), ' ' * 3, max_sz, ' ' * 5, singles, ' ' * 3, details)
    print()
