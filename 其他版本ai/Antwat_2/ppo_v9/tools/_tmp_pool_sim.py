"""临时脚本：分析对手池个体间余弦相似度"""
import json, os
import torch
import numpy as np
from collections import defaultdict

with open('ppo_v9/outputs/20260613_004120/league/pool.json') as f:
    pool = json.load(f)

opp_ids = pool['opponents']
checkpoints = pool['checkpoint_paths']
print(f'Pool size: {len(opp_ids)}')

genomes = {}
gen_info = {}
for oid in opp_ids:
    cp = checkpoints.get(oid, '')
    if not cp or not os.path.exists(cp):
        print(f'{oid}: checkpoint not found')
        continue
    try:
        ckpt = torch.load(cp, map_location='cpu', weights_only=True)
        sd = ckpt['policy_state_dict']
        weights = [param.cpu().numpy().flatten() for _, param in sd.items()]
        genome = np.concatenate(weights).astype(np.float32)
        genomes[oid] = genome
        gen_info[oid] = {
            'gen': ckpt.get('generation', '?'),
            'elo': ckpt.get('elo_rating', 0),
        }
    except Exception as e:
        print(f'{oid}: error - {e}')

short = lambda x: x.replace('gen_', 'g').replace('_ind_', '_')
ids = sorted(genomes.keys(), key=lambda x: gen_info[x]['gen'])

# Matrix header
print(f'\n=== 余弦相似度矩阵 (按gen排序) ===')
print(f'{"":12s}', end='')
for oid in ids:
    label = short(oid) + '(G' + str(gen_info[oid]["gen"]) + ')'
    print(f'{label:>16s}', end='')
print()

# Matrix body
for oid_i in ids:
    wi = genomes[oid_i]
    print(f'{short(oid_i):12s}', end='')
    for oid_j in ids:
        wj = genomes[oid_j]
        dot = np.dot(wi, wj)
        na = np.linalg.norm(wi)
        nb = np.linalg.norm(wj)
        cos = dot / (na * nb) if na > 0 and nb > 0 else 0.0
        if oid_i == oid_j:
            print(f'{"   -":>16s}', end='')
        else:
            print(f'{cos:16.3f}', end='')
    print()

# All pairs
pairs = []
for i, oid_i in enumerate(ids):
    for j, oid_j in enumerate(ids):
        if i >= j:
            continue
        wi, wj = genomes[oid_i], genomes[oid_j]
        cos = np.dot(wi, wj) / (np.linalg.norm(wi) * np.linalg.norm(wj))
        pairs.append((oid_i, oid_j, cos))
pairs.sort(key=lambda x: -x[2])

print(f'\n=== 摘要 ===')
print(f'总对数: {len(pairs)} (C(14,2)=91)')
oi, oj = pairs[0][0], pairs[0][1]
print(f'最高: {short(oi)}(G{gen_info[oi]["gen"]}) <-> {short(oj)}(G{gen_info[oj]["gen"]}) = {pairs[0][2]:.4f}')
oi, oj = pairs[-1][0], pairs[-1][1]
print(f'最低: {short(oi)}(G{gen_info[oi]["gen"]}) <-> {short(oj)}(G{gen_info[oj]["gen"]}) = {pairs[-1][2]:.4f}')
print(f'均值: {np.mean([p[2] for p in pairs]):.4f}')
print(f'中位: {np.median([p[2] for p in pairs]):.4f}')
print(f'标准差: {np.std([p[2] for p in pairs]):.4f}')

bins = [(0,0.2), (0.2,0.4), (0.4,0.6), (0.6,0.8), (0.8,1.0)]
print('\n分布:')
for lo, hi in bins:
    count = sum(1 for _,_,c in pairs if lo <= c < hi)
    bar = '#' * count
    print(f'  [{lo}-{hi}): {count:3d} {bar}')

print('\nTop 10:')
for k, (oid_i, oid_j, cos) in enumerate(pairs[:10]):
    gi, gj = gen_info[oid_i]['gen'], gen_info[oid_j]['gen']
    print(f'  #{k+1}: {short(oid_i)}(G{gi}) <-> {short(oid_j)}(G{gj}) = {cos:.4f}')

# Family (>0.6)
families = defaultdict(list)
for oid_i, oid_j, cos in pairs:
    if cos > 0.6:
        families[oid_i].append(oid_j)
        families[oid_j].append(oid_i)

visited = set()
family_list = []
for oid in ids:
    if oid in visited:
        continue
    if oid not in families:
        family_list.append([oid])
        visited.add(oid)
        continue
    queue = [oid]
    fam = []
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        fam.append(cur)
        for nb in families.get(cur, []):
            if nb not in visited:
                queue.append(nb)
    family_list.append(fam)

print(f'\n家族 (cos>0.6连通): {len(family_list)} 个')
for i, fam in enumerate(family_list):
    fshort = [s + '(G' + str(gen_info[s]["gen"]) + ')' for s in fam]
    print(f'  Fam{i+1} ({len(fam)}人): {", ".join(fshort)}')
