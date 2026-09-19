import json

BASE = '/root/autodl-tmp/AntWar/ppo_v10/outputs/20260615_130637'

with open(BASE + '/league/payoff.json') as f:
    payoff = json.load(f)
data = payoff['data']

ts_seeds = {
    16: ['gen_016_ind_010', 'gen_016_ind_018', 'gen_016_ind_040'],
    17: ['gen_017_ind_014', 'gen_017_ind_026', 'gen_017_ind_053'],
    18: ['gen_018_ind_006', 'gen_018_ind_007', 'gen_018_ind_018'],
    19: ['gen_019_ind_004', 'gen_019_ind_018', 'gen_019_ind_024'],
    20: ['gen_020_ind_000', 'gen_020_ind_003', 'gen_020_ind_011'],
    21: ['gen_021_ind_010', 'gen_021_ind_013', 'gen_021_ind_016'],
    22: ['gen_022_ind_003', 'gen_022_ind_011', 'gen_022_ind_018'],
    23: ['gen_023_ind_004', 'gen_023_ind_010', 'gen_023_ind_046'],
    24: ['gen_024_ind_006', 'gen_024_ind_011', 'gen_024_ind_028'],
}

for gen_num in sorted(ts_seeds.keys()):
    seeds = ts_seeds[gen_num]
    
    try:
        with open(BASE + '/generations/gen_%04d/battle_results.json' % gen_num) as f:
            br = json.load(f)
        rankings = br.get('rankings', [])
        ts_mu = {r['id']: r['trueskill_mu'] for r in rankings if 'trueskill_mu' in r}
        sorted_ids = sorted(ts_mu, key=lambda p: -ts_mu[p])
        rank_map = {pid: r+1 for r, pid in enumerate(sorted_ids)}
    except Exception as e:
        print('Gen%d: ERROR %s' % (gen_num, e))
        continue
    
    print('=' * 90)
    print('Gen%d (population=%d)' % (gen_num, len(sorted_ids)))
    print('-' * 60)
    
    cw_ids = []; cl_ids = []; st_ids = []
    
    for pid in sorted_ids:
        if pid in seeds:
            continue
        results = []; ok = True
        for sid in seeds:
            b = data.get(sid + '-' + pid, {})
            w = b.get('wins', 0); l = b.get('losses', 0)
            if w + l < 4: ok = False; break
            results.append((w, l))
        if not ok:
            continue
        if all(l == 0 for w, l in results):
            cw_ids.append(pid)
        elif all(w == 0 for w, l in results):
            cl_ids.append(pid)
        else:
            st_ids.append(pid)
    
    total = len(cw_ids) + len(cl_ids) + len(st_ids)
    if total == 0:
        print('no data')
        continue
    
    print('完胜(%d): %s' % (len(cw_ids), ', '.join(p.split('_')[-1] for p in cw_ids)))
    print('完败(%d): %s' % (len(cl_ids), ', '.join(p.split('_')[-1] for p in cl_ids)))
    print('可排序(%d/%.1f%%)' % (len(st_ids), len(st_ids)/total*100))
    print()
    print('--- 种子 vs 种群 ---')
    print('%-10s  排名  完胜 完败 可排序  mu' % 'Seed')
    print('-' * 44)
    
    for sid in seeds:
        sw = 0; sl = 0; ss = 0
        for opp in sorted_ids:
            if opp == sid: continue
            b = data.get(sid + '-' + opp, {})
            w = b.get('wins', 0); l = b.get('losses', 0)
            if w + l < 4: continue
            if l == 0: sw += 1
            elif w == 0: sl += 1
            else: ss += 1
        short = sid.split('_', 3)[-1] if '_' in sid else sid
        r = rank_map.get(sid, '?')
        mu = ts_mu.get(sid, 0)
        print('%-10s  %3s   %3d  %3d   %3d   %.2f' % (short, str(r), sw, sl, ss, mu))
    
    print()
