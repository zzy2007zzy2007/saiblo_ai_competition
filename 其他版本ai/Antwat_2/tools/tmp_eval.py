import json, os
BASE = '/root/autodl-tmp/AntWar/ppo_v10/outputs/20260616_215320/generations'
for g in [8]:
    gdir = BASE + '/gen_%04d' % g
    with open(gdir + '/evaluation.json') as f: ev = json.load(f)
    mu_map = {}
    try:
        with open(gdir + '/battle_results.json') as f:
            for s in json.load(f).get('seeds', []): mu_map[s['id']] = s.get('trueskill_mu', 0)
    except: pass
    print('===== Gen%d =====' % g)
    for sid, e in ev.items():
        zzy = e.get('rule_zzy25', {})
        a1w = zzy.get('agent1_wins', 0); a2w = zzy.get('agent2_wins', 0)
        ar = zzy.get('avg_rounds', 0)
        short = sid.split('_', 3)[-1]
        mu = mu_map.get(sid, 0)
        flag = ' ***ZZY!!!***' if a1w > 0 else ''
        print('  %s (mu=%.2f): %dW/%dL %.1fr%s' % (short, mu, a1w, a2w, ar, flag))
