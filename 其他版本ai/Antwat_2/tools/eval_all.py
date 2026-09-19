import json, os, sys

BASE = sys.argv[1]
for g in range(1, 200):
    gdir = BASE + '/gen_%04d' % g
    ev_file = gdir + '/evaluation.json'
    if not os.path.exists(ev_file):
        break
    with open(ev_file) as f:
        ev = json.load(f)
    print('===== Gen%d =====' % g)
    for sid, e in sorted(ev.items()):
        for agent_name, stats in e.items():
            if agent_name.startswith('_'):
                continue
            a1w = stats.get('agent1_wins', 0)
            a2w = stats.get('agent2_wins', 0)
            draws = stats.get('draws', 0)
            ar = stats.get('avg_rounds', 0)
            total = stats.get('total_battles', a1w + a2w + draws)
            short = sid.split('_', 3)[-1] if len(sid.split('_')) > 3 else sid
            flag = ' ***ZZY!!!***' if a1w > 0 else ''
            print('  %s vs %s: %dW/%dL/%dD (%.1fr, %d battles)%s' % (
                short, agent_name, a1w, a2w, draws, ar, total, flag))
    print()
