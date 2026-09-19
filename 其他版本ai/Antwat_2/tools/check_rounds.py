import json, sys

br_file = sys.argv[1]
with open(br_file) as f:
    br = json.load(f)
print('Keys:', list(br.keys()))
if 'rounds_data' in br:
    print('rounds_data len:', len(br['rounds_data']))
    # Count multi-action rounds for agent1 (GA)
    rounds_data = br['rounds_data']
    multi_ga = 0
    ga_act = 0
    for r in rounds_data:
        o1 = r.get('agent1_ops', [])
        n1 = 0 if (len(o1)==1 and o1[0].get('type')=='NOOP') else len(o1)
        if n1 > 0: ga_act += 1
        if n1 > 1:
            multi_ga += 1
            types = [op.get('type','?') for op in o1]
            print('  MULTI-GA: R%s: %s' % (r.get('round','?'), types))
    print('GA active rounds: %d/%d, multi-action: %d' % (ga_act, len(rounds_data), multi_ga))
else:
    print('No rounds_data')
    for k in br:
        v = br[k]
        if isinstance(v, list): print('%s: list[%d]' % (k, len(v)))
        elif isinstance(v, dict): print('%s: dict keys=%s' % (k, list(v.keys())[:5]))
        else: print('%s: %s' % (k, type(v).__name__))
