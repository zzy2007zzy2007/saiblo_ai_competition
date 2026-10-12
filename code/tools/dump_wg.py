"""Dump WinGraph from checkpoint."""
import sys, torch
from collections import Counter, defaultdict

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
wg = ckpt["win_graph"]

nodes = wg["nodes"]
print(f"config: n_max={wg['config']['n_max']} edge_games={wg['config']['edge_games']} threshold={wg['config']['win_threshold']}")
print(f"nodes: {len(nodes)}")
print(f"edges: {len(wg['edges'])}")
print()

# Node details: key = gen*2 (mean) or gen*2+1 (top1)
for nid in sorted(nodes.keys(), key=int):
    nd = nodes[nid]
    gen = int(nid) // 2
    kind = "mean" if int(nid) % 2 == 0 else "top1"
    ex = nd.get("example_win_rate", "?")
    print(f"  node {int(nid):2d} = gen_{gen:2d} {kind:4s}  ex_wr={ex}")

print()
print("edges (win_rate from -> to):")
by_src = defaultdict(list)
for e in wg["edges"]:
    by_src[e["from"]].append(e)
for src in sorted(by_src.keys(), key=int):
    for e in sorted(by_src[src], key=lambda x: x["to"]):
        print(f"  {int(e['from']):2d} -> {int(e['to']):2d}  wr={e['wr']:.3f}  games={e['games']}")

print()
print("wr distribution:")
buckets = Counter()
for e in wg["edges"]:
    b = int(e["wr"] * 10)
    buckets[b] += 1
for b in sorted(buckets):
    print(f"  [{b/10:.1f}, {(b+1)/10:.1f}): {buckets[b]} edges")
