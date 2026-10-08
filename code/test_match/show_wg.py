"""Show WinGraph contents from a checkpoint."""
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
wg = ckpt["win_graph"]
print("Config:", wg["config"])
print(f"\nNodes ({len(wg['nodes'])}):")
for gk in sorted(wg["nodes"], key=int):
    g = int(gk)
    n = wg["nodes"][gk]
    wer = n.get("example_win_rate")
    print(f"  gen={g:2d}  example_win_rate={wer}")
print(f"\nEdges ({len(wg['edges'])}):")
for entry in wg["edges"]:
    a, b, wa, wb, d = entry
    total = wa + wb + d
    wr = wa / total if total > 0 else 0.5
    print(f"  {a:2d} -> {b:2d}  wr={wr:.2f}  ({wa}/{wb}/{d})  edge_games={total}")
