import torch, itertools, sys
from pathlib import Path
CK = {
 "posnet_A_k5_m32": "training_history/vprior/posnet_A_k5_m32.pt",
 "valnet_A2": "training_history/inject_ex02/valnet_A2.pt",
 "three_mix_frozen": "training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt",
 "mix_frozen": "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt",
}
st = {}
for k, p in CK.items():
    if not Path(p).is_file():
        print(f"  (缺) {k}: {p}"); continue
    d = torch.load(p, map_location="cpu")
    st[k] = d
    print(f"  {k}: keys={sorted([x for x in d.keys() if x.endswith('_state')])} "
          f"meta_keys={sorted([x for x in d.keys() if not x.endswith('_state')])[:8]}")
def eq(a, b, state):
    if a not in st or b not in st: return None
    x, y = st[a].get(state), st[b].get(state)
    if x is None or y is None: return None
    if set(x.keys()) != set(y.keys()): return "keys differ"
    bad = [k for k in x if not torch.equal(x[k], y[k])]
    return f"{len(x)-len(bad)}/{len(x)} tensors identical" + ("" if not bad else f" (differ: {bad[:3]})")
print()
for s in ("class_state", "pos_state", "value_state"):
    for a, b in itertools.combinations([k for k in CK if k in st], 2):
        r = eq(a, b, s)
        if r: print(f"  {s:12s} {a} vs {b}: {r}")
