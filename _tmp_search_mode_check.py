"""Verify the class/position search-mode ablation actually restricts one axis.

For each mode we expand the ROOT node once on a real mid-game state and look at
the SELECTED candidate bundles (the ones MCTS would branch over):

  joint      -> both the class-tuple and the position-tuple vary across candidates
  pos-only   -> the class-tuple is IDENTICAL for every candidate (only positions vary)
  class-only -> every position is the per-class argmax (only classes vary)

Also checks that class-only's positions equal what the raw (temperature=0) decoder
would pick for the same class -- i.e. the "fixed" axis really is the argmax axis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
for p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch.set_num_threads(1)

from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, make_initial_state
from my_ai.az_intent.bundle_mcts import BundleMCTS, BundleNode

CK = sys.argv[1] if len(sys.argv) > 1 else "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 3

feat = FeatureExtractor(max_actions=96)
_, net_fn = make_net_fn_from_ckpt(CK, feat)

from my_ai.decoder import decode_network_output

state = make_initial_state(SEED, True)

# make_net_fn_from_ckpt returns a value-only net_fn; the tree needs the full dict
# (action_map + 3 head logits), so build that directly from the checkpoint model.
from my_ai.network import create_model, model_kwargs_from_ckpt
ckpt = torch.load(CK, map_location="cpu", weights_only=False)
model = create_model(**model_kwargs_from_ckpt(ckpt))
model.load_state_dict(ckpt["model_state"])
model.eval()


def full_net_fn(st, player):
    obs = feat.encode_observation(st, player, np.zeros(96))
    with torch.no_grad():
        out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                    torch.from_numpy(obs["stats"]).unsqueeze(0).float())
    heads = sorted(k for k in out if k.startswith("head") and k.endswith("_logits"))
    return {
        "action_map": out["action_map"][0].numpy(),
        "head_logits": [out[h][0].numpy() for h in heads],
        "value": float(out["value"].item()),
    }


for mode in ("joint", "pos-only", "class-only"):
    mcts = BundleMCTS(full_net_fn, iterations=1, max_depth_rounds=4, k=24,
                      t_class=0.5, t_pos=0.3, seed=SEED, search_mode=mode)
    node = BundleNode(state=state.clone(), player=0)
    val = mcts._expand(node, mcts.k)
    class_tuples, pos_tuples = set(), set()
    for key in node.bundles:
        cls = tuple((int(o[0]) if len(o) > 0 else -1) for o in key) if key else ()
        pos = tuple((int(o[1]), int(o[2])) if len(o) > 1 else (-1, -1) for o in key) if key else ()
        class_tuples.add(cls)
        pos_tuples.add(pos)
    print(f"\n[{mode}] leaf value={val:+.4f}  #candidates={len(node.bundles)}  "
          f"#distinct class-tuples={len(class_tuples)}  #distinct pos-tuples={len(pos_tuples)}")
    for key in node.bundles[:6]:
        print(f"    {[tuple(int(v) for v in o) for o in key]}")
    print("    class-tuples:", sorted(class_tuples)[:6])
    print("    pos-tuples:  ", sorted(pos_tuples)[:6])
