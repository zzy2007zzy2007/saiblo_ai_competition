"""Why does pos-only collapse? Print per-head class argmax / legality / sampled mass."""
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
from my_ai.az_intent.az_selfplay import make_initial_state
from my_ai.az_intent.bundle_mcts import head_class_probs, sample_bundle
from my_ai.decoder import make_class_mask, make_position_masks
from my_ai.network import create_model, model_kwargs_from_ckpt

CK = "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
SEED = 3

feat = FeatureExtractor(max_actions=96)
ckpt = torch.load(CK, map_location="cpu", weights_only=False)
model = create_model(**model_kwargs_from_ckpt(ckpt))
model.load_state_dict(ckpt["model_state"])
model.eval()

state = make_initial_state(SEED, True)
player = 0
obs = feat.encode_observation(state, player, np.zeros(96))
with torch.no_grad():
    out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                torch.from_numpy(obs["stats"]).unsqueeze(0).float())
heads = sorted(k for k in out if k.startswith("head") and k.endswith("_logits"))
net_out = {
    "action_map": out["action_map"][0].numpy(),
    "head_logits": [out[h][0].numpy() for h in heads],
    "value": float(out["value"].item()),
}

pm = make_position_masks(state, player, intent_decoding=True)
cm = make_class_mask(state, player, position_mask=pm, intent_decoding=True)
print("class_mask legal classes:", [i for i in range(24) if cm[i]])
for h in range(3):
    hl = net_out["head_logits"][h]
    am = int(np.argmax(hl))
    probs = head_class_probs(hl, 0.5)
    top = np.argsort(-probs)[:6]
    print(f"\nhead{h}: argmax={am} legal={bool(cm[am])}  "
          f"logit[min/mean/max]={hl.min():.3f}/{hl.mean():.3f}/{hl.max():.3f}")
    print(f"   t=0.5 top6 probs: " +
          ", ".join(f"c{int(i)}={probs[i]:.3f}{'(legal)' if cm[i] else ''}" for i in top))
    legal_mass = sum(probs[i] for i in range(24) if cm[i])
    print(f"   mass on legal classes = {legal_mass:.3f}")

# what does raw decode do?
from my_ai.decoder import decode_network_output
raw_ops = decode_network_output(dict(net_out), state, player, temperature=0.0, intent_decoding=True)
print("\nraw ops:", [(int(o.op_type), o.arg0, o.arg1) for o in raw_ops])

# what does joint sampling pick most often?
rng = np.random.default_rng(SEED)
class_probs = [head_class_probs(hl, 0.5) for hl in net_out["head_logits"]]
class_ids = [rng.choice(len(p), size=360, p=p).tolist() for p in class_probs]
from collections import Counter
agg: Counter = Counter()
for idx in range(360):
    ops, intents = sample_bundle(net_out, state, player, t_class=0.5, t_pos=0.3, rng=rng,
                                 position_mask=pm, class_mask=cm,
                                 class_probs=class_probs,
                                 class_ids=[cid[idx] for cid in class_ids])
    agg[tuple((int(o.op_type), o.arg0, o.arg1) for o in ops)] += 1
print("\njoint top-8 sampled keys:")
for kk, c in agg.most_common(8):
    print(f"   {c:4d}  {list(kk)}")
print("\nsampled class per head (head0 first 15):", class_ids[0][:15])
print("sampled class per head (head1 first 15):", class_ids[1][:15])
print("sampled class per head (head2 first 15):", class_ids[2][:15])

# pos-only with MASKED argmax instead of raw argmax -- does it have room?
masked_arg = [int(np.argmax(np.where(cm, hl, -1e9))) for hl in net_out["head_logits"]]
print("\nmasked argmax per head:", masked_arg, " legal:",
      [bool(cm[a]) for a in masked_arg])


def playable_argmax(hl, cm, state, player):
    """First class in descending logit order that decodes to a real (non-None) op."""
    from my_ai.decoder import decode_head
    order = np.argsort(-np.asarray(hl, dtype=np.float32))
    for cid in order[:8]:
        op = decode_head(hl, net_out["action_map"], cm, pm, state, player,
                         temperature=0.0, intent_decoding=True,
                         class_id=int(cid), pos_temperature=0.0)
        if op is not None:
            return int(cid)
    return None


print("playable argmax per head:",
      [playable_argmax(hl, cm, state, player) for hl in net_out["head_logits"]])

from my_ai.decoder import decode_head
for attempt in ("raw-argmax", "masked-argmax", "playable-argmax", "class11"):
    if attempt == "raw-argmax":
        cid = [int(np.argmax(hl)) for hl in net_out["head_logits"]]
    elif attempt == "masked-argmax":
        cid = masked_arg
    elif attempt == "playable-argmax":
        cid = [playable_argmax(hl, cm, state, player) or 23
               for hl in net_out["head_logits"]]
    else:
        cid = [11, 11, 11]
    rng2 = np.random.default_rng(SEED)
    keys = Counter()
    for _ in range(360):
        ops, _ = sample_bundle(net_out, state, player, t_class=0.5, t_pos=0.3, rng=rng2,
                               position_mask=pm, class_mask=cm, class_ids=cid)
        keys[tuple((int(o.op_type), o.arg0, o.arg1) for o in ops)] += 1
    print(f"pos-only({attempt}) pinned={cid}: {len(keys)} distinct candidate(s), top3="
          f"{[[list(kk), c] for kk, c in keys.most_common(3)]}")
