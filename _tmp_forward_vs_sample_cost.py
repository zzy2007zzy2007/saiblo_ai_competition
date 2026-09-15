"""成本模型：一次搜索的开销是在"前向次数"还是在"采样次数"上？

速度实测给出两条方程（每局我方回合）：joint 257F+17948S=850ms、
pos-only 257F+8712S=748ms、skip-hold 48.2F+4860S=178ms。
解出来 F≈2.5ms、S≈11µs（且能同时命中 skip-hold 那条，说明模型对）。
这里直接单测，确认 F、S 以及 F 内部（观测编码 vs 网络前向）各占多少。
"""
from __future__ import annotations

import statistics
import sys
import time
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
from my_ai.az_intent.bundle_mcts import sample_bundle
from my_ai.az_intent import bundle_mcts as bm
from my_ai.decoder import make_class_mask, make_position_masks
from my_ai.network import create_model, model_kwargs_from_ckpt

CK = "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
N_FWD = 300
N_SAMP = 360

feat = FeatureExtractor(max_actions=96)
ckpt = torch.load(CK, map_location="cpu", weights_only=False)
model = create_model(**model_kwargs_from_ckpt(ckpt))
model.load_state_dict(ckpt["model_state"])
model.eval()

state = make_initial_state(3, True)
for _ in range(40):
    for pl in (0, 1):
        if state.terminal:
            break
        obs = feat.encode_observation(state, pl, np.zeros(96))
        with torch.no_grad():
            out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                        torch.from_numpy(obs["stats"]).unsqueeze(0).float())
        from my_ai.decoder import decode_network_output
        state.apply_operation_list(pl, decode_network_output(out, state, pl,
                                                             temperature=0.0,
                                                             intent_decoding=True))
    if pl == 1 and not state.terminal:
        state.advance_round()

player = 0


def enc():
    return feat.encode_observation(state, player, np.zeros(96))


def fwd(obs):
    b = torch.from_numpy(obs["board"]).unsqueeze(0).float()
    s = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
    with torch.no_grad():
        return model(b, s)


# warmup
for _ in range(20):
    fwd(enc())

enc_t, fwd_t, full_t = [], [], []
for _ in range(N_FWD):
    t0 = time.perf_counter(); o = enc(); enc_t.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); out = fwd(o); fwd_t.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    o2 = enc(); out2 = fwd(o2)
    full_t.append(time.perf_counter() - t0)

print(f"encode_observation : {statistics.median(enc_t) * 1e3:7.3f} ms")
print(f"model forward      : {statistics.median(fwd_t) * 1e3:7.3f} ms")
print(f"net_fn 合计        : {statistics.median(full_t) * 1e3:7.3f} ms  "
      f"  <-- 这就是一次前向 F")

heads = sorted(k for k in out2 if k.startswith("head") and k.endswith("_logits"))
net_out = {"action_map": out2["action_map"][0].numpy(),
           "head_logits": [out2[h][0].numpy() for h in heads],
           "value": float(out2["value"].item())}
pm = make_position_masks(state, player, intent_decoding=True)
cm = make_class_mask(state, player, position_mask=pm, intent_decoding=True)

for tag, mode, pin, tpos in (("joint", "joint", "argmax", 0.3),
                             ("pos-only(argmax)", "pos-only", "argmax", 0.3),
                             ("class-only", "class-only", "argmax", 0.0)):
    m = bm.BundleMCTS(lambda *_: net_out, iterations=1, k=24, search_mode=mode,
                      pos_pin=pin, t_pos=tpos, seed=1)
    # replicate _expand's candidate generation once to time 360 sample_bundle calls
    rng = np.random.default_rng(7)
    cps = [bm.head_class_probs(hl, 0.5) for hl in net_out["head_logits"]]
    cids = [rng.choice(len(p), size=N_SAMP, p=p).tolist() for p in cps]
    t0 = time.perf_counter()
    for i in range(N_SAMP):
        sample_bundle(net_out, state, player, t_class=0.5, t_pos=tpos, rng=rng,
                      position_mask=pm, class_mask=cm,
                      class_probs=(None if mode == "pos-only" else cps),
                      class_ids=[c[i] for c in cids])
    dt = time.perf_counter() - t0
    print(f"360×sample_bundle [{tag:16s}]: {dt * 1e3:7.2f} ms  "
          f"({dt / N_SAMP * 1e6:5.2f} µs/次)")

print("\n--- 用这些数推 256 次迭代的搜索成本 ---")
F = statistics.median(full_t)
print(f"257 次前向 = {257 * F * 1e3:.0f} ms")
sys.stdout.flush()
