"""pos-only(playable) 为什么 0/32？只看不打搜：统计它实际会下什么。

对同一批真实局面，逐头比较：
  raw 的原始 argmax 类 / 掩码后 argmax 类 / playable_class（第一个真能执行的类）
并统计 playable 钉法在"位置 argmax"下产出的操作类型分布，与 raw 对比。
关键是看：强制可执行类到底把哪些动作塞进了棋盘（尤其 DOWNGRADE 的占比）。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
for p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch.set_num_threads(1)

from SDK.utils.features import FeatureExtractor
from SDK.utils.constants import OperationType
from my_ai.az_intent.az_selfplay import make_initial_state
from my_ai.az_intent.bundle_mcts import playable_class
from my_ai.decoder import (decode_network_output, decode_head, make_class_mask,
                           make_position_masks)
from my_ai.network import create_model, model_kwargs_from_ckpt

CK = "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
N_STATES = 120

feat = FeatureExtractor(max_actions=96)
ckpt = torch.load(CK, map_location="cpu", weights_only=False)
model = create_model(**model_kwargs_from_ckpt(ckpt))
model.load_state_dict(ckpt["model_state"])
model.eval()


def net_out(st, player):
    obs = feat.encode_observation(st, player, np.zeros(96))
    with torch.no_grad():
        out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                    torch.from_numpy(obs["stats"]).unsqueeze(0).float())
    heads = sorted(k for k in out if k.startswith("head") and k.endswith("_logits"))
    return {"action_map": out["action_map"][0].numpy(),
            "head_logits": [out[h][0].numpy() for h in heads]}


# ── gather real states from a raw self-play game ────────────────────────────
state = make_initial_state(3, True)
snaps, turn = [], 0
for _ in range(600):
    if state.terminal:
        break
    for player in (0, 1):
        if state.terminal:
            break
        turn += 1
        if turn % 3 == 0 and len(snaps) < N_STATES:
            snaps.append((state.clone(), player))
        obs = feat.encode_observation(state, player, np.zeros(96))
        with torch.no_grad():
            out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                        torch.from_numpy(obs["stats"]).unsqueeze(0).float())
        state.apply_operation_list(
            player, decode_network_output(out, state, player, temperature=0.0,
                                          intent_decoding=True))
    if player == 1 and not state.terminal:
        state.advance_round()

name = {int(t): t.name for t in OperationType}
raw_types, pin_types = Counter(), Counter()
raw_empty = 0
forced = 0                      # raw 空手、playable 钉法却出招
pin_head_classes = Counter()    # 每个头钉到的类
pin_down = 0                    # 钉法产生 DOWNGRADE 的回合数
raw_hold_all = 0

for st, pl in snaps:
    no = net_out(st, pl)
    pm = make_position_masks(st, pl, intent_decoding=True)
    cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
    raw_ops = decode_network_output(dict(no), st, pl, temperature=0.0, intent_decoding=True)
    for o in raw_ops:
        raw_types[name[int(o.op_type)]] += 1
    if not raw_ops:
        raw_empty += 1

    pinned, argmaxes = [], []
    for h in range(3):
        hl = no["head_logits"][h]
        argmaxes.append(int(np.argmax(hl)))
        c = playable_class(hl, no["action_map"], cm, pm, st, pl)
        pinned.append(c)
        pin_head_classes[c] += 1
    if all(a == 23 for a in argmaxes):
        raw_hold_all += 1

    ops = []
    pending = ()
    for h in range(3):
        op = decode_head(no["head_logits"][h], no["action_map"], cm.copy(),
                         pm.copy(), st, pl, class_id=pinned[h],
                         temperature=0.0, pos_temperature=0.0, intent_decoding=True)
        if op is None or not st.can_apply_operation(pl, op, pending):
            continue
        ops.append(op)
        pending = pending + (op,)
    for o in ops:
        pin_types[name[int(o.op_type)]] += 1
        if o.op_type == OperationType.DOWNGRADE_TOWER:
            pin_down += 1
    if not raw_ops and ops:
        forced += 1

n = len(snaps)
print(f"states={n}")
print(f"raw 空手的回合: {raw_empty}/{n} ({raw_empty / n:.1%})   "
      f"三头原始 argmax 全 HOLD: {raw_hold_all}/{n} ({raw_hold_all / n:.1%})")
print(f"raw 空手但 playable 钉法出招: {forced}/{n} ({forced / n:.1%})")
print(f"playable 钉法产生 DOWNGRADE 的回合: {pin_down}/{n} ({pin_down / n:.1%})")
print("\nraw 的操作类型分布:", dict(raw_types.most_common()))
print("playable 钉法的操作类型分布:", dict(pin_types.most_common()))
print("\nplayable 钉法每个头钉到的类（前 12）:")
tot = sum(pin_head_classes.values())
for c, k in pin_head_classes.most_common(12):
    print(f"   class {c:2d}: {k:5d} ({k / tot:.1%})")
print("\nsnap states side split:",
      Counter(pl for _, pl in snaps))
sys.stdout.flush()
