"""Phase 0 闸门：三网 net_fn 必须与二网 net_fn **逐位等价**。

三项检查：
  A. 加载器守卫：三网 ckpt 喂给 load_model_from_ckpt / load_split_models 必须**报错**
     （而不是静默返回没训过的 action_map）。
  B. 输出层逐位：同一批真实局面下，两个 net_fn 的 action_map / head_logits / value
     必须逐位相同（value 用精确相等比较）。
  C. 对局层：见 _tmp_position_choice_ab.py（同一脚本跑两个 ckpt，比较汇总数字）。

B 通过 ⇒ 搜索在相同 RNG 下会产生完全相同的候选 ⇒ 对局必然逐局相同（C 是独立复核）。
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
from my_ai.az_intent.az_selfplay import (load_model_from_ckpt, load_split_models,
                                         make_initial_state, make_net_fn_from_ckpt)
from my_ai.decoder import decode_network_output
from my_ai.network import create_model, model_kwargs_from_ckpt

CK2 = "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"          # 二网（现行）
CK3 = "training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt"    # 三网（新拆）

# ── A. 加载器守卫 ────────────────────────────────────────────────────────────
print("=== A. 加载器守卫 ===")
for name, fn in (("load_model_from_ckpt", load_model_from_ckpt),
                 ("load_split_models", load_split_models)):
    try:
        fn(CK3)
        print(f"  !! {name}(三网 ckpt) 竟然没报错 —— 守卫失效")
        sys.exit(1)
    except ValueError as e:
        print(f"  ✅ {name} 正确报错: {str(e)[:80]}...")
# 反过来：二网 ckpt 仍能正常加载
load_model_from_ckpt(CK2); load_split_models(CK2)
print("  ✅ 二网 ckpt 仍可被两个旧加载器读取（未破坏向后兼容）")

# ── B. 输出层逐位比较 ───────────────────────────────────────────────────────
print("\n=== B. net_fn 输出逐位比较 ===")
feat = FeatureExtractor(max_actions=96)
anchor2, netfn2 = make_net_fn_from_ckpt(CK2, feat)
anchor3, netfn3 = make_net_fn_from_ckpt(CK3, feat)
print(f"  二网 anchor 类型: {type(anchor2).__name__}   三网 anchor 类型: {type(anchor3).__name__}")

# 收集真实局面（raw 自对弈）
model = load_model_from_ckpt(CK2)
state = make_initial_state(3, True)
states, turn = [], 0
for _ in range(700):
    if state.terminal:
        break
    for pl in (0, 1):
        if state.terminal:
            break
        turn += 1
        if turn % 2 == 0 and len(states) < 80:
            states.append((state.clone(), pl))
        obs = feat.encode_observation(state, pl, np.zeros(96))
        with torch.no_grad():
            out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                        torch.from_numpy(obs["stats"]).unsqueeze(0).float())
        state.apply_operation_list(pl, decode_network_output(out, state, pl,
                                                            temperature=0.0,
                                                            intent_decoding=True))
    if pl == 1 and not state.terminal:
        state.advance_round()
print(f"  局面数 = {len(states)}")

n_am = n_hl = n_v = 0
worst_am = worst_hl = 0.0
for st, pl in states:
    a = netfn2(st, pl)
    b = netfn3(st, pl)
    am_same = np.array_equal(a["action_map"], b["action_map"])
    hl_same = all(np.array_equal(x, y) for x, y in zip(a["head_logits"], b["head_logits"]))
    v_same = a["value"] == b["value"]
    n_am += int(am_same); n_hl += int(hl_same); n_v += int(v_same)
    worst_am = max(worst_am, float(np.abs(a["action_map"] - b["action_map"]).max()))
    worst_hl = max(worst_hl, max(float(np.abs(x - y).max())
                                 for x, y in zip(a["head_logits"], b["head_logits"])))
N = len(states)
print(f"  action_map  逐位相同: {n_am}/{N}   (最大差 {worst_am:.3e})")
print(f"  head_logits 逐位相同: {n_hl}/{N}   (最大差 {worst_hl:.3e})")
print(f"  value       精确相等: {n_v}/{N}")

ok = (n_am == N and n_hl == N and n_v == N)
print(f"\nPhase 0 闸门: {'✅ 通过（三网与二网逐位等价）' if ok else '❌ 未通过，不许往下走'}")
sys.exit(0 if ok else 1)
