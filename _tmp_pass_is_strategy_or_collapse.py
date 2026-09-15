"""97% 空过：是"策略判断该空过"还是"类头/解码器坏了"？三个便宜测量。

(a) 1-ply 价值比较：在空过回合上，比较
       v_pass = -net_fn(state, 1-pl)          ← 空过分支（与搜索里空 bundle 子节点同约定）
       v_act(c) = -net_fn(post_c, 1-pl)       ← 每个合法类取其 argmax 位置后执行
    若价值头**系统性偏好出招**而策略却空过 ⇒ 类头/解码器是瓶颈（决定性）。
    若价值头也偏好空过 ⇒ **不能下结论**（既可能空过真合理，也可能价值头同盲区）。

(b) 空过回合的成因拆解：每个头的 raw argmax 是 HOLD？还是非法类（解码器不回退 ⇒ 空过）？
    以及 HOLD 的 logit 比"最好的合法非 HOLD 类"高多少（margin 小 ⇒ 这个空过很脆弱）。

(c) 外部裁判：rule_v4（比我们强）在**我们空过的那些回合**上是否出招？
    并与它的整体出招率对比 —— 只能说明"异常/正常"，不能证明"出招更好"。
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent
for p in (_REPO / "Ant-Game", _REPO / "code", _REPO / "其他版本ai" / "rule_v4"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

torch.set_num_threads(1)

from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.az_selfplay import make_initial_state
from my_ai.az_intent.bundle_mcts import BundleMCTS
from my_ai.decoder import (decode_head, decode_network_output, make_class_mask,
                           make_position_masks)
from my_ai.network import create_model, model_kwargs_from_ckpt

CK = "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
GAMES = (3, 11)
WANT = 110
HOLD = 23

feat = FeatureExtractor(max_actions=96)
ckpt = torch.load(CK, map_location="cpu", weights_only=False)
model = create_model(**model_kwargs_from_ckpt(ckpt))
model.load_state_dict(ckpt["model_state"])
model.eval()


def net_out_of(st, player):
    obs = feat.encode_observation(st, player, np.zeros(96))
    with torch.no_grad():
        out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                    torch.from_numpy(obs["stats"]).unsqueeze(0).float())
    heads = sorted(k for k in out if k.startswith("head") and k.endswith("_logits"))
    return {"action_map": out["action_map"][0].numpy(),
            "head_logits": [out[h][0].numpy() for h in heads],
            "value": float(out["value"].item())}


def net_fn(st, player):
    return net_out_of(st, player)


def collect(seed: int, want: int) -> list[tuple]:
    state = make_initial_state(seed, True)
    snaps, turn = [], 0
    for _ in range(700):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            turn += 1
            if turn % 2 == 0 and len(snaps) < want:
                snaps.append((state.clone(), player))
            obs = feat.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            state.apply_operation_list(player, decode_network_output(
                out, state, player, temperature=0.0, intent_decoding=True))
        if player == 1 and not state.terminal:
            state.advance_round()
    return snaps


from ai import AI as RuleAI

SNAPS: list[tuple] = []
for g in GAMES:
    SNAPS += collect(g, WANT)
print(f"states = {len(SNAPS)} (raw self-play seeds {GAMES}, stride 2)", flush=True)

rule_ai = RuleAI(seed=3)

# ── 先分类：pos-only 下这一回合是空过还是有选择 ────────────────────────────────
pass_turns, act_turns = [], []
for i, (st, pl) in enumerate(SNAPS):
    m = BundleMCTS(net_fn, iterations=256, max_depth_rounds=4, k=24, t_class=0.5,
                   t_pos=0.3, seed=1000 + i, search_mode="pos-only",
                   skip_single_candidate=True)
    res = m.search(st, pl, temperature=0.0)
    nb = len(res.bundles)
    (pass_turns if (nb == 0 or (nb == 1 and len(res.bundles[0]) == 0)) else act_turns).append((st, pl))
print(f"pass turns = {len(pass_turns)}   act/choice turns = {len(act_turns)}", flush=True)

# ── (a) 1-ply 价值比较（只在 pass 回合上）────────────────────────────────────
better = worse = equal = 0
margins = []
detail = Counter()
for st, pl in pass_turns:
    opp = 1 - pl
    v_pass = -net_out_of(st, opp)["value"]
    pm = make_position_masks(st, pl, intent_decoding=True)
    cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
    no = net_out_of(st, pl)
    best, best_cls = -9.9, None
    for c in range(24):
        if c == HOLD or not cm[c]:
            continue
        op = decode_head(no["head_logits"][0], no["action_map"], cm.copy(), pm.copy(),
                         st, pl, class_id=int(c), temperature=0.0, pos_temperature=0.0,
                         intent_decoding=True)
        if op is None or not st.can_apply_operation(pl, op, ()):
            continue
        post = st.clone()
        post.apply_operation_list(pl, [op])
        v = -net_out_of(post, opp)["value"]
        if v > best:
            best, best_cls = v, c
    detail["有合法动作可选"] += 1 if best_cls is not None else 0
    detail["无任何合法动作"] += 1 if best_cls is None else 0
    if best_cls is None:
        continue
    d = best - v_pass
    margins.append(d)
    if d > 1e-6:
        better += 1
    elif d < -1e-6:
        worse += 1
    else:
        equal += 1
    # (b) 顺便记录每头 argmax 的性质
    for h in range(3):
        hl = np.asarray(no["head_logits"][h], dtype=np.float64)
        am = int(np.argmax(hl))
        detail["头argmax==HOLD" if am == HOLD else
               ("头argmax非法" if not cm[am] else "头argmax合法可执行")] += 1

print(f"\n=== (a) 1-ply 价值比较（{len(margins)} 个 pass 回合有合法动作）===")
print(f"  价值头偏好出招 (v_act > v_pass): {better}  ({better / max(len(margins),1):.1%})")
print(f"  价值头偏好空过 (v_act < v_pass): {worse}  ({worse / max(len(margins),1):.1%})")
print(f"  持平                        : {equal}")
if margins:
    mg = np.asarray(margins)
    print(f"  margin (v_act - v_pass) 均值 {mg.mean():+.4f} 中位 {np.median(mg):+.4f} "
          f"[{mg.min():+.3f}, {mg.max():+.3f}]")
print(f"  成因拆解 {dict(detail)}")

# ── (b) HOLD vs 最好合法类的 logit margin ────────────────────────────────────
hold_margins = []
for st, pl in pass_turns:
    pm = make_position_masks(st, pl, intent_decoding=True)
    cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
    no = net_out_of(st, pl)
    for h in range(3):
        hl = np.asarray(no["head_logits"][h], dtype=np.float64)
        others = [hl[c] for c in range(24) if c != HOLD and cm[c]]
        if others:
            hold_margins.append(hl[HOLD] - max(others))
hm = np.asarray(hold_margins)
print(f"\n=== (b) HOLD 的 logit 减去'最好合法非 HOLD 类' ({hm.size} 个头次) ===")
print(f"  均值 {hm.mean():+.3f} 中位 {np.median(hm):+.3f} "
      f"[{hm.min():+.2f}, {hm.max():+.2f}]   >0 的比例 {(hm > 0).mean():.1%}")

# ── (c) rule_v4 在这些局面上的出招率 ─────────────────────────────────────────
def rule_acts(st, pl) -> bool:
    try:
        ops = rule_ai.choose_operations(st, pl)
    except Exception:
        return False
    return bool(ops)


p_on_pass = np.mean([rule_acts(st, pl) for st, pl in pass_turns]) if pass_turns else float("nan")
p_all = np.mean([rule_acts(st, pl) for st, pl in SNAPS])
p_on_act = np.mean([rule_acts(st, pl) for st, pl in act_turns]) if act_turns else float("nan")
print(f"\n=== (c) rule_v4 的出招率 ===")
print(f"  在我们 pass 的回合上出招: {p_on_pass:.1%}  (n={len(pass_turns)})")
print(f"  在我们有选择的回合上出招: {p_on_act:.1%}  (n={len(act_turns)})")
print(f"  在所有采样回合上出招（基线）: {p_all:.1%}  (n={len(SNAPS)})")

# 我方（raw 策略）整体出招率，作为对照
our = np.mean([bool(decode_network_output(net_out_of(st, pl), st, pl, temperature=0.0,
                                           intent_decoding=True)) for st, pl in SNAPS])
print(f"  我方 raw 策略在同样本上的出招率: {our:.1%}")
print("\npass-vs-collapse done")
