"""(a) 的无偏版本 + 仪器灵敏度对照。

问题：(a) 原来比的是 "max_c v_act(c) vs v_pass"。在 ~20 个类里取最大值，
**选择偏差**会把结果推向"出招更好"（噪声的最大值天然 > 0），即使所有动作其实中性。
实测那个 margin 均值 +0.0277 恰好和"全中性 + 取 20 个最大值"的预期一致 ⇒ 不能下结论。

无偏版本：
  ① 每个回合先算 mean_c v_act(c)（不选最大），再与 v_pass 配对比较 + t 统计量；
  ② 同时报 max 版（有偏）与 min 版（反方向有偏）作为对照——真效应应该三者同向。

灵敏度对照：取某个类，把它的**所有合法位置**各评估一次，量"同类的跨位置价值散布"。
文档记过 rel 价值对大部分位置输出 -0.0303 平齐 ⇒ 若跨类散布与"跨位置散布"同量级，
说明价值头分不出它该分的东西，+0.0277 就落在仪器灵敏度里。
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
from my_ai.az_intent.az_selfplay import make_initial_state
from my_ai.az_intent.bundle_mcts import BundleMCTS
from my_ai.decoder import (decode_head, decode_network_output, make_class_mask,
                           make_position_masks)
from my_ai.network import create_model, model_kwargs_from_ckpt

CK = "training_history/az_fixed/mix_r10p_vw_pol_frozen.pt"
GAMES = (3, 11)
WANT = 60
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


SNAPS: list[tuple] = []
for g in GAMES:
    SNAPS += collect(g, WANT)
print(f"states = {len(SNAPS)}", flush=True)

pass_turns = []
for i, (st, pl) in enumerate(SNAPS):
    m = BundleMCTS(net_fn, iterations=256, max_depth_rounds=4, k=24, t_class=0.5,
                   t_pos=0.3, seed=1000 + i, search_mode="pos-only",
                   skip_single_candidate=True)
    res = m.search(st, pl, temperature=0.0)
    nb = len(res.bundles)
    if nb == 0 or (nb == 1 and len(res.bundles[0]) == 0):
        pass_turns.append((st, pl))
print(f"pass turns = {len(pass_turns)}", flush=True)


def per_class_values(st, pl):
    """(v_pass, [(class, v_act)]，每个合法类取它 action_map 的 argmax 位置)"""
    opp = 1 - pl
    v_pass = -net_out_of(st, opp)["value"]
    pm = make_position_masks(st, pl, intent_decoding=True)
    cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
    no = net_out_of(st, pl)
    out = []
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
        out.append((c, -net_out_of(post, opp)["value"]))
    return v_pass, out


diffs_mean, diffs_max, diffs_min = [], [], []
for st, pl in pass_turns:
    v_pass, vs = per_class_values(st, pl)
    if not vs:
        continue
    vals = np.asarray([v for _, v in vs])
    diffs_mean.append(vals.mean() - v_pass)
    diffs_max.append(vals.max() - v_pass)
    diffs_min.append(vals.min() - v_pass)

dm = np.asarray(diffs_mean)
print(f"\n=== (a) 无偏版：mean_c v_act - v_pass（{dm.size} 个 pass 回合）===")
print(f"  均值 {dm.mean():+.4f}  SD {dm.std(ddof=1):.4f}  标准误 {dm.std(ddof=1) / np.sqrt(dm.size):.4f}")
print(f"  t = {dm.mean() / (dm.std(ddof=1) / np.sqrt(dm.size)):+.2f}   "
      f"偏好出招的回合占比 {(dm > 0).mean():.1%}")
print(f"  对照 max 版均值 {np.mean(diffs_max):+.4f}（有偏，天然偏正）")
print(f"  对照 min 版均值 {np.mean(diffs_min):+.4f}（反方向有偏，天然偏负）")

# ── 仪器灵敏度：同一个类、跨所有合法位置的价值散布 ───────────────────────────
pos_spreads = []
base_cls_spread = []
for st, pl in pass_turns[:40]:
    opp = 1 - pl
    v_pass = -net_out_of(st, opp)["value"]
    pm = make_position_masks(st, pl, intent_decoding=True)
    cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
    no = net_out_of(st, pl)
    # 选合法位置最多的那个类
    best_c, best_cells = None, 0
    for c in range(16):
        if cm[c] and pm[c].sum() > best_cells:
            best_c, best_cells = c, int(pm[c].sum())
    if best_c is None or best_cells < 3:
        continue
    cells = np.argwhere(pm[best_c])
    vals = []
    for x, y in cells[:24]:
        op = decode_head(no["head_logits"][0], no["action_map"], cm.copy(), pm.copy(),
                         st, pl, class_id=int(best_c), temperature=0.0,
                         pos_temperature=0.0, intent_decoding=True)
        if op is None:
            continue
        # 强制放到 (x,y)：直接用 _decode_tower_action 的结果不方便，改用引擎
        post = st.clone()
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType
        from my_ai.decoder import _decode_tower_action
        op2 = _decode_tower_action(st, pl, int(best_c), int(x), int(y))
        if op2 is None or not st.can_apply_operation(pl, op2, ()):
            continue
        post.apply_operation_list(pl, [op2])
        vals.append(-net_out_of(post, opp)["value"])
    if len(vals) >= 3:
        pos_spreads.append(np.std(vals))
        base_cls_spread.append(np.mean(vals) - v_pass)

ps = np.asarray(pos_spreads)
print(f"\n=== 仪器灵敏度：同一个类内部、跨合法位置的价值散布（{ps.size} 个局面）===")
print(f"  跨位置 SD 均值 {ps.mean():.4f}  中位 {np.median(ps):.4f}")
print(f"  同类内 '平均位置价值 - 空过价值' 均值 {np.mean(base_cls_spread):+.4f}")
print(f"\n对照：跨类 mean 差 {dm.mean():+.4f} / 跨类 max 差 {np.mean(diffs_max):+.4f}"
      f"  vs  跨位置 SD {ps.mean():.4f}")
print("\nunbiased pass-value done")
