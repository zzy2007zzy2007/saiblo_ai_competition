"""量 action_map 在"钉类的合法格"上的 std —— 用来定出 100x 尺度错配等效于把温度放大多少倍。

原始采样器 `_sample_position`：logits = (map / 100) / t_pos  ⇒ logit 尺度 = std(map)/100/t_pos
改后（2026-09-17 z-score）：    logits = z / t_pos           ⇒ logit 尺度 = 1/t_pos
两者等价 ⇒ std(map)/100 = 1，即需要 std(map) = 100。
实测 std(map) = s ⇒ **旧的 t_pos 等价于新的 t_pos × 100 / s**。

对每个"钉类有 >=2 个合法格"的回合，按头 0/1/2 各自的 argmax 钉类分别统计。
"""
import sys
from pathlib import Path
import numpy as np, torch
_REPO = Path(__file__).resolve().parent
for p in (_REPO/"Ant-Game", _REPO/"code"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
torch.set_num_threads(1)
from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
from my_ai.decoder import decode_network_output, make_class_mask, make_position_masks

CK = "training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt"
feat = FeatureExtractor(max_actions=96)
pol, _ = make_net_fn_from_ckpt(CK, feat)

stds, n = [], 0
for g in range(8):
    seed = 31 + g
    st = make_initial_state(seed, True)
    for _ in range(600):
        if st.terminal or n >= 120:
            break
        for pl in (0, 1):
            if st.terminal or n >= 120:
                break
            obs = feat.encode_observation(st, pl, np.zeros(96))
            with torch.no_grad():
                o = pol(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                        torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            am = o["action_map"][0].numpy()
            pm = make_position_masks(st, pl, intent_decoding=True)
            cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
            for h in (1, 2, 3):
                c = int(np.argmax(o["head%d_logits" % h][0].numpy()))
                if c < 0 or c >= len(cm) or not cm[c] or c == 23:
                    continue
                mask = pm[c]
                if mask.sum() < 2:
                    continue
                vals = am[c][mask]
                stds.append(float(vals.std()))
                n += 1
            st.apply_operation_list(pl, decode_network_output(o, st, pl, temperature=0.0,
                                                             intent_decoding=True))
        if pl == 1 and not st.terminal:
            st.advance_round()

s = float(np.mean(stds))
print("样本数 = %d（头槽位）" % len(stds))
print("action_map 在合法格上的 std: 均值 = %.3f   中位数 = %.3f   min=%.3f max=%.3f"
      % (s, float(np.median(stds)), float(np.min(stds)), float(np.max(stds))))
print()
print("⇒ 旧采样器 logits = (map/100)/t_pos 的等效新 t_pos = t_pos_old * 100 / std")
print("   std=%.2f ⇒ 放大倍数 100/std = %.1fx" % (s, 100.0 / s))
print("   所以旧的 t_pos=0.3  ≈ 新的 t_pos = %.2f" % (0.3 * 100.0 / s))
print("   （与实测标定对照：修复前的探索量等价于新的 t_pos≈4，见 experiment_log tpos_prior_quality）")
