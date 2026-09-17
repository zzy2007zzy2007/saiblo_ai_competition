"""先验质量诊断：地图够尖吗？尖的那个峰对齐价值吗？尖度随温度怎么变？

对每个 ckpt，在"位置维真的有得选"的参照局面上量（`--t-pos-list` 可一次扫多个温度）：
  ① 采样集中度 —— 用搜索真实的采样分布（合法格上 z-score 后 ÷ t_pos 的 softmax）算
     top1/top5 质量占比、有效候选数 exp(熵)。尖 ⇒ 小 k 就能覆盖好格、且两次抽样重合。
  ② 峰是否对齐价值 —— 地图 argmax 的格 vs 1-ply 全格价值 argmax 的格，一致率。
     （若峰尖在坏格上，那只是把 raw 变成"更自信的坏策略"。）

2026-09-17：参照局面的筛法改过。原先按"搜索候选数 >= 2"筛，但 z-score 修复后尖先验下
98% 的回合只剩 1 个候选 ⇒ 参照集既极小、又系统性偏向探索量大的配置。现改为按
"该回合头 0 的钉类有 >= 2 个合法格"筛，且**推演只用 raw 策略**（与搜索配置无关），
所以各 t_pos 共用同一批局面、前向也只需做一次。
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
_REPO = Path(__file__).resolve().parent
for p in (_REPO/"Ant-Game", _REPO/"code"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
torch.set_num_threads(1)
from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.az_selfplay import (load_three_models, make_initial_state,
                                         make_net_fn_from_ckpt)
from my_ai.az_intent.bundle_mcts import BundleMCTS
from my_ai.decoder import decode_head, decode_network_output, make_class_mask, make_position_masks

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--ref-games", type=int, default=6)
ap.add_argument("--max-ref", type=int, default=40)
ap.add_argument("--t-pos-list", type=str, default="0.3",
                help="逗号分隔的 t_pos 列表；在同一批参照局面上分别量")
a = ap.parse_args()
TPS = [float(x) for x in a.t_pos_list.split(",") if x.strip()]
feat = FeatureExtractor(max_actions=96)

# 参照局面：头 0 的钉类有 >= 2 个合法格的回合（= 位置维真的有得选）。推演只用 raw 策略，
# 所以与任何搜索配置无关 ⇒ 各 t_pos 共用同一批、前向也只做一次。
pol0, _nf0 = make_net_fn_from_ckpt(a.ckpts[0], feat)
refs = []
for g in range(a.ref_games):
    seed = 31+g; st = make_initial_state(seed, True)
    for _ in range(600):
        if st.terminal or len(refs) >= a.max_ref: break
        for pl in (0,1):
            if st.terminal or len(refs) >= a.max_ref: break
            obs = feat.encode_observation(st, pl, np.zeros(96))
            with torch.no_grad():
                o = pol0(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                         torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            pm = make_position_masks(st, pl, intent_decoding=True)
            cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
            c = int(np.argmax(o["head1_logits"][0].numpy()))
            if 0 <= c < len(cm) and cm[c] and c != 23 and int(pm[c].sum()) >= 2:
                refs.append((st.clone(), pl))
            st.apply_operation_list(pl, decode_network_output(o, st, pl, temperature=0.0,
                                                             intent_decoding=True))
        if pl == 1 and not st.terminal: st.advance_round()
print("参照局面 %d 个（头 0 钉类有 >=2 合法格）" % len(refs), flush=True)

for ck in a.ckpts:
    pol, _ = make_net_fn_from_ckpt(ck, feat)
    _ck = torch.load(ck, map_location="cpu", weights_only=False)
    if "pos_state" in _ck or "class_state" in _ck:          # 按内容判断，别按文件名
        _, _, vmodel = load_three_models(ck)
    else:
        vmodel = pol
    # 每个参照局面只做一次前向（地图/类头 + 逐候选格的价值）。t_pos 只改最后一步的
    # 归一化 ⇒ 各 t_pos 复用同一批记录，「峰对齐价值」也就能在所有温度上免费重算。
    recs = []
    for st, pl in refs:
        obs = feat.encode_observation(st, pl, np.zeros(96))
        with torch.no_grad():
            out = pol(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                      torch.from_numpy(obs["stats"]).unsqueeze(0).float())
        am = out["action_map"][0].numpy()
        hl = [out["head%d_logits" % h][0].numpy() for h in (1,2,3)]
        pm = make_position_masks(st, pl, intent_decoding=True)
        cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
        c = int(np.argmax(hl[0]))                       # 头 0 的钉类
        if not cm[c] or c == 23: continue
        cells = np.argwhere(pm[c])
        if len(cells) < 2: continue
        boards, stats, ok = [], [], []
        for (x,y) in cells:
            am2 = am.copy(); am2[c] = -1e9; am2[c, x, y] = 1e9
            op = decode_head(hl[0], am2, cm.copy(), pm.copy(), st, pl, class_id=int(c),
                             temperature=0.0, pos_temperature=0.0, intent_decoding=True)
            if op is None or not st.can_apply_operation(pl, op, ()): continue
            post = st.clone(); post.apply_operation_list(pl, [op])
            ob = feat.encode_observation(post, 1-pl, np.zeros(96))
            boards.append(ob["board"]); stats.append(ob["stats"]); ok.append((int(x),int(y)))
        if len(ok) < 2: continue
        vcell = []
        for i in range(0, len(boards), 128):
            with torch.no_grad():
                v = vmodel(torch.from_numpy(np.stack(boards[i:i+128])).float(),
                           torch.from_numpy(np.stack(stats[i:i+128])).float())["value"]
            vcell.extend((-v.squeeze(-1).numpy()).tolist())     # -value = 我方优势
        recs.append({"am": am, "c": c, "cells": np.asarray(ok, dtype=np.int64),
                     "vcell": np.asarray(vcell, dtype=np.float64)})
    print("  [%s] 可用参照局面 %d 个" % (Path(ck).name, len(recs)), flush=True)

    for tp in TPS:
        t1, t5, eff, align = [], [], [], 0
        for rc in recs:
            cells = rc["cells"]
            lv = rc["am"][rc["c"]][cells[:,0], cells[:,1]]
            lv = (lv - lv.mean()) / (lv.std() + 1e-8)  # 与采样器同一套 z-score
            vals = lv / tp
            vals = vals - vals.max(); pr = np.exp(vals); pr /= pr.sum()
            order = np.argsort(-pr)
            t1.append(float(pr[order[0]])); t5.append(float(pr[order[:5]].sum()))
            eff.append(float(np.exp(-(pr*np.log(pr+1e-12)).sum())))
            # 峰是否落在 1-ply 全格价值 argmax 上
            align += int(int(order[0]) == int(np.argmax(rc["vcell"])))
        print("%-26s t_pos=%-5.2f top1=%.2f top5=%.2f 有效候选=%5.1f  峰对齐价值=%s (%d)"
              % (Path(ck).name, tp, np.mean(t1), np.mean(t5), np.mean(eff),
                 ("%.0f%%" % (100.0*align/max(len(recs),1))), len(recs)), flush=True)
