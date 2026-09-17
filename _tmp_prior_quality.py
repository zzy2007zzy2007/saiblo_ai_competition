"""先验质量诊断：地图够尖吗？尖的那个峰对齐价值吗？

对每个 ckpt，在"会出招"的参照局面上量：
  ① 采样集中度 —— 用搜索真实的采样分布（map/100/t_pos 掩码后 softmax）算
     top1/top5 质量占比、有效候选数 exp(熵)。尖 ⇒ 小 k 就能覆盖好格、且两次抽样重合。
  ② 峰是否对齐价值 —— 地图 argmax 的格 vs 1-ply 全格价值 argmax 的格，一致率。
     （若峰尖在坏格上，那只是把 raw 变成"更自信的坏策略"。）
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
ap.add_argument("--t-pos", type=float, default=0.3)
a = ap.parse_args()
feat = FeatureExtractor(max_actions=96)

# 参照局面：基线 pos-only 配置下"有得选"的回合（每个 ckpt 共用同一批）
pol0, nf0 = make_net_fn_from_ckpt(a.ckpts[0], feat)
refs = []
for g in range(a.ref_games):
    seed = 31+g; st = make_initial_state(seed, True); gt = 0
    for _ in range(600):
        if st.terminal or len(refs) >= a.max_ref: break
        for pl in (0,1):
            if st.terminal or len(refs) >= a.max_ref: break
            gt += 1
            m = BundleMCTS(nf0, iterations=256, max_depth_rounds=4, k=24, sample_mult=15,
                           t_class=0.5, t_pos=0.3, seed=seed*1000+gt,
                           search_mode="pos-only", skip_single_candidate=True)
            r = m.search(st, pl, temperature=0.0)
            if len(set(r.bundles)) >= 2:
                refs.append((st.clone(), pl))
            obs = feat.encode_observation(st, pl, np.zeros(96))
            with torch.no_grad():
                o = pol0(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                         torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            st.apply_operation_list(pl, decode_network_output(o, st, pl, temperature=0.0,
                                                             intent_decoding=True))
        if pl == 1 and not st.terminal: st.advance_round()
print("参照局面 %d 个" % len(refs), flush=True)

for ck in a.ckpts:
    pol, _ = make_net_fn_from_ckpt(ck, feat)
    _ck = torch.load(ck, map_location="cpu", weights_only=False)
    if "pos_state" in _ck or "class_state" in _ck:          # 按内容判断，别按文件名
        _, _, vmodel = load_three_models(ck)
    else:
        vmodel = pol
    t1, t5, eff, align, n = [], [], [], 0, 0
    for st, pl in refs:
        obs = feat.encode_observation(st, pl, np.zeros(96))
        b = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        s = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        with torch.no_grad():
            out = pol(b, s)
        am = out["action_map"][0].numpy()
        hl = [out["head%d_logits" % h][0].numpy() for h in (1,2,3)]
        pm = make_position_masks(st, pl, intent_decoding=True)
        cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)
        c = int(np.argmax(hl[0]))                       # 头 0 的钉类
        if not cm[c] or c == 23: continue
        cells = np.argwhere(pm[c])
        if len(cells) < 2: continue
        lv = am[c][cells[:,0], cells[:,1]]
        lv = (lv - lv.mean()) / (lv.std() + 1e-8)      # 与采样器同一套 z-score
        vals = lv / a.t_pos
        vals = vals - vals.max(); pr = np.exp(vals); pr /= pr.sum()
        order = np.argsort(-pr)
        t1.append(float(pr[order[0]])); t5.append(float(pr[order[:5]].sum()))
        eff.append(float(np.exp(-(pr*np.log(pr+1e-12)).sum())))
        # ① vs ②：地图 argmax 是否 = 1-ply 全格价值 argmax
        map_cell = tuple(cells[order[0]])
        boards, stats, ops = [], [], []
        for (x,y) in cells:
            am2 = am.copy(); am2[c] = -1e9; am2[c, x, y] = 1e9
            op = decode_head(hl[0], am2, cm.copy(), pm.copy(), st, pl, class_id=int(c),
                             temperature=0.0, pos_temperature=0.0, intent_decoding=True)
            if op is None or not st.can_apply_operation(pl, op, ()): continue
            post = st.clone(); post.apply_operation_list(pl, [op])
            ob = feat.encode_observation(post, 1-pl, np.zeros(96))
            boards.append(ob["board"]); stats.append(ob["stats"]); ops.append((x,y))
        if not ops: continue
        n += 1
        best, bi = -9e9, 0
        for i in range(0, len(boards), 128):
            bb = torch.from_numpy(np.stack(boards[i:i+128])).float()
            ss = torch.from_numpy(np.stack(stats[i:i+128])).float()
            with torch.no_grad():
                v = vmodel(bb, ss)["value"].squeeze(-1).numpy()
            for j, val in enumerate(v):
                if -float(val) > best: best, bi = -float(val), i+j
        align += int(tuple(ops[bi]) == map_cell)
    print("%-34s top1=%.2f top5=%.2f 有效候选=%.1f  峰对齐价值=%s (%d)"
          % (Path(ck).name, np.mean(t1), np.mean(t5), np.mean(eff),
             ("%.0f%%" % (100.0*align/max(n,1))), n), flush=True)
