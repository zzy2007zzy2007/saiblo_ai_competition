"""采集"价值偏好"位置目标 —— 第一轮位置网训练用（见 docs/az_value_prior_plan.md）。

对每个出招决策点、每个头，若**钉类**（= 该头 argmax，与 `--search-mode pos-only` 一致）
有 >=2 个**可执行**的合法格，就记一条：

    (board, stats, player, head_idx, class_id, cells[(x,y)...], a[...])

其中 `a` = 对每个格子构造 post-state 后过价值网、取**对手视角价值的负**（= 我方优势）。
**存原始的 a，不存 probs** —— 目标在训练时算：合法格上 z-score 后 `softmax(z/τ)`，
这样 τ 可以事后调而不用重采。

为什么不用 MCTS visit 当目标：可负担设置下 visit 目标几乎不可复现（一致率 0~42%），
而价值偏好是**局面的确定性函数**（无 rng、无 seed 依赖），且每个出招回合约 0.8 秒
（对比 k=200/it=8192 的约 330 秒）。理由与实测见 docs/az_value_prior_plan.md §0/§0.1。

用法:
    python code/my_ai/az_intent/collect_value_prior.py \
        --checkpoint training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt \
        --games 400 --workers 16 --seed 910101 --out training_history/vprior/vp_400.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _worker(job) -> dict:
    ckpt, games, seed, max_rounds, native = job
    import torch
    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import (load_three_models, make_initial_state,
                                             make_net_fn_from_ckpt)
    from my_ai.decoder import (decode_head, decode_network_output, make_class_mask,
                               make_position_masks)

    feat = FeatureExtractor(max_actions=96)
    pol, _ = make_net_fn_from_ckpt(ckpt, feat)
    _, _, vmodel = load_three_models(ckpt)

    out = {"boards": [], "stats": [], "player": [], "head": [], "cls": [],
           "cells": [], "adv": []}
    n_turn = n_act = 0

    for g in range(games):
        st = make_initial_state(seed + g, native)
        for _ in range(max_rounds):
            if st.terminal:
                break
            for pl in (0, 1):
                if st.terminal:
                    break
                n_turn += 1
                obs = feat.encode_observation(st, pl, np.zeros(96))
                with torch.no_grad():
                    o = pol(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
                am = o["action_map"][0].numpy()
                hl = [o[f"head{h}_logits"].squeeze(0).numpy() for h in (1, 2, 3)]
                pm = make_position_masks(st, pl, intent_decoding=True)
                cm = make_class_mask(st, pl, position_mask=pm, intent_decoding=True)

                for h in range(3):
                    c = int(np.argmax(hl[h]))
                    if c < 0 or c >= len(cm) or not cm[c] or c == 23:
                        continue                      # 不可执行 / HOLD 无位置目标
                    cells = np.argwhere(pm[c])
                    if len(cells) < 2:
                        continue                      # 单格目标无信息
                    posts, keep = [], []
                    for (x, y) in cells:
                        am2 = am.copy(); am2[c] = -1e9; am2[c, x, y] = 1e9
                        op = decode_head(hl[h], am2, cm.copy(), pm.copy(), st, pl,
                                         class_id=int(c), temperature=0.0,
                                         pos_temperature=0.0, intent_decoding=True)
                        if op is None or not st.can_apply_operation(pl, op, ()):
                            continue
                        post = st.clone()
                        post.apply_operation_list(pl, [op])
                        ob = feat.encode_observation(post, 1 - pl, np.zeros(96))
                        posts.append((ob["board"], ob["stats"]))
                        keep.append((int(x), int(y)))
                    if len(keep) < 2:
                        continue
                    adv = []
                    for i in range(0, len(posts), 128):
                        bb = torch.from_numpy(np.stack([t[0] for t in posts[i:i + 128]])).float()
                        ss = torch.from_numpy(np.stack([t[1] for t in posts[i:i + 128]])).float()
                        with torch.no_grad():
                            v = vmodel(bb, ss)["value"].squeeze(-1).numpy()
                        adv.extend((-v).tolist())     # 对手将行棋 ⇒ 取负 = 我方优势
                    out["boards"].append(obs["board"])
                    out["stats"].append(obs["stats"])
                    out["player"].append(pl)
                    out["head"].append(h)
                    out["cls"].append(c)
                    out["cells"].append(np.asarray(keep, dtype=np.int16))
                    out["adv"].append(np.asarray(adv, dtype=np.float32))
                    n_act += 1

                ops = decode_network_output(o, st, pl, temperature=0.0, intent_decoding=True)
                st.apply_operation_list(pl, ops)
            if not st.terminal:
                st.advance_round()

    return {"out": out, "n_turn": n_turn, "n_act": n_act, "n_games": games}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=910101)
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--native-engine", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="只记录在元数据里；目标在训练时才由 a 算 softmax(z/tau)")
    args = ap.parse_args()

    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0)
           for i in range(args.workers)]
    jobs, s = [], args.seed
    for i, n in enumerate(per):
        if n <= 0:
            continue
        jobs.append((args.checkpoint, n, s, args.max_rounds, args.native_engine))
        s += n * 1000
    print("[collect] %d 局 / %d worker，seed 起 %d" % (args.games, len(jobs), args.seed),
          flush=True)

    import multiprocessing as mp
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            res = pool.map(_worker, jobs)
    else:
        res = [_worker(j) for j in jobs]

    boards = [b for r in res for b in r["out"]["boards"]]
    statss = [s_ for r in res for s_ in r["out"]["stats"]]
    players = [p for r in res for p in r["out"]["player"]]
    heads = [h for r in res for h in r["out"]["head"]]
    clss = [c for r in res for c in r["out"]["cls"]]
    cells = [c for r in res for c in r["out"]["cells"]]
    advs = [a for r in res for a in r["out"]["adv"]]
    n_turn = sum(r["n_turn"] for r in res)
    n_games = sum(r["n_games"] for r in res)

    N = len(cells)
    MAXC = max(len(c) for c in cells)
    board = np.zeros((N,) + boards[0].shape, dtype=np.float16)
    stat = np.zeros((N,) + statss[0].shape, dtype=np.float16)
    cell = np.full((N, MAXC, 2), -1, dtype=np.int16)
    adv = np.zeros((N, MAXC), dtype=np.float32)
    cnt = np.zeros(N, dtype=np.int16)
    for i in range(N):
        board[i] = boards[i]
        stat[i] = statss[i]
        cell[i, :len(cells[i])] = cells[i]
        adv[i, :len(advs[i])] = advs[i]
        cnt[i] = len(cells[i])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, board=board, stats=stat,
                        player=np.asarray(players, dtype=np.int8),
                        head=np.asarray(heads, dtype=np.int8),
                        cls=np.asarray(clss, dtype=np.int8),
                        cell=cell, adv=adv, cnt=cnt,
                        tau_meta=np.float32(args.tau), ckpt_meta=np.array(args.checkpoint))
    print("[collect] 完成：%d 局 / %d 回合 / **%d 个决策点**（%.1f/局）"
          % (n_games, n_turn, N, N / max(n_games, 1)), flush=True)
    print("[collect] 写出 %s   张量: board%s stats%s cell%s adv%s  最多格数=%d"
          % (args.out, board.shape, stat.shape, cell.shape, adv.shape, MAXC), flush=True)


if __name__ == "__main__":
    main()
