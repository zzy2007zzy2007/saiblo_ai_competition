"""采集"价值偏好"位置目标 —— 位置网训练用（见 docs/az_value_prior_plan.md、
docs/az_posnet_v2_plan.md）。

对每个出招决策点、每个头，取**一批"被记录的类"**，对每个类记一条：

    (board, stats, player, head_idx, class_id, cells[(x,y)...], a[...], weight)

其中 `a` = 对每个格子构造 post-state 后过价值网、取**对手视角价值的负**（= 我方优势）。
**存原始的 a，不存 probs** —— 目标在训练时算：合法格上 z-score 后 `softmax(z/τ)`，
这样 τ 可以事后调而不用重采。

**记录的类（`--multi-class-k K`）**：`K=1` = 只记该头 argmax 那一类（= 旧行为，逐位等价）；
`K>1` = **argmax 类（weight=1）+ 从其余可用类里随机抽 K-1 个（weight=λ）**。
动机（用户 2026-09-20）：实测每个 (回合, 头) 平均有 **18.9 个可用类**，而旧口径只记 1 个
⇒ 位置网 24 个通道里只有 1 个拿到过监督（`vp_400.npz` 的 cls 全是 17）。
"可用类"由 `class_mask`/`position_mask` 推（**不手写规则**）⇒ 天然与解码器的可执行性一致。

**`--max-cells M`**：每类最多取 M 个格子（0=不封顶）。用于把成本压回可控
（裸记"全部合法类 × 全部格"实测是 193.7x 行 / **51.5x 价值网前向**）。

**为什么不用 MCTS visit 当目标**：可负担设置下 visit 目标几乎不可复现（一致率 0~42%），
而价值偏好是**局面的确定性函数**（无 rng、无 seed 依赖）。理由与实测见
docs/az_value_prior_plan.md §0/§0.1。visit 目标是第二阶段的事（docs/az_posnet_v2_plan.md §4）。

**⚠️ 强制格校验（`_op_matches_cell`）**：把某格"顶到最高"只对**能正常解码**的类有效。
买不起的超武类会走"读通道 16 的自动降级"⇒ 每个格返回同一个降级操作 ⇒ 该行会**退化/错标**。
所以记录前必须校验"解码出的操作确实落在被顶起来的那一格上"（分类处理：DOWNGRADE 看 tower_id）。
另外还丢弃 `a` 极差为 0 的退化行。两者都计数并打印。

用法:
    python code/my_ai/az_intent/collect_value_prior.py \
        --checkpoint training_history/inject_ex02/valnet_A2.pt \
        --games 400 --workers 16 --seed 910101 --out training_history/vprior/vp_400.npz
    # 多类记录（第一阶段主配置）：
    ... --multi-class-k 5 --max-cells 32 --row-weight-lambda 0.2 --out .../vp_multi.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HOLD = 23
DOWNGRADE_CH = 16


def _op_matches_cell(op, st, pl, c: int, x: int, y: int) -> bool:
    """解码出的 op 是否**真的落在被顶起来的 (x,y) 上**？

    必须分类处理，因为各 op 的位置字段语义不同：
      * 0-15（塔意图）：空格 → `BUILD(x, y)`；己方塔 → `UPGRADE(tower_id, step)`
      * 16（降级）    ：`DOWNGRADE(tower_id)` —— **arg0 是 tower_id，不是 x/y**
      * 17-20（超武） ：`(op_type, x, y)`
    """
    from SDK.backend.model import OperationType as OT
    ot = int(op.op_type)
    if ot == int(OT.DOWNGRADE_TOWER):
        return int(c) == DOWNGRADE_CH          # 自动降级会挂到武器类上 ⇒ 直接判不符
    if ot == int(OT.UPGRADE_TOWER):
        tw = st.tower_at(x, y)
        return (tw is not None and tw.player == pl
                and int(tw.tower_id) == int(op.arg0) and 0 <= int(c) <= 15)
    return int(op.arg0) == int(x) and int(op.arg1) == int(y)


def _worker(job) -> dict:
    (ckpt, games, seed, max_rounds, native, progress_dir,
     K, M, lam) = job
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
    rng = np.random.default_rng(seed)          # K=1 且 M=0 时**不消耗** ⇒ 旧行为逐位不变

    # 逐局进度文件（照 az_selfplay.py 的 progress_path 做法）—— 否则一个小时的采集
    # 中间完全看不到进度，只能靠外推（用户 2026-09-19 指出这个缺口）。
    prog = None
    if progress_dir:
        prog = Path(progress_dir) / ("vp_progress_seed%05d.txt" % seed)
        prog.parent.mkdir(parents=True, exist_ok=True)
        with open(prog, "w", encoding="utf-8") as f:
            f.write("ckpt=%s games=%d seed=%d K=%d M=%d lam=%.3f\n"
                    % (ckpt, games, seed, K, M, lam))

    out = {"boards": [], "stats": [], "player": [], "head": [], "cls": [],
           "cells": [], "adv": [], "weight": []}
    n_turn = n_act = 0
    n_drop_cell = n_drop_degen = n_eval = 0          # 守卫丢弃数 / 退化行数 / 价值网前向数
    cls_hist = Counter()        # 按类累计**格数**（成本口径）
    cls_rows = Counter()        # 按类累计**行数**（权重/覆盖口径）
    t_start = time.time()

    for g in range(games):
        st = make_initial_state(seed + g, native)
        n_act_g = 0
        rounds = 0
        for _ in range(max_rounds):
            if st.terminal:
                break
            rounds += 1
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
                    c_arg = int(np.argmax(hl[h]))
                    if K <= 1:
                        picks = [(c_arg, 1.0)]
                    else:
                        usable = [c for c in range(len(cm))
                                  if c != HOLD and cm[c] and pm[c].any()]
                        picks = []
                        if c_arg in usable:
                            picks.append((c_arg, 1.0))
                            others = [c for c in usable if c != c_arg]
                            rng.shuffle(others)
                            picks += [(c, lam) for c in others[:K - 1]]

                    for (c, w) in picks:
                        if c < 0 or c >= len(cm) or not cm[c] or c == HOLD:
                            continue                  # 不可执行 / HOLD 无位置目标
                        cells = np.argwhere(pm[c])
                        if M > 0 and len(cells) > M:  # 成本封顶：随机取 M 格
                            sel = rng.choice(len(cells), size=M, replace=False)
                            cells = cells[np.sort(sel)]
                        if len(cells) < 2:
                            continue                  # 单格目标无信息
                        posts, keep = [], []
                        for (x, y) in cells:
                            am2 = am.copy(); am2[c] = -1e9; am2[c, x, y] = 1e9
                            op = decode_head(hl[h], am2, cm.copy(), pm.copy(), st, pl,
                                             class_id=int(c), temperature=0.0,
                                             pos_temperature=0.0, intent_decoding=True)
                            if op is None or not st.can_apply_operation(pl, op, ()):
                                continue
                            if not _op_matches_cell(op, st, pl, int(c), int(x), int(y)):
                                n_drop_cell += 1       # 顶格没生效（如自动降级）⇒ 丢弃
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
                            n_eval += len(posts[i:i + 128])
                        if float(np.ptp(adv)) <= 0.0:
                            n_drop_degen += 1         # 全相等 ⇒ 该行无信息（退化/错标）
                            continue
                        out["boards"].append(obs["board"])
                        out["stats"].append(obs["stats"])
                        out["player"].append(pl)
                        out["head"].append(h)
                        out["cls"].append(c)
                        out["cells"].append(np.asarray(keep, dtype=np.int16))
                        out["adv"].append(np.asarray(adv, dtype=np.float32))
                        out["weight"].append(float(w))
                        cls_hist[c] += len(keep)
                        cls_rows[c] += 1
                        n_act += 1
                        n_act_g += 1

                ops = decode_network_output(o, st, pl, temperature=0.0, intent_decoding=True)
                st.apply_operation_list(pl, ops)
            if not st.terminal:
                st.advance_round()

        if prog is not None:
            line = ("seed=%d game %3d/%d  rounds=%3d  rows(累计)=%4d  rows(本局)=%2d  "
                    "价值网前向(本 worker)=%d  丢弃(顶格/退化)=%d/%d  用时=%.0fs\n"
                    % (seed, g + 1, games, rounds, n_act, n_act_g, n_eval,
                       n_drop_cell, n_drop_degen, time.time() - t_start))
            with open(prog, "a", encoding="utf-8") as f:
                f.write(line)
            print("[collect] " + line.rstrip(), flush=True)

    return {"out": out, "n_turn": n_turn, "n_act": n_act, "n_games": games,
            "n_drop_cell": n_drop_cell, "n_drop_degen": n_drop_degen, "n_eval": n_eval,
            "cls_hist": cls_hist, "cls_rows": cls_rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=910101)
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--native-engine", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress-dir", type=str, default=None,
                    help="逐局进度文件目录（默认 = --out 所在目录）")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="只记录在元数据里；目标在训练时才由 a 算 softmax(z/tau)")
    ap.add_argument("--multi-class-k", type=int, default=1,
                    help="每个 (回合, 头) 记录几个类。1 = 只记 argmax 类（**逐位等价于旧行为**）；"
                         ">1 = argmax 类(weight=1) + 随机抽 K-1 个可用类(weight=lambda)。"
                         "见 docs/az_posnet_v2_plan.md §3")
    ap.add_argument("--max-cells", type=int, default=0,
                    help="每类最多取几格（0=不封顶）。用于压成本（裸记全部是 51.5x 价值网前向）")
    ap.add_argument("--row-weight-lambda", type=float, default=0.2,
                    help="非 argmax 类的行权重（闪电通道从占 100%% 掉到 ~1/K，必须加权）")
    args = ap.parse_args()

    pdir = args.progress_dir or str(Path(args.out).parent)
    per = [args.games // args.workers + (1 if i < args.games % args.workers else 0)
           for i in range(args.workers)]
    jobs, s = [], args.seed
    for i, n in enumerate(per):
        if n <= 0:
            continue
        jobs.append((args.checkpoint, n, s, args.max_rounds, args.native_engine, pdir,
                     args.multi_class_k, args.max_cells, args.row_weight_lambda))
        s += n * 1000
    print("[collect] %d 局 / %d worker，seed 起 %d；K=%d M=%d lambda=%.3f"
          % (args.games, len(jobs), args.seed, args.multi_class_k, args.max_cells,
             args.row_weight_lambda), flush=True)

    import multiprocessing as mp
    t0 = time.time()
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
    weights = [w for r in res for w in r["out"]["weight"]]
    n_turn = sum(r["n_turn"] for r in res)
    n_games = sum(r["n_games"] for r in res)
    n_drop_cell = sum(r["n_drop_cell"] for r in res)
    n_drop_degen = sum(r["n_drop_degen"] for r in res)
    n_eval = sum(r["n_eval"] for r in res)
    hist = Counter(); rows_hist = Counter()
    for r in res:
        hist.update(r["cls_hist"])
        rows_hist.update(r["cls_rows"])

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
                        weight=np.asarray(weights, dtype=np.float32),
                        tau_meta=np.float32(args.tau), ckpt_meta=np.array(args.checkpoint),
                        k_meta=np.int32(args.multi_class_k),
                        m_meta=np.int32(args.max_cells),
                        lambda_meta=np.float32(args.row_weight_lambda))
    print("[collect] 完成：%d 局 / %d 回合 / **%d 行**（%.1f 行/局）  用时 %.0fs"
          % (n_games, n_turn, N, N / max(n_games, 1), time.time() - t0), flush=True)
    print("[collect] 成本：价值网前向 %d（%.0f/局）  丢弃：顶格校验 %d / 退化 %d"
          % (n_eval, n_eval / max(n_games, 1), n_drop_cell, n_drop_degen), flush=True)
    tot_rows = max(sum(rows_hist.values()), 1)
    print("[collect] **类覆盖**：涉及 %d 个类；行数占比 = %s"
          % (len(rows_hist), ", ".join("%d:%.1f%%" % (c, 100.0 * n / tot_rows)
                                       for c, n in rows_hist.most_common(8))), flush=True)
    print("[collect] 写出 %s   张量: board%s stats%s cell%s adv%s  最多格数=%d"
          % (args.out, board.shape, stat.shape, cell.shape, adv.shape, MAXC), flush=True)


if __name__ == "__main__":
    main()
