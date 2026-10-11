"""采集"**位置教师**"数据（路线 A②，2026-10-11 **修正版**）：用引擎安静 rollout 给位置候选打分。

⚠️ 本文件修掉了两个**真 bug**（2026-10-11 由用户追问"A② 为什么每决策只有 ~2 格可解码"逼出，详见 `docs/experiment_log.md` 第 86b 条）：

1. **局面从不推进**（原版只对 `post = st.clone()` 落子/推进）⇒ 一局里所有决策看**同一个冻结局面**
   ⇒ 数据是"3 头 × 2 玩家 × 256 回合 × 同一个局面"。修法：按历史那个**有效**的采集器
   （`collect_value_prior.py:195-198`）——**把网络的束真的打到 `st` 上**，每轮 `st.advance_round()`。
2. **坐标轴序写反**：位置掩码是 `mask[ch, x, y]`（`decoder.py:192` 用 `HIGHLAND_CELLS` 的 `(x, y)` 索引）
   ⇒ `np.argwhere(pm[c])` 给的**就是 (x, y)**。原版写成取反再双交换 ⇒ 传给 `_op_matches_cell` 的坐标是错的
   ⇒ 33 格里只有 `x==y` 的对角格能过检查（实测 32/33 判"位置不符"）⇒ 这才是"只有 2 格"的来源。
   修法：**全程 `[x, y]`**（设格 `am[c,x,y]`、校验 `_op_matches_cell(..., x, y)`、存盘 `cell=[x, y]`）
   —— 与 `train_value_prior.py:140` 的 `am[..., xy[0], xy[1]]` 口径一致。

另外两处口径决定：
* **只钉塔类 0–16**（部署 MC 菜单的 `_econ_rank` 就是 `range(0, 16|17)`）：超武类 17–20 里
  **类 18 的 op 位置字段不适用 `_op_matches_cell`**（实测 271 格、14,092 次全判"位置不符"、成功 0）⇒ 整类白采，直接跳过；
* 采样：先按位置图值降序**尝试到 collect_cap 个成功格**，再**只记录分散的 M 个**（前 2 + 分位分散）⇒ 目标是有覆盖的分布。

用法：
  python -u code/my_ai/az_intent/collect_pos_rollout_teacher.py \
    --checkpoint checkpoints/grafted_sdk_A.pt --games 30 --workers 8 \
    --max-cells 12 --collect-cap 24 --rollout-rounds 256 --seed 25000 \
    --out training_history/posroll2/cell30.npz
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
import sys as _sys

for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

HOLD = 23
PIN_MAX_CLASS = 16          # 只钉 0..16（= 部署 MC 菜单的经济类范围；避开类 18 的位置口径坑）


def _worker(job) -> dict:
    (ckpt, games, seed, max_rounds, native, M, collect_cap, rollout_rounds, probe_classes, heads) = job
    import torch

    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.collect_value_prior import _op_matches_cell
    from my_ai.az_intent.mc_quiet import quiet_value
    from my_ai.decoder import (decode_head, decode_network_output, make_class_mask,
                               make_position_masks)

    feat = FeatureExtractor(max_actions=96)
    pol, _ = make_net_fn_from_ckpt(ckpt, feat)

    out = {"board": [], "stats": [], "player": [], "head": [], "cls": [],
           "cell": [], "adv": [], "weight": []}
    stats_cnt = collections.Counter()
    n_turn = n_pos_dec = n_rows = n_cell_drop = n_eval = 0
    deltas = []
    pin_hist = collections.Counter()
    head_hist = collections.Counter()
    rounds_seen = 0
    t0 = time.time()

    for g in range(games):
        st = make_initial_state(seed + g, native)
        for rnd in range(int(max_rounds)):
            if st.terminal:
                break
            rounds_seen = rnd
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

                for h in range(max(1, int(heads))):
                    # ① 廉价定类：logit 降序里第一个"能解出可执行 op"的**塔类**（0..16，排除 HOLD）
                    c_pin = None
                    for c in np.argsort(-hl[h])[:max(1, int(probe_classes))]:
                        c = int(c)
                        if c == HOLD or not (0 <= c <= PIN_MAX_CLASS):
                            continue
                        if not cm[c] or not pm[c].any():
                            continue
                        probe = decode_head(hl[h], am, cm.copy(), pm.copy(), st, pl,
                                            class_id=c, temperature=0.0,
                                            pos_temperature=0.0, intent_decoding=True)
                        if probe is not None and st.can_apply_operation(pl, probe, ()):
                            c_pin = c
                            break
                    if c_pin is None:
                        continue
                    cells_all = np.argwhere(pm[c_pin])              # (x, y) —— 与 [ch, x, y] 口径一致
                    if len(cells_all) < 2:
                        continue
                    n_pos_dec += 1
                    vals_all = np.array([float(am[c_pin, x, y]) for (x, y) in cells_all])
                    order_all = np.argsort(-vals_all)

                    rows_cells, rows_adv = [], []
                    for i in order_all:
                        if len(rows_adv) >= max(2, int(collect_cap)):
                            break
                        x, y = int(cells_all[i][0]), int(cells_all[i][1])      # ← 不再交换
                        am2 = am.copy()
                        am2[c_pin] = -1e9
                        am2[c_pin, x, y] = 1e9
                        op = decode_head(hl[h], am2, cm.copy(), pm.copy(), st, pl,
                                         class_id=int(c_pin), temperature=0.0,
                                         pos_temperature=0.0, intent_decoding=True)
                        if op is None or not st.can_apply_operation(pl, op, ()):
                            n_cell_drop += 1
                            continue
                        if not _op_matches_cell(op, st, pl, int(c_pin), x, y):   # ← 同一个 (x,y)
                            n_cell_drop += 1
                            continue
                        post = st.clone()
                        post.apply_operation_list(pl, [op])
                        if pl == 1:
                            post.advance_round()        # 与 MC 的评分口径一致（p1 出招后回合推进）
                        s = float(quiet_value(post, pl, max_rounds=int(rollout_rounds)))
                        n_eval += 1
                        rows_cells.append([x, y])
                        rows_adv.append(s)
                    if len(rows_adv) < 2:
                        continue
                    # 只记录**分散**的 M 个（前 2 + 按排名分位分散）⇒ 目标是有覆盖的分布
                    n_c = len(rows_adv)
                    if int(M) <= 0:                     # 0 = 不抽样，全部记录（忠实版）
                        sel = list(range(n_c))
                        rows_cells = [rows_cells[j] for j in sel]
                        rows_adv = [rows_adv[j] for j in sel]
                        a = np.asarray(rows_adv, dtype=np.float32)
                        if float(np.ptp(a)) <= 0.0:
                            stats_cnt["all_equal"] += 1
                        else:
                            srt = np.sort(a)[::-1]
                            deltas.append(float(srt[0] - srt[1]))
                        out["board"].append(obs["board"]); out["stats"].append(obs["stats"])
                        out["player"].append(int(pl)); out["head"].append(int(h))
                        out["cls"].append(int(c_pin)); out["cell"].append(np.asarray(rows_cells, dtype=np.int16))
                        out["adv"].append(a); out["weight"].append(np.ones(len(rows_adv), dtype=np.float32))
                        pin_hist[int(c_pin)] += 1; n_rows += 1
                        continue
                    sel = [0, 1]
                    if int(M) > len(sel) and n_c > len(sel):
                        for q in np.linspace(0, n_c - 1, num=int(M)).astype(int):
                            if int(q) not in sel:
                                sel.append(int(q))
                            if len(sel) >= int(M):
                                break
                    sel = sorted(set(sel))[:max(2, int(M))]
                    rows_cells = [rows_cells[j] for j in sel]
                    rows_adv = [rows_adv[j] for j in sel]
                    a = np.asarray(rows_adv, dtype=np.float32)
                    if float(np.ptp(a)) <= 0.0:
                        stats_cnt["all_equal"] += 1
                    else:
                        srt = np.sort(a)[::-1]
                        deltas.append(float(srt[0] - srt[1]))
                    out["board"].append(obs["board"])
                    out["stats"].append(obs["stats"])
                    out["player"].append(int(pl))
                    out["head"].append(int(h))
                    out["cls"].append(int(c_pin))
                    out["cell"].append(np.asarray(rows_cells, dtype=np.int16))
                    out["adv"].append(a)
                    out["weight"].append(np.ones(len(rows_adv), dtype=np.float32))
                    pin_hist[int(c_pin)] += 1
                    head_hist[int(h)] += 1
                    n_rows += 1

                # ⚠️ 修正 1：把网络的束**真的打到局面上**并推进回合（历史 `collect_value_prior.py:195-198` 口径）
                ops = decode_network_output(o, st, pl, temperature=0.0, intent_decoding=True)
                st.apply_operation_list(pl, ops)
            if not st.terminal:
                st.advance_round()
        if True:   # 每局都打印（worker 只有 3~4 局，原来 %5 等于没有中间输出）
            print(f"  [worker seed={seed}] {g+1}/{games} 局（回合到 {rounds_seen}），位置决策 {n_pos_dec}，"
                  f"行 {n_rows}，rollout {n_eval}，落格失效 {n_cell_drop}，{time.time()-t0:.0f}s", flush=True)

    return dict(out=out, n_turn=n_turn, n_pos=n_pos_dec, n_rows=n_rows,
                n_cell_drop=n_cell_drop, n_eval=n_eval, deltas=deltas,
                stats_cnt=dict(stats_cnt), pin_hist=dict(pin_hist), head_hist=dict(head_hist),
                secs=time.time() - t0,
                rounds_seen=rounds_seen)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=25000)
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--max-cells", type=int, default=0, help="每决策**记录**的格数；0 = 不抽样、全部记录（忠实版默认）")
    ap.add_argument("--collect-cap", type=int, default=64, help="每决策**尝试**到多少个成功格为止（塔类的合法格本来只有 ~33 个 ⇒ 64 ≈ 全部）")
    ap.add_argument("--rollout-rounds", type=int, default=256)
    ap.add_argument("--probe-classes", type=int, default=8)
    ap.add_argument("--heads", type=int, default=3, help="在头 0..heads-1 上采（默认 3 = 全采，忠实版）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import multiprocessing as mp

    njobs = max(1, int(args.workers))
    per = [args.games // njobs + (1 if i < args.games % njobs else 0) for i in range(njobs)]
    jobs, s = [], int(args.seed)
    for k in per:
        if k <= 0:
            continue
        jobs.append((args.checkpoint, k, s, args.max_rounds, True, args.max_cells,
                     args.collect_cap, args.rollout_rounds, args.probe_classes, args.heads))
        s += k
    print(f"[posroll2] ckpt={args.checkpoint} 局数={args.games} workers={len(jobs)} "
          f"max_cells={args.max_cells} collect_cap={args.collect_cap} heads={args.heads} seed={args.seed}", flush=True)

    t0 = time.time()
    merged = {"board": [], "stats": [], "player": [], "head": [], "cls": [],
              "cell": [], "adv": [], "weight": []}
    n_turn = n_pos = n_rows = n_eval = n_cell_drop = 0
    deltas = []
    stats_cnt = collections.Counter()
    pin_hist = collections.Counter()
    head_hist = collections.Counter()
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        results = []
        with mp.Pool(len(jobs)) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                results.append(r)
                print(f"[posroll2] 一个 worker 完成（累计 {len(results)}/{len(jobs)}）", flush=True)
    for r in results:
        for k in merged:
            merged[k].extend(r["out"][k])
        n_turn += r["n_turn"]; n_pos += r["n_pos"]; n_rows += r["n_rows"]
        n_eval += r["n_eval"]; n_cell_drop += r["n_cell_drop"]; deltas += r["deltas"]
        stats_cnt.update(r["stats_cnt"]); pin_hist.update(r["pin_hist"]); head_hist.update(r["head_hist"])

    def _pad(rows, dtype, fill):
        """ragged ⇒ (N, C) padding（配 `cnt` 记录每行有效长度；行数不固定是因为"能解码的候选数"逐决策不同）"""
        C = max((len(r) for r in rows), default=1)
        arr = np.full((len(rows), C), fill, dtype=dtype)
        for i, r in enumerate(rows):
            arr[i, :len(r)] = r
        return arr

    def _pad2(rows, dtype, fill):
        """ragged 的 (n, 2) 坐标 ⇒ (N, C, 2)"""
        C = max((len(r) for r in rows), default=1)
        arr = np.full((len(rows), C, 2), fill, dtype=dtype)
        for i, r in enumerate(rows):
            arr[i, :len(r)] = r
        return arr

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        board=np.asarray(merged["board"], dtype=np.float16),
        stats=np.asarray(merged["stats"], dtype=np.float16),
        player=np.asarray(merged["player"], dtype=np.int8),
        head=np.asarray(merged["head"], dtype=np.int8),
        cls=_pad(merged["cls"], np.int16, -1),
        cell=_pad2(merged["cell"], np.int16, -1),             # (x, y)
        adv=_pad(merged["adv"], np.float32, 0.0),
        weight=_pad(merged["weight"], np.float32, 0.0),
        cnt=np.asarray([len(a) for a in merged["adv"]], dtype=np.int16),
    )
    d = np.asarray(deltas, dtype=np.float64) if deltas else np.zeros(0)
    summary = {
        "games": args.games, "workers": len(jobs), "max_cells": args.max_cells,
        "collect_cap": args.collect_cap, "seed": args.seed,
        "turns": n_turn, "position_decisions": n_pos, "rows": n_rows,
        "rollout_evals": n_eval, "cell_drops": n_cell_drop,
        "rows_all_equal": int(stats_cnt.get("all_equal", 0)),
        "pin_hist": dict(sorted(pin_hist.items(), key=lambda kv: -kv[1])[:8]),
        "head_hist": {str(k): v for k, v in sorted(head_hist.items())},
        "cells_per_row_mean": float(np.mean([len(a) for a in merged["adv"]])) if merged["adv"] else 0.0,
        "delta_median": float(np.median(d)) if d.size else None,
        "frac_delta_ge_0.05": float((d >= 0.05).mean()) if d.size else None,
        "frac_delta_ge_0.10": float((d >= 0.10).mean()) if d.size else None,
        "secs": round(time.time() - t0, 1),
    }
    Path(str(args.out) + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== 采集完成（修正版：真实对局 + [x,y] 口径）=====")
    print(f"  输出 = {args.out}（{Path(args.out).stat().st_size/1e6:.1f} MB）")
    print(f"  回合={n_turn} 位置决策={n_pos} 训练行={n_rows}（{summary['cells_per_row_mean']:.1f} 格/行）"
          f" rollout={n_eval} 落格失效={n_cell_drop} 用时={summary['secs']}s")
    print(f"  钉类分布：{summary['pin_hist']}")
    print(f"  **按头采到的行数**：{summary['head_hist']}（= 只采头 0 会损失的比例）")
    if d.size:
        print(f"  行内 |top1-top2|：中位 {summary['delta_median']:.3f}；≥0.05 占 {summary['frac_delta_ge_0.05']*100:.1f}%、"
              f"≥0.10 占 {summary['frac_delta_ge_0.10']*100:.1f}%；全平行 {summary['rows_all_equal']}/{n_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
