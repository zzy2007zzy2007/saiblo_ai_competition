"""采集**跨类**教师：对每个决策，给"若干候选类各自的最佳落格"打**安静 rollout** 分。

动机（2026-10-11 第 86 轮，读代码得到）：
- MC 的候选菜单来自 `_econ_rank`：对每个经济类 c 取
  `v_c = max(AM[c] over c 的合法格)`（`bundle_mcts.py:_econ_rank`），**按 v_c 跨类排序**，取前 `MC_TOPK` 个进菜单；
  代码注释自己写着"跨类 map 最大值不可比" ⇒ 菜单实际长期被类 0/16 占据 ⇒ 这是**升级候选进不了菜单**的机制根源。
- 所以**位置头决定菜单组成** ⇒ 把位置头训成"**per-class 最大值与 rollout 打分同序**"，菜单就会变好 ⇒ 一条真通道。
- 与路线 A 的区别：A 的目标是"同一类内部选哪一格"（**2 格二选一 ⇒ 不可学** ✗）；
  本目标跨**类**比较，类间分差是**量级差**（建造 vs HOLD/升级）⇒ 目标是可学的 ✓。

产物：npz，每个决策一行：
  `board (28,19,19) fp16 / stats (42,) fp16 / player / cls (N,) / cell (N,2)【(row,col)=(y,x)】/ score (N,) / cnt`
用法：
  python -u code/my_ai/az_intent/collect_class_rollout_teacher.py \
    --checkpoint checkpoints/grafted_sdk_A.pt --games 150 --workers 8 \
    --top-classes 5 --rollout-rounds 256 --seed 23000 --out training_history/posroll/class150.npz
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
LIGHT = 17


def _worker(job) -> dict:
    (ckpt, games, seed, max_rounds, native, top_classes, rollout_rounds) = job
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

    out = {"board": [], "stats": [], "player": [], "cls": [], "cell": [], "score": [], "cnt": []}
    n_turn = n_dec = n_rows = n_eval = n_drop = 0
    gaps = []
    t0 = time.time()

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
                # 候选类 = 经济类里的"per-class 最大值"前 top_classes（复刻 _econ_rank 的口径，排除闪电）
                rank = []
                for c in range(0, LIGHT):
                    if not cm[c] or not pm[c].any():
                        continue
                    v = float(np.where(pm[c], am[c], -np.inf).max())
                    if np.isfinite(v):
                        rank.append((v, int(c)))
                rank.sort(key=lambda x: -x[0])
                if len(rank) < 1:
                    continue
                # 候选集**复刻部署的 MC 菜单**：HOLD(23) + 经济类前 k。
                # ⚠️ 2026-10-11 实测：只取"经济类前 5"时**类间分差 100% 为 0**（它们实质是同一个动作）
                #    ⇒ 目标退化；必须把 HOLD 放进来（HOLD vs 行动 才是决定性的比较）。
                cands = [HOLD] + [c for _v, c in rank[:max(1, int(top_classes))]]
                n_dec += 1

                cells, scores, keep = [], [], []
                for c in cands:
                    if c == HOLD:
                        # HOLD = 空操作：不打任何 op，直接对"当前局面"做安静 rollout（p1 仍需推进回合）
                        post = st.clone()
                        if pl == 1:
                            post.advance_round()
                        s = float(quiet_value(post, pl, max_rounds=int(rollout_rounds)))
                        n_eval += 1
                        keep.append(int(c)); cells.append([-1, -1]); scores.append(s)
                        continue
                    # ⚠️ 2026-10-11 实测：**只试该类 argmax 的那一个格会得到行 0**（那个格解码失败整个类就废，
                    #    实测落格失效率 ~90%）。⇒ 按值降序**多试几格直到成功**（复刻部署的取值口径：掩码内最大值）。
                    # ⚠️ 2026-10-11 修：掩码是 `mask[ch, x, y]` ⇒ `np.argwhere` 给的就是 **(x, y)**
                    #    （旧版把变量命名成 ys/xs 再交换去校验 ⇒ 只有 x==y 的对角格能过 `_op_matches_cell`）
                    cells_xy = np.argwhere(pm[c])
                    if len(cells_xy) == 0:
                        continue
                    vals = am[c][cells_xy[:, 0], cells_xy[:, 1]]
                    op = None
                    x = y = -1
                    for i in np.argsort(-vals)[:8]:
                        x, y = int(cells_xy[i][0]), int(cells_xy[i][1])
                        am2 = am.copy()
                        am2[c] = -1e9
                        am2[c, x, y] = 1e9
                        _op = decode_head(hl[0], am2, cm.copy(), pm.copy(), st, pl, class_id=int(c),
                                          temperature=0.0, pos_temperature=0.0, intent_decoding=True)
                        if _op is None or not st.can_apply_operation(pl, _op, ()):
                            n_drop += 1
                            continue
                        if not _op_matches_cell(_op, st, pl, int(c), x, y):
                            n_drop += 1
                            continue
                        op = _op
                        break
                    if op is None:
                        continue
                    post = st.clone()
                    post.apply_operation_list(pl, [op])
                    if pl == 1:
                        post.advance_round()
                    s = float(quiet_value(post, pl, max_rounds=int(rollout_rounds)))
                    n_eval += 1
                    keep.append(int(c))
                    cells.append([x, y])                 # (x, y) 口径（与掩码/train_pos_class_rank 一致）
                    scores.append(s)
                if len(scores) < 2:
                    continue
                a = np.asarray(scores, dtype=np.float32)
                srt = np.sort(a)[::-1]
                gaps.append(float(srt[0] - srt[1]))
                out["board"].append(obs["board"])
                out["stats"].append(obs["stats"])
                out["player"].append(int(pl))
                out["cls"].append(np.asarray(keep, dtype=np.int16))
                out["cell"].append(np.asarray(cells, dtype=np.int16))
                out["score"].append(a)
                out["cnt"].append(int(len(keep)))
                n_rows += 1

                # ⚠️ 2026-10-11 修（原版缺失）：把该玩家网络的束**真的打到局面上**。
                #    必须**顺序**应用（p1 决策时必须已经看到 p0 的操作），所以不能用 resolve_turn。
                ops = decode_network_output(o, st, pl, temperature=0.0, intent_decoding=True)
                st.apply_operation_list(pl, ops)
            if not st.terminal:          # 双方都操作完 ⇒ **每回合唯一一次结算**（README §9 / 桥接 L719）
                st.advance_round()
        if True:                         # 每局都打印（worker 只有 2~12 局，原来 %5 等于没有中间输出）
            print(f"  [worker seed={seed}] {g+1}/{games} 局，决策 {n_dec}，行 {n_rows}，"
                  f"rollout {n_eval}，{time.time()-t0:.0f}s", flush=True)

    return dict(out=out, n_turn=n_turn, n_dec=n_dec, n_rows=n_rows, n_eval=n_eval,
                n_drop=n_drop, gaps=gaps, secs=time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=23000)
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--top-classes", type=int, default=5)
    ap.add_argument("--rollout-rounds", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import multiprocessing as mp

    njobs = max(1, int(args.workers))
    per = [args.games // njobs + (1 if i < args.games % njobs else 0) for i in range(njobs)]
    jobs, s = [], int(args.seed)
    for k in per:
        if k <= 0:
            continue
        jobs.append((args.checkpoint, k, s, args.max_rounds, True, args.top_classes, args.rollout_rounds))
        s += k
    print(f"[clsroll] ckpt={args.checkpoint} 局数={args.games} workers={len(jobs)} "
          f"top_classes={args.top_classes} seed={args.seed}", flush=True)

    t0 = time.time()
    merged = {"board": [], "stats": [], "player": [], "cls": [], "cell": [], "score": [], "cnt": []}
    n_dec = n_rows = n_eval = n_drop = 0
    gaps = []
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        results = []
        with mp.Pool(len(jobs)) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                results.append(r)
                print(f"[clsroll] 一个 worker 完成（累计 {len(results)}/{len(jobs)}）", flush=True)
    for r in results:
        for k in merged:
            merged[k].extend(r["out"][k])
        n_dec += r["n_dec"]; n_rows += r["n_rows"]; n_eval += r["n_eval"]; n_drop += r["n_drop"]
        gaps += r["gaps"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
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

    np.savez_compressed(
        args.out,
        board=np.asarray(merged["board"], dtype=np.float16),
        stats=np.asarray(merged["stats"], dtype=np.float16),
        player=np.asarray(merged["player"], dtype=np.int8),
        cls=_pad(merged["cls"], np.int16, -1),
        cell=_pad2(merged["cell"], np.int16, -1),             # (x, y)
        score=_pad(merged["score"], np.float32, 0.0),
        cnt=np.asarray([len(a) for a in merged["score"]], dtype=np.int16),
    )
    garr = np.asarray(gaps, dtype=np.float64) if gaps else np.zeros(0)
    summary = {
        "games": args.games, "workers": len(jobs), "top_classes": args.top_classes,
        "seed": args.seed, "decisions": n_dec, "rows": n_rows, "rollout_evals": n_eval,
        "cell_drops": n_drop,
        "gap_top1_top2_median": float(np.median(garr)) if garr.size else None,
        "gap_top1_top2_mean": float(garr.mean()) if garr.size else None,
        "frac_gap_ge_0.05": float((garr >= 0.05).mean()) if garr.size else None,
        "frac_gap_ge_0.20": float((garr >= 0.20).mean()) if garr.size else None,
        "frac_gap_eq_0": float((garr <= 1e-9).mean()) if garr.size else None,
        "secs": round(time.time() - t0, 1),
    }
    Path(str(args.out) + ".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n===== 跨类教师采集完成 =====")
    print(f"  输出 = {args.out}（{Path(args.out).stat().st_size/1e6:.1f} MB）")
    print(f"  决策={n_dec}  训练行={n_rows}  rollout={n_eval}  落格失效={n_drop}  用时={summary['secs']}s")
    if garr.size:
        print(f"  **类间分差（top1-top2）**：中位 {summary['gap_top1_top2_median']:.3f}、均值 {summary['gap_top1_top2_mean']:.3f}；"
              f"≥0.05 占 {summary['frac_gap_ge_0.05']*100:.1f}%，≥0.20 占 {summary['frac_gap_ge_0.20']*100:.1f}%，"
              f"=0 占 {summary['frac_gap_eq_0']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
