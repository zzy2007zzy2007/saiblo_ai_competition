"""采集"**位置教师**"数据（路线 A 修订版，2026-10-10）—— 用**引擎安静 rollout**给位置候选打分。

为什么这么做（见 `docs/prereg_20261010_actionnet_retrain.md` §7）：
- 原计划想直接用**搜索的 visit 分布**当教师，但实测（29,303 个决策）**100% 的决策 top1 ≤ 1.2×top2、
  熵中位 1.000** ⇒ 搜索的 visit 在位置维上基本是**硬币**，只能教噪声（这正是历史关掉 `pos_distill` 的原因）。
- 改用**用户提的思路**：先**廉价**定类（这里用"按 logit 降序找第一个**真能解出非 None 操作**的类" =
  文档里的 `playable` 钉法），再对**位置图得分最高的前 M 个格**逐格做**安静 rollout**
  （`mc_quiet.quiet_value`：打这一手后双方冻结推进到终局，取终局 HP 差）⇒ 得到**确定性、有区分度**的软目标。

产物：`npz`，schema 与历史 `training_history/vprior/vp_*.npz` **一致**
（`boards/stats/player/head/cls/cells/adv/weight`）⇒ 训练**直接复用** `train_value_prior.py`：
    python -u code/my_ai/az_intent/train_value_prior.py --ckpt <三网 ckpt> --data <本 npz> \
      --epochs 60 --batch-size 256 --lr 1e-3 --target-tau 1.0 --t-pos 1.0 --out <新 pos_state 三网 ckpt>

用法（试点 50 局）：
  python -u code/my_ai/az_intent/collect_pos_rollout_teacher.py \
    --checkpoint checkpoints/grafted_sdk_A.pt --games 50 --workers 8 \
    --max-cells 4 --rollout-rounds 256 --seed 20000 --out training_history/posroll/pilot50.npz
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

import numpy as np

# ⚠️ 必须**模块级**设置（与 `collect_value_prior.py:47-50` 同一约定）：Windows 的 spawn 子进程会
#    重新导入 `__main__`，只有模块级代码才会在子进程里再执行一次 —— 否则子进程报
#    `ModuleNotFoundError: No module named 'SDK'`（2026-10-10 实测踩到）。
_REPO = Path(__file__).resolve().parents[3]
import sys as _sys

for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

HOLD = 23


def _worker(job) -> dict:
    (ckpt, games, seed, max_rounds, native, M, rollout_rounds, probe_classes) = job
    import torch

    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import (load_three_models, make_initial_state,
                                             make_net_fn_from_ckpt)
    from my_ai.az_intent.collect_value_prior import _op_matches_cell
    from my_ai.az_intent.mc_quiet import quiet_value
    from my_ai.decoder import (decode_head, make_class_mask, make_position_masks)

    feat = FeatureExtractor(max_actions=96)
    pol, _ = make_net_fn_from_ckpt(ckpt, feat)
    _c, _p, _v = load_three_models(ckpt)   # 只为确认这是三网 ckpt（本脚本只用 pol）

    # ⚠️ 键名必须是 `board`（单数）—— `train_value_prior.py:82` 读 `d["board"]`（2026-10-10 踩到）
    out = {"board": [], "stats": [], "player": [], "head": [], "cls": [],
           "cells": [], "adv": [], "weight": []}
    stats_cnt = collections.Counter()
    n_turn = n_pos_dec = n_rows = n_cell_drop = n_eval = 0
    deltas = []                 # 每行的 |top1-top2|（判定"教师有没有区分度"）
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
                for h in range(3):
                    # ① 廉价定类：按 logit 降序，取第一个**真能解出非 None 操作**的类（≠HOLD）
                    order = np.argsort(-hl[h])
                    c_pin = None
                    for c in order[:max(1, int(probe_classes))]:
                        c = int(c)
                        if c == HOLD or not (0 <= c < len(cm)) or not cm[c] or not pm[c].any():
                            continue
                        probe = decode_head(hl[h], am, cm.copy(), pm.copy(), st, pl,
                                            class_id=c, temperature=0.0,
                                            pos_temperature=0.0, intent_decoding=True)
                        if probe is not None and st.can_apply_operation(pl, probe, ()):
                            c_pin = c
                            break
                    if c_pin is None:
                        continue
                    cells_all = np.argwhere(pm[c_pin])
                    if len(cells_all) < 2:
                        continue
                    # ② 按位置图得分**降序扫描**，连续取前 M 个"真能解出可执行 op"的格
                    #     ⚠️ 2026-10-10 实测坑：先截前 M 格再试解码 ⇒ 多数格解不出 ⇒ rows=0
                    #     （每局 3072 个位置决策但只 1536 次 rollout、训练行 0）
                    vals_all = np.array([float(am[c_pin, y, x]) for (y, x) in cells_all])
                    keep_idx = np.argsort(-vals_all)
                    n_pos_dec += 1

                    rows_cells, rows_adv = [], []
                    for i in keep_idx:
                        if len(rows_adv) >= max(2, int(M)):
                            break
                        x, y = int(cells_all[i][1]), int(cells_all[i][0])
                        am2 = am.copy()
                        am2[c_pin] = -1e9
                        am2[c_pin, y, x] = 1e9
                        op = decode_head(hl[h], am2, cm.copy(), pm.copy(), st, pl,
                                         class_id=int(c_pin), temperature=0.0,
                                         pos_temperature=0.0, intent_decoding=True)
                        if op is None or not st.can_apply_operation(pl, op, ()):
                            n_cell_drop += 1
                            continue
                        if not _op_matches_cell(op, st, pl, int(c_pin), int(x), int(y)):
                            n_cell_drop += 1
                            continue
                        post = st.clone()
                        post.apply_operation_list(pl, [op])
                        if pl == 1:
                            post.advance_round()      # 与 MC 的评分口径一致（p1 出招后回合推进）
                        s = float(quiet_value(post, pl, max_rounds=int(rollout_rounds)))
                        n_eval += 1
                        rows_cells.append([int(x), int(y)])
                        rows_adv.append(s)
                    if len(rows_adv) < 2:
                        continue
                    a = np.asarray(rows_adv, dtype=np.float32)
                    if float(np.ptp(a)) <= 0.0:
                        stats_cnt["all_equal"] += 1
                    else:
                        srt = np.sort(a)[::-1]
                        deltas.append(float(srt[0] - srt[1]))
                    out["boards"].append(obs["board"])
                    out["stats"].append(obs["stats"])
                    out["player"].append(int(pl))
                    out["head"].append(int(h))
                    out["cls"].append(int(c_pin))
                    out["cells"].append(np.asarray(rows_cells, dtype=np.int16))
                    out["adv"].append(a)
                    out["weight"].append(np.ones(len(rows_adv), dtype=np.float32))
                    n_rows += 1
        if (g + 1) % 5 == 0:
            print(f"  [worker seed={seed}] {g+1}/{games} 局，位置决策 {n_pos_dec}，行 {n_rows}，"
                  f"rollout 次数 {n_eval}，{time.time()-t0:.0f}s", flush=True)

    return dict(out=out, n_turn=n_turn, n_pos=n_pos_dec, n_rows=n_rows,
                n_cell_drop=n_cell_drop, n_eval=n_eval, deltas=deltas,
                stats_cnt=dict(stats_cnt), secs=time.time() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--max-cells", type=int, default=4)
    ap.add_argument("--rollout-rounds", type=int, default=256)
    ap.add_argument("--probe-classes", type=int, default=8)
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
                     args.rollout_rounds, args.probe_classes))
        s += k
    print(f"[posroll] ckpt={args.checkpoint} 局数={args.games} workers={len(jobs)} "
          f"max_cells={args.max_cells} rollout_rounds={args.rollout_rounds} seed={args.seed}",
          flush=True)

    t0 = time.time()
    results = []
    if len(jobs) == 1:
        results = [_worker(jobs[0])]
    else:
        with mp.Pool(len(jobs)) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                results.append(r)
                print(f"[posroll] 一个 worker 完成（累计 {len(results)}/{len(jobs)}）", flush=True)

    merged = {"board": [], "stats": [], "player": [], "head": [], "cls": [],
              "cells": [], "adv": [], "weight": []}
    n_turn = n_pos = n_rows = n_eval = n_cell_drop = 0
    deltas = []
    stats_cnt = collections.Counter()
    for r in results:
        for k in merged:
            merged[k].extend(r["out"][k])
        n_turn += r["n_turn"]; n_pos += r["n_pos"]; n_rows += r["n_rows"]
        n_eval += r["n_eval"]; n_cell_drop += r["n_cell_drop"]; deltas += r["deltas"]
        stats_cnt.update(r["stats_cnt"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        board=np.asarray(merged["board"], dtype=np.float16),
        stats=np.asarray(merged["stats"], dtype=np.float16),
        player=np.asarray(merged["player"], dtype=np.int8),
        head=np.asarray(merged["head"], dtype=np.int8),
        cls=np.asarray(merged["cls"], dtype=np.int16),
        cells=np.asarray(merged["cells"], dtype=np.int16),
        adv=np.asarray(merged["adv"], dtype=np.float32),
        weight=np.asarray(merged["weight"], dtype=np.float32),
    )

    # ---- 信号闸门（试点专用）：rollout 分数到底有没有区分度 ----
    d = np.asarray(deltas, dtype=np.float64) if deltas else np.zeros(0)
    summary = {
        "games": args.games, "workers": len(jobs), "max_cells": args.max_cells,
        "rollout_rounds": args.rollout_rounds, "seed": args.seed,
        "turns": n_turn, "position_decisions": n_pos, "rows": n_rows,
        "rollout_evals": n_eval, "cell_drops": n_cell_drop,
        "rows_all_equal": int(stats_cnt.get("all_equal", 0)),
        "delta_n": int(d.size),
        "delta_median": float(np.median(d)) if d.size else None,
        "delta_mean": float(d.mean()) if d.size else None,
        "frac_delta_ge_0.05": float((d >= 0.05).mean()) if d.size else None,
        "frac_delta_ge_0.10": float((d >= 0.10).mean()) if d.size else None,
        "frac_delta_eq_0": float((d <= 1e-9).mean()) if d.size else None,
        "secs": round(time.time() - t0, 1),
    }
    sp = str(args.out) + ".summary.json"
    Path(sp).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== 采集完成 =====")
    print(f"  输出 = {args.out}（{Path(args.out).stat().st_size/1e6:.1f} MB）；摘要 = {sp}")
    print(f"  回合数={n_turn}  位置决策={n_pos}  训练行={n_rows}  rollout 次数={n_eval}  "
          f"落格失效丢弃={n_cell_drop}  用时={summary['secs']}s")
    print(f"  **全平的决策 = {summary['rows_all_equal']}/{n_rows}**"
          f"（若 ≈100% ⇒ 安静 rollout 在位置维上也没区分度 ⇒ 路线 A 判死）")
    if d.size:
        print(f"  |top1-top2|：中位 {summary['delta_median']:.3f}，均值 {summary['delta_mean']:.3f}；"
              f"≥0.05(=1 HP) 占 {summary['frac_delta_ge_0.05']*100:.1f}%，"
              f"≥0.10(=2 HP) 占 {summary['frac_delta_ge_0.10']*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
