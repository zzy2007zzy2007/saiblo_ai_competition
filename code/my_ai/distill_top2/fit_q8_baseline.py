"""可行性基线：用 GBM 从「状态特征 + 计划特征」预测 `mean_terminal_score`（8 回合期望值）。

回答的问题：这份标签里有多少**状态内**（同状态下区分不同候选计划）的可解释信号？
- 绝对 R² 会被"状态间差异"灌水（不同回合分数差几千），必须看状态内残差。
- 决策只看状态内排序，所以最关键的是 **top-1 一致率**和**成对排序正确率**。

切分方式：**按状态随机切分**（不是按 seed）。按 seed 切分时模型要跨对局泛化，
之前实测 test R² = −0.24，那测的是"跨对局迁移"而不是"状态内判别"。

用法：
    python fit_q8_baseline.py --data training_history/distill_top2/q8/q8_dataset_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor

SKIP_KEYS = {"seed", "round", "player", "key", "ops", "mean", "std", "n"}


def load(path: Path):
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    feature_names = [key for key in rows[0]
                     if key not in SKIP_KEYS and isinstance(rows[0][key], (int, float))]
    matrix = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for index, row in enumerate(rows):
        for column, name in enumerate(feature_names):
            matrix[index, column] = float(row.get(name, 0.0))
    state_ids = np.array([
        row["seed"] * 1_000_000 + row["round"] * 10 + row["player"] for row in rows
    ], dtype=np.int64)
    ys = np.array([row["mean"] for row in rows], dtype=np.float64)
    heurs = np.array([row["heuristic"] for row in rows], dtype=np.float64)
    is_hold = np.array([row["key"] == "hold" for row in rows], dtype=bool)
    return matrix, feature_names, state_ids, ys, heurs, is_hold, rows


def r2(truth: np.ndarray, guess: np.ndarray) -> float:
    ss_res = float(np.sum((truth - guess) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = (np.sum(ra * ra) * np.sum(rb * rb)) ** 0.5
    return float(np.sum(ra * rb) / den) if den > 1e-12 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_history/distill_top2/q8/q8_dataset_v2.jsonl")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--leaves", type=int, default=63)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matrix, feature_names, state_ids, ys, heurs, is_hold, rows = load(Path(args.data))
    unique_states = np.unique(state_ids)
    print(f"[数据] {matrix.shape[0]} 行 × {matrix.shape[1]} 特征, {len(unique_states)} 个状态")

    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(unique_states)
    n_test = max(1, int(len(unique_states) * args.test_frac))
    test_states = set(shuffled[:n_test].tolist())
    test_mask = np.array([int(s) in test_states for s in state_ids])
    train_mask = ~test_mask
    print(f"[切分] 按状态随机：train {train_mask.sum()} 行 / test {test_mask.sum()} 行 "
          f"({len(unique_states) - n_test} / {n_test} 个状态)")

    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter, learning_rate=args.lr, max_leaf_nodes=args.leaves,
        early_stopping=False, random_state=0)
    model.fit(matrix[train_mask], ys[train_mask])
    pred = model.predict(matrix)
    print("[训练] 完成")

    print(f"\n=== 绝对 R²（参照，含状态间差异）===")
    print(f"train = {r2(ys[train_mask], pred[train_mask]):.5f}")
    print(f"test  = {r2(ys[test_mask], pred[test_mask]):.5f}")

    resid_true: list[float] = []
    resid_pred: list[float] = []
    top1 = top1_hold = 0
    n_states = 0
    rho: list[float] = []
    pair_total = pair_correct = 0
    pair_weighted_total = pair_weighted_correct = 0.0
    act_total = act_hit = 0
    hold_total = hold_ok = 0
    always_hold_agree = 0

    for state in test_states:
        mask = state_ids == state
        truth = ys[mask]
        guess = pred[mask]
        bonus = heurs[mask]
        holds = is_hold[mask]
        if len(truth) < 5:
            continue
        n_states += 1
        resid_true.extend((truth - truth.mean()).tolist())
        resid_pred.extend((guess - guess.mean()).tolist())
        rho.append(spearman(truth, guess))

        truth_pick = int(np.argmax(truth + bonus))
        guess_pick = int(np.argmax(guess + bonus))
        top1 += int(truth_pick == guess_pick)
        top1_hold += int(holds[truth_pick])
        always_hold_agree += int(holds[truth_pick])  # 永远 hold 的一致性

        truth_acts = not holds[truth_pick]
        guess_acts = not holds[guess_pick]
        if truth_acts:
            act_total += 1
            act_hit += int(guess_acts)
        else:
            hold_total += 1
            hold_ok += int(not guess_acts)

        # 成对排序正确率（同状态内两两比较）
        order = np.argsort(-truth)
        top = order[: min(12, len(order))]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = top[i], top[j]
                if truth[a] == truth[b]:
                    continue
                correct = int((guess[a] - guess[b]) > 0)
                weight = float(truth[a] - truth[b])
                pair_total += 1
                pair_correct += correct
                pair_weighted_total += weight
                pair_weighted_correct += weight * correct

    resid_true_arr = np.array(resid_true)
    resid_pred_arr = np.array(resid_pred)
    print(f"\n=== 状态内指标（{n_states} 个测试状态）===")
    print(f"状态内残差 R²         = {r2(resid_true_arr, resid_pred_arr):.4f}")
    print(f"状态内 Spearman 中位  = {statistics.median(rho):+.4f}")
    print(f"top-1 一致率（含先验）= {top1}/{n_states} = {100.0 * top1 / n_states:.1f}%"
          f"   （其中真值是 hold 的 {top1_hold} 个）")
    print(f"该动时动了（召回）    = {act_hit}/{act_total}"
          + (f" = {100.0 * act_hit / act_total:.1f}%" if act_total else ""))
    print(f"该 hold 时 hold      = {hold_ok}/{hold_total}"
          + (f" = {100.0 * hold_ok / hold_total:.1f}%" if hold_total else ""))
    print(f"\n前 12 名内部成对排序正确率 = {100.0 * pair_correct / max(pair_total, 1):.1f}%"
          f"（按分差加权 {100.0 * pair_weighted_correct / max(pair_weighted_total, 1e-9):.1f}%）")
    spread = [float(ys[state_ids == s].max() - ys[state_ids == s].min()) for s in test_states]
    print(f"状态内真实分数离散度（max-min）均值 = {statistics.mean(spread):.0f}")
    print(f"\n（参照：亚军的决策分差中位 ~71；单次 rollout σ 中位 111，SE(125) 中位 9.9）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
