"""训练「计划条件化的实体网络」预测 8 回合期望值（`mean_terminal_score`）。

评价规范（与 `fit_q8_baseline.py` 完全一致，便于对比）：
- 按**状态**随机切分（`group` 字段）
- 状态内 Spearman / top-1 一致率（含 +80 先验）/ 该动时动了 / 该 hold 时 hold
- **必须与「永远 hold」基线并列**：任何"复刻亚军决策"的说法都要先打过它

用法：
    python train_q8_value.py --data training_history/distill_top2/q8/q8_ent.npz \
        --target resid --epochs 40 --out checkpoints/q8_value_resid.pt
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "code"))
from my_ai.entity_transformer.q8_value_model import PlanConditionedEntityValue, count_params  # noqa: E402

TENSOR_KEYS = ("tower_cont", "tower_type", "tower_pos", "tower_mask",
               "ant_cont", "ant_kind", "ant_behavior", "ant_pos", "ant_mask",
               "state_feats", "plan_feats", "plan_pos")
INT_KEYS = {"tower_type", "tower_pos", "ant_kind", "ant_behavior", "ant_pos", "plan_pos"}
BOOL_KEYS = {"tower_mask", "ant_mask"}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = (np.sum(ra * ra) * np.sum(rb * rb)) ** 0.5
    return float(np.sum(ra * rb) / den) if den > 1e-12 else float("nan")


def to_batch(data: dict[str, np.ndarray], index: np.ndarray, device: torch.device) -> dict:
    batch = {}
    for key in TENSOR_KEYS:
        array = data[key][index]
        tensor = torch.from_numpy(np.ascontiguousarray(array))
        if key in INT_KEYS:
            tensor = tensor.long()
        elif key in BOOL_KEYS:
            tensor = tensor.bool()
        else:
            tensor = tensor.float()
        batch[key] = tensor.to(device)
    return batch


def evaluate(model, data, index: np.ndarray, groups: np.ndarray, device, target_mean: float,
             target_std: float, group_scale: dict[int, float] | None = None) -> dict:
    """group_scale：`resid_z` 目标时每个状态的标准差，用来把 z 值换回原始分数尺度。

    ⚠️ 不做这步的话，预测值（±0.5 的 z 值）永远打不过 hold 的 +80 加分，
       会误判成"模型只会 hold"——这是标度错，不是模型能力问题。
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(index), 2048):
            chunk = index[start:start + 2048]
            output = model(to_batch(data, chunk, device)).squeeze(-1)
            preds.append((output * target_std + target_mean).cpu().numpy())
    pred = np.concatenate(preds)

    truth_all = data["value"][index]
    heur_all = data["heuristic"][index]
    hold_all = data["is_hold"][index]
    group_all = data["group"][index]

    rho, top1, n_states = [], 0, 0
    act_hit = act_total = hold_ok = hold_total = 0
    pair_total = pair_correct = 0
    for group in np.unique(group_all):
        mask = group_all == group
        scale = 1.0 if group_scale is None else float(group_scale.get(int(group), 1.0))
        truth = truth_all[mask]
        guess = pred[mask] * scale
        bonus = heur_all[mask]
        holds = hold_all[mask]
        if len(truth) < 5:
            continue
        n_states += 1
        rho.append(spearman(truth, guess))
        truth_pick = int(np.argmax(truth + bonus))
        guess_pick = int(np.argmax(guess + bonus))
        top1 += int(truth_pick == guess_pick)
        if holds[truth_pick]:
            hold_total += 1
            hold_ok += int(holds[guess_pick])
        else:
            act_total += 1
            act_hit += int(not holds[guess_pick])
        order = np.argsort(-truth)[: min(12, len(truth))]
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                a, b = order[i], order[j]
                if truth[a] == truth[b]:
                    continue
                pair_total += 1
                pair_correct += int((guess[a] - guess[b]) > 0)
    return {
        "n_states": n_states,
        "rho_median": float(statistics.median(rho)) if rho else float("nan"),
        "top1": top1 / max(n_states, 1),
        "act_recall": act_hit / act_total if act_total else float("nan"),
        "hold_precision": hold_ok / hold_total if hold_total else float("nan"),
        "pairwise": pair_correct / max(pair_total, 1),
        "always_hold_baseline": hold_total / max(n_states, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_history/distill_top2/q8/q8_ent.npz")
    parser.add_argument("--target", choices=["abs", "resid", "resid_z"], default="resid_z")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--split-by", choices=["group", "seed"], default="group",
                        help="group=按 (seed,round,player) 随机切；seed=按整局切（更严格的泛化测试）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="checkpoints/q8_value.pt")
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = np.load(args.data)
    data = {key: raw[key] for key in raw.files}
    print(f"[数据] {data['value'].shape[0]} 行, state {data['state_feats'].shape[1]} 维, "
          f"plan {data['plan_feats'].shape[1]} 维, device={device}")

    groups = data["group"]
    units = groups // 1_000_000 if args.split_by == "seed" else groups
    unique = np.unique(units)
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(unique)
    n_test = max(1, int(len(unique) * args.test_frac))
    test_groups = np.isin(units, shuffled[:n_test])
    train_index = np.flatnonzero(~test_groups)
    test_index = np.flatnonzero(test_groups)
    print(f"[切分] 按 {args.split_by}: train {len(train_index)} 行 "
          f"({len(unique) - n_test} 个单元) / test {len(test_index)} 行 ({n_test} 个单元)")

    # 目标
    label = data["value"].astype(np.float64)
    group_scale: dict[int, float] | None = None
    if args.target in ("resid", "resid_z"):
        group_values: dict[int, list[float]] = {}
        for group, value in zip(groups.tolist(), label.tolist()):
            group_values.setdefault(group, []).append(value)
        means = {key: float(np.mean(values)) for key, values in group_values.items()}
        spreads = {key: float(np.std(values)) or 1.0 for key, values in group_values.items()}
        target = np.array([value - means[group] for value, group in zip(label, groups.tolist())])
        if args.target == "resid_z":
            # 每个状态内再除以自身标准差 ⇒ 所有状态等权，模型只关心"状态内谁更好"。
            # （不做这一步时，离散度大的状态会主导 MSE，模型倾向输出常数、只挑 hold。）
            target = np.array([residual / spreads[group]
                               for residual, group in zip(target, groups.tolist())])
            group_scale = spreads
    else:
        target = label
    target_mean = float(target[train_index].mean())
    target_std = float(target[train_index].std()) or 1.0
    target_norm = (target - target_mean) / target_std
    print(f"[目标] {args.target}: mean={target_mean:.2f} std={target_std:.2f}")

    model = PlanConditionedEntityValue(
        d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        state_dim=data["state_feats"].shape[1], plan_dim=data["plan_feats"].shape[1]).to(device)
    print(f"[模型] {count_params(model)/1e6:.2f}M 参数")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    log_handle = open(args.log, "w", encoding="utf-8") if args.log else None

    def log(message: str) -> None:
        print(message, flush=True)
        if log_handle:
            log_handle.write(message + "\n")
            log_handle.flush()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rng = np.random.default_rng(args.seed * 1000 + epoch)
        order = epoch_rng.permutation(train_index)
        losses = []
        t0 = time.time()
        for start in range(0, len(order), args.batch):
            chunk = order[start:start + args.batch]
            batch = to_batch(data, chunk, device)
            output = model(batch).squeeze(-1)
            loss = F.mse_loss(output, torch.from_numpy(target_norm[chunk]).float().to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            metrics = evaluate(model, data, test_index, groups, device, target_mean, target_std,
                               group_scale)
            metrics["epoch"] = epoch
            metrics["train_loss"] = float(np.mean(losses))
            metrics["seconds"] = round(time.time() - t0, 1)
            history.append(metrics)
            log(f"[ep {epoch:3d}] loss={metrics['train_loss']:.5f} "
                f"rho={metrics['rho_median']:+.3f} top1={100*metrics['top1']:.1f}% "
                f"(hold基线 {100*metrics['always_hold_baseline']:.1f}%) "
                f"act={100*metrics['act_recall']:.1f}% hold_ok={100*metrics['hold_precision']:.1f}% "
                f"pair={100*metrics['pairwise']:.1f}% ({metrics['seconds']}s)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "target_mean": target_mean,
        "target_std": target_std,
        "state_dim": int(data["state_feats"].shape[1]),
        "plan_dim": int(data["plan_feats"].shape[1]),
        "history": history,
    }, out_path)
    log(f"[保存] {out_path}")
    if log_handle:
        log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
