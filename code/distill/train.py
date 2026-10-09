"""Train a model on BC data collected from ExampleAI.

Uses ss_train's SSDataset and ss_supervised_update with class-balanced
sampling, label smoothing, weight decay, and head diversity loss.

Usage:
    python code/distill/train.py distill_data/ --out distill_model.pt --epochs 30
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
from my_ai.network import create_model
from my_ai.ss_train import SSDataset, ss_supervised_update, save_checkpoint
from utils.logger import get_logger


def compute_class_weights(dataset: SSDataset, max_weight: float = 5.0) -> torch.Tensor | None:
    """Compute inverse-frequency class weights, clipped to ``max_weight``.

    Clipping prevents rare classes from dominating the loss (which
    causes the model to collapse to predicting the highest-weight class).
    """
    labels = dataset.class_label  # (T, N_heads)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels) * labels.shape[1]
    class_weights = np.ones(24, dtype=np.float32)
    for cid, cnt in zip(unique, counts):
        freq = cnt / total
        if freq > 0:
            class_weights[int(cid)] = 1.0 / freq
    # Clip to max_weight
    class_weights = np.clip(class_weights, None, max_weight)
    # Normalize so mean weight = 1
    class_weights = class_weights / class_weights.mean()
    return torch.from_numpy(class_weights)


def main():
    parser = argparse.ArgumentParser(description="BC distillation training")
    parser.add_argument("data_dir", type=str, help="directory with .npz files")
    parser.add_argument("--out", type=str, default="distill_model.pt",
                        help="output checkpoint path")
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--p-hold", type=float, default=1.0,
                        help="HOLD downsampling ratio (1.0 = keep all)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="label smoothing factor")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW weight decay")
    parser.add_argument("--bias-decay", type=float, default=1e-3,
                        help="extra L2 penalty on policy head biases")
    parser.add_argument("--lambda-div", type=float, default=0.01,
                        help="head diversity loss weight")
    # 2026-10-19（本 goal 第 1 轮）：消融"那 +42pp 来自哪个头"
    parser.add_argument("--lambda-class", type=float, default=1.0,
                        help="class-head loss weight (0 = 只训动作图)")
    parser.add_argument("--lambda-map", type=float, default=1.0,
                        help="action-map loss weight (0 = 只训类头)")
    parser.add_argument("--lambda-soft", type=float, default=0.5,
                        help="soft-target KL loss weight (0=disable)")
    parser.add_argument("--soft-temperature", type=float, default=2.0,
                        help="temperature for soft-target softmax")
    parser.add_argument("--no-class-weights", action="store_true",
                        help="disable inverse-frequency class weighting")
    parser.add_argument("--max-class-weight", type=float, default=5.0,
                        help="clip class weights to this maximum (prevents collapse)")
    parser.add_argument("--log", type=str, default=None,
                        help="log file path")
    # 2026-10-19（AZ 环预注册 `prereg_20261019_goal2_az_loop.md`）：**微调**而不是从零重训
    parser.add_argument("--init", type=str, default=None,
                        help="从该 ckpt 初始化（支持 model_state（单模型）或 class_state（三网部署格式））")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    log = None
    if args.log:
        log = get_logger(args.log)

    # ── Data ──────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    npz_paths = sorted(data_dir.glob("distill_*.npz"))
    if not npz_paths:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Found {len(npz_paths)} .npz files")
    ds = SSDataset(npz_paths, p_hold=args.p_hold)
    print(f"  Total samples: {len(ds)}")

    # Class weights (inverse-frequency, clipped)
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(ds, max_weight=args.max_class_weight)
        print(f"  Class weights: min={class_weights.min():.3f}, "
              f"max={class_weights.max():.3f}, "
              f"mean={class_weights.mean():.3f}")
        # Show per-class weights
        classes_with_weight = [(i, class_weights[i].item()) for i in range(24)]
        classes_with_weight.sort(key=lambda x: -x[1])
        print(f"  Top-5 weighted classes: "
              f"{[(f'c{c}', f'{w:.1f}') for c, w in classes_with_weight[:5]]}")

    # ── Model ─────────────────────────────────────────────────────────
    model = create_model(num_heads=args.num_heads)
    print(f"Model: {model.count_parameters():,} params, {args.num_heads} heads")

    if args.init:
        import torch as _torch
        ck = _torch.load(args.init, map_location="cpu")
        if isinstance(ck, dict) and "model_state" in ck:
            sd = ck["model_state"]
        elif isinstance(ck, dict) and "class_state" in ck:
            sd = ck["class_state"]
        else:
            sd = ck
        sd = {k: v for k, v in sd.items() if hasattr(v, "shape")}
        cur = model.state_dict()
        miss = [k for k in cur if k not in sd]
        extra = [k for k in sd if k not in cur]
        shape_bad = [k for k in cur if k in sd and tuple(sd[k].shape) != tuple(cur[k].shape)]
        keep = {k: v for k, v in sd.items() if k in cur and tuple(sd[k].shape) == tuple(cur[k].shape)}
        cur.update(keep)
        model.load_state_dict(cur)
        print(f"  --init={args.init}: 载入 {len(keep)}/{len(cur)} 个张量；缺 {len(miss)}；多 {len(extra)}；形状不符 {len(shape_bad)}")
        if miss[:3]:
            print(f"    缺失示例: {miss[:3]}")

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nTraining: epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}")
    print(f"  label_smoothing={args.label_smoothing}, weight_decay={args.weight_decay}, "
          f"bias_decay={args.bias_decay}")
    print(f"  lambda_div={args.lambda_div}, lambda_soft={args.lambda_soft}, "
          f"soft_temp={args.soft_temperature}")
    print(f"  class_weights={'yes' if class_weights is not None else 'no'}")
    ret = ss_supervised_update(model, ds, device,
                               epochs=args.epochs, lr=args.lr,
                               batch_size=args.batch_size,
                               label_smoothing=args.label_smoothing,
                               weight_decay=args.weight_decay,
                               bias_decay=args.bias_decay,
                               lambda_div=args.lambda_div,
                               lambda_class=args.lambda_class,
                               lambda_map=args.lambda_map,
                               lambda_soft=args.lambda_soft,
                               soft_temperature=args.soft_temperature,
                               class_weights=class_weights,
                               log=log)

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    mean = model.get_parameters_as_vector()
    save_checkpoint(out_path, mean, model, 0)
    print(f"\nSaved: {out_path}")
    print(f"  cls={ret['class_loss']:.4f}  map={ret['map_loss']:.4f}  "
          f"tot={ret['total_loss']:.4f}  samples={ret['samples']}")


if __name__ == "__main__":
    main()
