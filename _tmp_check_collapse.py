"""Check the feature-collapse status of the checkpoints we actually use.

Replicates docs/az_training_validation_results.md §20's metric: the cross-SAMPLE
std of `spatial_feat` (the resblocks output).  A healthy model gives a different
feature map per board (std ~0.01-0.08); a collapsed one gives an identical map
for every board (std ~0).

States come from the stored real observations (warm_polonly npz) so we measure on
genuine, diverse positions.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / "Ant-Game"), str(REPO / "code")]


def spatial_feat_std(model, boards):
    with torch.no_grad():
        x = model.initial_conv(boards)
        for b in model.resblocks:
            x = b(x)
    # std over the batch dimension, per channel/position -> summarise
    return float(x.std(dim=0).mean().item()), float(x.abs().max().item())


def main():
    from my_ai.network import create_model, model_kwargs_from_ckpt

    npz = sorted((REPO / "training_history/az_fixed/warm_polonly").glob("*.npz"))
    boards = []
    for f in npz[:8]:
        d = np.load(f)
        boards.append(d["board"][:16])  # 16 frames x 8 games = 128 states
        d.close()
    b = torch.from_numpy(np.concatenate(boards)).float()
    print(f"probe states: {b.shape[0]} from {min(8, len(npz))} games\n")

    cands = [
        ("az_fixed/gen0120_bn_init", "current line hotstart"),
        ("az_fixed/az_r10", "current line POLICY (used in every recent arm)"),
        ("az_cpp/az_r69", "historically 46.9% policy"),
        ("az_cpp/az_r85", "historically 43.8% model"),
        ("az_cpp/az_r62", "historically 31.2% policy"),
    ]
    print(f"{'checkpoint':30s} {'cross-sample std':>17} {'max|act|':>10}  status        note")
    for name, note in cands:
        p = REPO / "training_history" / f"{name}.pt"
        if not p.exists():
            print(f"{name:30s} {'MISSING':>17}")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        try:
            m = create_model(**model_kwargs_from_ckpt(ck))
            m.load_state_dict(ck["model_state"])
        except Exception as e:
            print(f"{name:30s} LOAD FAIL: {e}")
            continue
        m.eval()
        std, mx = spatial_feat_std(m, b)
        status = "COLLAPSED" if std < 1e-4 else ("degraded" if std < 1e-3 else "healthy")
        print(f"{name:30s} {std:>17.8f} {mx:>10.3f}  {status:12s}  {note}")


if __name__ == "__main__":
    main()
