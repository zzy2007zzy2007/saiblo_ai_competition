"""Test if actual parameter displacement exceeds random-walk expectation."""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch


def analyze(out_dir: str, label: str):
    pts = sorted(Path(out_dir).glob("gen_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"\n{'='*65}")
    print(f"  {label}  ({len(pts)} checkpoints)")
    print(f"{'='*65}")

    vecs = []
    for pt in pts:
        data = torch.load(pt, map_location="cpu", weights_only=True)
        m = data.get("mean", data["top2_params"][1]).numpy().flatten()
        vecs.append(m)
    vecs = np.array(vecs)

    steps = [np.linalg.norm(vecs[i] - vecs[i-1]) for i in range(1, len(vecs))]
    mean_step = np.mean(steps)
    std_step = np.std(steps)

    N = len(steps)  # number of steps
    actual_disp = np.linalg.norm(vecs[-1] - vecs[0])

    # Random walk expectation: E[||disp||] ≈ mean_step * sqrt(N)
    rw_expected = mean_step * np.sqrt(N)
    rw_upper = (mean_step + std_step) * np.sqrt(N)

    ratio = actual_disp / rw_expected if rw_expected > 0 else float('inf')

    print(f"  Checkpoints: {pts[0].stem} → {pts[-1].stem}")
    print(f"  Mean step norm:   {mean_step:.4f} ± {std_step:.4f}")
    print(f"  Steps (N):        {N}")
    print(f"  Actual displacement:         {actual_disp:.4f}")
    print(f"  Random walk expected:        {rw_expected:.4f}")
    print(f"  Upper bound (mean+std):      {rw_upper:.4f}")
    print(f"  Ratio (actual / rw):         {ratio:.3f}")
    if ratio > 2.0:
        print(f"  >> SIGNAL: displacement is {ratio:.1f}x random walk, clear direction")
    elif ratio > 1.5:
        print(f"  > WEAK SIGNAL: moderate direction, some consistency")
    elif ratio > 1.0:
        print(f"  ~ NOISE: barely above random walk")
    else:
        print(f"  < COLLAPSE: less than random walk, oscillating/backtracking")


analyze("training_history/20260701_004058", "Training 6 (elite pool, stable)")
analyze("training_history/20260630_182516", "Training 3 (collapsed after gen_11)")
analyze("training_history/20260630_215620", "Training 4 (restart, oscillating)")
