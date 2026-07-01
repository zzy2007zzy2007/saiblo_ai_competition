"""Analyze ES parameter trajectory: are we stuck in local optimum or random walk?"""
from __future__ import annotations
import sys, math
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

OUT_DIR = Path("training_history/20260701_004058")

# ——— load all checkpoint mean vectors ———
ckpts: list[tuple[int, np.ndarray]] = []
for pt in sorted(OUT_DIR.glob("gen_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
    gen = int(pt.stem.split("_")[1])
    data = torch.load(pt, map_location="cpu", weights_only=True)
    mean = data.get("mean", data["top2_params"][1]).numpy()  # top2 mean if exists
    ckpts.append((gen, mean))

# ——— 1. consecutive cosine similarity ———
print("=" * 65)
print("Consecutive generation mean vector cosine similarity")
print("  cos≈1.0 → tiny change (stuck local optimum)")
print("  cos≈0.0 → random direction (random walk)")
print("  cos>0.7 → consistent direction (meaningful gradient)")
print("=" * 65)
cos_sims = []
for i in range(1, len(ckpts)):
    a = ckpts[i - 1][1].flatten()
    b = ckpts[i][1].flatten()
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    cos_sims.append(cos)
    print(f"  gen {ckpts[i-1][0]:3d} → {ckpts[i][0]:3d}:  cos={cos:.4f}  "
          f"norm={np.linalg.norm(ckpts[i-1][1]-ckpts[i][1]):.4f}")
print(f"\n  avg cos = {np.mean(cos_sims):.4f} ± {np.std(cos_sims):.4f}")

# ——— 2. Early vs Late direction consistency ———
# Check if mean shift directions are correlated (i.e. pointing same way)
half = len(ckpts) // 2
early_shifts = []
late_shifts = []
for i in range(1, len(ckpts)):
    shift = ckpts[i][1] - ckpts[i - 1][1]
    if i <= half:
        early_shifts.append(shift)
    else:
        late_shifts.append(shift)

# Within-group correlation
def avg_cos(shifts):
    dots = []
    for i in range(len(shifts)):
        for j in range(i + 1, len(shifts)):
            a, b = shifts[i].flatten(), shifts[j].flatten()
            dots.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return np.mean(dots), np.std(dots)

e_cos, e_std = avg_cos(early_shifts)
l_cos, l_std = avg_cos(late_shifts)
print(f"\n  Early-gen direction self-consistency:  avg_cos={e_cos:.4f} ± {e_std:.4f}")
print(f"  Late-gen  direction self-consistency:  avg_cos={l_cos:.4f} ± {l_std:.4f}")

# ——— 3. Total displacement vs path length ———
start = ckpts[0][1].flatten()
end = ckpts[-1][1].flatten()
total_displacement = np.linalg.norm(end - start)
total_path = sum(np.linalg.norm(ckpts[i][1].flatten() - ckpts[i - 1][1].flatten())
                 for i in range(1, len(ckpts))
                 if np.linalg.norm(ckpts[i][1].flatten() - ckpts[i - 1][1].flatten()) < 5)  # filter outliers
efficiency = total_displacement / total_path if total_path > 0 else 0
print(f"\n  Total displacement (start→end): {total_displacement:.4f}")
print(f"  Total path length:               {total_path:.4f}")
print(f"  Efficiency (disp/path):          {efficiency:.4f}")
print(f"    ≈1.0 → straight line (clear gradient direction)")
print(f"    ≈0.0 → random walk (back and forth)")

# ——— 4. Local norm ratio ———
stuck_count = 0
for i in range(max(1, len(ckpts) - 10), len(ckpts)):
    if np.linalg.norm(ckpts[i][1] - ckpts[i - 1][1]) < 0.01:
        stuck_count += 1
print(f"\n  Last {min(10, len(ckpts))} gens with ||shift||<0.01: {stuck_count}")
