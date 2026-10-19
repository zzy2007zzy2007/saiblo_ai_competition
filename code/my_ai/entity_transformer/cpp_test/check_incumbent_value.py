"""检查 M6c 原 value_state 的原始输出范围，用于确定 AZAI_VALUE_TANH 的取值。"""
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "Ant-Game"))

from my_ai.az_intent.az_selfplay import load_three_models  # noqa: E402

ckpt = str(_REPO / "checkpoints" / "posnet_A_k5_m32.pt")
cls_m, pos_m, val_m = load_three_models(ckpt)
val_m.eval()
print("value model type:", type(val_m).__name__)

d = _REPO / "training_history" / "distill_top2" / "lowq_f64"
board = np.load(d / "board.npy", mmap_mode="r")
stats = np.load(d / "stats.npy", mmap_mode="r")
lab = np.load(d / "label.npy")

idx = np.arange(0, 20000, 1)
outs = []
with torch.no_grad():
    for s in range(0, len(idx), 512):
        bi = idx[s:s + 512]
        b = torch.from_numpy(np.asarray(board[bi], np.float32))
        st = torch.from_numpy(np.asarray(stats[bi], np.float32))
        o = val_m(b, st)
        if isinstance(o, dict):
            o = o["value"]
        outs.append(o.reshape(-1).cpu().numpy())
raw = np.concatenate(outs)
print("incumbent raw value: min=%.3f max=%.3f mean=%.3f std=%.3f" % (raw.min(), raw.max(), raw.mean(), raw.std()))
print("incumbent tanh(raw): min=%.3f max=%.3f std=%.3f" % (np.tanh(raw).min(), np.tanh(raw).max(), np.tanh(raw).std()))
y = lab[idx]
print("label (target):      min=%.3f max=%.3f mean=%.3f std=%.3f" % (y.min(), y.max(), y.mean(), y.std()))


def r2(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return 1.0 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()


print("incumbent raw  R2 vs label: %.4f" % r2(y, raw))
print("incumbent tanh R2 vs label: %.4f" % r2(y, np.tanh(raw)))
