"""诊断：蒸馏候选 (`grafted_sdk_A_visitmarg.pt`) 的位置图**坏在哪**（冒烟 0 建塔）。

对照三个 ckpt 在**同一状态**上的 `action_map`：
  * 数值健全性（finite / min / max / mean / std）；
  * 全图 argmax 的格与值；
  * 精确并列（tie）的比例；
  * 在**合法类掩码 + 位置掩码**下能不能解出操作（用 `decode_head` 逐类试）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    import os
    os.environ.setdefault("AZAI_BAN_CLASSES", "17")
    os.environ.setdefault("AZAI_BAN_OPS", "21")

    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import make_initial_state, load_three_models
    from my_ai.decoder import decode_head, make_class_mask, make_position_masks

    feat = FeatureExtractor(max_actions=96)
    st = make_initial_state(90000, True)
    for _ in range(80):
        st.advance_round()
    pmask = make_position_masks(st, 0, intent_decoding=True)
    cmask = make_class_mask(st, 0, position_mask=pmask, intent_decoding=True)
    print(f"  合法类数={int(cmask.sum())}；类 0 合法格数={int(pmask[0].sum())}；类 16 合法格数={int(pmask[16].sum())}")

    import torch
    from SDK.utils.features import FeatureExtractor as _FE
    obs = _FE(max_actions=96).encode_observation(st, 0, np.zeros(96))
    bb = torch.from_numpy(np.asarray(obs["board"], dtype=np.float32)).unsqueeze(0)
    ss = torch.from_numpy(np.asarray(obs["stats"], dtype=np.float32)).unsqueeze(0)

    for name in ("grafted_sdk_A.pt", "grafted_sdk_A_visitmarg.pt"):
        _cm, pm, _vm = load_three_models(f"checkpoints/{name}")
        pm.eval()
        with torch.no_grad():
            am = pm(bb, ss)["action_map"][0].numpy().astype(np.float64)
        fin = np.isfinite(am)
        print(f"\n  === {name}")
        print(f"    finite={fin.all()}；min={np.nanmin(am):+.4f} max={np.nanmax(am):+.4f} "
              f"mean={np.nanmean(am):+.4f} std={np.nanstd(am):.4f}")
        flat = am.reshape(am.shape[0], -1)
        idx = np.unravel_index(int(np.nanargmax(flat)), flat.shape)
        print(f"    全图 argmax：类 {idx[0]} 格 ({idx[1] // 19}, {idx[1] % 19})，值 {np.nanmax(flat):+.4f}")
        for c in (0, 16):
            m = am[c]
            mx = np.nanmax(m)
            ties = int((m == mx).sum())
            legal = pmask[c] if c < pmask.shape[0] else np.zeros_like(m, dtype=bool)
            print(f"    类 {c:2d}：max={mx:+.4f} 并列个数={ties}；合法格内 max="
                  f"{np.nanmax(np.where(legal, m, -np.inf)):+.4f}；能否解出 op="
                  f"{decode_head(am[c], am, cmask, pmask, st, 0, class_id=c, temperature=0.0, pos_temperature=0.0, intent_decoding=True) is not None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
