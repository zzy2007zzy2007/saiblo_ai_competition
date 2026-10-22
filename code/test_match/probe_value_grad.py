"""用 **autograd** 直接测「价值网对各输入的敏感度」——不需要改任何状态。

为什么要这版：
  · 之前的 `probe_value_coin_slope.py` 用"给 clone 加金币"做干预，但
    `state.coins[0] = x` 在 facade 上是**写进缓存列表**，`_cache.clear()` 会退回引擎真值
    ⇒ 干预实际无效 ⇒ 那次测出的 ∂V/∂coins ≡ 0 **是假象**（2026-10-22 发现）。
  · 正确做法（用户提议的等价形式）：把观测当 `requires_grad=True` 的张量，前向一次、反传一次，
    `∂V/∂输入` 就是解析梯度。给输入接"固定偏置层"再对这个偏置求导，与直接对输入求导完全等价。

输出：
  1. stats 42 维里，**金币两维**的 |∂V/∂x|（并给出全体分位数做参照）
  2. board 28 通道里，每通道的平均 |∂V/∂x|
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

from SDK.utils.features import FeatureExtractor                      # noqa: E402
from my_ai.az_intent.az_selfplay import load_three_models           # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade        # noqa: E402

CKPT = REPO / "checkpoints/posnet_A_k5_m32.pt"
N_STATES = 12
EVERY = 25


def main() -> int:
    feat = FeatureExtractor(max_actions=96)
    _cls, _pos, value_model = load_three_models(str(CKPT))
    value_model.eval()

    grads_stats, grads_board = [], []
    coin_idx = None
    for seed in (7, 11):
        st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
        for r in range(0, 260):
            if r % EVERY == 0 and r >= 40:
                s = np.asarray(feat.encode_stats(st, 0), dtype=np.float32).ravel()
                b = np.asarray(feat.encode_board(st, 0), dtype=np.float32)
                if coin_idx is None:
                    n_summary = len(feat.summarize(st, 0).values)
                    coin_idx = (n_summary, n_summary + 1)   # extras 的前两项就是双方金币/300
                    print(f"[索引] summarize 维度 {n_summary} ⇒ stats 里金币两维 = {coin_idx}")
                st_t = torch.tensor(s, requires_grad=True).unsqueeze(0)
                bd_t = torch.tensor(b, requires_grad=True).unsqueeze(0)
                out = value_model(bd_t, st_t)
                v = (out["value"] if isinstance(out, dict) else out).sum()
                gs, gb = torch.autograd.grad(v, [st_t, bd_t])
                grads_stats.append(gs.detach().numpy().ravel())
                grads_board.append(np.abs(gb.detach().numpy()[0]).mean(axis=(1, 2)))
            st.advance_round()
            if st.terminal:
                break

    gs = np.stack(grads_stats)                 # (n, 42)
    gb = np.stack(grads_board)                 # (n, 28)
    mag = np.abs(gs).mean(axis=0)
    order = np.argsort(-mag)
    print(f"\n样本 {len(gs)} 个局面")
    print(f"[stats] |∂V/∂x| 的分布：中位 {np.median(mag):.6f}  p90 {np.quantile(mag,0.9):.6f}  最大 {mag.max():.6f}")
    print(f"  金币两维（{coin_idx[0]}, {coin_idx[1]}）的 |∂V/∂x| = "
          f"{mag[coin_idx[0]]:.6f} / {mag[coin_idx[1]]:.6f}"
          f"  ⇒ 分位 {100.0*(mag < mag[coin_idx[0]]).mean():.0f}% / {100.0*(mag < mag[coin_idx[1]]).mean():.0f}%")
    print("  最敏感的 6 个 stats 维度：")
    for i in order[:6]:
        tag = " <- 金币" if i in coin_idx else ""
        print(f"    dim{i:<3} |grad|={mag[i]:.6f}{tag}")

    print(f"[符号] ∂V/∂(我方金币) 均值 {gs[:, coin_idx[0]].mean():+.5f}"
          f"（正=囤钱好，负=囤钱坏）")
    print(f"[符号] ∂V/∂(敌方金币) 均值 {gs[:, coin_idx[1]].mean():+.5f}")
    print(f"       逐局面看：我方为正 {int((gs[:, coin_idx[0]] > 0).sum())}/{len(gs)}"
          f"，敌方为正 {int((gs[:, coin_idx[1]] > 0).sum())}/{len(gs)}")
    print(f"\n[board] 各通道平均 |∂V/∂x|（前 8）:")
    bmag = gb.mean(axis=0)
    for i in np.argsort(-bmag)[:8]:
        print(f"    ch{i:<3} |grad|={bmag[i]:.6f}")


def _accepts_two(model) -> bool:
    import inspect
    try:
        params = list(inspect.signature(model.forward).parameters)
        return len(params) >= 3          # self, board, stats
    except (TypeError, ValueError):
        return True


def _forward_via_hook(model, board, stats):  # pragma: no cover - 兜底
    raise RuntimeError("模型 forward 签名不是 (board, stats)，需要单独适配")


if __name__ == "__main__":
    raise SystemExit(main())
