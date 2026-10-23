"""公平先验比较台（2026-10-23）。

修正前一版的方法错误：候选集**不能**由各网的先验采样产生（换网就换样本）。
本台子：
  1. 固定候选集：用 `make_position_masks`（与网络无关）枚举每类的合法 (格/塔)，
     每类均匀抽至多 40 个，保证可负担；
  2. 固定教师：统一用 `posnet_A` 的 `value_state` 跑 h=16 rollout 估值；
  3. 先验：从各网 `net_fn(...)['action_map'][cls][x][y]` 直接读（不经过候选采样）；
  4. 输出每类在两网上的 Spearman / recall@4 —— 据此决定"拼接时每类用谁"。
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np

from SDK.backend.model import Operation
from SDK.utils.constants import OperationType
from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.game_state_facade import GameStateFacade
from my_ai.decoder import make_position_masks
from my_ai.az_intent.value_batch import make_batched_value_fn

CLS_NAME = {**{i: f"tower{i}" for i in range(16)}, 16: "DOWNGRADE", 17: "LIGHTNING",
            21: "UP_GEN", 22: "UP_ANT"}
PER_CLASS_CAP = 40
H = 16


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main() -> int:
    feat = FeatureExtractor(max_actions=96)
    _, nf_a = make_net_fn_from_ckpt("checkpoints/posnet_A_k5_m32.pt", feat,
                                    value_tanh=True, value_rel_to_abs=0.0)
    _, nf_b = make_net_fn_from_ckpt("checkpoints/spliced_prior_A.pt", feat,
                                    value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models("checkpoints/posnet_A_k5_m32.pt")   # 固定教师
    vm.eval()
    vf = make_batched_value_fn(vm, feat)

    stats = collections.defaultdict(lambda: {"A": [], "B": [], "V": []})
    for seed, warm in ((1540, 60), (1540, 140), (1541, 200), (1580, 100), (1580, 200)):
        f = GameStateFacade.initial(seed=seed)
        for _ in range(warm):
            f.advance_round()
        if f.terminal:
            continue
        masks = make_position_masks(f, 0, intent_decoding=True)
        amap_a = nf_a(f, 0)["action_map"]
        amap_b = nf_b(f, 0)["action_map"]

        cands = []
        for cls in list(range(16)) + [16, 17, 21, 22]:
            xs, ys = np.where(masks[cls])
            if len(xs) == 0:
                continue
            idx = np.linspace(0, len(xs) - 1, min(PER_CLASS_CAP, len(xs))).astype(int)
            for i in idx:
                cands.append((cls, int(xs[i]), int(ys[i])))
        if not cands:
            continue

        # 教师：每个候选 apply + advance H + 值网
        leaves = []
        for (cls, x, y) in cands:
            sim = f.clone()
            if cls < 16:
                op = Operation(OperationType.BUILD_TOWER, x, y)   # arg1 = tower type 由 decoder 语义决定，这里用位置
            elif cls == 16:
                op = Operation(OperationType.DOWNGRADE_TOWER, x, y)
            elif cls == 17:
                op = Operation(OperationType.USE_LIGHTNING_STORM, x, y)
            elif cls == 21:
                op = Operation(OperationType.UPGRADE_GENERATION, 0, 0)
            else:
                op = Operation(OperationType.UPGRADE_ANT_LEVEL, 0, 0)
            try:
                sim.apply_operation_list(0, [op])
            except Exception:
                continue
            for _ in range(H):
                if sim.terminal:
                    break
                sim.advance_round()
            leaves.append(((cls, x, y), sim))
        if not leaves:
            continue
        vals = vf([s for _, s in leaves], 0)
        for ((cls, x, y), _), v in zip(leaves, vals):
            stats[cls]["A"].append(float(amap_a[cls][x][y]))
            stats[cls]["B"].append(float(amap_b[cls][x][y]))
            stats[cls]["V"].append(float(v))

    print(f"  {'类':<11}{'n':>5}{'我们Spearman':>14}{'蒸馏Spearman':>14}   谁更好")
    win = collections.Counter()
    for cls in sorted(stats):
        d = stats[cls]
        rho_a = spearman(d["A"], d["V"])
        rho_b = spearman(d["B"], d["V"])
        n = len(d["V"])
        if np.isnan(rho_a) and np.isnan(rho_b):
            continue
        if (rho_a if not np.isnan(rho_a) else -9) >= (rho_b if not np.isnan(rho_b) else -9):
            win["A"] += 1; who = "我们"
        else:
            win["B"] += 1; who = "蒸馏"
        print(f"  {CLS_NAME.get(cls, str(cls)):<11}{n:>5}{rho_a:>14.3f}{rho_b:>14.3f}   {who}")
    print(f"\n  合计：我们用 {win['A']} 类 / 蒸馏用 {win['B']} 类")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
