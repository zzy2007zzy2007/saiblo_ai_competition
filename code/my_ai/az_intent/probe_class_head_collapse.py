"""量：部署 ckpt 的**类头**在真实中局上都押什么类？以及**位置网各通道**的顶格质量。

目的（用户 2026-10-21 提出）：
  ① 类头是不是塌缩到闪电（类 17）⇒ 若是，统一搜索里它已无作用；
  ② 位置网的"塔通道"和"闪电通道"分别靠谱吗。

做法：用日志重放真实对局到若干回合，取两侧视角共 N 个局面：
  · 类头 argmax 分布（head0）
  · 位置网类 0（建 Basic）与类 17（闪电）的 argmax 是否落在**合法**格上、其 top1 的 action_map 值
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [_REPO / "code", _REPO / "code" / "cpp_engine", _REPO / "Ant-Game", _REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.backend.model import Operation  # noqa: E402
from SDK.utils.constants import OperationType  # noqa: E402
from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from my_ai.decoder import make_class_mask, make_position_masks  # noqa: E402

LOGS = _REPO / "match_results"
SEED = 732


def load_ops(path: Path) -> dict:
    out = {}
    for line in path.open(encoding="utf-8", errors="replace"):
        if '"rank":1,' not in line or "base_heuristic" not in line:
            continue
        rec = json.loads(line)
        ops = []
        if rec["key"] != "hold":
            for seg in rec["key"].split("|"):
                if seg.startswith("F:"):
                    continue
                parts = seg.split(":")
                if len(parts) < 2:
                    continue
                v = [int(x) for x in parts[:3]] + [-1] * (3 - min(3, len(parts)))
                ops.append((v[0], v[1], v[2]))
        out[(rec["round"], rec["player"])] = ops
    return out


feat = FeatureExtractor(max_actions=96)
_, net_fn = make_net_fn_from_ckpt(str(_REPO / "checkpoints/posnet_A_k5_m32.pt"), feat,
                                  value_tanh=True, value_rel_to_abs=0.0)
ops0 = load_ops(LOGS / f"ai0_runnerup_seed{SEED}.stderr.log")
ops1 = load_ops(LOGS / f"ai1_ai3_linear_seed{SEED}.stderr.log")

facade = GameStateFacade.initial(SEED)
head_cls = collections.Counter()
light_ok = collections.Counter()      # 闪电通道：argmax 是否在合法集内
build_ok = collections.Counter()      # 建塔通道（类 0）：argmax 是否在合法集内
samples = 0
for r in range(400):
    if facade.terminal:
        break
    for player, table in ((0, ops0), (1, ops1)):
        ops = table.get((r, player), [])
        if ops:
            facade.apply_operation_list(player, [
                Operation(OperationType(int(t)), int(a0), int(a1)) for t, a0, a1 in ops])
    facade.advance_round()
    if r % 20 != 0 or r < 40:
        continue
    for player in (0, 1):
        out = net_fn(facade, player)
        hl = np.asarray(out["head_logits"][0], dtype=np.float32)
        am = np.asarray(out["action_map"], dtype=np.float32)
        masks = make_position_masks(facade, player, intent_decoding=True)
        cmask = make_class_mask(facade, player, position_mask=masks, intent_decoding=True)
        legal_classes = np.where(cmask)[0]
        if legal_classes.size == 0:
            continue
        masked = np.where(cmask, hl, -1e9)
        top = int(np.argmax(masked))
        head_cls[top] += 1
        samples += 1
        for cid, counter in ((17, light_ok), (0, build_ok)):
            if not cmask[cid]:
                counter["该类不合法"] += 1
                continue
            y, x = np.unravel_index(int(np.argmax(np.where(masks[cid], am[cid], -1e9))),
                                    am[cid].shape)
            counter["argmax合法" if masks[cid][y, x] else "argmax非法"] += 1

print(f"[样本] {samples} 个局面")
print(f"\n=== 类头（head0）argmax 分布（top 8）===")
tot = sum(head_cls.values())
NAMES = {17: "闪电", 16: "降级", 0: "建Basic", 23: "HOLD"}
for cid, cnt in head_cls.most_common(8):
    print(f"  类 {cid:>2} ({NAMES.get(cid, '?'):<8}) {cnt:>4} 次  {100*cnt/tot:5.1f}%")
print(f"\n  其中闪电(17)占比 = {100*head_cls[17]/tot:.1f}%")

print(f"\n=== 位置网通道：argmax 是否落在合法格 ===")
for name, counter in (("闪电通道(17)", light_ok), ("建塔通道(0)", build_ok)):
    print(f"  {name}: {dict(counter)}")
