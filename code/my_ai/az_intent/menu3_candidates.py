"""方法侧 M5 的候选菜单（预注册 `docs/prereg_20261005_menu_class_search.md`）。

给 `BundleMCTS(candidate_fn=...)` 用：**把"出哪个类"交回搜索**，但候选只有 3 个：
  ① **闪电**（类 17，放在位置网通道 17 的最热合法格）
  ② **HOLD**（空 bundle）
  ③ **一个经济类**（类 0..16 里"位置网最热格值最高"的那一个，放在它自己的最热合法格）

契约（见 `bundle_mcts.__init__` 注释）：`candidate_fn(state, player, k) -> list[tuple[(op_type,arg0,arg1), ...]] | None`
—— 返回非空列表则**完全接管**该节点的候选（先验均匀），返回 None 则回落原采样。

需要"当前位置网输出"（`action_map` / `head_logits`）。由于契约里没有 net_out，本模块用一个
**holder**（调用方在 `net_fn` 包装里塞最近一次输出）来取；`_expand` 内部先算 net_fn 再算 candidate_fn，
所以 holder 里就是**当前节点**的输出。
"""
from __future__ import annotations

import numpy as np

LIGHTNING_CLS = 17
HOLD_CLS = 23
SPEND_CLASSES = tuple(range(0, 17))

_PM_CACHE: dict = {}


def _pm(state, player):
    """位置掩码（缓存，避免每回合重算）。"""
    key = (id(state), int(player))
    hit = _PM_CACHE.get(key)
    if hit is None:
        from my_ai.decoder import make_position_masks
        hit = make_position_masks(state, player, intent_decoding=True)
        if len(_PM_CACHE) > 64:
            _PM_CACHE.clear()
        _PM_CACHE[key] = hit
    return hit


def _op_tuple(cid: int, head_logits, am, cm, pm, state, player):
    """把一个类解成 (op_type, arg0, arg1)；解不出来返回 None。"""
    from my_ai.decoder import decode_head
    hl = np.asarray(head_logits, dtype=np.float32)
    op = decode_head(hl, am, np.array(cm, copy=True), np.array(pm, copy=True),
                     state, player, class_id=int(cid), temperature=0.0,
                     pos_temperature=0.0, intent_decoding=True)
    if op is None:
        return None
    return (int(op.op_type), int(op.arg0), int(op.arg1))


def make_menu3_fn(holder: dict):
    """holder: {"out": <最近一次 net_fn 的输出 dict>}；由调用方维护。"""

    def f(state, player, k):
        out = holder.get("out")
        if not out:
            return None
        from my_ai.decoder import make_class_mask
        am = np.asarray(out["action_map"], dtype=np.float32)
        hl = np.asarray(out["head_logits"][0], dtype=np.float32)   # 三个 head 的类 logits（常数头 ⇒ 基本相同）
        pm = _pm(state, player)
        cm = make_class_mask(state, player, position_mask=pm, intent_decoding=True)

        menu = []
        # ① 闪电
        if cm[LIGHTNING_CLS] and pm[LIGHTNING_CLS].any():
            t = _op_tuple(LIGHTNING_CLS, hl, am, cm, pm, state, player)
            if t is not None:
                menu.append((t,))
        # ② HOLD（空 bundle 永远是合法候选）
        menu.append(())
        # ③ 经济类：位置网最热格值最高者
        best, best_v = None, -np.inf
        for cid in SPEND_CLASSES:
            if not cm[cid] or not pm[cid].any():
                continue
            v = float(np.where(pm[cid], am[cid], -np.inf).max())
            if v > best_v:
                best, best_v = cid, v
        if best is not None:
            t = _op_tuple(best, hl, am, cm, pm, state, player)
            if t is not None:
                menu.append((t,))
        # 去重（HOLD 可能与"解不出来的类"重复）
        uniq = []
        for b in menu:
            if b not in uniq:
                uniq.append(b)
        return uniq if len(uniq) >= 2 else None

    return f
