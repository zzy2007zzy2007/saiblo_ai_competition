"""过滤闸门（`_block_legal_gate`）的**确定性**探针 —— 2026-10-28。

验的是 `docs/illegal_op_strict_mode_20261028.md` §5.3 那个真非法操作的**机制**本身：

    一块**本来合法**的多 op 候选 [(13,塔,-1), (11,建塔格)]（先降级退钱、再建塔）
    被 `AZAI_NO_DOWNGRADE` / `AZAI_BAN_OPS=13` 删掉降级之后 ⇒ 剩下的建塔"钱不够" ⇒ **半块非法**。

⚠️ 一个容易踩的坑（探针第一版就踩了）：**建塔的合法性检查本身含金币**
（`can_apply_dry` 会算 cost 与 coins）⇒ `can_apply(建塔, pending=[])` 为 False 时，
"随便找个空格子"是找不到的。必须用 `pending=[降级 op]` 才能问出"退钱之后这个格子能不能建"。
这正是这个 bug 的形态：**整块合法（pending 里有降级）而半块非法（pending 为空）**。

探针要证明四件事：

1. **闸门抓得住**：过滤后的半块被判非法（reason = `build_no_gold(...)`），而过滤前的整块合法
   —— 这是"整块合法、半块非法"的存在性证明，也是闸门存在的理由；
2. **闸门不误杀**：过滤前/过滤后都合法的块返回空（不许把好块丢成 HOLD）；
3. **闸门的计数是分开的**：`A.ILLEGAL_STATS`（口径 = "真的发出去了多少非法 op"）**不被闸门污染**
   （否则 `RESULT ... illegal=` 会虚高）；
4. **检查器坏掉时 fail-closed**（宁可 HOLD，也不能 fail-open 把可能非法的 op 发出去）。

不依赖搜索、不依赖网络、也不起任何子进程：只用引擎 + facade 把局面推到一个"钱不够"的状态。

用法：`python code/test_match/probe_filter_gate.py`（失败返回 1）
"""
from __future__ import annotations

import sys

_REAL_OUT = sys.stdout          # ⚠️ 下面 import 会执行 `sys.stdout = sys.stderr`（协议保护）
MAP = 19                        # 地图边长 = 2*EDGE-1（`SDK.utils.constants.EDGE=10`，见 faulty_ai.py:31-33）


def say(msg: str) -> None:
    print(msg, file=_REAL_OUT, flush=True)


def main() -> int:
    import az_bridge_ai as A             # noqa: E402  （会把 sys.stdout 指到 stderr）
    from SDK.backend.model import Operation as Op
    from SDK.utils.constants import OperationType as OT
    from SDK.utils.constants import TOWER_UPGRADE_TREE

    ok_all = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        say(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    def build_cell(facade, pl: int, pending=()):
        """扫出一个**在 pending 之后**建塔合法的格子（不用解码器，避免 x/y 布局歧义）。"""
        for x in range(MAP):
            for y in range(MAP):
                if facade.can_apply_operation(pl, Op(OT.BUILD_TOWER, x, y), pending=pending):
                    return x, y
        return None

    def upgrade_op(facade, pl: int):
        for t in facade.towers_of(pl):
            for tg in TOWER_UPGRADE_TREE.get(t.tower_type, ()):
                op = Op(OT.UPGRADE_TOWER, int(t.tower_id), int(tg))
                if facade.can_apply_operation(pl, op):
                    return op
        return None

    # ── 走到「降级退钱 + 建塔 = 合法，但只剩建塔 = 非法」的局面 ────────────────────────
    # 引擎口径（探针实测）：建塔价 = build_tower_cost(已有塔数)：15/30/45/90…；降级退钱 =
    # build_tower_cost(n-1) * REFUND_RATIO（实测 54/81…）⇒ "退的钱 + 手上的钱 ≥ 建塔价 > 手上的钱"
    # 的窗口一旦出现，就是 §5.3 的形态（那两局是 coins=3/30 vs cost=45）。
    pl = 0
    hit = None
    for seed in (7, 11, 23, 31):
        facade = A.GameState.initial(seed=seed)
        for rnd in range(200):
            tws = facade.towers_of(pl)
            coins = int(facade.coins[pl])
            cost = int(facade.build_tower_cost(facade.tower_count(pl)))
            if tws and coins >= cost - 100:      # 粗筛：退钱最多也就这个量级
                for t in tws:
                    down = Op(OT.DOWNGRADE_TOWER, int(t.tower_id), -1)
                    if not facade.can_apply_operation(pl, down):
                        continue
                    cell = build_cell(facade, pl, pending=[down])
                    if cell is None:
                        continue
                    bld = Op(OT.BUILD_TOWER, int(cell[0]), int(cell[1]))
                    if facade.can_apply_operation(pl, bld, pending=[]):
                        continue                 # 半块也合法 ⇒ 不是我们要找的局面
                    hit = {"seed": seed, "round": rnd, "coins": coins, "cost": cost,
                           "tower": int(t.tower_id), "tower_type": int(t.tower_type),
                           "tower_level": int(getattr(t, "level", -1)),
                           "cell": cell, "raw": [down, bld], "half": [bld],
                           "income": int(facade.downgrade_tower_income(t.tower_type, t))}
                    break
                if hit:
                    break
            # 局面驱动：付得起就建塔（把钱花光），否则能升级就升级，否则空过
            cell0 = build_cell(facade, pl) if coins >= cost else None
            if cell0 is not None:
                facade.apply_operation_list(
                    pl, [Op(OT.BUILD_TOWER, int(cell0[0]), int(cell0[1]))])
            else:
                up = upgrade_op(facade, pl)
                if up is not None:
                    facade.apply_operation_list(pl, [up])
            facade.advance_round()
        if hit:
            break

    if hit is None:
        check("构造出「整块合法 / 半块非法」的局面", False,
              "（探针自己的构造失败，不是被验代码的问题）")
        return 1
    facade = A.GameState.initial(seed=hit["seed"])
    say(f"局面: seed={hit['seed']} round={hit['round']} coins={hit['coins']} "
        f"build_cost={hit['cost']} 塔 id={hit['tower']} type={hit['tower_type']} "
        f"level={hit['tower_level']} 降级退钱={hit['income']} cell={hit['cell']}")
    check("构造出「整块（降级+建塔）合法」的局面", True,
          f"raw={[(int(o.op_type), int(o.arg0), int(o.arg1)) for o in hit['raw']]}")
    check("构造出「半块（只剩建塔）非法」的局面", True,
          f"half={[(int(o.op_type), int(o.arg0), int(o.arg1)) for o in hit['half']]}")
    say("（下面所有闸门调用的局面由探针自己重建 ⇒ 需要把同一局面再推到那一回合）")

    # ── 把局面重建出来（同一 seed 重放同一套动作到那一回合）─────────────────────────────
    facade = A.GameState.initial(seed=hit["seed"])
    for rnd in range(hit["round"] + 1):
        tws = facade.towers_of(pl)
        coins = int(facade.coins[pl])
        cost = int(facade.build_tower_cost(facade.tower_count(pl)))
        if rnd == hit["round"]:
            break
        cell0 = build_cell(facade, pl) if coins >= cost else None
        if cell0 is not None:
            facade.apply_operation_list(
                pl, [Op(OT.BUILD_TOWER, int(cell0[0]), int(cell0[1]))])
        else:
            up = upgrade_op(facade, pl)
            if up is not None:
                facade.apply_operation_list(pl, [up])
        facade.advance_round()
    say(f"重建后: round={facade.round_index} coins={list(facade.coins)} "
        f"towers={len(facade.towers_of(pl))}")

    down = Op(OT.DOWNGRADE_TOWER, int(hit["tower"]), -1)
    bld = Op(OT.BUILD_TOWER, int(hit["cell"][0]), int(hit["cell"][1]))

    # ── ① 闸门对"过滤后的半块"必须报非法 ─────────────────────────────────────────
    n_ill_before = int(A.ILLEGAL_STATS.total)
    bad_half = A._block_legal_gate(facade, pl, [bld])
    check("闸门：过滤后的半块被判非法（⇒ 调用方会整块丢弃）", len(bad_half) == 1,
          f"bad={[(b.get('op'), b.get('reason')) for b in bad_half]}")
    if bad_half:
        reason = str(bad_half[0].get("reason") or "")
        check("闸门给出的 reason 是 build_no_gold（钱不够）", reason.startswith("build_no_gold"), reason)
    check("闸门**没有**污染 ILLEGAL_STATS（口径 = 真发出去的非法 op）",
          int(A.ILLEGAL_STATS.total) == n_ill_before,
          f"ILLEGAL_STATS.total={A.ILLEGAL_STATS.total}（调用前 {n_ill_before}）")

    # ── ② 闸门对"过滤前本来就合法"的整块不许报非法（不误杀）────────────────────────
    bad_raw = A._block_legal_gate(facade, pl, [down, bld])
    check("闸门：过滤前的整块判为合法（不误杀，不会把好块丢成 HOLD）", not bad_raw,
          f"bad={[(b.get('op'), b.get('reason')) for b in bad_raw]}")

    # ── ③ HOLD（空块）必须合法（过滤把整块删空的路径）────────────────────────────────
    check("闸门：空块（HOLD）判为合法", not A._block_legal_gate(facade, pl, []))

    # ── ④ 检查器坏掉时必须 fail-closed（绝不 fail-open 发出可能非法的 op）────────────
    class _Broken:
        _g = None

        def can_apply_operation(self, *a, **k):
            raise RuntimeError("checker down")

    bad_broken = A._block_legal_gate(_Broken(), pl, [bld])
    check("闸门：检查器失效 ⇒ fail-closed（按非法处理）", bool(bad_broken),
          f"bad={[(b.get('op'), b.get('reason')) for b in bad_broken]}")

    say("")
    say("=== 全部 PASS ===" if ok_all else "=== 有 FAIL ===")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
