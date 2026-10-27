#!/usr/bin/env python3
"""探针（2026-10-27）：把"非法操作在本地到底发生了什么"钉死成可复现的实验。

回答四个问题（报告 `docs/illegal_op_strict_mode_20261028.md` 的证据来源）：

  ① pybind 到底暴露了什么接口？`cold_handle_rule_illegal` 在这条路径上**有没有**被读？
     （`binding.cpp` 里 `apply_operation_list` 无条件调 `apply_operation_list_cold`，
      而 `cold_handle_rule_illegal` 只在 `round_read_from_judger` 里被读 ⇒ 猜它惰性）
  ② `apply_operation_list` 的**返回值**是什么？（`match_via_sdk` 老代码 `if il0 or il1:` 的判据）
  ③ 冷路径遇到非法 op 是"跳过"还是"报错"？（状态是否被改、金币是否被扣）
  ④ 新加的 `check_ops` 能不能真的抓到非法 op；`can_apply_dry` vs `can_apply_cold` 的
     判定是否一致、代价各是多少（影子校验的成本）。

用法：
    python -u code/test_match/probe_illegal_op_mode.py            # 全部检查
    python -u code/test_match/probe_illegal_op_mode.py --verbose  # 额外打印每步细节
失败返回码 1（可直接当回归测试）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "test_match"))
sys.path.insert(0, str(REPO / "code" / "cpp_engine"))
sys.path.insert(0, str(REPO / "Ant-Game"))

import os
os.add_dll_directory(str(REPO / "code" / "cpp_engine"))
os.add_dll_directory(r"C:/mingw64/bin")

# ⚠️ 先 import match_via_sdk（它做 sys.path / DLL 目录的原始设置），再 import native_game
import match_via_sdk as MVS  # noqa: E402
import native_game  # noqa: E402
from SDK.backend.model import Operation  # noqa: E402
from SDK.utils.constants import (  # noqa: E402
    PLAYER_BASES,
    OperationType,
    TowerType,
)
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

SEED = 7
FAILS: list[str] = []
VERBOSE = "--verbose" in sys.argv


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def _op(t: int, a0: int = -1, a1: int = -1) -> Operation:
    return Operation(OperationType(int(t)), int(a0), int(a1))


def _snapshot(state) -> tuple:
    return (tuple(int(c) for c in state.coins),
            tuple((int(t.tower_id), int(t.player), int(t.x), int(t.y), int(t.tower_type))
                  for t in state.towers),
            tuple(int(b.hp) for b in state.bases),
            bool(state.terminal))


def _first_legal_build(state, player: int) -> Operation | None:
    """从决策器的位置掩码里挑第一个**确实能落**的建塔格（逐个 can_apply 复核）。

    ⚠️ 掩码布局是 `mask[class, x, y]`（decoder.py:190）⇒ `nonzero()` 返回 **(xs, ys)**。
    写成 `ys, xs = ...` 会**把 x/y 调换**（本仓库 `az_bridge_ai._probe_upgrade_once:931` 就是那样，
    于是它那条"建塔对照"其实一直是非法 op —— 正是本次改动要抓的那类 silent bug）。
    """
    from my_ai.decoder import make_position_masks
    pm = make_position_masks(state, player, intent_decoding=True)
    xs, ys = (pm[0] > 0).nonzero()
    for x, y in zip(xs.tolist(), ys.tolist()):
        op = _op(OperationType.BUILD_TOWER, int(x), int(y))
        if state.can_apply_operation(player, op):
            return op
    return None


# ── ① pybind 暴露的接口 ────────────────────────────────────────────────────────
def part1_api() -> None:
    print("\n== ① pybind 接口 ==", flush=True)
    names = sorted(n for n in dir(native_game.NativeGame) if not n.startswith("_"))
    print(f"NativeGame 暴露的方法: {names}", flush=True)
    check("NativeGame 没有 round_read_from_judger（⇒ cold flag 的唯一下游不存在）",
          "round_read_from_judger" not in names)
    check("有 can_apply_cold / can_apply_dry（影子校验可用）",
          "can_apply_cold" in names and "can_apply_dry" in names)
    import inspect
    doc = native_game.NativeGame.apply_operation_list.__doc__ or ""
    if VERBOSE:
        print(f"  apply_operation_list doc: {doc!r}", flush=True)


# ── ② cold=True / False 在 facade 路径上是否真的不同 ──────────────────────────
def part2_cold_inert() -> None:
    print("\n== ② cold_handle_rule_illegal 在 facade 路径上是否惰性 ==", flush=True)
    st_cold = GameStateFacade.initial(seed=SEED, cold_handle_rule_illegal=True)
    st_strict = GameStateFacade.initial(seed=SEED, cold_handle_rule_illegal=False)

    # 非法 op：在我方基地格上建塔（官方 "TowerBuild: attempt to build a tower ... there is already a camp"）
    bx, by = PLAYER_BASES[0]
    bad = _op(OperationType.BUILD_TOWER, bx, by)
    # 合法 op：第一个确实能落的建塔格
    good = _first_legal_build(st_cold, 0)

    check("非法 op（基地格建塔）被 can_apply_dry 判非法",
          not st_cold.can_apply_operation(0, bad))
    check("非法 op 被 can_apply_cold 判非法",
          not st_cold._g.can_apply_cold(0, [int(bad.op_type), bad.arg0, bad.arg1], []))

    before = _snapshot(st_cold)
    ret = st_cold.apply_operation_list(0, [bad])
    after = _snapshot(st_cold)
    check("apply_operation_list 返回 None（老代码 `if il0:` 永远不成立）", ret is None,
          f"return={ret!r}")
    check("冷路径对非法 op 静默跳过（状态不变、不抛异常）", before == after)

    # 同一个非法 op 打进 cold=False 的引擎：看行为是否与 cold=True 完全一致
    before_s = _snapshot(st_strict)
    ret_s = st_strict.apply_operation_list(0, [bad])
    after_s = _snapshot(st_strict)
    check("cold=False 的引擎对同一个非法 op 的行为与 cold=True **逐位相同**（flag 惰性）",
          (before == before_s) and (after == after_s) and (ret == ret_s),
          f"cold=True after={after == before}  cold=False after={after_s == before_s}")

    # 合法 op 两边都要能落（确保上面不是"什么都拒绝"造成的假象）
    if good is not None:
        c2 = st_cold.clone()
        n0, c0 = len(c2.towers_of(0)), int(c2.coins[0])
        c2.apply_operation_list(0, [good])
        check("对照：合法建塔在冷路径上确实生效（塔数 +1、金币减少）",
              len(c2.towers_of(0)) == n0 + 1 and int(c2.coins[0]) < c0,
              f"{n0}->{len(c2.towers_of(0))} coins {c0}->{int(c2.coins[0])}")
    else:
        check("对照：合法建塔（找不到合法格，跳过）", True)


# ── ③ 升级 op 的两种参数形式（历史 bug 现场）────────────────────────────────
def part3_upgrade_form() -> None:
    print("\n== ③ 升级 op 参数形式：正确 (tower_id, target_type) vs 错误 (x, y) ==", flush=True)
    st = GameStateFacade.initial(seed=SEED, cold_handle_rule_illegal=False)
    good = _first_legal_build(st, 0)
    if good is None:
        check("建一座塔用于升级测试（找不到合法格，跳过）", True)
        return
    st.apply_operation_list(0, [good])
    for _ in range(80):           # 攒钱（每回合有收入）——升级 level-0 塔要 60
        if int(st.coins[0]) >= 120:
            break
        st.advance_round()
    if not st.towers_of(0):
        check("建塔在冷路径上生效（塔数 > 0）", False, f"coins={list(st.coins)}")
        return
    tw = st.towers_of(0)[0]
    right = _op(OperationType.UPGRADE_TOWER, int(tw.tower_id), int(TowerType.HEAVY))
    wrong = _op(OperationType.UPGRADE_TOWER, int(tw.x), int(tw.y))
    ok_r = st.can_apply_operation(0, right)
    ok_w = st.can_apply_operation(0, wrong)
    check("正确的升级 op (tower_id, target_type) 被判合法", ok_r,
          f"tower_id={tw.tower_id} coins={int(st.coins[0])} "
          f"reason={MVS.op_reason(st, 0, right)}")
    check("错误的升级 op (x, y) 被判**非法**（历史 bug 的确切形态）", not ok_w,
          f"(x,y)=({tw.x},{tw.y})")
    # 冷路径对"错误形式"是静默跳过 ⇒ 正是历史上"塔等级永远 0"却不报错的原因
    c = st.clone()
    lv0 = [int(t.level) for t in c.towers_of(0)]
    c.apply_operation_list(0, [wrong])
    lv1 = [int(t.level) for t in c.towers_of(0)]
    check("冷路径下 (x,y) 形式的升级 op 被静默丢弃（等级不变）", lv0 == lv1, f"{lv0} -> {lv1}")


# ── ④ 影子校验器：抓不抓得到 + 代价 ─────────────────────────────────────────
def part4_guard() -> None:
    print("\n== ④ 新的影子校验（check_ops）与两种 can_apply 的代价 ==", flush=True)
    st = GameStateFacade.initial(seed=SEED, cold_handle_rule_illegal=False)
    # 造一个中局状态：连续 40 回合"能建就建"，让塔/蚁数量接近实战
    built = 0
    for _ in range(40):
        g = _first_legal_build(st, 0)
        if g is not None:
            st.apply_operation_list(0, [g])
            built += 1
        st.advance_round()
    for _ in range(200):          # 攒钱到能升级（探针要覆盖"合法升级 op"这一类）
        if int(st.coins[0]) >= 300:
            break
        st.advance_round()
    print(f"  中局状态: round={st.round_index} 塔={len(st.towers)} 蚁={len(st.ants)} "
          f"金币={list(st.coins)}  build次数={built}", flush=True)

    bx, by = PLAYER_BASES[0]
    bad_list = [_op(OperationType.BUILD_TOWER, bx, by)]          # 基地格建塔
    if not st.towers_of(0):       # 确保至少有一座 tower 可升级（否则"合法升级"这条没法测）
        g = _first_legal_build(st, 0)
        if g is not None:
            st.apply_operation_list(0, [g])
    good_list = ([_op(OperationType.UPGRADE_TOWER, int(st.towers_of(0)[0].tower_id),
                      int(TowerType.HEAVY))] if st.towers_of(0) else [])

    stats = MVS.IllegalStats()
    logs: list[str] = []
    t0 = time.perf_counter()
    caught = MVS.check_ops(st, 0, bad_list, stats, seed=SEED, round_index=st.round_index,
                           tag="probe", cold=False, log=logs.append)
    dt_bad = (time.perf_counter() - t0) * 1000.0
    check("check_ops 抓到非法 op", len(caught) == 1 and stats.total == 1,
          f"records={caught}")
    check("check_ops 打印了含 ILLEGAL 的单行日志（下游 _tmp_ladder 靠它计数）",
          any("ILLEGAL" in x and "reason=" in x for x in logs),
          logs[0] if logs else "(无日志)")
    check("汇总行**不含**大写 ILLEGAL（否则 0 非法也会被数成 1）",
          "ILLEGAL" not in stats.summary(SEED, False), stats.summary(SEED, False)[:120])
    if VERBOSE or caught:
        for x in logs:
            print(f"    | {x}", flush=True)

    if good_list:
        stats2 = MVS.IllegalStats()
        caught2 = MVS.check_ops(st, 0, good_list, stats2, seed=SEED,
                                round_index=st.round_index, tag="probe", cold=False,
                                log=logs.append)
        check("check_ops 不误报合法 op（升级）", not caught2 and stats2.total == 0)
        # 代价：两种 can_apply 各跑 200 次
        op = good_list[0]
        raw = [int(op.op_type), int(op.arg0), int(op.arg1)]
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            st.can_apply_operation(0, op, pending=())
        dt_dry = (time.perf_counter() - t0) / n * 1000.0
        t0 = time.perf_counter()
        for _ in range(n):
            st._g.can_apply_cold(0, raw, [])
        dt_cold = (time.perf_counter() - t0) / n * 1000.0
        print(f"  can_apply_dry={dt_dry * 1000:.1f}us  can_apply_cold={dt_cold * 1000:.1f}us "
              f"（每 op 的校验成本；一局 ~400 回合 × 2 侧 × ~1.5 op ⇒ "
              f"{dt_cold * 1200 / 1000:.1f}s 量级）", flush=True)
        check("clone 版校验的成本可接受（< 5 ms/op）", dt_cold < 5.0, f"{dt_cold:.2f}ms/op")
    else:
        check("合法 op 不误报（无塔可升级，跳过）", True)

    # 空操作列表（HOLD）要计进 holds 而不是 checked
    stats3 = MVS.IllegalStats()
    MVS.check_ops(st, 1, [], stats3, seed=SEED, round_index=0, log=logs.append)
    check("HOLD（空操作列表）被计成 holds 而非非法",
          stats3.total == 0 and stats3.holds[1] == 1 and stats3.checked == 0)


def main() -> int:
    print(f"python={sys.executable}", flush=True)
    part1_api()
    part2_cold_inert()
    part3_upgrade_form()
    part4_guard()
    print("", flush=True)
    if FAILS:
        print(f"=== 有 {len(FAILS)} 项 FAIL: {FAILS} ===", flush=True)
        return 1
    print("=== 全部通过 ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
