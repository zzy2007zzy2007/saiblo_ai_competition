"""`RunnerUpOracle` 自对弈校验：两个亚军（P0/P1）通过本查询器走完整协议互相对打。

这是协议层**最硬**的检验：
  * 如果消息顺序/行格式有任何一处不对，AI 会立刻 `[cpp_sdk] ... failed` + 读端错位，
    不可能连续跑几十回合；
  * 只有两边都真在出招（造塔/放蚂蚁/降级…）并且局面自己演化（金币、HP 变化），
    才能说明「我们喂给它的局面」被它当真了；
  * 顺带覆盖 `player=0`（P0 的读序与 P1 不同：先出招、后收对手行）。

用法：python code/test_match/probe_oracle_selfplay.py [seed] [rounds]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (REPO_ROOT / "code", REPO_ROOT / "code" / "cpp_engine", REPO_ROOT / "Ant-Game",
           REPO_ROOT / "code" / "test_match"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from test_match.runnerup_oracle import OracleError, RunnerUpOracle  # noqa: E402

try:
    from SDK.utils.constants import OperationType  # noqa: E402
except Exception:  # noqa: BLE001
    OperationType = None


def op_name(op) -> str:
    try:
        return f"{OperationType(int(op.op_type)).name}{op.to_protocol_tokens()[1:]}"
    except Exception:  # noqa: BLE001
        return str(tuple(op.to_protocol_tokens()))


def check_legal(st, player: int, ops) -> tuple[list, list]:
    accepted, illegal = [], []
    for op in ops:
        (accepted if st.can_apply_operation(player, op, accepted) else illegal).append(op)
    return accepted, illegal


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    print(f"[selfplay] seed={seed} max_rounds={max_rounds} 双方都是 ai_cpp_lure_v4", flush=True)

    o0 = RunnerUpOracle(seed=seed, player=0, timeout=10.0)
    o1 = RunnerUpOracle(seed=seed, player=1, timeout=10.0)
    n_act0 = n_act1 = 0
    n_ill0 = n_ill1 = 0
    try:
        for _ in range(max_rounds):
            if st.terminal:
                print(f"[selfplay] 终局：round={st.round_index} winner={st.winner}", flush=True)
                break
            rnd = st.round_index
            t0 = time.monotonic()
            # P0 先出招（官方协议：同一回合 P0 先动）
            ops0 = o0.ask(st)
            o1.set_opponent_ops(rnd, ops0)          # P1 在 ask 前登记本轮对手行
            ops1 = o1.ask(st)
            o0.set_opponent_ops(rnd, ops1)          # P0 下一回合开头才投递
            ms = (time.monotonic() - t0) * 1000.0

            a0, i0 = check_legal(st, 0, ops0)
            a1, i1 = check_legal(st, 1, ops1)
            n_act0 += 1 if a0 else 0
            n_act1 += 1 if a1 else 0
            n_ill0 += len(i0)
            n_ill1 += len(i1)
            print(f"  r{rnd:03d} p0={len(a0)}ops {' '.join(op_name(o) for o in a0) or '-'} | "
                  f"p1={len(a1)}ops {' '.join(op_name(o) for o in a1) or '-'} | "
                  f"coins={list(st.coins)} hp={[int(b.hp) for b in st.bases]} "
                  f"ants={len(st.ants)} towers={len(st.towers)} {ms:.0f}ms", flush=True)
            if i0 or i1:
                print(f"       !! 非法 p0={[op_name(o) for o in i0]} p1={[op_name(o) for o in i1]}",
                      flush=True)
            st.apply_operation_list(0, a0)
            st.apply_operation_list(1, a1)
            st.advance_round()
    except OracleError as exc:
        print(f"[selfplay] ✗ 协议失败 round={st.round_index}：{type(exc).__name__}: {exc}",
              flush=True)
        return 1
    finally:
        o0.close()
        o1.close()

    print(f"\n{'='*78}")
    print(f"[selfplay] 跑完 round={st.round_index} terminal={st.terminal} winner={st.winner}")
    print(f"[selfplay] 出招回合数 p0={n_act0} p1={n_act1}  非法操作 p0={n_ill0} p1={n_ill1}")
    print(f"[selfplay] coins={list(st.coins)} base_hp={[int(b.hp) for b in st.bases]} "
          f"ants={len(st.ants)} towers={len(st.towers)}")
    print(f"[selfplay] oracle 统计 p0={o0.stats} p1={o1.stats}")
    print(f"{'='*78}")
    # 判据（协议保真）：每回合都拿到回包、零非法操作、至少一方真在出招、局面自己演化了。
    # ⚠️ 不要求「双方都出招」：亚军 v4 的 hold 先验（80 分）让它对空局面大量空过，
    #    实测它会连续十几回合不动（真实对局 seed8 也是 r0-r17 全程 action_count=0）。
    all_ok = (o0.stats["ok"] == o0.stats["asks"] and o1.stats["ok"] == o1.stats["asks"]
              and o0.stats["timeouts"] == 0 and o1.stats["timeouts"] == 0)
    evolved = len(st.ants) > 0 or len(st.towers) > 0 or list(st.coins) != [50, 50]
    acted = (n_act0 + n_act1) > 0
    ok = all_ok and evolved and acted and n_ill0 == 0 and n_ill1 == 0
    print(f"[selfplay] 判定：{'PASS' if ok else 'FAIL'}"
          f"（每回合都收到回包={all_ok} 局面演化={evolved} 有出招={acted} "
          f"零非法={n_ill0 + n_ill1 == 0}）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
