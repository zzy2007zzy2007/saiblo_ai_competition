"""验证 `RunnerUpOracle`：seed=10 连续喂 30 回合，逐回合打印它的出招与合法性。

判据（任务要求）：**≥25/30 回合成功**。
"成功" = 本回合拿到回包 **且** 回包里的操作全部通过 `GameStateFacade.can_apply_operation`
（cold 那条路的 dry-run，和桥接判非法用的是同一个入口）。

用法：python code/test_match/probe_oracle_verify.py [seed] [rounds]
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (REPO_ROOT / "code", REPO_ROOT / "code" / "cpp_engine", REPO_ROOT / "Ant-Game",
           REPO_ROOT / "code" / "test_match"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402
from test_match.runnerup_oracle import (  # noqa: E402
    OracleError,
    RunnerUpOracle,
)


def op_text(op) -> str:
    """`op.to_protocol_tokens()` = 真正写到线上去的那几个整数。"""
    return "(" + ",".join(str(t) for t in op.to_protocol_tokens()) + ")"


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    print(f"[verify] seed={seed} rounds={rounds} 查询方=player 1（亚军）；对手=空过", flush=True)
    print(f"[verify] 局面：ghost 起点 coins={list(st.coins)} hp={[b.hp for b in st.bases]}", flush=True)

    results: list[dict] = []
    ms_list: list[float] = []
    t_start = time.monotonic()

    oracle = RunnerUpOracle(seed=seed, player=1, timeout=2.0)
    try:
        for _ in range(rounds):
            if st.terminal:
                print(f"[verify] 对局在第 {st.round_index} 回合已终局，提前停止", flush=True)
                break
            rnd = st.round_index
            coins = list(st.coins)
            hp = [int(b.hp) for b in st.bases]
            t0 = time.monotonic()
            try:
                ops = oracle.ask(st)
            except OracleError as exc:
                print(f"  r{rnd:02d} ✗ {type(exc).__name__}: {exc}", flush=True)
                results.append({"round": rnd, "ok": False, "why": type(exc).__name__})
                break                      # 会话作废 ⇒ 后面的回合没有意义
            ms = (time.monotonic() - t0) * 1000.0
            ms_list.append(ms)

            # 合法性：逐个过 can_apply_operation（pending = 本轮已接受的操作）
            accepted, illegal = [], []
            for op in ops:
                if st.can_apply_operation(1, op, accepted):
                    accepted.append(op)
                else:
                    illegal.append(op)

            applied = [op_text(o) for o in accepted]
            ok = not illegal
            results.append({"round": rnd, "ok": ok, "n_ops": len(ops),
                            "n_illegal": len(illegal), "ms": ms})
            flag = "✓" if ok else "✗非法"
            print(f"  r{rnd:02d} {flag} ops={len(ops)} 合法={len(accepted)} "
                  f"{' '.join(applied) if applied else '(空过)'} "
                  f"{ms:6.1f}ms coins={coins} hp={hp}", flush=True)
            if illegal:
                print(f"       !! 非法操作：{[op_text(o) for o in illegal]}", flush=True)

            st.apply_operation_list(1, accepted)
            st.apply_operation_list(0, [])          # 对手空过
            st.advance_round()
    finally:
        oracle.close()

    total = len(results)
    ok_n = sum(1 for r in results if r["ok"])
    rate = (100.0 * ok_n / total) if total else 0.0
    print(f"\n{'='*78}")
    print(f"[verify] 成功 {ok_n}/{total} = {rate:.1f}%   判据 ≥25/30 ⇒ "
          f"{'PASS' if ok_n >= 25 else 'FAIL'}")
    if ms_list:
        print(f"[verify] 每回合耗时 ms: min={min(ms_list):.1f} "
              f"median={statistics.median(ms_list):.1f} max={max(ms_list):.1f}")
    print(f"[verify] 总墙钟 {time.monotonic() - t_start:.1f}s   oracle.stats={oracle.stats}")
    print(f"[verify] 亚军 stderr: {oracle.stderr_path}")
    print(f"{'='*78}")
    return 0 if ok_n >= 25 else 1


if __name__ == "__main__":
    raise SystemExit(main())
