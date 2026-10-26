"""证据生成器：dump 亚军回包的**原始字节**（含 4 字节大端长度前缀）。

跑 25 回合自对弈，只在回包非空时打印 prefix/payload 的 hex —— 用来给协议文档提供
「AI→裁判」方向的真实字节证据（不是推测）。见 `docs/runnerup_oracle_protocol.md` §1。
"""
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game",
           REPO / "code" / "test_match"):
    sys.path.insert(0, str(_p))

from my_ai.az_intent.game_state_facade import GameStateFacade
from test_match.runnerup_oracle import RunnerUpOracle

N_EMPTY = 0


class DumpOracle(RunnerUpOracle):
    def _read_packet(self) -> bytes:
        global N_EMPTY
        header = self._read(4, "长度前缀")
        size = struct.unpack(">I", header)[0]
        payload = self._read(size, "payload") if size else b""
        if len(payload) <= 2:
            N_EMPTY += 1
            if N_EMPTY == 1:
                print(f"  [raw] 空回包 prefix={header.hex(' ')} size={size} "
                      f"payload={payload.hex(' ')} text={payload!r}", flush=True)
        else:
            print(f"  [raw] 非空回包 prefix={header.hex(' ')} size={size} "
                  f"payload={payload.hex(' ')} text={payload!r}", flush=True)
        return payload


def main() -> int:
    seed, rounds = 10, int(sys.argv[1]) if len(sys.argv) > 1 else 25
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    o0 = DumpOracle(seed=seed, player=0)
    o1 = DumpOracle(seed=seed, player=1)
    try:
        for _ in range(rounds):
            if st.terminal:
                break
            rnd = st.round_index
            ops0 = o0.ask(st)
            o1.set_opponent_ops(rnd, ops0)
            ops1 = o1.ask(st)
            o0.set_opponent_ops(rnd, ops1)
            st.apply_operation_list(0, ops0)
            st.apply_operation_list(1, ops1)
            st.advance_round()
    finally:
        o0.close()
        o1.close()
    print(f"[raw] done round={st.round_index} 空回包 {N_EMPTY} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
