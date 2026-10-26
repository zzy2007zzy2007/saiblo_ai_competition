"""⚠️ 已废弃（2026-10-26）：这个探针**协议顺序是错的**，只能跑 round 0，round 1 必死。

它的 bug：P1 的 stdin 上 init 之后第一条必须是「对手操作块」，**第 0 回合没有局面文本**；
这里却在第 0 回合先发了 `state_to_text()`，于是整条流错位一条消息，亚军在自己 round 0 的
`sync_round` 里就抛 `[cpp_sdk] sync_round failed: invalid ant row` 并退出。
逐行推演见 `docs/runnerup_oracle_protocol.md` §5。

**请用 `code/test_match/runnerup_oracle.py`（`RunnerUpOracle`）**，它按真实桥接的顺序实现，
已验证 seed=10 连续 30 回合 30/30。本文件保留仅作为反例。

亚军出招查询器（复刻桥接循环，2026-10-26）。

关键修正：**从 round 0 开始连续喂局面**（亚军 C++ SDK 自己维护回合计数，
跳到中间回合只喂一条会让它一直等前面那些消息 ⇒ receive_opponent failed / 超时）。

用法：python probe_runnerup_oracle.py [seed] [max_round]
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game",
           REPO / "Ant-Game" / "SDK", REPO / "code" / "test_match"):
    sys.path.insert(0, str(_p))

os.add_dll_directory(r"C:/mingw64/bin")

from my_ai.az_intent.game_state_facade import GameStateFacade
from test_match.match_via_sdk import ProcIO, send_line, state_to_text, parse_ops_text

RUNNERUP = str(REPO / "其他版本ai" / "saiblo-30th-AI" / "Game1" / "antgame_ai_cpp" / "cpp_lure_v4" / "build" / "ai_cpp_lure_v4.exe")


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_round = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    proc = subprocess.Popen([RUNNERUP], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            cwd=str(Path(RUNNERUP).parent), bufsize=0)
    io = ProcIO(proc, "runnerup", strip_cr=True)
    send_line(io, f"1 {seed}\n")          # init：我们是 P1

    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    ok = tot = 0
    for rnd in range(max_round):
        if st.terminal:
            print(f"  [round {rnd}] 对局结束")
            break
        try:
            send_line(io, state_to_text(st))   # ① 局面（传 facade 本体）
            send_line(io, "0")                 # ② 假 p0 本回合不出招
            reply = io.read_packet()           # ③ 它的出招（长度前缀包）
        except Exception as exc:
            print(f"  [round {rnd}] ✗ 通信失败 {type(exc).__name__}: {exc}")
            break
        if isinstance(reply, bytes):
            reply = reply.decode("utf-8", errors="replace")
        ops = parse_ops_text(reply)
        tot += 1
        applied = []
        try:
            st.apply_operation_list(1, ops)    # ④ 应用并推进
            ok += 1
            applied = [(int(o.op_type), o.arg0, o.arg1) for o in ops]
        except Exception as exc:
            print(f"  [round {rnd}] ✗ 应用失败 {type(exc).__name__}: {exc}")
        if rnd < 3 or rnd % 10 == 0:
            print(f"  [round {rnd}] ops={applied}（共 {len(ops)} 个）", flush=True)
        st.advance_round()

    try:
        proc.stdin.close()
    except Exception:
        pass
    proc.terminate()
    print(f"\n  结果：{ok}/{tot} 回合成功拿到并应用它的操作")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
