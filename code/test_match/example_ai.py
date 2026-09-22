"""最小可运行的协议 AI 示例（照抄改 `decide()` 就行）。

配套文档：`docs/protocol_ai_spec.md`。这个文件的价值是**把 §1–§4 的收发顺序写对**——
那是接入时唯一容易搞错的地方（收裸文本 / 发长度前缀 / 谁先手时的读写次序）。
默认行为：**永远空过**（合法、最保守）。想看它真出招，加 `--act-round=N --x=.. --y=..`。

跑法：
    python code/test_match/match_via_sdk.py --ai0=code/test_match/example_ai.py --ai1=<对手> 7
    python code/test_match/conformance.py --ai0=code/test_match/example_ai.py    # 一致性体检
"""
from __future__ import annotations

import struct
import sys

# ⚠️ stdout 是"4 字节长度前缀"的二进制协议通道：任何漏出去的 print 都会把后续包冲掉。
# 先把真 stdout 存下来，再把 sys.stdout 指到 stderr —— 漏出的输出全部去 stderr（裁判会收进日志）。
_PROTO_OUT = sys.stdout.buffer
sys.stdout = sys.stderr


def log(msg: str) -> None:
    sys.stderr.write(f"[example_ai] {msg}\n")
    sys.stderr.flush()


def send_ops(ops: list[tuple[int, ...]]) -> None:
    """回一回合的操作。空操作 = 只有一行 `0`。"""
    lines = [str(len(ops))]
    lines.extend(" ".join(str(t) for t in op) for op in ops)
    data = ("\n".join(lines) + "\n").encode("utf-8")
    _PROTO_OUT.write(struct.pack(">I", len(data)))   # 4 字节大端长度前缀
    _PROTO_OUT.write(data)
    _PROTO_OUT.flush()


def recv_line() -> str | None:
    """读一行裸文本。对局结束时返回 None。"""
    raw = sys.stdin.buffer.readline()
    if not raw:
        return None
    return raw.decode("utf-8", errors="replace").rstrip("\r\n")


def read_ops() -> list[tuple[int, ...]] | None:
    """读对手的操作文本：首行数量 + 若干行 token。"""
    head = recv_line()
    if head is None:
        return None
    n = int(head.strip())
    out: list[tuple[int, ...]] = []
    for _ in range(n):
        ln = recv_line()
        if ln is None:
            return None
        parts = [int(x) for x in ln.split()]
        if parts:
            out.append(tuple(parts))
    return out


def read_state() -> dict | None:
    """按 docs/protocol_ai_spec.md §3 的结构消费局面文本。

    **必须严格按这个顺序和计数读**——否则你会和裁判错位（表现是后面全乱）。
    """
    line = recv_line()
    if line is None:
        return None
    st: dict = {"round": int(line.strip()), "towers": [], "ants": []}

    for _ in range(int(recv_line().strip())):          # tower_count
        t = [int(x) for x in recv_line().split()]
        st["towers"].append({"tower_id": t[0], "player": t[1], "x": t[2], "y": t[3],
                             "type": t[4], "cooldown": t[5], "hp": t[6]})

    for _ in range(int(recv_line().strip())):          # ant_count
        a = [int(x) for x in recv_line().split()]
        st["ants"].append({"ant_id": a[0], "player": a[1], "x": a[2], "y": a[3],
                           "hp": a[4], "level": a[5], "age": a[6],
                           "status": a[7], "behavior": a[8], "kind": a[9]})

    st["coins"] = [int(x) for x in recv_line().split()[:2]]
    camp = [int(x) for x in recv_line().split()]
    st["base_hp"] = camp[:2]

    cd_rows = int(recv_line().strip())                 # 恒为 2
    st["cooldowns"] = [[int(x) for x in recv_line().split()] for _ in range(cd_rows)]

    for _ in range(int(recv_line().strip())):          # effect_count
        e = [int(x) for x in recv_line().split()]
        st.setdefault("effects", []).append({"weapon": e[0], "player": e[1],
                                             "x": e[2], "y": e[3], "turns": e[4]})
    return st


def decide(state: dict | None, player: int, round_idx: int,
           act_round: int, act_xy: tuple[int, int]) -> list[tuple[int, ...]]:
    """←←← **在这里填你的算法。** 返回 [(op_type, arg0, arg1), ...]，空列表 = 空过。

    `state` 为 None 表示第 0 回合且你先手（官方协议这时还没给你局面，见规范 §2）。
    """
    if state is None:
        return []                       # 第 0 回合空过是最省事的正确做法
    if round_idx % 50 == 0:
        log(f"round={round_idx} coins={state['coins']} base_hp={state['base_hp']} "
            f"towers={len(state['towers'])} ants={len(state['ants'])}")
    if act_round >= 0 and round_idx == act_round:
        x, y = act_xy
        return [(11, x, y)]             # 11 = BUILD_TOWER, 参数是 (x, y)
    return []


def main() -> int:
    act_round = -1
    act_xy = (7, 10)
    for a in sys.argv[1:]:
        if a.startswith("--act-round="):
            act_round = int(a.split("=", 1)[1])
        elif a.startswith("--x="):
            act_xy = (int(a.split("=", 1)[1]), act_xy[1])
        elif a.startswith("--y="):
            act_xy = (act_xy[0], int(a.split("=", 1)[1]))

    init = recv_line()
    if init is None:
        log("没收到 init 行")
        return 1
    player, seed = (int(x) for x in init.split())
    log(f"player={player} seed={seed} act_round={act_round} act_xy={act_xy}")

    state: dict | None = None
    round_idx = 0
    try:
        while True:
            # ↓ 这个读写次序就是协议的全部难点（规范 §2）：先手方"先写后读"，后手方"先读后写"
            if player == 0:
                send_ops(decide(state, player, round_idx, act_round, act_xy))
                opp = read_ops()
                if opp is None:
                    break
            else:
                opp = read_ops()
                if opp is None:
                    break
                send_ops(decide(state, player, round_idx, act_round, act_xy))
            st = read_state()
            if st is None:
                break
            state = st
            round_idx += 1
    except BrokenPipeError:
        log("裁判关掉了管道（正常收尾）")
    log(f"结束: rounds={round_idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
