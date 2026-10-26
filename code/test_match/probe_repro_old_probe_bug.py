"""证据生成器：用真实字节推演旧探针为什么报 "invalid ant row"。

把旧探针（`probe_runnerup_oracle.py`）实际写进亚军 stdin 的字节流按
`sdk.hpp:1059-1214` 的读序重放一遍，打印每一步读了哪一行、当成什么用。不猜，只重放。
结论见 `docs/runnerup_oracle_protocol.md` §5。
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code" / "cpp_engine"))

from test_match.match_via_sdk import state_to_text
from my_ai.az_intent.game_state_facade import GameStateFacade

seed = 10
st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
state0 = state_to_text(st)
st.apply_operation_list(0, [])
st.apply_operation_list(1, [])
st.advance_round()
state1 = state_to_text(st)

# 旧探针 probe_runnerup_oracle.py 的写法：每回合 send state_to_text(st) 再 send "0"
stream = state0 + "0\n" + state1 + "0\n"
lines = stream.split("\n")
if lines and lines[-1] == "":
    lines.pop()

print(f"旧探针写出的字节流 = state(r0) + '0\\n' + state(r1) + '0\\n'，共 {len(lines)} 行")
print(f"state(r0) 的 9 行: {state0.split(chr(10))[:9]}")
print(f"state(r1) 的前 3 行: {state1.split(chr(10))[:3]}")
print()

i = 0


def nxt() -> str:
    global i
    if i >= len(lines):
        raise EOFError("流结束")
    line = lines[i]
    print(f"  第{i + 1:2d}行 读到 {line!r}")
    i += 1
    return line


def stoi(line: str) -> int:
    """模拟 C++ std::stoi：只解析**前缀**，遇到空格/非数字就停。

    ⇒ `std::stoi("50 50")` = 50（而 Python `int("50 50")` 会抛）。
    这正是这条坑的来源：金币行 '50 50' 被当成 ant_count 读成 50。
    """
    s = line.strip()
    if not s:
        raise ValueError("stoi: 空串")
    j = 1 if s[0] in "+-" else 0
    while j < len(s) and s[j].isdigit():
        j += 1
    if j == 0 or (j == 1 and s[0] in "+-"):
        raise ValueError(f"stoi: 无法转换 {line!r}")
    return int(s[:j])


print("P1 的读序（ai_cpp_lure_v4.cpp:180-186：receive_opponent → self → sync_round）")
print("① receive_opponent (sdk.hpp:1059-1085)")
count = stoi(nxt())
print(f"   ⇒ 对手操作数 = {count}（把局面文本的第 1 行「回合号 0」当成了操作数！）")
for _ in range(count):
    nxt()
print("② perform_self_turn：它先写出自己的操作（探针在这一步成功收到 '0\\n'）")
print("③ sync_round (sdk.hpp:1087-1214)")
print(f"   round_index  ← {nxt()!r}")
tower_count = stoi(nxt())
print(f"   tower_count  ← {tower_count}")
for _ in range(tower_count):
    nxt()
ant_count = stoi(nxt())
print(f"   ant_count    ← {ant_count}   ⚠️ 第 4 行是金币行 '50 50'，stoi 只吃前缀 50")
for k in range(ant_count):
    row = nxt()
    ints = [int(t) for t in row.split()]
    if len(ints) < 8:
        print(f"\n   ✗ 第 {k + 1} 个「蚂蚁行」={row!r} 只有 {len(ints)} 个整数 < 8"
              f" ⇒ sdk.hpp:1137 抛 invalid ant row")
        print("   （此时探针已收到 round 0 的回包 '0\\n'，所以它以为 round 0 成功了；"
              "真相是亚军在 round 0 自己的 sync_round 里就死了）")
        break
