"""给 probe_runnerup_oracle.py 补上 init 握手行（`{player} {seed}`）。

协议：AI 启动后先收一行 `{player} {seed}`（match_via_sdk.py:294），之后才是每回合的局面文本。
"""
from pathlib import Path

p = Path("code/test_match/probe_runnerup_oracle.py")
t = p.read_text(encoding="utf-8")

if "init 握手" not in t:
    anchor = "    ok = 0\n"
    assert anchor in t, "锚点未找到"
    ins = (
        "    # init 握手：裁判在启动后先发 `{player} {seed}`（见 match_via_sdk.py:294）\n"
        '    send_line(io, f"1 {seed}\\n")\n'
    )
    t = t.replace(anchor, ins + anchor, 1)
    p.write_text(t, encoding="utf-8")
    print("  已加 init 握手")
else:
    print("  已有 init 握手")
