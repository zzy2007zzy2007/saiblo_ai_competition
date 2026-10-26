"""再试一种消息格式：state 与对手 ops 都用**4 字节长度前缀包**（send_packet）。

依据：
  · `read_packet` 说明回复是长度前缀包（错误信息 "need 4B" 证实）；
  · 请求侧如果也是包格式，那么 send_line 会让对端把长度头当文本读 ⇒ 报 EOF。
"""
from pathlib import Path

p = Path("code/test_match/probe_runnerup_oracle.py")
t = p.read_text(encoding="utf-8")

old = '''    send_line(io, text)
    send_line(io, opponent_ops)
    reply = io.read_packet()'''
new = '''    send_packet(io, text)
    send_packet(io, opponent_ops)
    reply = io.read_packet()'''

if "send_packet(io, text)" not in t:
    assert old in t, "ask 锚点未找到"
    t = t.replace(old, new, 1)
    # init 握手也要用包格式
    t = t.replace('send_line(io, f"1 {seed}\\n")', 'send_packet(io, f"1 {seed}\\n")', 1)
    p.write_text(t, encoding="utf-8")
    print("  已改为 send_packet（含 init）")
else:
    print("  已是 send_packet")
