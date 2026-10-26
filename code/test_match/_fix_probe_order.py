"""修正探针的消息顺序：round_text → 对手 ops 行 → 读回复（与裁判一致）。"""
from pathlib import Path

p = Path("code/test_match/probe_runnerup_oracle.py")
t = p.read_text(encoding="utf-8")

old = '''def ask(io: ProcIO, text: str) -> list:
    send_line(io, text)
    reply = io.read_packet()'''
new = '''def ask(io: ProcIO, text: str, opponent_ops: str = "0") -> list:
    # 裁判顺序：先发局面文本，再发"对手的出招"（无操作就发 "0"），最后才读回复
    send_line(io, text)
    send_line(io, opponent_ops)
    reply = io.read_packet()'''

if "opponent_ops" not in t:
    assert old in t, "ask 锚点未找到"
    t = t.replace(old, new, 1)
    # 去掉读取之后额外发 "0" 的旧逻辑
    t = t.replace('''        # 模拟裁判对 P1 的"同步"消息（发 P0 的空出招）
        try:
            send_line(io, "0")
        except Exception:
            pass
''', "")
    p.write_text(t, encoding="utf-8")
    print("  已修正消息顺序")
else:
    print("  已是正确顺序")
