"""按 docs/protocol_ai_spec.md 修正：
   · 裁判→AI 是裸文本行（send_line），无长度前缀
   · AI→裁判 是 4 字节长度前缀包（read_packet）
并把等待时间放宽（亚军冷启动可能较慢）。
"""
from pathlib import Path

p = Path("code/test_match/probe_runnerup_oracle.py")
t = p.read_text(encoding="utf-8")

# ① init / state / opponent 全部改回裸文本行
t = t.replace('send_packet(io, f"1 {seed}\\n")', 'send_line(io, f"1 {seed}\\n")')
t = t.replace("    send_packet(io, text)\n    send_packet(io, opponent_ops)",
              "    send_line(io, text)\n    send_line(io, opponent_ops)")

# ② 放宽读取超时（若 ProcIO 支持 timeout 参数）
if "timeout" not in t:
    t = t.replace("    reply = io.read_packet()", "    reply = io.read_packet()   # 内部有超时；见 ProcIO")

p.write_text(t, encoding="utf-8")
print("  已改回 send_line（init/state/opponent）")
