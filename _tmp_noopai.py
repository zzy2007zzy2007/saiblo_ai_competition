"""空操作假 AI（**驱动层验证专用**）：每回合都回"0 个操作"，不建塔不放技能。

用途：给验证批次造一个**必然很快终局**的对手——亚军 C++ 会在几十回合内推掉它的基地
⇒ 引擎真的判终局（`verdict=engine`）、整局 ~1-2 min（纯 C++，不吃 torch/GPU）。

协议（抄 `code/test_match/match_via_sdk.py` 的 ProcIO/run_match，见 `docs/protocol_ai_spec.md`）：
  · 启动后先收到一行 `"<player> <seed>\\n"`；
  · 之后每回合收到"对手操作/局面文本"若干行，**用 4 字节大端长度前缀 + payload** 回包；
  · payload `"0\\n"` = 本回合 0 个操作（`parse_ops_text` 只读第一行的整数）。
所有回合的回包都是同一个 `"0\\n"` ⇒ 对每一行输入都回一包也不会错位（多出来的包会被后续回合消费）；
这样实现最简单，也**不会**因为我们猜错"每回合收到几行"而把对局挂死。
"""
import struct
import sys

_OUT = sys.stdout.buffer
_OK = b"0\n"


def respond() -> None:
    _OUT.write(struct.pack(">I", len(_OK)) + _OK)
    _OUT.flush()


respond()                      # 第 0 回合（启动时只收到 "player seed" 一行）
for _line in sys.stdin.buffer:  # 之后每收到一行就回一包（见文件头为什么这样做不会错位）
    respond()
