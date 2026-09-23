"""验证 bridge 的分帧修复：payload 长度恰好 13 字节（= 0x0D）时不能再错位。

背景（2026-09-23）：`strip_cr` 曾在**读取层**删掉流里所有 0x0D 字节，而这条流带着**二进制的 4 字节
长度前缀** ⇒ 前缀 `00 00 00 0D` 被删成 `00 00 00` ⇒ 桥读到 `00 00 00 32` = 50（'2'=0x32），
于是去等 50 字节、只等到 12 ⇒ 整局被超时掐断。
本测试用一个"回包恰好 13 字节"的假 AI 直接验证收发层。
"""
import importlib.util
import struct
import subprocess
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mvs_t", REPO / "code" / "test_match" / "match_via_sdk.py")
mvs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mvs)

PAYLOAD = "2\n13 10\n13 7\n"          # 恰好 13 字节 ⇒ 长度前缀 = 00 00 00 0D
assert len(PAYLOAD.encode()) == 13, len(PAYLOAD.encode())

FAKE = (
    "import struct,sys\n"
    f"d={PAYLOAD.encode()!r}\n"
    "sys.stdout.buffer.write(struct.pack('>I',len(d))+d)\n"
    "sys.stdout.buffer.flush()\n"
    "import time; time.sleep(5)\n"
)

print("payload =", repr(PAYLOAD), " len =", len(PAYLOAD.encode()), "= 0x%02X" % len(PAYLOAD.encode()))
fails = 0
for strip_cr in (False, True):        # 两种都不该错位（strip_cr 现在已废弃）
    proc = subprocess.Popen([sys.executable, "-c", FAKE], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    io = mvs.ProcIO(proc, f"fake(strip_cr={strip_cr})", strip_cr=strip_cr)
    try:
        got = io.read_packet()
        ok = got.decode() == PAYLOAD
        print(f"  strip_cr={strip_cr}: 收到 {len(got)} 字节, 内容一致 = {ok}")
        fails += 0 if ok else 1
    except Exception as exc:  # noqa: BLE001
        print(f"  strip_cr={strip_cr}: 读失败 {type(exc).__name__}: {exc}")
        fails += 1
    finally:
        proc.kill()

print()
print("修复验证:", "PASS" if fails == 0 else f"FAIL ({fails} 个 case)")
sys.exit(1 if fails else 0)
