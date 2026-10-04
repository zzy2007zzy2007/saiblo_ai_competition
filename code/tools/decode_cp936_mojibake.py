# -*- coding: utf-8 -*-
"""把「UTF-8 字节被当成 Windows CP936 解码」产生的乱码还原回中文。

背景（2026-10-05 实测）：
  * 乱码长这样：``鎸?docs/...锛堟湰娆′换鍔?...``（每 2 个乱码字对应原文 3 字节的 1 个字符）
  * 还原 = 把乱码按 **Windows CP936** 编回字节，再按 UTF-8 解码
  * 为什么不能用 Python 的 `gbk`/`gb18030`：
      - Windows CP936 把「未定义但格式合法」的字节对映到**私用区** U+E000..，Python 的
        `gbk` 表里没有这些项 ⇒ `gbk` 直接报 `UnicodeEncodeError`
      - Windows 把单字节 0x80 映成 `€`(U+20AC)，而 `gb18030` 把 U+20AC 编成 A2E3 ⇒ 字节错位
    ⇒ 必须用 Windows 自己的转换表（本脚本用 ctypes 调 `MultiByteToWideChar`）
  * 每次「非法字节对」被 CP936 丢掉时会留下一个 ASCII `?`，它吃掉了**两个**字节：
    一个 UTF-8 续字节 0x80..0xBF（补完当前字符）+ 一个 ASCII 字节 < 0x40（空格/数字/标点）。
    字符串末尾的单字节残尾（如 `。` 的第 3 字节 0x82）则只丢 **1** 个字节。
    这些字节无法从乱码本身唯一确定 —— 本脚本给出全部候选并打印上下文，由调用者用
    `--picks` 按语义定夺。

用法::

    # 1) 先看有哪些「丢字节」的位置和候选
    python code/tools/decode_cp936_mojibake.py --file garbled.txt
    # 2) 按语义选好后给死（0x 可省；末尾位置只给一个字节；none 表示不补）
    python code/tools/decode_cp936_mojibake.py --file garbled.txt \
        --picks 0=89:20,1=8e:20,2=84:20,3=a5:37,4=a2:20,5=82:none --out fixed.txt
    # 3) 独立验证：fixed.txt -> UTF-8 -> CP936 必须逐字符等于 garbled.txt
    python code/tools/decode_cp936_mojibake.py --verify garbled.txt fixed.txt
"""
import argparse
import codecs
import collections
import ctypes
import glob
import math
import os
import sys

CP = 936
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.MultiByteToWideChar.restype = ctypes.c_int
_k32.MultiByteToWideChar.argtypes = [ctypes.c_uint, ctypes.c_ulong, ctypes.c_char_p,
                                     ctypes.c_int, ctypes.c_wchar_p, ctypes.c_int]


def cp936_decode(data: bytes) -> str:
    """Windows CP936 解码（非法序列 -> '?'，与当初弄乱时用的是同一个 API）。"""
    if not data:
        return ""
    n = _k32.MultiByteToWideChar(CP, 0, data, len(data), None, 0)
    buf = ctypes.create_unicode_buffer(n + 1)
    _k32.MultiByteToWideChar(CP, 0, data, len(data), buf, n)
    return buf[:n]


def cp936_inverse_table() -> dict:
    """字符 -> 当初它在 CP936 里占的字节。"""
    table = {}
    for b in range(0x00, 0x100):
        s = cp936_decode(bytes([b]))
        if len(s) == 1:
            table.setdefault(s, bytes([b]))
    for lead in range(0x81, 0xFF):
        for trail in range(0x40, 0xFF):
            if trail == 0x7F:
                continue
            s = cp936_decode(bytes([lead, trail]))
            if len(s) == 1 and s != "?":
                table.setdefault(s, bytes([lead, trail]))
    return table


def split_lost(text: str, table: dict):
    """切成 [已知字节段, 已知字节段, ...] 和每个 '?' 的字符下标。"""
    segs, lost, cur = [], [], bytearray()
    for i, ch in enumerate(text):
        if ch == "?":
            segs.append(bytes(cur))
            cur = bytearray()
            lost.append(i)
        else:
            cur += table.get(ch, b"?")
    segs.append(bytes(cur))
    return segs, lost


def pending_of(b: bytes):
    """b 末尾那个没写完的 UTF-8 序列（b''=正好在边界；None=末尾不是合法前缀）。"""
    if not b:
        return b""
    if b[-1] < 0x80:
        return b""
    if 0x80 <= b[-1] <= 0xBF:
        j, steps = len(b) - 1, 0
        while j >= 0 and 0x80 <= b[j] <= 0xBF and steps < 3:
            j -= 1
            steps += 1
        if j < 0 or not (0xC2 <= b[j] <= 0xF4):
            return None
        cand = b[j:]
    else:
        cand = b[-1:]
    dec = codecs.getincrementaldecoder("utf-8")()
    try:
        dec.decode(cand, False)
    except UnicodeDecodeError:
        return None
    return b"" if dec.getstate()[0] == b"" else cand


def candidates(pend: bytes, has_next: bool):
    """所有 (补完的字符, 该字节, 后面那个被吃掉的 ASCII 字节或 None)。"""
    out = []
    for b0 in range(0x80, 0xC0):
        dec = codecs.getincrementaldecoder("utf-8")()
        try:
            text = dec.decode(pend + bytes([b0]), False)
        except UnicodeDecodeError:
            continue
        if dec.getstate()[0] or not text:
            continue
        if has_next:
            for b1 in list(range(0x00, 0x40)) + [0x7F]:
                out.append((text[-1], b0, b1))
        else:
            out.append((text[-1], b0, None))
    return out


def corpus_freq(paths):
    freq = collections.Counter()
    for pat in paths:
        for p in glob.glob(pat, recursive=True):
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    freq.update(fh.read())
            except OSError:
                pass
    return freq


def parse_picks(spec: str):
    picks = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        j, val = item.split("=")
        if ":" in val:
            b0, b1 = val.split(":")
            b1 = None if b1.lower() in ("none", "-", "") else int(b1, 16)
        else:
            b0, b1 = val, None
        picks[int(j)] = (int(b0, 16), b1)
    return picks


def decode(text: str, picks=None, freq=None, verbose=True):
    table = cp936_inverse_table()
    segs, lost = split_lost(text, table)
    pends = [pending_of(s) for s in segs[:-1]] + [pending_of(segs[-1])]
    picks = dict(picks or {})
    report = []
    for j, pos in enumerate(lost):
        cands = candidates(pends[j], j + 1 < len(segs))
        if not cands:
            report.append("j%d 在字符 #%d：找不到候选（末尾字节段 = %s）"
                          % (j, pos, segs[j][-6:].hex(" ")))
            continue
        if freq is not None:
            cands.sort(key=lambda t: -freq.get(t[0], 0))
        if j not in picks:
            top = cands[0]
            b1 = top[2] if top[2] is not None else None
            picks[j] = (top[1], b1)
            report.append("j%d 在字符 #%d：按语料频率猜 %s(0x%02X)+%s，候选 %s"
                          % (j, pos, top[0], top[1],
                             "无" if b1 is None else repr(chr(b1)),
                             " / ".join("%s(0x%02X)" % (c[0], c[1]) for c in cands[:5])))
        else:
            ch = [c[0] for c in cands if c[1] == picks[j][0]]
            report.append("j%d 在字符 #%d：指定 %s(0x%02X)  %s"
                          % (j, pos, ch[0] if ch else "?", picks[j][0],
                             " ".join(segs[j][-6:].hex(" ").split()) + " | ? | "
                             + (segs[j + 1][:6].hex(" ") if j + 1 < len(segs) else "<末尾>")))
    buf = segs[0]
    for j in range(len(lost)):
        b0, b1 = picks[j]
        buf += bytes([b0]) + (b"" if b1 is None else bytes([b1])) + segs[j + 1]
    try:
        text_out = buf.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SystemExit("拼出来的字节流不是合法 UTF-8：%s" % exc)
    if verbose:
        print("\n".join(report), file=sys.stderr)
    return text_out, picks


def verify(original: str, fixed: str) -> bool:
    """把修好的中文再走一遍当初的事故：UTF-8 编码 -> CP936 解码，看是否逐字符还原乱码。"""
    return cp936_decode(fixed.encode("utf-8")) == original


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="乱码文件（UTF-8 保存）；不给则用 --text")
    ap.add_argument("--text", help="乱码字符串本身")
    ap.add_argument("--picks", help="如 0=89:20,1=8e:20,5=82:none")
    ap.add_argument("--corpus", action="append", default=None,
                    help="算字符频率用的 glob（可多次）；给了就在没指定 picks 时用它排序")
    ap.add_argument("--out", help="把还原后的中文写到这个文件（UTF-8）")
    ap.add_argument("--verify", nargs=2, metavar=("乱码", "中文"),
                    help="只做独立验证：中文 -> UTF-8 -> CP936 是否等于乱码文件")
    args = ap.parse_args()

    if args.verify:
        a, b = args.verify
        orig = open(a, encoding="utf-8").read() if os.path.exists(a) else a
        fixed = open(b, encoding="utf-8").read() if os.path.exists(b) else b
        ok = verify(orig.strip("\n"), fixed.strip("\n"))
        print("round-trip 逐字符一致: %s" % ok)
        return 0 if ok else 1

    if args.file:
        text = open(args.file, encoding="utf-8").read().strip("\n")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read().strip("\n")

    freq = corpus_freq(args.corpus) if args.corpus else None
    fixed, picks = decode(text, parse_picks(args.picks) if args.picks else None, freq)
    print("picks: " + ",".join("%d=%02x:%s" % (j, b0, "none" if b1 is None else "%02x" % b1)
                               for j, (b0, b1) in sorted(picks.items())), file=sys.stderr)
    print("round-trip 逐字符一致: %s" % verify(text, fixed), file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(fixed)
        print("written %s" % args.out, file=sys.stderr)
    print(fixed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
