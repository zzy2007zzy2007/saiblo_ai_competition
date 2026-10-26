"""真实桥接的线上字节 dump + `state_to_text` 对 `sdk.hpp` 的严格复验（2026-10-26）。

做三件事：
1. **拦截真实对局**（`match_via_sdk.run_match`）里裁判↔AI 的每一条消息——不改桥接一行代码，
   只把 `send_line` / `ProcIO.read_packet` 换成本文件的记录版（`run_match` 按模块全局名查表）。
2. 把发给亚军（player 1）的消息按**顺序**列出来 ⇒ 这就是协议的确切报文序列（真值，不是推测）。
3. 用一份**逐行照抄 `antgame_sdk/sdk.hpp:1087-1214 recv_round_state`** 的严格解析器
   复验每一条局面文本：必须正好吃掉全部行、且 `round_index` 递增。
   `state_to_text()` 与官方格式是否一致，由这个断言回答，不靠肉眼。

用法：python code/test_match/probe_wire_trace.py [seed] [rounds]
输出：match_results/wire_trace_seed<N>.jsonl（逐条消息）+ 控制台上的可读序列
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (REPO_ROOT / "code", REPO_ROOT / "code" / "cpp_engine", REPO_ROOT / "Ant-Game",
           REPO_ROOT / "code" / "test_match"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import match_via_sdk as M  # noqa: E402

EVENTS: list[dict] = []


# ── 严格镜像：sdk.hpp:1087-1214 recv_round_state ─────────────────────────────
class StrictStateParser:
    """逐行照抄 C++ 的解析顺序与必需字段数；任何偏差都当场抛。"""

    def __init__(self, text: str, expect_round: int | None = None):
        self.lines = text.split("\n")
        if self.lines and self.lines[-1] == "":
            self.lines.pop()            # 末尾换行
        self.i = 0
        self.expect_round = expect_round

    def _next(self) -> str:
        if self.i >= len(self.lines):
            raise ValueError(f"行不够：已消费 {self.i} 行，共 {len(self.lines)} 行")
        line = self.lines[self.i]
        self.i += 1
        return line

    def _ints(self, line: str) -> list[int]:
        return [int(t) for t in line.split()]

    def parse(self) -> dict:
        st: dict = {}
        st["round_index"] = int(self._next())
        if self.expect_round is not None and st["round_index"] != self.expect_round:
            raise ValueError(f"round_index={st['round_index']} 期望 {self.expect_round}")
        n_tower = int(self._next())
        st["towers"] = []
        for _ in range(n_tower):
            row = self._ints(self._next())
            if len(row) < 6:            # sdk.hpp:1107  matched < 6 -> "invalid tower row"
                raise ValueError(f"invalid tower row: {row}")
            st["towers"].append(row)
        n_ant = int(self._next())
        st["ants"] = []
        for _ in range(n_ant):
            row = self._ints(self._next())
            if len(row) < 8:            # sdk.hpp:1137  matched < 8 -> "invalid ant row"
                raise ValueError(f"invalid ant row: {row}")
            st["ants"].append(row)
        coins = self._ints(self._next())
        if len(coins) != 2:             # sdk.hpp:1157  必须正好 2
            raise ValueError(f"invalid coin row: {coins}")
        st["coins"] = coins
        camp = self._ints(self._next())
        if len(camp) < 2:               # sdk.hpp:1171
            raise ValueError(f"invalid camp row: {camp}")
        st["camps_hp"] = camp[:2]
        st["speed_lv"] = camp[2:4] if len(camp) >= 4 else None
        st["anthp_lv"] = camp[4:6] if len(camp) >= 6 else None
        n_cd = int(self._next())
        st["weapon_cooldowns"] = []
        for _ in range(min(n_cd, 2)):
            row = self._ints(self._next())
            if len(row) < 4:            # sdk.hpp:1191
                raise ValueError(f"invalid cooldown row: {row}")
            st["weapon_cooldowns"].append(row)
        n_fx = int(self._next())
        st["active_effects"] = []
        for _ in range(n_fx):
            row = self._ints(self._next())
            if len(row) != 5:           # sdk.hpp:1208  != 5 -> "invalid effect row"
                raise ValueError(f"invalid effect row: {row}")
            st["active_effects"].append(row)
        if self.i != len(self.lines):
            raise ValueError(f"还剩 {len(self.lines) - self.i} 行没被消费："
                             f"{self.lines[self.i:][:3]}")
        return st


def main() -> int:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    # p0 用 rv4（Python 壳，已知 good control），避免依赖 magica 的 e:\ 运行时目录
    M.AI0_EXE = (REPO_ROOT / "code" / "test_match" / "rv4_pkg" / "main.py").resolve()

    orig_send_line = M.send_line
    orig_read_packet = M.ProcIO.read_packet

    def send_line(io, text: str) -> None:      # noqa: ANN001
        EVENTS.append({"dir": "judge->ai", "ai": io._label, "text": text})
        orig_send_line(io, text)

    def read_packet(self):                     # noqa: ANN001
        data = orig_read_packet(self)
        EVENTS.append({"dir": "ai->judge", "ai": self._label,
                       "text": data.decode("utf-8", errors="replace")})
        return data

    M.send_line = send_line
    M.ProcIO.read_packet = read_packet
    print(f"[trace] seed={seed} rounds={rounds} AI0={M.AI0_EXE.name} AI1=runnerup", flush=True)
    keep = REPO_ROOT / "match_results" / "wire_trace"
    result = M.run_match(seed, keep, max_rounds=rounds)
    print(f"[trace] run_match: rounds={result.get('rounds')} exc={result.get('exception')}",
          flush=True)

    out = REPO_ROOT / "match_results" / f"wire_trace_seed{seed}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for ev in EVENTS:
            fh.write(json.dumps({"seed": seed, **ev}, ensure_ascii=False) + "\n")
    print(f"[trace] {len(EVENTS)} 条消息 → {out}", flush=True)

    # ── 只看发给亚军的流（这就是 RunnerUpOracle 必须复刻的序列）──────────────
    # ⚠️ 不能用「首行是数字就算操作块」来分类：一条 round=N 且正好 N+1 行的局面文本
    #    会被误判（实测 round=14/15 行时发生）。真桥接对 P1 的流必然是严格交替的
    #    `init, 对手操作块, 局面, 对手操作块, 局面, ...`（match_via_sdk.py:425-491），
    #    所以这里用**位置状态机**，内容只用来断言。
    ru = [ev for ev in EVENTS if ev["ai"] == "ai_cpp_lure_v4"]
    print(f"\n=== 发给亚军(ai_cpp_lure_v4)的消息序列，共 {len(ru)} 条 ===")
    expect_round = 1
    expect_kind = "ops"
    bad = 0
    for idx, ev in enumerate(ru):
        text = ev["text"]
        nbytes = len(text.encode("utf-8"))
        lines = [ln for ln in text.split("\n") if ln != ""]
        if ev["dir"] == "ai->judge":
            print(f"  [{idx:02d}] AI->裁判 {nbytes:4d}B  回包原文 {text!r}")
            continue
        if idx == 0:      # init 行
            print(f"  [{idx:02d}] 裁判->AI {nbytes:4d}B  init {text!r}")
            assert text.count(" ") == 1 and text.strip().split()[0] == "1", text
            continue
        if expect_kind == "ops":
            try:
                count = int(lines[0].strip())
                if len(lines) != count + 1:
                    raise ValueError(f"操作数 {count} 但来了 {len(lines) - 1} 行")
                for row in lines[1:]:
                    toks = [int(t) for t in row.split()]
                    if not 1 <= len(toks) <= 3:        # sdk.hpp:1072-1082
                        raise ValueError(f"操作行字段数 {len(toks)} 不合法: {row!r}")
                print(f"  [{idx:02d}] 裁判->AI {nbytes:4d}B  对手操作块 ✓ count={count} "
                      f"全文={text!r}")
            except Exception as exc:  # noqa: BLE001
                bad += 1
                print(f"  [{idx:02d}] 裁判->AI {nbytes:4d}B  对手操作块 ✗ {exc} 原文={text!r}")
            expect_kind = "state"
        else:
            try:
                st = StrictStateParser(text, expect_round=expect_round).parse()
                expect_round += 1
                print(f"  [{idx:02d}] 裁判->AI {nbytes:4d}B  局面文本 ✓严格解析通过 "
                      f"round={st['round_index']} 塔={len(st['towers'])} 蚁={len(st['ants'])} "
                      f"金币={st['coins']} 基地={st['camps_hp']} "
                      f"冷却行={len(st['weapon_cooldowns'])} 效果={len(st['active_effects'])} "
                      f"({len(lines)} 行全消费)")
            except Exception as exc:  # noqa: BLE001
                bad += 1
                print(f"  [{idx:02d}] 裁判->AI {nbytes:4d}B  局面文本 ✗ 解析失败: {exc}")
                print(f"        原文={text!r}")
            expect_kind = "ops"
    print(f"\n[trace] 局面文本严格解析失败 {bad} 条 / 共 {max(expect_round - 1, 0)} 条")
    print(f"[trace] 结论：{'PASS' if bad == 0 else 'FAIL'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
