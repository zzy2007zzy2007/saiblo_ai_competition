"""探针：把**同一条真实出招序列**分别喂给两个引擎，比较每回合局面。

引擎 A：官方 C++ 判题器 `main.exe`（judger 协议，`Ant-Game/game/src/main.cpp`）。
         用 `Ant-Game/game/src/*.cpp` 原样编译，输出到 `code/cpp_engine/build_probe/`。
引擎 B：我们 in-process 的 `GameStateFacade`（= `code/cpp_engine/native_game.pyd`，
         pybind11 绑定 `code/cpp_engine/src/game.cpp`）。`match_via_sdk.py` 用的就是它。

出招序列来源：`match_results/ladder_logs/**` 里的对局日志，逐回合的
`[round N] pX(name) 原文='...' -> [...]` 行。日志里同时有每回合的完整塔快照
（`[round N] towers=[...]`），可以做第三个（也是最强）对照。

用法（三选一，可叠加）：
  # 只跟日志里的 RESULT 行/塔快照对拍（不需要编译任何东西）
  python code/test_match/probe_replay_engines.py <log>

  # 加上官方判题器（先按 build_cmds 编译出 judge_O0/judge_O2）
  python code/test_match/probe_replay_engines.py <log> \
      --judge code/cpp_engine/build_probe/judge_O0.exe \
      --judge code/cpp_engine/build_probe/judge_O2.exe \
      --replay-dir code/cpp_engine/build_probe/replay

  # 跳过 facade（只跑官方判题器）
  python code/test_match/probe_replay_engines.py <log> --judge ... --no-facade

编译官方判题器（-O0 = 官方 Makefile 的默认，无 -O 参数）：
  bash -lc 'cd /d/2026智能体大赛_新2_distill && export TMP=$PWD/code/cpp_engine/build_probe/tmp && \
    /c/mingw64/bin/g++.exe -std=c++17 -g -I Ant-Game/game/include Ant-Game/game/src/*.cpp \
    -o code/cpp_engine/build_probe/judge_O0.exe'
把 `-g` 换成 `-O2` 就是 judge_O2.exe（= 我们 native_game.pyd 的编译选项）。
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 日志行格式（match_via_sdk.py:448 打印；原文是 Python repr，可能含 \n 转义）
RE_SIDE = re.compile(r"\[round (\d+)\] p([01])\(([^)]*)\) 原文=(.*) -> (\[.*\])\s*$")
RE_TOWERS = re.compile(r"\[round (\d+)\] towers=(\[.*\])\s*$")
RE_ANTS = re.compile(r"\[round (\d+)\] ants=(\[.*\])\s*$")
RE_RESULT = re.compile(
    r"RESULT .*?rounds=(\d+) .*?base_hp=(\d+),(\d+)")
RE_SEED = re.compile(r"对战 seed=(\d+)")


# ─────────────────────────── 日志解析 ───────────────────────────
def parse_log(path: Path):
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = RE_SEED.search(txt)
    seed = int(m.group(1)) if m else None
    per_round_text: dict[int, dict[int, str]] = {}
    log_towers: dict[int, list] = {}
    log_ants: dict[int, list] = {}
    for line in txt.splitlines():
        m = RE_SIDE.search(line)
        if m:
            rnd, side, _who, raw = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            try:
                text = ast.literal_eval(raw)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(text, str):
                continue
            per_round_text.setdefault(rnd, {})[side] = text
            continue
        m = RE_TOWERS.search(line)
        if m:
            try:
                log_towers[int(m.group(1))] = ast.literal_eval(m.group(2))
            except Exception:  # noqa: BLE001
                pass
            continue
        m = RE_ANTS.search(line)
        if m:
            try:
                log_ants[int(m.group(1))] = ast.literal_eval(m.group(2))
            except Exception:  # noqa: BLE001
                pass
    res = RE_RESULT.search(txt)
    result = None
    if res:
        result = {"rounds": int(res.group(1)),
                  "hp": (int(res.group(2)), int(res.group(3)))}
    return {"seed": seed, "ops_text": per_round_text,
            "towers": log_towers, "ants": log_ants, "result": result}


# ───────────────────── 出招文本 → Operation ─────────────────────
def parse_ops_text(text: str, Operation, OperationType):
    """逐字复刻 match_via_sdk.py:560 parse_ops_text。"""
    lines = text.strip().split("\n")
    if not lines:
        return []
    try:
        count = int(lines[0].strip())
    except ValueError:
        return []
    ops = []
    for i in range(1, count + 1):
        if i >= len(lines):
            break
        parts = [int(x) for x in lines[i].split()]
        if not parts:
            continue
        op_type = OperationType(parts[0])
        if len(parts) == 1:
            ops.append(Operation(op_type))
        elif len(parts) == 2:
            ops.append(Operation(op_type, parts[1]))
        else:
            ops.append(Operation(op_type, parts[1], parts[2]))
    return ops


def ops_to_text(ops) -> str:
    """规范化为协议文本（judger 只认这个格式）。"""
    if not ops:
        return "0\n"
    toks = [list(op.to_protocol_tokens()) for op in ops]
    return f"{len(toks)}\n" + "\n".join(" ".join(str(t) for t in row) for row in toks) + "\n"


# ─────────────────────── 引擎 B：facade 重放 ───────────────────────
def run_facade(rows, n_rounds: int, seed: int, cold: bool = True):
    from my_ai.az_intent.game_state_facade import GameStateFacade
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=cold)
    out = []
    for r in range(n_rounds):
        for p in (0, 1):
            st.apply_operation_list(p, rows[r][p])
        st.advance_round()
        out.append({
            "round": r + 1,
            "hp": (int(st.bases[0].hp), int(st.bases[1].hp)),
            "coins": (int(st.coins[0]), int(st.coins[1])),
            "towers": [(int(t.tower_id), int(t.player), int(t.x), int(t.y),
                        int(t.tower_type), int(t.hp)) for t in st.towers],
            "ants": [(int(a.ant_id), int(a.player), int(a.x), int(a.y), int(a.hp),
                      int(a.level), int(a.kind)) for a in st.ants],
            "terminal": bool(st.terminal),
            "winner": None if st.winner is None else int(st.winner),
        })
    return out


# ─────────────────── 引擎 A：官方判题器重放 ───────────────────
def _packet(obj) -> bytes:
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def run_judge(exe: Path, rows, n_rounds: int, seed: int, replay_path: Path):
    if replay_path.exists():
        replay_path.unlink()
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    init = {
        "player_list": [1, 1],
        "player_num": 2,
        "config": {"random_seed": seed,
                   "movement_policy": "enhanced",
                   "cold_handle_rule_illegal": True},
        "replay": str(replay_path),
        "time": 0,
    }
    buf = bytearray(_packet(init))
    for r in range(n_rounds):
        for p in (0, 1):
            buf += _packet({"player": p, "content": ops_to_text(rows[r][p]), "time": 0})
    # 收尾包：如果对局在喂完的回合数内没有自然结束，判题器会读这一包 ⇒ is_end →
    # 跳出主循环 → send_end_info() → dump_result(replay)。若已自然结束则此包不会被读。
    buf += _packet({"player": -1, "content": json.dumps({"player": 0, "error": 0}),
                    "time": 0})
    t0 = time.time()
    # stdout=DEVNULL：判题器每回合都往 stdout 写整份局面（440 回合 ≈ 9MB），且官方
    # main.cpp 没有把 stdout 切二进制模式（\n 会被翻成 \r\n，长度前缀就废了），我们
    # 只关心它落盘的 replay.json，所以直接丢掉 stdout。
    proc = subprocess.run([str(exe)], input=bytes(buf), stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, timeout=1800)
    dt = time.time() - t0
    if not replay_path.exists():
        raise RuntimeError(
            f"judge produced no replay (rc={proc.returncode})\n"
            f"stderr={proc.stderr.decode('utf-8', 'replace')[-2000:]}")
    data = json.loads(replay_path.read_text(encoding="utf-8"))
    out = []
    for i, entry in enumerate(data):
        rs = entry.get("round_state", {})
        camps = rs.get("camps")
        out.append({
            "round": i + 1,
            "hp": (int(camps[0]), int(camps[1])) if camps else None,
            "coins": tuple(int(c) for c in rs["coins"]) if rs.get("coins") else None,
            "towers_raw": rs.get("towers", []),
            "ants_raw": rs.get("ants", []),
            "winner": rs.get("winner"),
        })
    return out, proc, dt


# ─────────────────────────── 对比 ───────────────────────────
def first_div_hp(a, b, label_a, label_b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i]["hp"] != b[i]["hp"]:
            return (i + 1, a[i]["hp"], b[i]["hp"])
    if len(a) != len(b):
        return (n + 1, f"len {len(a)}", f"len {len(b)}")
    return None


def first_div_towers(facade_rows, log_towers):
    """facade 的塔快照 vs 日志里每回合的完整塔快照。日志 key = 回合号（1 起）。"""
    for row in facade_rows:
        r = row["round"]
        if r not in log_towers:
            continue
        ours = sorted(row["towers"])
        theirs = sorted(tuple(int(v) for v in t) for t in log_towers[r])
        if ours != theirs:
            only_ours = [t for t in ours if t not in theirs]
            only_log = [t for t in theirs if t not in ours]
            return (r, only_ours, only_log)
    return None


def fmt_hp(hp):
    return f"({hp[0]},{hp[1]})" if isinstance(hp, (tuple, list)) else str(hp)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--judge", action="append", default=[], type=Path)
    ap.add_argument("--replay-dir", type=Path, default=Path("C:/Temp/probe_build/replay"),
                    help="必须放在**纯 ASCII** 路径下：官方判题器是 MinGW 窄字符 build，"
                         "replay 路径含中文时 std::ofstream 静默失败（replay 文件不会生成）")
    ap.add_argument("--no-facade", action="store_true")
    ap.add_argument("--max-rounds", type=int, default=None)
    ap.add_argument("--trace-round", type=int, default=None,
                    help="额外打印该回合 facade 与日志的完整塔/蚁快照")
    args = ap.parse_args()

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    info = parse_log(args.log)
    seed = info["seed"]
    print(f"[log] {args.log}")
    print(f"[log] seed={seed}  RESULT={info['result']}")
    if seed is None:
        print("!! 日志里没有 '对战 seed=N'，无法确定种子")
        return 2

    rounds_with_ops = sorted(info["ops_text"])
    n_rounds = (rounds_with_ops[-1] + 1) if rounds_with_ops else 0
    if info["result"]:
        n_rounds = min(n_rounds, info["result"]["rounds"])
    if args.max_rounds:
        n_rounds = min(n_rounds, args.max_rounds)
    missing = [r for r in range(n_rounds) if 0 not in info["ops_text"].get(r, {})
               or 1 not in info["ops_text"].get(r, {})]
    print(f"[log] rounds with both sides' ops: {n_rounds}"
          + (f"  (missing ops in {len(missing)} rounds, treated as pass)" if missing else ""))
    print(f"[log] logged tower snapshots: {len(info['towers'])} rounds, "
          f"ant snapshots: {len(info['ants'])} rounds")

    rows = []
    for r in range(n_rounds):
        d = info["ops_text"].get(r, {})
        rows.append({p: parse_ops_text(d.get(p, "0\n"), Operation, OperationType)
                     for p in (0, 1)})

    report = {}

    # ---- 引擎 B ----
    facade_rows = None
    if not args.no_facade:
        t0 = time.time()
        facade_rows = run_facade(rows, n_rounds, seed)
        print(f"\n[facade] native_game.pyd  {n_rounds} rounds in {time.time()-t0:.2f}s")
        print(f"[facade] final hp={fmt_hp(facade_rows[-1]['hp'])} "
              f"coins={facade_rows[-1]['coins']} "
              f"terminal={facade_rows[-1]['terminal']} winner={facade_rows[-1]['winner']}")
        report["facade_final"] = {"hp": list(facade_rows[-1]["hp"]),
                                  "rounds": n_rounds}
        d = first_div_towers(facade_rows, info["towers"])
        if d is None:
            print("[facade vs log] 每回合塔快照完全一致 ✅")
        else:
            r, only_ours, only_log = d
            print(f"[facade vs log] 首个塔快照不一致：round {r}"
                  f"\n    ours only: {only_ours[:6]}\n    log  only: {only_log[:6]}")
            report["facade_vs_log_towers_first_div"] = r
        if info["result"]:
            loghp = info["result"]["hp"]
            ok = facade_rows[-1]["hp"] == loghp
            print(f"[facade vs RESULT] hp: facade={fmt_hp(facade_rows[-1]['hp'])} "
                  f"log={loghp}  -> {'一致 ✅' if ok else '不一致 ❌'}")
            report["facade_vs_result_hp"] = ok

    # ---- 引擎 A ----
    judge_results = {}
    for exe in args.judge:
        if not exe.exists():
            print(f"\n[judge {exe.name}] !! 不存在，跳过")
            continue
        rp = args.replay_dir / f"replay_{exe.stem}_{args.log.stem}.json"
        rows_j, proc, dt = run_judge(exe, rows, n_rounds, seed, rp)
        judge_results[exe.name] = rows_j
        print(f"\n[judge {exe.name}] {len(rows_j)} rounds in {dt:.2f}s  rc={proc.returncode}")
        if rows_j:
            print(f"[judge {exe.name}] final hp={fmt_hp(rows_j[-1]['hp'])} "
                  f"coins={rows_j[-1]['coins']} winner={rows_j[-1]['winner']}")
        if info["result"]:
            loghp = info["result"]["hp"]
            ok = bool(rows_j) and tuple(rows_j[-1]["hp"]) == loghp
            print(f"[judge {exe.name} vs RESULT] hp={fmt_hp(rows_j[-1]['hp']) if rows_j else None} "
                  f"log={loghp}  -> {'一致 ✅' if ok else '不一致 ❌'}")
            report[f"judge_{exe.name}_vs_result_hp"] = ok
        if len(rows_j) != n_rounds:
            print(f"[judge {exe.name}] 注意：回合数 {len(rows_j)} != 喂入 {n_rounds}")
        report[f"judge_{exe.name}_final"] = {
            "hp": list(rows_j[-1]["hp"]) if rows_j else None,
            "rounds": len(rows_j)}

    # ---- facade vs judge ----
    if facade_rows is not None:
        for name, rows_j in judge_results.items():
            d = first_div_hp(facade_rows, rows_j, "facade", name)
            if d is None:
                print(f"\n[facade vs {name}] 每回合 base_hp 完全一致 ✅")
            else:
                r, a, b = d
                print(f"\n[facade vs {name}] 首个 base_hp 不一致：round {r} "
                      f"facade={fmt_hp(a)} judge={fmt_hp(b)}")
                report[f"facade_vs_{name}_first_div"] = r
            # coins
            for i in range(min(len(facade_rows), len(rows_j))):
                if rows_j[i]["coins"] and tuple(facade_rows[i]["coins"]) != rows_j[i]["coins"]:
                    print(f"[facade vs {name}] 首个 coins 不一致：round {i+1} "
                          f"facade={facade_rows[i]['coins']} judge={rows_j[i]['coins']}")
                    report[f"facade_vs_{name}_first_coin_div"] = i + 1
                    break
            else:
                print(f"[facade vs {name}] 每回合 coins 完全一致 ✅")

    # ---- 多 judge 互比 ----
    names = list(judge_results)
    if len(names) >= 2:
        a, b = names[0], names[1]
        d = first_div_hp(judge_results[a], judge_results[b], a, b)
        if d is None:
            print(f"\n[{a} vs {b}] 每回合 base_hp 完全一致（编译选项不影响）✅")
        else:
            r, x, y = d
            print(f"\n[{a} vs {b}] 首个 base_hp 不一致：round {r} {a}={fmt_hp(x)} {b}={fmt_hp(y)}")

    if args.trace_round:
        r = args.trace_round
        print(f"\n===== trace round {r} =====")
        if facade_rows:
            row = facade_rows[r - 1]
            print(f"facade towers: {sorted(row['towers'])}")
            print(f"facade ants  : {sorted(row['ants'])}")
        print(f"log    towers: {sorted(tuple(int(v) for v in t) for t in info['towers'].get(r, []))}")
        print(f"log    ants  : {sorted(info['ants'].get(r, []))}")

    out = args.replay_dir / f"probe_summary_{args.log.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[summary] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
