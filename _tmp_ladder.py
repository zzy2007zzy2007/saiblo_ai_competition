"""阶梯实验的并行运行器：并行跑 `code/test_match/match_via_sdk.py`（bridge），汇总。

用途（见 docs/champ_ladder_plan.md）：把"我们 ↔ rule_v4 ↔ 冠军/亚军"接到同一把尺子上。

用法:
  python _tmp_ladder.py --tag=A1_rv4_self --jobs=8 \
      --ai0=code/test_match/rv4_pkg/main.py --ai1=code/test_match/rv4_pkg/main.py \
      7 7r 8 8r            # 7 = AI0 先手；7r = AI1 先手（镜像）

设计要点（都是被历史坑出来的，别删）:
  1) 胜负**只读 bridge 自己打的 RESULT 行**，绝不在这里重推一遍
     （记忆 feedback_known_good_control_for_harness 第四条：重打判定把"败"记成了"平"）。
  2) 报 `verdict=`：`engine` = 引擎终局时算好的官方 5 级级联；`hp_tiebreak` = 循环提前停下时
     bridge 的兜底，**不是官方规则**。官方规则**从不判平局** ⇒ 出现 `draw` 即证明判词非官方。
  3) ⚠️ **镜像对称（配对分=1.0、方差 0）是必要不充分的**：若两个 AI 都永远空过，游戏会打满
     512 回合，官方级联第⑤级"全相同判 P0 胜"会让 P0 必胜 ⇒ 签名照样完美通过。
     **所以必须同时报"双方真的出招了"**（每局非空操作数），否则那只是"僵局伪装成对称"。
  4) 子进程设 PYTHONIOENCODING=utf-8，否则日志是 GBK、UTF-8 读全是乱码。
"""
from __future__ import annotations

import collections
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOG_ROOT = REPO / "match_results" / "ladder_logs"
TIMEOUT = 7200

# 本脚本自己的 stdout/stderr 会被 run_logged 重定向进文件，此时 Python 用 **cp936** ⇒
# 中文变乱码、emoji 直接 UnicodeEncodeError 崩掉（实测踩到：打印 "⚠️" 时异常退出，
# 把整个汇总吃掉）。强制 UTF-8 + replace，任何字符都不会再让脚本死。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

RESULT_RE = re.compile(
    r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) (?:winner_side=(\S+) )?verdict=(\S+) "
    r"engine_winner=(\S+) rounds=(\S+) terminal=(\S+) base_hp=(\d+),(\d+) coins=(\d+),(\d+)")
OPS_RE = re.compile(r"\[round (\d+)\] p(\d)\((\w+)\) .*-> \[(.*)\]$")


def parse_cli(argv: list[str]) -> tuple[dict, list[str]]:
    cfg = {"tag": "ladder", "jobs": 8, "ai0": None, "ai1": None, "reanalyze": False}
    toks: list[str] = []
    for a in argv:
        if a.startswith("--tag="):
            cfg["tag"] = a.split("=", 1)[1]
        elif a == "--reanalyze":
            cfg["reanalyze"] = True
        elif a.startswith("--jobs="):
            cfg["jobs"] = int(a.split("=", 1)[1])
        elif a.startswith("--ai0="):
            cfg["ai0"] = a.split("=", 1)[1]
        elif a.startswith("--ai1="):
            cfg["ai1"] = a.split("=", 1)[1]
        else:
            toks.append(a)
    return cfg, toks


def run_one(token: str, log_path: Path, fwd: list[str], sem: threading.Semaphore,
            results: dict, lock: threading.Lock) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-u", "code/test_match/match_via_sdk.py", "--trace-ops", *fwd, token]
    with sem:
        t0 = time.time()
        rc = -1
        try:
            with open(log_path, "wb") as f:
                p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                   cwd=str(REPO), timeout=TIMEOUT)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            rc = -2
        except Exception as exc:  # noqa: BLE001
            rc = -3
            print(f"[{token}] EXC {type(exc).__name__}: {exc}", flush=True)
        dt = time.time() - t0
        with lock:
            results[token] = (rc, dt)
            print(f"[{token}] rc={rc}  {dt / 60:.1f} min", flush=True)


def read_game(log_path: Path, token: str) -> dict:
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    g: dict = {"token": token, "winner": None, "winner_side": None, "verdict": None,
               "engine_winner": None,
               "rounds": None, "terminal": None, "hp": None, "coins": None,
               "p0_label": None, "p1_label": None, "same_label": False, "illegal": 0,
               "ops": collections.Counter(), "act": collections.Counter(),
               "hist": collections.defaultdict(collections.Counter)}
    for ln in txt.splitlines():
        if "ILLEGAL" in ln:
            g["illegal"] += 1
        m = RESULT_RE.search(ln)
        if m:
            g["p0_label"], g["p1_label"] = m.group(2), m.group(3)
            g["same_label"] = (m.group(2) == m.group(3))
            g["winner"] = m.group(4)
            g["winner_side"] = m.group(5)          # 新格式才有；旧日志为 None
            g["verdict"], g["engine_winner"] = m.group(6), m.group(7)
            g["rounds"], g["terminal"] = m.group(8), m.group(9)
            g["hp"] = (int(m.group(10)), int(m.group(11)))
            g["coins"] = (int(m.group(12)), int(m.group(13)))
            # 旧日志（无 winner_side）的兜底：HP 不等时按 HP 判侧（与官方级联第①级一致）。
            # ⚠️ 这只是读旧日志用的近似，不是"重新判胜负"；HP 相等时留 None，不猜。
            if g["winner_side"] is None and g["verdict"] != "aborted":
                h0, h1 = g["hp"]
                g["winner_side"] = "p0" if h0 > h1 else ("p1" if h1 > h0 else None)
        m = OPS_RE.search(ln)
        if m:
            who, ops = "p" + m.group(2), m.group(4)
            n = ops.count("Operation(")
            g["ops"][who] += n
            if n:
                g["act"][who] += 1
            for t in re.findall(r"<OperationType\.(\w+):", ops):
                g["hist"][who][t] += 1
    return g


def main() -> int:
    cfg, tokens = parse_cli(sys.argv[1:])
    if not tokens:
        print("用法: python _tmp_ladder.py --tag=NAME [--jobs=8] [--ai0=..] [--ai1=..] 7 7r 8 8r ...")
        return 2
    out_dir = LOG_ROOT / cfg["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fwd: list[str] = []
    if cfg["ai0"]:
        fwd.append(f"--ai0={cfg['ai0']}")
    if cfg["ai1"]:
        fwd.append(f"--ai1={cfg['ai1']}")

    print(f"[ladder] tag={cfg['tag']}  jobs={cfg['jobs']}", flush=True)
    print(f"[ladder] ai0={cfg['ai0']}", flush=True)
    print(f"[ladder] ai1={cfg['ai1']}", flush=True)
    print(f"[ladder] tokens={tokens}  logs={out_dir}", flush=True)

    results: dict[str, tuple[int, float]] = {}
    if cfg["reanalyze"]:
        # 只重新解析已存在的日志（改了汇总逻辑后不必重跑对局）
        print("[ladder] --reanalyze：跳过对局，直接解析已有日志", flush=True)
    else:
        lock = threading.Lock()
        sem = threading.Semaphore(cfg["jobs"])
        t0 = time.time()
        threads = [threading.Thread(target=run_one,
                                    args=(t, out_dir / f"{t}.log", fwd, sem, results, lock))
                   for t in tokens]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        dts = [d for _, d in results.values()]
        if dts:
            print(f"\n[ladder] 全部结束 wall={wall:.1f}s  单局 min/med/max = "
                  f"{min(dts)/60:.1f}/{sorted(dts)[len(dts)//2]/60:.1f}/{max(dts)/60:.1f} min",
                  flush=True)

    games = []
    for t in tokens:
        g = read_game(out_dir / f"{t}.log", t)
        g["rc"] = results.get(t, (None, 0))[0]
        games.append(g)

    print("\n=== 逐局 ===", flush=True)
    print("%-6s %-10s %-10s %-10s %-12s %-6s %-11s %-4s %-30s %-30s %s" % (
        "token", "p0", "p1", "winner", "verdict", "rounds", "hp", "ill",
        "p0 act/ops", "p1 act/ops", "rc"), flush=True)
    for g in games:
        print("%-6s %-10s %-10s %-10s %-12s %-6s %-11s %-4d %-30s %-30s %s" % (
            g["token"], g["p0_label"], g["p1_label"], g["winner"], g["verdict"],
            g["rounds"], str(g["hp"]), g["illegal"],
            f"{g['act']['p0']}/{g['ops']['p0']} {dict(g['hist']['p0'].most_common(3))}",
            f"{g['act']['p1']}/{g['ops']['p1']} {dict(g['hist']['p1'].most_common(3))}",
            g["rc"]), flush=True)

    invalid = [g["token"] for g in games if not g["winner"] or g["winner"] == "INVALID"]
    wins = collections.Counter(g["winner"] for g in games
                               if g["winner"] and g["winner"] != "INVALID")
    print(f"\n=== 胜负汇总: {dict(wins)} ===", flush=True)
    if invalid:
        print(f"  !! 无效局（异常终止/截断，不计分）: {invalid}", flush=True)
    verdicts = collections.Counter(g["verdict"] for g in games if g["verdict"])
    print(f"判词来源: {dict(verdicts)}  (engine=官方5级级联; hp_tiebreak=非官方兜底)", flush=True)
    if verdicts.get("hp_tiebreak"):
        print("  !! 有 hp_tiebreak 的局 => 那些局的胜负不是官方规则算的，别当结论用。", flush=True)
    if wins.get("draw"):
        print("  !! 出现 draw => 判词一定不是官方的（官方级联第5级恒判 P0，从不平局）。", flush=True)

    # ---- 僵局检查：永远空过会让"对称签名"假通过 ----
    dead = [g["token"] for g in games if g["ops"]["p0"] + g["ops"]["p1"] <= 2]
    print(f"僵局嫌疑（全场双方操作总数 ≤2）: {dead if dead else '无'}", flush=True)

    # ---- 先手侧统计（来自 RESULT 的 winner_side，与标签无关 ⇒ 两个 AI 同名时也能用）----
    side = collections.Counter()
    for g in games:
        if not g["winner"] or g["winner"] == "INVALID":
            continue
        side[g["winner_side"] or "不可判(HP 相等且无 winner_side)"] += 1
    print(f"\n=== 先手侧胜负: {dict(side)} ===", flush=True)
    if side.get("不可判(HP 相等且无 winner_side)"):
        print("  !! 有局判不出侧别（旧日志且 HP 相等）——请用新 bridge 重跑以获得 winner_side。",
              flush=True)

    # ---- 镜像配对（AI0 得分 / 2）----
    # ⚠️ 只有**两个 AI 不同**时这个读数才成立（那时 N/Nr 是先手互换的两盘棋）。
    # 若 AI0 == AI1：N 与 Nr 里执先手的都是同一个程序 ⇒ 两局是**同一盘棋**、胜负必然相同，
    # 而 winner 只有标签、区分不出是哪一边 ⇒ 得分恒为 2.0（我就这样把 A1 误读过一次）。
    seeds_seen: dict[int, list[dict]] = collections.defaultdict(list)
    for g in games:
        s = re.match(r"(\d+)", g["token"])
        if s:
            seeds_seen[int(s.group(1))].append(g)
    pairs = {s: v for s, v in seeds_seen.items() if len(v) == 2}
    if pairs and any(g["same_label"] for g in games):
        print(f"\n=== 镜像配对: {len(pairs)} 对 —— 两个 AI 相同（同标签），配对分**不可用**；"
              f"改用上面的「先手侧胜负」。这里只做**确定性检查** ===", flush=True)
        for s in sorted(pairs):
            tok = [g["token"] for g in pairs[s]]
            readouts = {(g["rounds"], str(g["hp"]), str(g["coins"]), g["winner"]) for g in pairs[s]}
            same = "读数逐字段相同 ✓" if len(readouts) == 1 else f"读数不同! {readouts}"
            print(f"  seed {s}: {tok} -> {same}", flush=True)
    elif pairs:
        print(f"\n=== 镜像配对（AI0 得分 / 2）: {len(pairs)} 对 ===", flush=True)
        vals: list[float] = []
        for s in sorted(pairs):
            sc = 0.0
            ok = True
            for g in pairs[s]:
                if not g["winner"] or g["winner"] == "INVALID":
                    ok = False
                    break
                ai0_label = g["p1_label"] if g["token"].endswith("r") else g["p0_label"]
                sc += 0.5 if g["winner"] == "draw" else (1.0 if g["winner"] == ai0_label else 0.0)
            print(f"  seed {s}: AI0 得分 = {sc if ok else '?'}   ({[g['token'] for g in pairs[s]]})",
                  flush=True)
            if ok:
                vals.append(sc)
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            print(f"  均值 = {mean:.3f}  方差 = {var:.4f}   "
                  f"（等强度 AI 的预期 = 1.000 / 0.0000）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
