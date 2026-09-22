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

RESULT_RE = re.compile(
    r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) verdict=(\S+) engine_winner=(\S+) "
    r"rounds=(\S+) terminal=(\S+) base_hp=(\d+),(\d+) coins=(\d+),(\d+)")
OPS_RE = re.compile(r"\[round (\d+)\] p(\d)\((\w+)\) .*-> \[(.*)\]$")


def parse_cli(argv: list[str]) -> tuple[dict, list[str]]:
    cfg = {"tag": "ladder", "jobs": 8, "ai0": None, "ai1": None}
    toks: list[str] = []
    for a in argv:
        if a.startswith("--tag="):
            cfg["tag"] = a.split("=", 1)[1]
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
    g: dict = {"token": token, "winner": None, "verdict": None, "engine_winner": None,
               "rounds": None, "terminal": None, "hp": None, "coins": None,
               "p0_label": None, "p1_label": None, "illegal": 0,
               "ops": collections.Counter(), "act": collections.Counter(),
               "hist": collections.defaultdict(collections.Counter)}
    for ln in txt.splitlines():
        if "ILLEGAL" in ln:
            g["illegal"] += 1
        m = RESULT_RE.search(ln)
        if m:
            g["p0_label"], g["p1_label"] = m.group(2), m.group(3)
            g["winner"], g["verdict"], g["engine_winner"] = m.group(4), m.group(5), m.group(6)
            g["rounds"], g["terminal"] = m.group(7), m.group(8)
            g["hp"] = (int(m.group(9)), int(m.group(10)))
            g["coins"] = (int(m.group(11)), int(m.group(12)))
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
              f"{min(dts)/60:.1f}/{sorted(dts)[len(dts)//2]/60:.1f}/{max(dts)/60:.1f} min", flush=True)

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

    wins = collections.Counter(g["winner"] for g in games if g["winner"])
    print(f"\n=== 胜负汇总: {dict(wins)} ===", flush=True)
    verdicts = collections.Counter(g["verdict"] for g in games if g["verdict"])
    print(f"判词来源: {dict(verdicts)}  (engine=官方5级级联; hp_tiebreak=非官方兜底)", flush=True)
    if verdicts.get("hp_tiebreak"):
        print("  ⚠️ 有 hp_tiebreak 的局 ⇒ 那些局的胜负不是官方规则算的，别当结论用。", flush=True)
    if wins.get("draw"):
        print("  ⚠️ 出现 draw ⇒ 判词一定不是官方的（官方级联第⑤级恒判 P0，从不平局）。", flush=True)

    # ---- 僵局检查：永远空过会让"对称签名"假通过 ----
    dead = [g["token"] for g in games if g["ops"]["p0"] + g["ops"]["p1"] <= 2]
    print(f"僵局嫌疑（全场双方操作总数 ≤2）: {dead if dead else '无'}", flush=True)

    # ---- 镜像配对（AI0 得分 / 2）----
    seeds_seen: dict[int, list[dict]] = collections.defaultdict(list)
    for g in games:
        s = re.match(r"(\d+)", g["token"])
        if s:
            seeds_seen[int(s.group(1))].append(g)
    pairs = {s: v for s, v in seeds_seen.items() if len(v) == 2}
    if pairs:
        print(f"\n=== 镜像配对（AI0 得分 / 2）: {len(pairs)} 对 ===", flush=True)
        vals: list[float] = []
        for s in sorted(pairs):
            sc = 0.0
            ok = True
            for g in pairs[s]:
                if not g["winner"]:
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
                  f"（完全相同 AI 的预期 = 1.000 / 0.0000）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
