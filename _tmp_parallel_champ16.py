"""冠军(magica_v3_O0) vs 亚军(ai_cpp_lure_v4) 的并行完整对局（512 回合），
裁判 = 我们的 code/cpp_engine（match_via_sdk.py 里的 GameStateFacade）。

相对 08-10 的 `_tmp_parallel_matches.py`：
  1) 传 `--trace-ops` ⇒ **双方每回合的实际操作都写进各自的种子日志**（用户 2026-09-22 要求）；
  2) 子进程设 `PYTHONIOENCODING=utf-8` ⇒ 否则日志被写成 GBK，UTF-8 读全是乱码
     （旧脚本的中文汇总正则就是因为这个才全失效的）；
  3) 汇总只用 ASCII 模式解析（`base_hp:` / `[round N] p0(\w+)` / `OperationType.XXX`）。

**镜像局**：seed token 带后缀 `r` 表示**亚军先手**（`7` = 冠军先手，`7r` = 亚军先手）。
胜负按"谁执 p0"判定（不能硬编码 hp[0]=冠军，镜像局会算反）。

用法:
  python _tmp_parallel_champ16.py 7 8 ... 22        # 冠军先手那半
  python _tmp_parallel_champ16.py 7r 8r ... 22r     # 亚军先手那半
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
LOG_DIR = REPO / "match_results" / "champ16_logs"

# 同 _tmp_ladder.py：被 run_logged 重定向到文件时 Python 用 cp936 ⇒ 中文乱码 / emoji 崩溃。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
TOKENS = sys.argv[1:] or [str(s) for s in range(7, 23)]
TIMEOUT = 7200  # 单局上限；多路并行有 CPU 争抢，留足余量

results: dict[str, tuple[int, float]] = {}
lock = threading.Lock()


def run_one(token: str) -> None:
    log = LOG_DIR / f"seed{token}.log"
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    rc = -1
    try:
        with open(log, "wb") as f:
            p = subprocess.run(
                [sys.executable, "-u", "code/test_match/match_via_sdk.py", "--trace-ops", token],
                stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(REPO), timeout=TIMEOUT,
            )
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = -2
    except Exception as exc:  # noqa: BLE001
        rc = -3
        print(f"[seed {token}] EXC {type(exc).__name__}: {exc}", flush=True)
    dt = time.time() - t0
    with lock:
        results[token] = (rc, dt)
        print(f"[seed {token}] rc={rc}  {dt / 60:.1f} min", flush=True)


def parse(token: str) -> dict:
    """从一份种子日志里抠出：p0 是谁、双方操作构成、回合数、残血。"""
    txt = (LOG_DIR / f"seed{token}.log").read_text(encoding="utf-8", errors="replace")
    g = {"token": token, "rounds": 0, "hp": None, "coins": None, "illegal": 0, "exc": 0,
         "p0role": None, "ops": collections.Counter(), "act": collections.Counter(),
         "hist": collections.defaultdict(collections.Counter),
         "towers_first": None, "towers_last": 0}
    for ln in txt.splitlines():
        if "ILLEGAL" in ln:
            g["illegal"] += 1
        if "EXC " in ln:
            g["exc"] += 1
        m = re.search(r"\[round (\d+)\] (p\d)\((\w+)\) .*-> \[(.*)\]$", ln)
        if m:
            rnd, who, role, ops = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            if who == "p0" and g["p0role"] is None:
                g["p0role"] = role          # 谁执 p0（先手）
            g["rounds"] = max(g["rounds"], rnd + 1)
            n = ops.count("Operation(")
            g["ops"][role] += n
            if n:
                g["act"][role] += 1
            for t in re.findall(r"<OperationType\.(\w+):", ops):
                g["hist"][role][t] += 1
        m = re.search(r"\[round (\d+)\] towers=\[(.*)\]\s*$", ln)
        if m:
            body = m.group(2).strip()
            g["towers_last"] = body.count("(")
            if body and g["towers_first"] is None:
                g["towers_first"] = int(m.group(1))
        m = re.search(r"base_hp: \[(\d+), (\d+)\]\s+coins: \[(\d+), (\d+)\]", ln)
        if m:
            g["hp"] = (int(m.group(1)), int(m.group(2)))
            g["coins"] = (int(m.group(3)), int(m.group(4)))
    # hp[0] 属于先手那一方（p0role）
    if g["hp"] and g["p0role"]:
        p1role = "runnerup" if g["p0role"] == "champion" else "champion"
        hp_by_role = {g["p0role"]: g["hp"][0], p1role: g["hp"][1]}
        g["hp_by_role"] = hp_by_role
        if hp_by_role["champion"] > hp_by_role["runnerup"]:
            g["winner"] = "champion"
        elif hp_by_role["runnerup"] > hp_by_role["champion"]:
            g["winner"] = "runnerup"
        else:
            g["winner"] = "draw"
    else:
        g["hp_by_role"] = None
        g["winner"] = "?"
    return g


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[run] {len(TOKENS)} 局并行, tokens={TOKENS}, trace_ops=on, log_dir={LOG_DIR}", flush=True)
    t0 = time.time()
    threads = [threading.Thread(target=run_one, args=(t,)) for t in TOKENS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    dts = [d for _, d in results.values()]
    print(f"\n[run] 全部结束: wall={wall:.1f}s  单局 min/median/max = "
          f"{min(dts) / 60:.1f}/{sorted(dts)[len(dts) // 2] / 60:.1f}/{max(dts) / 60:.1f} min", flush=True)

    games = [parse(t) for t in TOKENS]
    wins = collections.Counter(g["winner"] for g in games)
    tot = collections.Counter()

    fmt = "%6s  %6s  %-4s  %-9s  %-9s  %2s/%-3s %-26s %2s/%-3s %-26s %5s %3s %3s"
    print("\n=== 逐局 ===", flush=True)
    print(fmt % ("seed", "rounds", "first", "hp(champ,run)", "coins", "act", "ops",
                 "hist", "act", "ops", "hist", "twr1", "ill", "exc"), flush=True)
    for g, tok in zip(games, TOKENS):
        c, r = g["hist"]["champion"], g["hist"]["runnerup"]
        tot.update(c); tot.update(r)
        print(fmt % (tok, g["rounds"], (g["p0role"] or "?")[:4],
                     str(g["hp_by_role"] and (g["hp_by_role"]["champion"], g["hp_by_role"]["runnerup"])),
                     str(g["coins"]),
                     g["act"]["champion"], g["ops"]["champion"], dict(c.most_common()),
                     g["act"]["runnerup"], g["ops"]["runnerup"], dict(r.most_common()),
                     g["towers_first"], g["illegal"], g["exc"]), flush=True)

    print(f"\n=== 汇总: {dict(wins)}  (n={len(games)}) ===", flush=True)
    print(f"总操作类型分布: {dict(tot.most_common())}", flush=True)
    print(f"可疑局(非法/异常/未跑满500回合): "
          f"{[g['token'] for g in games if g['illegal'] or g['exc'] or g['rounds'] < 500]}", flush=True)


if __name__ == "__main__":
    main()
