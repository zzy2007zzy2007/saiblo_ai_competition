"""阶梯实验的并行运行器：并行跑 `code/test_match/match_via_sdk.py`（bridge），汇总。

用途（见 docs/champ_ladder_plan.md）：把"我们 ↔ rule_v4 ↔ 冠军/亚军"接到同一把尺子上。

用法:
  python _tmp_ladder.py --tag=A1_rv4_self --jobs=8 \
      --ai0=code/test_match/rv4_pkg/main.py --ai1=code/test_match/rv4_pkg/main.py \
      7 7r 8 8r            # 7 = AI0 先手；7r = AI1 先手（镜像）

过夜批次（2026-10-29 新增，**默认全关 ⇒ 行为与改动前逐位相同**；见 docs/ladder_resilience_20261029.md）:
  python _tmp_ladder.py --tag=NIGHT --jobs=5 --ai0=... --ai1=... \
      --retry-invalid 2 --final-sweep --watchdog-stall-min 90 7 7r 8 8r ...
  · `--retry-invalid N`：一批跑完后扫该 tag 的日志目录
        （`match_results/ladder_logs/<tag>/<token>.log`），**凡没有 `verdict=engine` 的 token**
        （timeout / aborted / 空壳日志 / 没有 RESULT 行 / 没日志）重跑，最多 N 轮；
        **只重跑失败的**，已 engine 的日志绝不被覆盖；重跑前把失败局日志复制留档到
        `<dir>/_attempts/<token>.attempt<k>.log`。每轮打印：第几轮 / 待跑 token 列表 / 重跑前 verdict 计数。
  · `--final-sweep`：主循环（含 `--retry-invalid`）结束后**再扫一遍**，仍有非 engine 的 token
        ⇒ **再跑最后一次**（不占 `--retry-invalid` 的轮数）。用途：机器夜里休眠导致超时、
        醒来时重试轮数已经用完 ⇒ 这一轮通常在早上执行，把漏的补完。
  · `--watchdog-stall-min M`：运行期间监控**自己启动的对局**是否卡住：该局 progress 文件
        （`training_history/progress/<tag>/az_progress_*<token>*.txt`）或该局日志 mtime 连续
        M 分钟不前进 ⇒ **只按自己的 Popen 句柄那一个 PID 精确杀**整个进程树
        （`taskkill /PID <pid> /T /F`，**绝不按镜像名/命令行通配符批量杀**），
        记 `verdict=timeout_watchdog`，并按 `--retry-invalid` 的规则参与重试。
        顺序严格是：**先停调度 → 再杀子进程 → 用已死的根 PID 复查后代闭包 → 恢复调度**
        （踩过的坑：先杀对局会让驱动立刻抢信号量、把下一波 token 拉起来）。
  · 收尾汇总表（任一开关打开时打印，全关时不打印 ⇒ 默认 stdout 与改动前一致）：
        `token | verdict | rounds | 墙钟(min) | 尝试次数 | 是否有效` + `engine K/N` +
        仍无效的 token 列表（早上人工补跑就照这个列表给命令行）。

设计要点（都是被历史坑出来的，别删）:
  1) 胜负**只读 bridge 自己打的 RESULT 行**，绝不在这里重推一遍
     （记忆 feedback_known_good_control_for_harness 第四条：重打判定把"败"记成了"平"）。
  2) 报 `verdict=`：`engine` = 引擎终局时算好的官方 5 级级联；`hp_tiebreak` = 循环提前停下时
     bridge 的兜底，**不是官方规则**。官方规则**从不判平局** ⇒ 出现 `draw` 即证明判词非官方。
  3) ⚠️ **镜像对称（配对分=1.0、方差 0）是必要不充分的**：若两个 AI 都永远空过，游戏会打满
     512 回合，官方级联第⑤级"全相同判 P0 胜"会让 P0 必胜 ⇒ 签名照样完美通过。
     **所以必须同时报"双方真的出招了"**（每局非空操作数），否则那只是"僵局伪装成对称"。
  4) 子进程设 PYTHONIOENCODING=utf-8，否则日志是 GBK、UTF-8 读全是乱码。
  5) 判有效一律以**日志**为准（`verdict=engine`），进程内记忆的判词只用于"日志里没有 RESULT 行"
     的那一类（被看门狗杀掉的局就是这种）。
"""
from __future__ import annotations

import collections
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOG_ROOT = REPO / "match_results" / "ladder_logs"
# 单局墙钟上限（秒）。默认 7200 = 改动前的值（**不设环境变量 ⇒ 行为逐字不变**）。
# 2026-10-28：冠军 `magica_v3_O0.exe` 是 O0 编译、又是全项目最重的 MCTS，16 局基线必须放宽，
#   否则一局"慢但没死"的对局会被整局掐断、记成 verdict=timeout（= 白跑）。用 `LADDER_TIMEOUT` 覆盖。
TIMEOUT = int(os.environ.get("LADDER_TIMEOUT", "7200"))
# 有效性门槛（2026-10-29）：照抄 `ab_search_modes.py` / `analyze_paired.py` §4 —— 只有
# **engine** 算有效局；timeout / aborted / illegal_strict / hp_tiebreak / 无 RESULT / 无日志 一律无效。
VALID_VERDICT = "engine"
# 逐回合进度文件根（`code/test_match/progress.py` 写的）。
PROGRESS_ROOT = REPO / "training_history" / "progress"

# 2026-10-31（Bug ① 修复）：进度目录解析统一走 progress.progress_scan_dirs()（含 stamp + 旧目录回退）。
#   顶层 import（不再在 run_one 里延迟 import），这样看门狗 `_progress_candidates` 也能用同一个解析。
sys.path.insert(0, str(REPO / "code/test_match"))
import progress as _pg  # noqa: E402

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
    cfg = {"tag": "ladder", "jobs": 8, "ai0": None, "ai1": None, "reanalyze": False,
           "rounds": None,
           # ── 2026-10-29 过夜批次三件套（默认全关）────────────────────────────
           "retry_invalid": 0, "final_sweep": False, "watchdog_stall_min": 0.0}
    toks: list[str] = []
    i, n = 0, len(argv)
    while i < n:
        a = argv[i]
        # 新参数**同时支持** `--retry-invalid 2` 与 `--retry-invalid=2`（老参数只有 `=` 形式，不动）
        if a in ("--retry-invalid", "--watchdog-stall-min") and i + 1 < n:
            try:
                val = float(argv[i + 1])
            except ValueError:
                print(f"[ladder] !! {a} 需要一个数字，收到 {argv[i + 1]!r}", flush=True)
                raise SystemExit(2)
            cfg["retry_invalid" if a == "--retry-invalid" else "watchdog_stall_min"] = (
                int(val) if a == "--retry-invalid" else val)
            i += 2
            continue
        if a.startswith("--tag="):
            cfg["tag"] = a.split("=", 1)[1]
        elif a == "--reanalyze":
            cfg["reanalyze"] = True
        elif a.startswith("--rounds="):
            # 2026-10-28：透传给 `match_via_sdk.py`（冒烟/截断用）。**不给 ⇒ 命令行一个字节都不变**。
            cfg["rounds"] = a.split("=", 1)[1]
        elif a.startswith("--jobs="):
            cfg["jobs"] = int(a.split("=", 1)[1])
        elif a.startswith("--ai0="):
            cfg["ai0"] = a.split("=", 1)[1]
        elif a.startswith("--ai1="):
            cfg["ai1"] = a.split("=", 1)[1]
        elif a.startswith("--retry-invalid="):
            cfg["retry_invalid"] = int(a.split("=", 1)[1])
        elif a.startswith("--watchdog-stall-min="):
            cfg["watchdog_stall_min"] = float(a.split("=", 1)[1])
        elif a in ("--final-sweep", "--final-sweep=1"):
            cfg["final_sweep"] = True
        else:
            toks.append(a)
        i += 1
    return cfg, toks


def _archive_stderr(log_path: Path, token: str, labels: tuple[str, ...] = ()) -> None:
    """把该局两个 AI 的 stderr 复制进 `<tag>/stderr/`。

    动机（2026-10-06）：`match_results/ai{side}_{label}_seed{seed}.stderr.log` **不按 tag 分目录**，
    跨批跑同一个 seed 会**互相覆盖** ⇒ 事后无法复核"那一局到底慢在哪、有没有原生栈"。
    ⚠️ 只复制、不移动/不删除原文件 ⇒ 不改变既有工具的行为。
    """
    try:
        base = re.sub(r"r$", "", token)          # 镜像 token 11r → seed 11
        dst = log_path.parent / "stderr"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        pats = ([f"ai?_*_seed{base}.stderr.log"] if not labels
                else [f"ai?_*_seed{base}.stderr.log"])
        for pat in pats:
            for src in sorted(LOG_ROOT.parent.glob(pat)):
                # 只归档本局实际用的两个 AI 的日志（否则会把同 seed 下别的臂的旧日志也卷进来，
                # 污染 dec_max 汇总 —— 2026-10-06）
                if labels and not any(lbl in src.name for lbl in labels):
                    continue
                shutil.copy2(src, dst / src.name)
                n += 1
        if n == 0:
            print(f"[ladder] ⚠️ stderr 归档：seed {base} 没找到 ai?_*_seed{base}.stderr.log", flush=True)
    except Exception as exc:  # noqa: BLE001
        # ⚠️ 不静默：第一版这里 `pass`，结果把「常量名写错」吞了整整一轮（2026-10-06）
        print(f"[ladder] ⚠️ stderr 归档失败（不影响对局）: {type(exc).__name__}: {exc}", flush=True)


import time as _time_mod
_TAG = "ladder"        # 2026-10-23：当前 tag（run_one 里写 status 用）
_TOKENS: list = []      # 2026-10-23：进度 status 用
_T0 = _time_mod.time()

# ─────────────────── 2026-10-29：过夜批次三件套的进程内状态 ───────────────────
# 全部只在 main() 里初始化/收尾；`run_one`（工作线程）读写下面几个 dict 时**持 main 传进来的 lock**。
_FEATURES_ON = False                       # 三个开关任一打开 ⇒ 打印重试/看门狗/收尾汇总表区块
_MEM_VERDICT: dict[str, str] = {}          # token → 最近一次尝试的判词（日志没 RESULT 行时用）
_ATTEMPTS: dict[str, int] = {}             # token → 本进程内已尝试次数
_WALL_MIN: dict[str, float] = {}           # token → 累计墙钟（分钟，含重跑）
_STATE: dict[str, dict] = {}               # token → {verdict, valid, rounds, attempts, minutes, rc}
_KILLED_AT: dict[str, float] = {}          # token → 看门狗下手的时间戳（判断"这次尝试是被杀的"）
# 调度闸门：默认 open（不阻塞、行为与改动前一致）。看门狗命中时先 clear ⇒ 不再拉起新 token。
_SCHED_RUN = threading.Event()
_SCHED_RUN.set()
# 看门狗的表：token → {popen, log, started, last_seen, last_advance, prog_base, stalled}
_WATCH: dict[str, dict] = {}
_WATCH_LOCK = threading.Lock()
_WD_STOP = threading.Event()


# ───────────────────────── 日志 / 进度文件判定 ─────────────────────────
def _progress_candidates(tag: str, token: str) -> list[Path]:
    """该 token 的逐回合进度文件：`<progress>/<tag>_<stamp>/az_progress_*<token>*.txt` 的**精确**版本
    （含旧无后缀目录 `<tag>` 的回退，向后兼容）。

    ⚠️ 不能直接 `glob("az_progress_*10*.txt")`：token `10` 会把镜像局 `10r` 的文件一起捞进来
    （`10` 是 `10r` 的前缀）⇒ 会拿**别人那局**的 mtime 判断卡死。这里的判据是
    "seed 段以 token 开头，且 token 之后紧跟的东西**不是字母/数字**"。
    """
    out: list[Path] = []
    for d in _pg.progress_scan_dirs(tag):
        if not d.is_dir():
            continue
        for f in d.glob("az_progress_*.txt"):
            stem = f.stem
            if not stem.startswith("az_progress_seed"):
                continue
            rest = stem[len("az_progress_seed"):]
            if not rest.startswith(token):
                continue
            tail = rest[len(token):]
            if tail == "" or not tail[0].isalnum():   # `10` 后面只能是 `_`/结尾，`10r` 的 `r` 不算
                out.append(f)
    return out


def _last_nonempty_line(p: Path) -> str:
    """文件最后一条非空行（只读尾巴 512 字节，走 `rb` 不猜编码）。"""
    try:
        with p.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 512))
            tail = fh.read().decode("utf-8", "replace")
        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def _parse_log(log_path: Path, token: str) -> dict | None:
    """读日志 → `read_game` 的 dict；读不动返回 None（解析失败不能把整批带崩）。"""
    try:
        return read_game(log_path, token)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[ladder] ⚠️ 解析 {log_path.name} 失败: {type(exc).__name__}: {exc}", flush=True)
        return None


def _judge_token(token: str, out_dir: Path) -> tuple[str, bool]:
    """该 token 现在是否有效：(verdict, 是否有效)。

    **日志是唯一权威**：只有 `verdict=engine` 算有效。日志里没有判词（空壳 / 被看门狗杀掉 /
    根本没有 RESULT 行）时才用本进程记下的杀因（`_MEM_VERDICT`）。
    """
    lp = out_dir / f"{token}.log"
    if not lp.is_file():
        return _MEM_VERDICT.get(token, "no_log"), False
    g = _parse_log(lp, token)
    if g is None:
        return "unreadable", False
    v = g.get("verdict")
    if v is None:
        return _MEM_VERDICT.get(token, "no_result"), False
    return v, (v == VALID_VERDICT)


# ───────────────────────── 进程表 / 后代闭包（只读）─────────────────────────
def _process_table() -> list[tuple[int, int]]:
    """返回 `[(pid, ppid)]`（Toolhelp32 快照，**只读**，毫秒级，不碰任何进程）。

    为什么不用 `wmic`/PowerShell：两者一次要 100–500ms，且 `wmic` 在新 Windows 上已被移除。
    """
    if os.name != "nt":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_char * 260),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k32.Process32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32)]
        k32.Process32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        snap = k32.CreateToolhelp32Snapshot(0x00000002, 0)   # TH32CS_SNAPPROCESS
        if not snap or int(snap) == int(ctypes.c_void_p(-1).value):
            return []
        out: list[tuple[int, int]] = []
        try:
            entry = PROCESSENTRY32()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
            ok = k32.Process32First(snap, ctypes.byref(entry))
            while ok:
                out.append((int(entry.th32ProcessID), int(entry.th32ParentProcessID)))
                ok = k32.Process32Next(snap, ctypes.byref(entry))
        finally:
            k32.CloseHandle(snap)
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"[watchdog] ⚠️ 进程表快照失败（不影响对局）: {type(exc).__name__}: {exc}", flush=True)
        return []


def _descendants(root_pid: int) -> list[int]:
    """`root_pid` **及所有存活后代**（按 `ParentProcessId` 链递归；只读）。

    ⚠️ Windows **不会重新挂父**：根进程死后，孤儿子进程的 PPID 仍然指向那个**已死的 PID**
    ⇒ 「杀完用已死的根 PID 复查后代闭包」是可行的，这也是本函数能在杀后继续用的依据。
    """
    table = _process_table()
    if not table:
        return []
    kids: dict[int, list[int]] = collections.defaultdict(list)
    for pid, ppid in table:
        kids[ppid].append(pid)
    seen = {int(root_pid)}
    stack = [int(root_pid)]
    while stack:
        cur = stack.pop()
        for c in kids.get(cur, []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    alive = {pid for pid, _ppid in table}
    return sorted(p for p in seen if p in alive)


# ───────────────────────── 看门狗 ─────────────────────────
def _watch_register(token: str, proc: subprocess.Popen, log_path: Path) -> None:
    """把**本进程刚启动的**这一局登记进看门狗表（只登记，不干预）。"""
    base: dict[str, float] = {}
    try:
        for p in _progress_candidates(_TAG, token):
            try:
                base[str(p)] = p.stat().st_mtime
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        pass
    now = time.time()
    with _WATCH_LOCK:
        _WATCH[token] = {"token": token, "popen": proc, "log": log_path, "started": now,
                         "last_seen": 0.0, "last_advance": now, "prog_base": base,
                         "stalled": False, "stall_min": 0.0}


def _watch_unregister(token: str) -> None:
    with _WATCH_LOCK:
        _WATCH.pop(token, None)


def _watch_snapshot(rec: dict, tag: str) -> tuple[float, bool]:
    """(该局最新的 mtime, 是否已终局)。

    活跃度 = max(日志 mtime, 本次尝试新写过的 progress 文件 mtime) ⇒ 任一在前进就不算卡。
    已终局的局**不再更新** progress 文件，mtime 会一直变老 ⇒ 必须靠最后一行的 `terminal`
    排除（否则会对已完成的局狂杀）——判据沿用 `code/test_match/_ladder_watchdog.ps1`。
    """
    cur = 0.0
    terminal = False
    try:
        cur = max(cur, rec["log"].stat().st_mtime)
    except OSError:
        pass
    for p in _progress_candidates(tag, rec["token"]):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m <= rec["prog_base"].get(str(p), 0.0):
            continue        # 本次尝试还没写过它（上一轮/上一批的旧文件，不算活跃也不算终局）
        cur = max(cur, m)
        if _last_nonempty_line(p).startswith("terminal"):
            terminal = True
    return cur, terminal


def _kill_game_tree(token: str, rec: dict, why: str) -> None:
    """精确杀掉**这一局**的进程树：只用 `Popen` 句柄里的那一个 PID（`taskkill /PID <pid> /T /F`）。

    ⚠️ **绝不按镜像名/命令行通配符批量杀**（用户明确要求）。`taskkill /IM python.exe` 这类写法
       会把本机**别人正在跑的对局**一起杀掉 —— 2026-10 刚因为这种写法误杀过别人的进程。
       Popen 句柄持有的 PID 在进程活着期间不会被回收 ⇒ 按 PID 杀是精确且安全的。
    ⚠️ 顺序：调用方已经**先停调度**（`_SCHED_RUN.clear()`），本函数只负责"杀 + 复查闭包"。
    """
    p = rec.get("popen")
    if p is None or p.poll() is not None:      # 已经自己退了 ⇒ 什么都不用做
        return
    pid = int(p.pid)
    before = _descendants(pid)                 # 杀前闭包（含根）：留档 + 补杀白名单
    # ⚠️ 必须**在动手之前**打时间戳：run_one 线程会在进程死掉的那一刻醒来并据此判定
    #    "这次尝试是被看门狗杀的"（否则有竞态，会被记成 no_result）。
    with _STATE_LOCK_FOR_KILL:
        _KILLED_AT[token] = time.time()
    print(f"[watchdog] {token}: {why} ⇒ 精确 kill PID {pid} 及其子进程"
          f"（taskkill /PID {pid} /T /F，不碰其它进程）", flush=True)
    print(f"[watchdog] {token}: 杀前闭包 {len(before)} 个 PID: {before}", flush=True)
    try:
        out = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                             capture_output=True, timeout=120)
        msg = ((out.stdout or b"") + (out.stderr or b"")).decode("gbk", "replace").strip()
        for ln in msg.splitlines():
            print(f"[watchdog] taskkill rc={out.returncode}: {ln}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[watchdog] ⚠️ taskkill 失败，退化为 Popen.kill(): {type(exc).__name__}: {exc}",
              flush=True)
        try:
            p.kill()
        except Exception:                    # noqa: BLE001
            pass
    # 等根进程真的死掉（不抢 run_one 线程的 p.wait，只用 poll）
    for _ in range(120):
        if p.poll() is not None:
            break
        time.sleep(0.5)
    else:
        print(f"[watchdog] ⚠️ PID {pid} 60s 仍未退出，补一刀", flush=True)
        try:
            p.kill()
        except Exception:                    # noqa: BLE001
            pass
        for _ in range(60):
            if p.poll() is not None:
                break
            time.sleep(0.5)
    # ── 用**已死的根 PID** 复查后代闭包 ──
    left = [x for x in _descendants(pid) if x != pid]
    if not left:
        print(f"[watchdog] {token}: 用已死根 PID {pid} 复查后代闭包 ⇒ 0 个存活（干净）",
              flush=True)
        return
    print(f"[watchdog] {token}: ⚠️ 根 PID {pid} 已死，但闭包里还有 {len(left)} 个存活: {left}",
          flush=True)
    for x in left:
        if x not in before:
            # 不在杀前闭包里 ⇒ 不能证明是我们的后代，**不碰**（宁可漏杀）
            print(f"[watchdog] {token}: (跳过 PID {x}：不在杀前闭包里，不敢碰)", flush=True)
            continue
        try:
            o2 = subprocess.run(["taskkill", "/PID", str(x), "/T", "/F"],
                                capture_output=True, timeout=60)
            print(f"[watchdog] {token}: 补杀 PID {x} rc={o2.returncode}", flush=True)
        except Exception as exc:             # noqa: BLE001
            print(f"[watchdog] {token}: ⚠️ 补杀 PID {x} 失败: {type(exc).__name__}: {exc}",
                  flush=True)
    left2 = [x for x in _descendants(pid) if x != pid]
    print(f"[watchdog] {token}: 用已死根 PID {pid} 复查后代闭包 ⇒ 还剩 {len(left2)} 个"
          f"{'：' + str(left2) if left2 else '（干净）'}", flush=True)


def _watchdog_loop(tag: str, stall_min: float) -> None:
    """看门狗主循环：只盯 `_WATCH` 里**本进程启动的**对局，命中就精确清掉那一局。"""
    stall_s = stall_min * 60.0
    poll = max(2.0, min(30.0, stall_s / 6.0))
    print(f"[watchdog] 启动：tag={tag}  阈值 {stall_min:g} min（progress/日志 mtime 连续这么久"
          f"不前进 ⇒ 判卡死）  每 {poll:.0f}s 检查一次", flush=True)
    print("[watchdog] 只按本进程 Popen 的**那一个 PID** 精确杀（taskkill /PID <pid> /T /F）；"
          "先停调度再杀，杀完用已死根 PID 复查后代闭包", flush=True)
    while not _WD_STOP.wait(poll):
        now = time.time()
        with _WATCH_LOCK:
            items = list(_WATCH.items())
        fired: list[tuple[str, dict, float]] = []
        for token, rec in items:
            try:
                cur, terminal = _watch_snapshot(rec, tag)
            except Exception as exc:         # noqa: BLE001  看门狗自己绝不能把批次带崩
                print(f"[watchdog] ⚠️ {token} 取活跃度失败: {type(exc).__name__}: {exc}",
                      flush=True)
                continue
            if cur > rec["last_seen"] + 1e-6:
                rec["last_seen"] = cur
                rec["last_advance"] = now
                continue
            idle = now - rec["last_advance"]
            if terminal:
                continue                     # 已终局（最后一轮不更新文件）：不判卡死
            if idle >= stall_s and not rec["stalled"]:
                fired.append((token, rec, idle))
        if not fired:
            continue
        _SCHED_RUN.clear()                   # ① 先停调度（不再拉起新 token）
        print(f"[watchdog] 命中 {len(fired)} 局 ⇒ 先停调度（_SCHED_RUN=closed），再逐局精杀",
              flush=True)
        try:
            for token, rec, idle in fired:
                rec["stalled"] = True
                rec["stall_min"] = idle / 60.0
                _kill_game_tree(token, rec, f"progress/日志已 {idle / 60:.1f} min 没有更新"
                                            f"（阈值 {stall_min:g} min）")
        finally:
            _SCHED_RUN.set()                 # ③ 复查完闭包才恢复调度
            print("[watchdog] 恢复调度（_SCHED_RUN=open）", flush=True)
    print("[watchdog] 停止", flush=True)


# `_KILLED_AT` 的锁单独一把（看门狗线程写、run_one 线程读；不与 main 的打印锁混用）
_STATE_LOCK_FOR_KILL = threading.Lock()


def run_one(token: str, log_path: Path, fwd: list[str], sem: threading.Semaphore,
            results: dict, lock: threading.Lock, labels: tuple[str, ...] = ()) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    # 2026-10-23（用户要求）：把 tag 传下去，match_via_sdk 用它写逐回合进度文件
    #   文件：training_history/progress/<tag>_seed<seed>.txt
    env["DSH_PROGRESS_TAG"] = str(_TAG)
    cmd = [sys.executable, "-u", "code/test_match/match_via_sdk.py", "--trace-ops", *fwd, token]
    with sem:
        # ⚠️ 调度闸门必须放在**信号量内**：否则被看门狗杀掉的那局一释放 sem，等在 sem 上的线程
        #    就会立刻把下一波 token 拉起来（我们踩过这个坑）。默认 open ⇒ 不阻塞、行为不变。
        _SCHED_RUN.wait()
        with lock:
            _ATTEMPTS[token] = _ATTEMPTS.get(token, 0) + 1
            attempt = _ATTEMPTS[token]
        t0 = time.time()
        rc = -1
        wd = {"hard": False}
        try:
            with open(log_path, "wb") as f:
                # 2026-10-29：从 `subprocess.run` 改成 `Popen` —— 看门狗必须拿到**自己的 PID**。
                #   超时分支仍按旧语义：只杀直接子进程、rc 记 -2（默认路径读数与改动前一致）。
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env,
                                        cwd=str(REPO))
                _watch_register(token, proc, log_path)
                try:
                    rc = proc.wait(timeout=TIMEOUT)
                except subprocess.TimeoutExpired:
                    wd["hard"] = True
                    try:
                        proc.kill()
                    except Exception:        # noqa: BLE001
                        pass
                    try:
                        proc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        pass
                    rc = -2
                finally:
                    _watch_unregister(token)
            _archive_stderr(log_path, token, labels)
        except Exception as exc:  # noqa: BLE001
            _watch_unregister(token)
            _archive_stderr(log_path, token, labels)
            rc = -3
            print(f"[{token}] EXC {type(exc).__name__}: {exc}", flush=True)
        dt = time.time() - t0
        g = _parse_log(log_path, token) or {}
        verdict = g.get("verdict")
        rounds = g.get("rounds")
        if verdict != VALID_VERDICT:
            with _STATE_LOCK_FOR_KILL:
                killed_at = _KILLED_AT.get(token, 0.0)
            if killed_at >= t0:
                verdict = "timeout_watchdog"
            elif wd["hard"]:
                verdict = "timeout_hard"
            elif verdict is None:
                verdict = "no_result"
        valid = (verdict == VALID_VERDICT)
        with lock:
            results[token] = (rc, dt)
            _MEM_VERDICT[token] = str(verdict)
            _WALL_MIN[token] = _WALL_MIN.get(token, 0.0) + dt / 60.0
            _STATE[token] = {"verdict": verdict, "valid": valid, "rounds": rounds,
                             "attempts": attempt, "minutes": _WALL_MIN[token], "rc": rc}
            print(f"[{token}] rc={rc}  {dt / 60:.1f} min", flush=True)
            if _FEATURES_ON:
                print(f"[{token}]   第 {attempt} 次尝试：verdict={verdict}"
                      f"{'' if valid else '  ⇒ 无效，等重试'}"
                      f"  墙钟累计 {_WALL_MIN[token]:.1f} min", flush=True)
            # 2026-10-23（用户要求）：整批进度文件 training_history/progress/<tag>/status.txt
            try:
                _tag = str(_TAG)
                _tok_list = _TOKENS[0] if _TOKENS else []
                _total = len(_tok_list)
                _d = REPO / "match_results/ladder_logs" / _tag
                _w = _l = _inv = 0
                _recs = []   # (完成时间 mtime, 文本)
                for _tok in _tok_list:
                    _lg = _d / f"{_tok}.log"
                    if not _lg.exists():
                        continue
                    _g = read_game(_lg, _tok)
                    _ws = _g.get("winner_side")
                    if _ws is None:
                        _inv += 1
                        continue
                    # read_game 的 winner_side 是 'p0'/'p1'（新格式）或按 HP 兜底推的
                    _ws = str(_ws).lower()
                    if _ws not in ("p0", "p1"):
                        _inv += 1
                        continue
                    _ours_p0 = "az_bridge_ai" in str(_g.get("p0_label") or "")
                    _won = (_ws == "p0") == _ours_p0
                    if _won:
                        _w += 1
                    else:
                        _l += 1
                    # 2026-10-31（Bug ② 修复）：按【完成时间】（日志 mtime）降序取前 6，
                    #   而不是按 token 顺序取前 6（原来把"最先完成"当成了"最近完成"）。
                    _recs.append((_lg.stat().st_mtime,
                                  f"{_tok} {'胜' if _won else '负'} "
                                  f"hp={_g.get('hp')} ({_g.get('rounds')} 回合)"))
                _recent = [t for _, t in sorted(_recs, reverse=True)[:6]]
                _done = _w + _l
                _inflight = []
                for _d in _pg.progress_scan_dirs(_tag):
                    for _f in _d.glob("az_progress_seed*.txt"):
                        try:
                            _last = _f.read_text(encoding="utf-8").strip().splitlines()[-1]
                            if _last.startswith("round="):
                                _inflight.append(int(_last.split("=")[1].split()[0]))
                        except Exception:
                            pass
                _pg.write_status(_tag, total=_total, done=_done, wins=_w, losses=_l,
                                 invalid=_inv, started=_T0, inflight_rounds=_inflight,
                                 recent=_recent)
            except Exception:
                pass


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


def _archive_attempt(token: str, out_dir: Path) -> None:
    """重跑前把上一轮的失败日志**复制**留证据（只复制，不删/不改原日志）。"""
    k = _ATTEMPTS.get(token, 0)
    src = out_dir / f"{token}.log"
    if k <= 0 or not src.is_file():
        return
    try:
        dst_dir = out_dir / "_attempts"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{token}.attempt{k}.log"
        shutil.copy2(src, dst)
        print(f"[retry] {token} 的失败日志已留档: {dst.relative_to(REPO).as_posix()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[retry] ⚠️ {token} 留档失败（不影响重跑）: {type(exc).__name__}: {exc}", flush=True)


def _disp_w(text: str) -> int:
    """字符串的**显示宽度**（CJK 全角算 2 列）。表头是中文，用 `% -Ns` 对不齐，所以要自己算。"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    n = width - _disp_w(text)
    if n <= 0:
        return text
    return (" " * n + text) if right else (text + " " * n)


def _print_final_table(tokens: list[str], out_dir: Path) -> None:
    """收尾汇总表：每个 token 一行 + `engine K/N` + 仍无效的 token 列表。

    ⚠️ 只在三个开关任一打开时调用（全关 ⇒ 默认 stdout 与改动前一致）。
    """
    rows: list[tuple[str, str, str, float, int, bool]] = []
    for token in tokens:
        verdict, valid = _judge_token(token, out_dir)
        st = _STATE.get(token) or {}
        rounds = st.get("rounds")
        if rounds is None:
            try:
                rounds = read_game(out_dir / f"{token}.log", token).get("rounds")
            except Exception:  # noqa: BLE001
                rounds = None
        rows.append((token, str(verdict or "-"),
                     "-" if rounds is None else str(rounds),
                     _WALL_MIN.get(token, 0.0), _ATTEMPTS.get(token, 0), valid))
    n_ok = sum(1 for r in rows if r[5])
    widths = (7, 16, 7, 10, 9, 10)
    head = ("token", "verdict", "rounds", "墙钟(min)", "尝试次数", "是否有效")
    print("\n=== 收尾汇总（每个 token 一行；判有效 = 日志里 verdict=engine）===", flush=True)
    print(" | ".join(_pad(h, w) for h, w in zip(head, widths)), flush=True)
    print("-" * (_disp_w(" | ".join(_pad(h, w) for h, w in zip(head, widths)))), flush=True)
    for token, verdict, rounds, wall, att, valid in rows:
        print(" | ".join((
            _pad(token, widths[0]),
            _pad(verdict, widths[1]),
            _pad(rounds, widths[2], right=True),
            _pad(f"{wall:.1f}", widths[3], right=True),
            _pad(str(att), widths[4], right=True),
            _pad("✓ 有效" if valid else "✗ 无效", widths[5]),
        )), flush=True)
    print(f"\nengine {n_ok}/{len(rows)}", flush=True)
    bad = [f"{t}({v})" for t, v, _r, _w, _a, ok in rows if not ok]
    if bad:
        print("仍无效的 token（早上人工补跑就照这个列表给命令行）: " + ", ".join(bad), flush=True)
    else:
        print("仍无效的 token: 无（全部 engine）", flush=True)


def main() -> int:
    cfg, tokens = parse_cli(sys.argv[1:])
    if not tokens:
        print("用法: python _tmp_ladder.py --tag=NAME [--jobs=8] [--ai0=..] [--ai1=..] 7 7r 8 8r ...")
        print("      过夜批次可加: --retry-invalid N  --final-sweep  --watchdog-stall-min M")
        return 2
    out_dir = LOG_ROOT / cfg["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fwd: list[str] = []
    if cfg["ai0"]:
        fwd.append(f"--ai0={cfg['ai0']}")
    if cfg["ai1"]:
        fwd.append(f"--ai1={cfg['ai1']}")
    if cfg.get("rounds"):
        fwd.append(f"--rounds={cfg['rounds']}")

    _labels = tuple(Path(x.split("=", 1)[1]).stem for x in fwd if x.startswith(("--ai0=", "--ai1=")))
    print(f"[ladder] tag={cfg['tag']}  jobs={cfg['jobs']}", flush=True)
    print(f"[ladder] ai0={cfg['ai0']}", flush=True)
    print(f"[ladder] ai1={cfg['ai1']}", flush=True)
    print(f"[ladder] tokens={tokens}  logs={out_dir}", flush=True)
    # 2026-10-31（Bug ① 修复）：本批共享一个目录时间戳。外层 run_logged 设了就沿用；
    #   没设（直接跑）就由本驱动兜底生成一次，随 env 传进每个 match_via_sdk 子进程 ⇒ 整批同目录。
    os.environ.setdefault("DSH_PROGRESS_DIR_STAMP", time.strftime("%Y%m%d_%H%M%S"))
    global _T0, _TAG, _FEATURES_ON
    _TAG = str(cfg.get("tag", "ladder"))
    _TOKENS.clear(); _TOKENS.append(list(tokens))
    _T0 = time.time()
    # 三个新开关全关 ⇒ 不打印任何新区块（默认 stdout 与改动前一致）
    _FEATURES_ON = bool(cfg["retry_invalid"] > 0 or cfg["final_sweep"]
                        or cfg["watchdog_stall_min"] > 0)
    if _FEATURES_ON:
        print(f"[ladder] 过夜批次开关: retry_invalid={cfg['retry_invalid']}  "
              f"final_sweep={cfg['final_sweep']}  watchdog_stall_min="
              f"{cfg['watchdog_stall_min']:g}", flush=True)

    results: dict[str, tuple[int, float]] = {}
    if cfg["reanalyze"]:
        # 只重新解析已存在的日志（改了汇总逻辑后不必重跑对局）
        print("[ladder] --reanalyze：跳过对局，直接解析已有日志", flush=True)
    else:
        lock = threading.Lock()
        sem = threading.Semaphore(cfg["jobs"])

        def run_batch(task_tokens: list[str], is_retry: bool = False) -> None:
            """跑一批 token。`is_retry=True` 用于重试轮/final-sweep：

            **重跑前再判一次**（万一这一瞬间日志已经变成 `verdict=engine`，比如人工补了）
            ⇒ 有效就跳过并告警。成功局的结果文件**绝不**被重跑覆盖。
            """
            if is_retry:
                keep: list[str] = []
                for t in task_tokens:
                    v, ok = _judge_token(t, out_dir)
                    if ok:
                        print(f"[retry] {t} 已有 verdict={v}（有效）⇒ 跳过，不覆盖", flush=True)
                        with lock:
                            _STATE[t] = {**(_STATE.get(t) or {}), "verdict": v, "valid": True}
                        continue
                    keep.append(t)
                task_tokens = keep
            if not task_tokens:
                print("[retry] 待跑清单为空（都已经是 engine）⇒ 不启动任何对局", flush=True)
                return
            threads = [threading.Thread(target=run_one,
                                        args=(t, out_dir / f"{t}.log", fwd, sem, results, lock,
                                              _labels))
                       for t in task_tokens]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        def invalid_tokens() -> list[tuple[str, str]]:
            """当前仍然无效的 token（+ 判词）。**日志是唯一权威**。"""
            out: list[tuple[str, str]] = []
            for t in tokens:
                v, ok = _judge_token(t, out_dir)
                if not ok:
                    out.append((t, v))
            return out

        wd_thread = None
        if cfg["watchdog_stall_min"] > 0:
            _WD_STOP.clear()
            wd_thread = threading.Thread(target=_watchdog_loop,
                                         args=(_TAG, float(cfg["watchdog_stall_min"])),
                                         name="ladder-watchdog", daemon=True)
            wd_thread.start()
        t0 = time.time()
        try:
            # ───────────────────── 第一批 ─────────────────────
            run_batch(tokens)

            # ───────────────── ① --retry-invalid N 轮 ─────────────────
            if cfg["retry_invalid"] > 0:
                for rnd in range(1, cfg["retry_invalid"] + 1):
                    bad = invalid_tokens()
                    if not bad:
                        print(f"\n[retry] 第 {rnd}/{cfg['retry_invalid']} 轮：没有无效局了，停止重试",
                              flush=True)
                        break
                    cnt = collections.Counter(v for _t, v in bad)
                    print(f"\n=== 重试第 {rnd}/{cfg['retry_invalid']} 轮："
                          f"{len(bad)} 局没有 verdict=engine ===", flush=True)
                    print(f"    重跑前 verdict 计数分布: {dict(cnt)}", flush=True)
                    print(f"    待跑 token: {', '.join(t for t, _v in bad)}", flush=True)
                    for t, _v in bad:
                        _archive_attempt(t, out_dir)
                    run_batch([t for t, _v in bad], is_retry=True)
                else:
                    left = invalid_tokens()
                    if left:
                        print(f"\n[retry] ⚠️ 已用满 {cfg['retry_invalid']} 轮，仍有 {len(left)} 局无效: "
                              f"{', '.join(f'{t}({v})' for t, v in left)}", flush=True)

            # ───────────────── ② --final-sweep（不占重试轮数）─────────────────
            if cfg["final_sweep"]:
                bad = invalid_tokens()
                print(f"\n=== final-sweep（主循环 + 重试 {cfg['retry_invalid']} 轮之后）："
                      f"{len(bad)} 局没有 verdict=engine ===", flush=True)
                if bad:
                    cnt = collections.Counter(v for _t, v in bad)
                    print(f"    当前 verdict 计数分布: {dict(cnt)}", flush=True)
                    print(f"    待跑 token: {', '.join(t for t, _v in bad)}", flush=True)
                    print("    （这一轮通常在早上执行：机器夜里休眠导致超时、醒来时重试轮数已用完 "
                          "⇒ 把漏的补完；不占 --retry-invalid 的轮数）", flush=True)
                    for t, _v in bad:
                        _archive_attempt(t, out_dir)
                    run_batch([t for t, _v in bad], is_retry=True)
                else:
                    print("    没有无效局 ⇒ final-sweep 不跑任何对局", flush=True)
        finally:
            if wd_thread is not None:
                _WD_STOP.set()
                wd_thread.join(timeout=120)
        wall = time.time() - t0
        dts = [d for _, d in results.values()]
        if dts:
            print(f"\n[ladder] 全部结束 wall={wall:.1f}s  单局 min/med/max = "
                  f"{min(dts)/60:.1f}/{sorted(dts)[len(dts)//2]/60:.1f}/{max(dts)/60:.1f} min",
                  flush=True)

    # ───────────────── ④ 收尾汇总表（结算前；仅开关打开时）─────────────────
    if _FEATURES_ON:
        _print_final_table(tokens, out_dir)

    games = []
    for t in tokens:
        if not (out_dir / f"{t}.log").exists():
            print(f"[ladder] ⚠️ 缺日志 {t}.log（该局未跑/被杀），跳过汇总", flush=True)
            continue
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
