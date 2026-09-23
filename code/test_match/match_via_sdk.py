#!/usr/bin/env python3
"""冠军(magica) vs 亚军(cpp_lure_v4) 本地对战
用Python SDK(Ant-Game/SDK/backend/engine.py)做裁判，
两个C++ AI作为外部进程，通过stdin/stdout协议交互。
"""
from __future__ import annotations

import copy
import json
import os
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "Ant-Game"))
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "code" / "cpp_engine"))
os.add_dll_directory(str(REPO_ROOT / "code" / "cpp_engine"))
os.add_dll_directory(r"C:/mingw64/bin")

from my_ai.az_intent.game_state_facade import GameStateFacade as GameState  # noqa: E402
from SDK.utils.constants import (  # noqa: E402
    MAX_ROUND,
    OperationType,
    PLAYER_BASES,
    SuperWeaponType,
    TowerType,
)
from SDK.backend.model import Operation  # noqa: E402

CHAMPION_EXE = (REPO_ROOT / "其他版本ai" / "ant-war2-magica-v3" / "magica_v3_O0.exe")
RUNNERUP_EXE = (REPO_ROOT / "其他版本ai" / "saiblo-30th-AI" / "Game1"
                / "antgame_ai_cpp" / "cpp_lure_v4" / "build" / "ai_cpp_lure_v4.exe")

# 命令行 --ai0=/--ai1= 覆盖（默认 = 冠军/亚军）。用来跑任意两个 AI 的对局，
# 例如阶梯计划的 A1（rule_v4 打自己）与 A2（rule_v4 打冠军/亚军）。
AI0_EXE: Path | None = None
AI1_EXE: Path | None = None

# 通用接入口的其余部分：每个 AI 还可以带自己的 argv / env / cwd。
# 目的：**任意设计、任意语言**的协议 AI 都能当参数接进来，不必迁就我们的技术路径。
# 协议本体见 docs/protocol_ai_spec.md；最小示例见 code/test_match/example_ai.py。
AI_SPEC: dict[int, dict] = {
    0: {"args": (), "env": {}, "cwd": None},
    1: {"args": (), "env": {}, "cwd": None},
}


def _parse_env(spec: str) -> dict[str, str]:
    """`--ai0-env=K=V;K2=V2` → dict。"""
    out: dict[str, str] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"环境变量要用 K=V 形式，收到 {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = v
    return out


def _label_of(exe: Path) -> tuple[str, bool]:
    """AI 的角色标签 + 是否是 magica 那份需要特殊 env 的二进制。

    默认那两个二进制保持原有标签 champion/runnerup（文件名与判词都不变）；其他 AI 用
    文件名的 stem 当标签（清理掉 Windows 文件名非法字符）。
    """
    is_magica = "magica" in exe.name.lower()
    if exe == CHAMPION_EXE:
        return "champion", is_magica
    if exe == RUNNERUP_EXE:
        return "runnerup", is_magica
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in exe.stem)
    return safe, is_magica


TIMEOUT_SECONDS = 300.0
# 2026-09-22：120 → 300。原值在机器被别的进程抢 CPU 时会把整局掐断
# （实测 DET_seed11_run1：第 84 回合亚军那步 >120s，报
#  "timed out reading ai_cpp_lure_v4 (need 50B, have 12B)"，整局作废）。
# 它只是"卡死检测"的门限，不影响正常对局的结果。
TRACE_OPS = "--trace-ops" in sys.argv

DUMP_ROUND = None
DUMP_PATH = ""
DUMPED = False
for _arg in sys.argv:
    if _arg.startswith("--dump-round="):
        try:
            DUMP_ROUND = int(_arg.split("=", 1)[1].split(":", 1)[0])
            DUMP_PATH = _arg.split(":", 1)[1] if ":" in _arg else "state_dump.json"
        except (ValueError, IndexError):
            pass


def normalize_lines(text: str) -> str:
    """归一化行分隔符：\r\n 或单独 \r -> \n。
    runnerup(C++)在Windows文本模式下输出会带\r，而magica只接受\n。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


class ProcIO:
    """线程读取子进程stdout（Windows 兼容，os.set_blocking 在管道上不可用）。

    strip_cr=True 时把读取到的字节流中所有 \r 剥掉（\r\n -> \n）。
    用于 runnerup：它的 C++ stdout 是 Windows 文本模式，写入时 \n 被转成 \r\n，
    但长度前缀仍按 \n 计数，导致流错位；剥掉 \r 后前缀与实际字节数对齐。
    """

    def __init__(self, proc: subprocess.Popen[bytes], label: str, strip_cr: bool = False):
        import threading
        self._proc = proc
        self._label = label
        # ⚠️ 已废弃：读取层**绝不**删字节（会破坏二进制的长度前缀）。
        # \r 规范化已移到 run_match 里对 **payload 解码后的文本**做。保留参数只为不改调用点。
        self._strip_cr = strip_cr
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._done = threading.Event()

        def _reader() -> None:
            try:
                while True:
                    # os.read returns available bytes (BufferedReader.read blocks
                    # until the full n on Windows, which stalls on small outputs)
                    chunk = os.read(self._proc.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    # ⚠️ 绝不在这一层删字节：这条流里带着**二进制的 4 字节长度前缀**，
                    # 删掉任何一个 0x0D 都会让整条流错位。
                    # 实证（2026-09-23）：我方回包 "2\n13 10\n13 7\n" = **13 字节 = 0x0D**，
                    # 前缀 `00 00 00 0D` 里的 0x0D 被删 ⇒ 桥读到 `00 00 00 32` = 50（'2'=0x32），
                    # 于是它去等 50 字节、只等到 12 字节 ⇒ 这局被超时掐断。
                    # 08-10 那个"seed 11 第 84 回合 need 50B have 12B"也是同一个 bug。
                    # ⇒ \r 的规范化只能作用于 **payload 解码后的文本**（见 run_match）。
                    with self._lock:
                        self._buf.extend(chunk)
            finally:
                self._done.set()

        threading.Thread(target=_reader, daemon=True).start()

    def read(self, size: int, timeout: float = TIMEOUT_SECONDS) -> bytes:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if len(self._buf) >= size:
                    out = bytes(self._buf[:size])
                    del self._buf[:size]
                    return out
                have = len(self._buf)
                head = bytes(self._buf[:64])
            if time.monotonic() > deadline:
                # dump 原始字节：用来分辨"AI 写包有 bug"还是"我们的分帧错位"
                # （实测 seed11 第 84 回合：声明 50B、只到 12B）
                raise TimeoutError(
                    f"timed out reading {self._label} (need {size}B, have {have}B) "
                    f"buf_head_hex={head.hex()} buf_head_repr={head!r}")
            if self._done.is_set():
                with self._lock:
                    if len(self._buf) == 0:
                        raise EOFError(f"{self._label} exited (code {self._proc.poll()})")
            time.sleep(0.01)

    def read_packet(self) -> bytes:
        """读4字节长度前缀+payload"""
        size = struct.unpack(">I", self.read(4))[0]
        return self.read(size)


def write_all(stream, payload: bytes) -> None:
    stream.write(payload)
    stream.flush()


def send_line(io: ProcIO, text: str) -> None:
    write_all(io._proc.stdin, text.encode("utf-8"))


def send_packet(io: ProcIO, text: str) -> None:
    data = text.encode("utf-8")
    write_all(io._proc.stdin, struct.pack(">I", len(data)) + data)


def ops_to_text(ops: list[Operation]) -> str:
    lines = [str(len(ops))]
    lines.extend(" ".join(str(t) for t in op.to_protocol_tokens()) for op in ops)
    return "\n".join(lines) + "\n"


def state_to_text(state: GameState) -> str:
    """生成与官方协议一致的回合局面文本"""
    lines: list[str] = []
    lines.append(str(state.round_index))

    towers = state.towers
    lines.append(str(len(towers)))
    for t in towers:
        lines.append(f"{t.tower_id} {t.player} {t.x} {t.y} {int(t.tower_type)} {int(t.cooldown_clock)} {t.hp}")

    ants = state.ants
    lines.append(str(len(ants)))
    for a in ants:
        lines.append(
            f"{a.ant_id} {a.player} {a.x} {a.y} {a.hp} {a.level} {a.age} "
            f"{int(a.status)} {int(a.behavior)} {int(a.kind)}"
        )

    lines.append(f"{state.coins[0]} {state.coins[1]}")

    b0, b1 = state.bases[0], state.bases[1]
    lines.append(
        f"{b0.hp} {b1.hp} {b0.generation_level} {b1.generation_level} "
        f"{b0.ant_level} {b1.ant_level}"
    )

    # 武器冷却：每行一个玩家，按武器ID 1-4顺序
    lines.append("2")
    for p in (0, 1):
        cds = [
            int(state.weapon_cooldowns[p, SuperWeaponType.LIGHTNING_STORM]),
            int(state.weapon_cooldowns[p, SuperWeaponType.EMP_BLASTER]),
            int(state.weapon_cooldowns[p, SuperWeaponType.DEFLECTOR]),
            int(state.weapon_cooldowns[p, SuperWeaponType.EMERGENCY_EVASION]),
        ]
        lines.append(" ".join(str(c) for c in cds))

    effects = state.active_effects
    lines.append(str(len(effects)))
    # 官方 C++ 引擎(Output::add_active_effects)按 player0->player1、再按武器类型
    # 顺序输出效果；Python 列表是插入顺序，必须重排后输出，否则 AI 内部模拟分歧。
    for e in sorted(effects, key=lambda e: (e.player, int(e.weapon_type))):
        lines.append(f"{int(e.weapon_type)} {e.player} {e.x} {e.y} {e.remaining_turns}")

    return "\n".join(lines) + "\n"


def start_ai(exe: Path, player: int, seed: int, stderr_path: Path, extra_env: dict | None = None,
             strip_cr: bool = False, argv: tuple[str, ...] = (), cwd: Path | None = None,
             ) -> tuple[subprocess.Popen, ProcIO]:
    """启动一个 AI。

    **通用接入口**：`argv`（额外命令行参数）、`cwd`（工作目录）、`extra_env`（环境变量）
    都是参数 ⇒ 任意设计、任意语言写的协议 AI 都能当参数接进来，不必迁就我们的技术路径。
    详细协议见 `docs/protocol_ai_spec.md`。
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if exe.suffix == ".py":
        # 协议 AI 的 Python 壳：用 sys.executable 起，并把 Ant-Game 根放进 PYTHONPATH。
        # 抄自 code/test_match/run_cpp_ai_match.py:65-70（code/test_match/rv4_pkg/main.py
        # 就是靠这个跑的）——本文件原来只支持可执行文件，所以 rule_v4 这类 Python AI 进不来。
        env["PYTHONPATH"] = str(REPO_ROOT / "Ant-Game") + os.pathsep + env.get("PYTHONPATH", "")
        cmd = [sys.executable, str(exe), *argv]
    else:
        cmd = [str(exe), *argv]
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else str(exe.parent),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_path.open("wb"),
        env=env,
    )
    io = ProcIO(proc, exe.stem, strip_cr=strip_cr)
    send_line(io, f"{player} {seed}\n")
    return proc, io


def run_match(seed: int, keep_dir: Path, max_rounds: int = MAX_ROUND,
              runnerup_first: bool = False) -> dict:
    """跑一局。runnerup_first=True（命令行 token 后缀 r）= 第二个 AI(AI1) 执先手(player0)。

    默认 AI0 = 冠军 magica_v3_O0、AI1 = 亚军 ai_cpp_lure_v4；可用 --ai0=/--ai1= 覆盖
    （A1 的 rule_v4 打自己、A2 的 rule_v4 打冠军/亚军都靠这个）。
    """
    global DUMPED
    keep_dir.mkdir(parents=True, exist_ok=True)
    a0 = AI0_EXE or CHAMPION_EXE
    a1 = AI1_EXE or RUNNERUP_EXE
    p0_exe, p1_exe = (a1, a0) if runnerup_first else (a0, a1)
    p0_label, p0_magica = _label_of(p0_exe)
    p1_label, p1_magica = _label_of(p1_exe)

    stderr0 = keep_dir / f"ai0_{p0_label}_seed{seed}.stderr.log"
    stderr1 = keep_dir / f"ai1_{p1_label}_seed{seed}.stderr.log"

    def ai_env(is_magica: bool) -> dict:
        if is_magica:
            return {"ANTWAR_MAGICA_IO_TRACE": "1",
                    "ANTWAR_MAGICA_RUNTIME_ARTIFACT_DIR": r"e:\magica_runtime"}
        return {"ANTGAME_CPP_BASELINE_DEBUG": "summary"}

    # AI 的 argv/env/cwd 跟着 **AI 的身份**（ai0/ai1）走，不跟着"谁执先手"走
    s0 = AI_SPEC[0] if not runnerup_first else AI_SPEC[1]
    s1 = AI_SPEC[1] if not runnerup_first else AI_SPEC[0]
    env0 = {**ai_env(p0_magica), **s0["env"]}
    env1 = {**ai_env(p1_magica), **s1["env"]}
    proc0, io0 = start_ai(p0_exe, 0, seed, stderr0, extra_env=env0,
                          strip_cr=not p0_magica, argv=s0["args"], cwd=s0["cwd"])
    proc1, io1 = start_ai(p1_exe, 1, seed, stderr1, extra_env=env1,
                          strip_cr=not p1_magica, argv=s1["args"], cwd=s1["cwd"])

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    result: dict = {"seed": seed, "champion_first": p0_magica,
                    "p0_label": p0_label, "p1_label": p1_label}
    _OPS_HISTORY: list[list[list[int]]] = []
    # DUMP_ROUND==0 时快照初始状态（整局重放用），否则等待对局推进到该回合
    saved_snapshot = copy.deepcopy(state) if DUMP_ROUND == 0 else None
    rounds = 0
    try:
        while not state.terminal and rounds < max_rounds:
            # ===== 回合开始：player0（先手）先操作 =====
            payload0 = io0.read_packet()
            text0 = payload0.decode("utf-8", errors="replace")
            if not p0_magica:
                # \r 规范化只作用于 **payload 解码后的文本**（原来在读取层删字节 ⇒ 会破坏
                # 二进制的长度前缀，见 ProcIO._reader 的注释与 2026-09-23 的实证）。
                text0 = normalize_lines(text0)
            # player1 收到 player0 的操作（magica只接受\n结尾，归一化\r）
            send_line(io1, normalize_lines(text0) if p1_magica else text0)
            payload1 = io1.read_packet()
            text1 = payload1.decode("utf-8", errors="replace")
            if not p1_magica:
                text1 = normalize_lines(text1)
            # player0 收到 player1 的操作
            send_line(io0, normalize_lines(text1) if p0_magica else text1)

            # 解析双方操作
            ops0 = parse_ops_text(text0)
            ops1 = parse_ops_text(text1)
            if TRACE_OPS:
                print(f"  [round {rounds}] p0({p0_label}) 原文={text0.strip()!r} -> {ops0}", flush=True)
                print(f"  [round {rounds}] p1({p1_label}) 原文={text1.strip()!r} -> {ops1}", flush=True)

            il0 = state.apply_operation_list(0, ops0)
            il1 = state.apply_operation_list(1, ops1)
            _OPS_HISTORY.append([op.to_protocol_tokens() for op in ops0] + [None] + [op.to_protocol_tokens() for op in ops1])
            if il0 or il1:
                print(f"  !! [round {rounds}] ILLEGAL p0({p0_label})={il0} p1({p1_label})={il1} terminal={state.terminal} winner={state.winner}", flush=True)
                if TRACE_OPS:
                    print(f"     金币={state.coins}", flush=True)
                    print(f"     塔列表={[(t.tower_id, t.player, t.x, t.y, int(t.tower_type), t.hp) for t in state.towers]}", flush=True)
                    for op in il0 + il1:
                        print(f"     can_apply(op={op}): {state.can_apply_operation(0, op, ops0) if op in il0 else state.can_apply_operation(1, op, ops1)}", flush=True)
            state.advance_round()
            rounds += 1

            # 状态快照（对拍用）：推进到目标回合时保存一份深拷贝
            if DUMP_ROUND is not None and not DUMPED and saved_snapshot is None and rounds == DUMP_ROUND:
                saved_snapshot = copy.deepcopy(state)
                print(f"  [snapshot] round {rounds} saved", flush=True)

            # 发送局面给两个AI
            round_text = state_to_text(state)
            if TRACE_OPS and state.active_effects:
                print(f"  [round {rounds}] active_effects={[(int(e.weapon_type), e.player, e.x, e.y, e.remaining_turns) for e in state.active_effects]}", flush=True)
            if TRACE_OPS:
                print(f"  [round {rounds}] towers={[(t.tower_id, t.player, t.x, t.y, int(t.tower_type), t.hp) for t in state.towers]}", flush=True)
            if TRACE_OPS and state.ants:
                print(f"  [round {rounds}] ants={[(a.ant_id, a.player, a.x, a.y, a.hp, a.level, int(a.kind)) for a in state.ants][:40]}", flush=True)
            send_line(io0, round_text)
            send_line(io1, round_text)

        # 延迟 dump：对局结束后写出快照 + 完整操作历史（对拍用）
        if DUMP_ROUND is not None and not DUMPED:
            import cpp_probe_match as _cpm
            DUMPED = True
            snapshot = saved_snapshot if saved_snapshot is not None else state
            data = _cpm.dump_state(snapshot)
            data["seed"] = seed
            data["snapshot_round"] = DUMP_ROUND
            data["ops_history"] = _OPS_HISTORY
            Path(DUMP_PATH).write_text(json.dumps(data), encoding="utf-8")
            print(f"  [dump] snapshot_r{DUMP_ROUND} ops_logged={len(_OPS_HISTORY)} -> {DUMP_PATH}", flush=True)

        result["rounds"] = rounds
        result["terminal"] = bool(state.terminal)
        result["winner"] = state.winner
        result["base_hp"] = [int(b.hp) for b in state.bases]
        result["coins"] = [int(c) for c in state.coins]
        return result
    except Exception as exc:
        import traceback
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["rounds"] = rounds
        result["base_hp"] = [int(b.hp) for b in state.bases]
        print(f"  EXC {exc}", flush=True)
        print(f"  proc0({p0_label}).poll={proc0.poll()}  proc1({p1_label}).poll={proc1.poll()}", flush=True)
        return result
    finally:
        for proc in (proc0, proc1):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()


def parse_ops_text(text: str) -> list[Operation]:
    lines = text.strip().split("\n")
    if not lines:
        return []
    try:
        count = int(lines[0].strip())
    except ValueError:
        return []
    ops: list[Operation] = []
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


def main() -> int:
    global AI0_EXE, AI1_EXE
    # 种子格式：整数 或 整数+r（如 11r 表示第二个 AI(AI1) 执先手）
    # 另有 --rounds=N（回合上限）、--ai0=路径、--ai1=路径（默认 = 冠军/亚军）
    seeds: list[tuple[int, bool]] = []
    max_rounds = MAX_ROUND
    for arg in sys.argv[1:]:
        if arg.startswith("--rounds="):
            max_rounds = int(arg.split("=")[1])
        elif arg.startswith("--ai0="):
            # ⚠️ 必须 resolve()：start_ai 会 cwd=exe.parent，相对路径会再叠一层
            # （曾把 code/test_match/x.py 变成 code/test_match/code/test_match/x.py，AI 直接 exit 2）
            AI0_EXE = Path(arg.split("=", 1)[1]).resolve()
        elif arg.startswith("--ai1="):
            AI1_EXE = Path(arg.split("=", 1)[1]).resolve()
        elif arg.startswith("--ai0-args="):
            AI_SPEC[0]["args"] = tuple(shlex.split(arg.split("=", 1)[1]))
        elif arg.startswith("--ai1-args="):
            AI_SPEC[1]["args"] = tuple(shlex.split(arg.split("=", 1)[1]))
        elif arg.startswith("--ai0-env="):
            AI_SPEC[0]["env"] = _parse_env(arg.split("=", 1)[1])
        elif arg.startswith("--ai1-env="):
            AI_SPEC[1]["env"] = _parse_env(arg.split("=", 1)[1])
        elif arg.startswith("--ai0-cwd="):
            AI_SPEC[0]["cwd"] = Path(arg.split("=", 1)[1]).resolve()
        elif arg.startswith("--ai1-cwd="):
            AI_SPEC[1]["cwd"] = Path(arg.split("=", 1)[1]).resolve()
        elif arg.isdigit():
            seeds.append((int(arg), False))
        elif arg.endswith("r") and arg[:-1].isdigit():
            seeds.append((int(arg[:-1]), True))
    if not seeds:
        seeds = [(7, False)]
    keep_dir = REPO_ROOT / "match_results"
    keep_dir.mkdir(parents=True, exist_ok=True)

    a0, a1 = AI0_EXE or CHAMPION_EXE, AI1_EXE or RUNNERUP_EXE
    print(f"[bridge] AI0={a0}", flush=True)
    print(f"[bridge] AI1={a1}", flush=True)
    for _i, _p in ((0, a0), (1, a1)):
        _s = AI_SPEC[_i]
        print(f"[bridge] AI{_i} spec: args={list(_s['args'])} cwd={_s['cwd']} env={_s['env']}",
              flush=True)
    # 裁判身份必须显式可见：main.exe 与我们的 cpp_engine 不是风格差异，是"正常对局"与
    # "两个 AI 全程静默弃权"的分界（2026-08-10），而且三份引擎源码逐字节相同、源码比对也发现不了。
    print("[bridge] judge=cpp_engine (in-process GameStateFacade)", flush=True)
    print(f"[bridge] max_rounds={max_rounds}  seeds={seeds}", flush=True)

    wins: dict[str, int] = {}
    for seed, runnerup_first in seeds:
        p0_name = _label_of(a1 if runnerup_first else a0)[0]
        print(f"\n{'='*70}\n 对战 seed={seed}  {p0_name} 先手  "
              f"AI0={_label_of(a0)[0]} vs AI1={_label_of(a1)[0]}\n{'='*70}", flush=True)
        r = run_match(seed, keep_dir, max_rounds, runnerup_first=runnerup_first)
        print(f"  回合: {r.get('rounds')}  终局: {r.get('terminal')}", flush=True)
        print(f"  base_hp: {r.get('base_hp')}  coins: {r.get('coins')}", flush=True)
        if r.get("exception"):
            print(f"  异常: {r['exception']}", flush=True)
            print(r.get("traceback", ""), flush=True)
            # ⚠️ **异常终止的局不是结果**：绝不能报成一个胜者（实测 seed11 第 84 回合超时，
            # 老代码却打出 winner=runnerup ⇒ 一个截断的局被当成战果）。标 INVALID 且不计分。
            _hp = r.get("base_hp") or [0, 0]
            _cn = r.get("coins") or [0, 0]
            print(f"  RESULT seed={seed} p0={r.get('p0_label', 'p0')} p1={r.get('p1_label', 'p1')} "
                  f"winner=INVALID verdict=aborted exc={type(r.get('exception')).__name__} "
                  f"rounds={r.get('rounds')} terminal={bool(r.get('terminal'))} "
                  f"base_hp={_hp[0]},{_hp[1]} coins={_cn[0]},{_cn[1]}", flush=True)
            continue

        winner = r.get("winner")           # 引擎自己的 winner；None = 还没判
        hp = r.get("base_hp") or [0, 0]
        p0_lab = r.get("p0_label", "p0")
        p1_lab = r.get("p1_label", "p1")
        # ⚠️ 判词来源必须标出来。引擎在终局时已算好完整级联（HP→击杀数→超武次数少者→总时间少者→判 P0），
        # 我们只在"循环提前停下、引擎还没判"时退回"只比 HP"——那条**不是官方规则**（官方也从不判平局）。
        if winner is not None and winner in (0, 1):
            verdict = "engine"
            winner_ai = p0_lab if winner == 0 else p1_lab
            winner_side = "p0" if winner == 0 else "p1"
        else:
            verdict = "hp_tiebreak"
            winner_ai = p0_lab if hp[0] > hp[1] else (p1_lab if hp[1] > hp[0] else "draw")
            winner_side = ("p0" if hp[0] > hp[1] else ("p1" if hp[1] > hp[0] else "draw"))
        wins[winner_ai] = wins.get(winner_ai, 0) + 1
        print(f"  >>> {winner_ai}({winner_side}) 胜 <<<", flush=True)
        # winner_side 是**标签之外**的独立信息：两个 AI 同名时（例如 A1 里都是 main）
        # 只有它能告诉你赢的是哪一侧。标签用于区分不同 AI，侧别用于镜像配对与侧偏检查。
        print(f"  RESULT seed={seed} p0={p0_lab} p1={p1_lab} winner={winner_ai} "
              f"winner_side={winner_side} verdict={verdict} engine_winner={winner} "
              f"rounds={r.get('rounds')} terminal={bool(r.get('terminal'))} "
              f"base_hp={hp[0]},{hp[1]} "
              f"coins={(r.get('coins') or [0, 0])[0]},{(r.get('coins') or [0, 0])[1]}", flush=True)

    print(f"\n{'='*70}\n汇总: {json.dumps(wins, ensure_ascii=False)}\n{'='*70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
