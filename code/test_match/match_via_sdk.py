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
    MAP_SIZE,
    MAX_ACTIONS,
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


def magica_env() -> dict[str, str]:
    """magica（冠军）**专用**的启动环境：`PATH` 前置一个可用的 mingw DLL 目录 + 两个诊断变量。

    为什么必须动 `PATH`（2026-10-28 实测，`docs/us_vs_champion_20261028.md` §2）：
      * `magica_v3_O0.exe` 是 **mingw 动态链接**的，而它的目录里**一个 DLL 都没带**
        （只有 exe + main.cpp + readme.md）⇒ Windows 加载器只能**沿 PATH** 找
        `libstdc++-6.dll` / `libgcc_s_seh-1.dll` / `libwinpthread-1.dll`；
      * 本机 PATH 里 **anaconda 的 mingw 目录**（`D:/anaconda3/Library/mingw-w64/bin`）排在
        `C:/mingw64/bin` **前面**，那份 libstdc++ 与它的导入表不匹配 ⇒ 进程**刚起来就退出**，
        退出码 `3221225785` = `0xC0000139`（STATUS_ENTRYPOINT_NOT_FOUND）；
      * bridge 侧只看到 `EOFError` / `OSError [Errno 22]`（写一个已经死掉的 stdin），
        极易被误读成"协议不对/它太慢"。
    实测（`_tmp_mag_deps2.sh` 的三次试验）：干净 PATH、或把 `C:/mingw64/bin`（或亚军 build
    目录那三份 DLL）放最前 ⇒ 正常输出 `[Magica] hello ... ReadInit 成功` 并进入协议。

    ⚠️ 只改**这个子进程**的环境（调用方 `os.environ` 不动）：我们自己的进程跑 torch/pybind，
    改全局 PATH 有把 `cpp_engine` 的 DLL 搜索顺序带偏的风险。
    可用 `MVS_MAGICA_DLL_DIR` 显式指定（换机器时最省事）。
    """
    out: dict[str, str] = {
        "ANTWAR_MAGICA_IO_TRACE": "1",
        "ANTWAR_MAGICA_RUNTIME_ARTIFACT_DIR": r"e:\magica_runtime",
    }
    cands: list[Path] = []
    _ov = os.environ.get("MVS_MAGICA_DLL_DIR", "").strip()
    if _ov:
        cands.append(Path(_ov))
    cands.append(Path("C:/mingw64/bin"))
    cands.append(RUNNERUP_EXE.parent)          # 亚军那三份 DLL 与它 ABI 相容（实测能起来）
    for d in cands:
        try:
            if d.is_dir() and (d / "libstdc++-6.dll").is_file():
                out["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                sys.stderr.write(f"[bridge] magica PATH 前置 {d}（避开 anaconda mingw 的 0xC0000139）\n")
                sys.stderr.flush()
                return out
        except OSError:
            continue
    sys.stderr.write("[bridge] ⚠️ 没找到可用的 mingw DLL 目录 ⇒ magica 可能 0xC0000139 直接死\n")
    sys.stderr.flush()
    return out


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


# ⚠️ 可调（2026-10-06）：原来是写死的 300.0。它决定「一次读包最多等多久」⇒
#    一次**算得太久**的决策会被记成 aborted，从而**伪装成无效局**（曾经误导过一次归因）。
#    用 MVS_TIMEOUT_SECONDS 覆盖；超时局在 RESULT 里标 verdict=timeout（见下）。
TIMEOUT_SECONDS = float(os.environ.get("MVS_TIMEOUT_SECONDS", "300"))
# 2026-09-22：120 → 300。原值在机器被别的进程抢 CPU 时会把整局掐断
# （实测 DET_seed11_run1：第 84 回合亚军那步 >120s，报
#  "timed out reading ai_cpp_lure_v4 (need 50B, have 12B)"，整局作废）。
# 它只是"卡死检测"的门限，不影响正常对局的结果。
TRACE_OPS = "--trace-ops" in sys.argv
# 2026-10-23：逐回合进度 trace（code/test_match/progress.py）
_TRACE = None
_SEED_TOKEN = None   # 2026-10-23：原始 seed token（如 "7r"），进度文件名用它

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
# 也支持环境变量（`_tmp_ladder.py` 没法给 AI 传额外命令行参数）
if DUMP_ROUND is None:
    _de = os.environ.get("AZAI_DUMP_ROUND", "")
    if _de:
        try:
            DUMP_ROUND = int(_de.split(":", 1)[0])
            DUMP_PATH = _de.split(":", 1)[1] if ":" in _de else "state_dump.json"
        except (ValueError, IndexError):
            DUMP_ROUND = None


# ══════════════════════════════════════════════════════════════════════════════════════
# 非法操作：严格模式 + 逐条影子校验（2026-10-27）
#
# 引擎参数 `cold_handle_rule_illegal`（官方 `Ant-Game/game/src/game.cpp:2618` 同名逻辑）：
#   False = **官方默认**、平台判题用的就是这个 ⇒ 非法操作走 `apply_operation`，一旦失败就
#           `set_AI_state_IO(player)`（`is_end=true, winner=对手`）⇒ 发非法操作的一方**直接判负**；
#   True  = 冷处理 ⇒ 非法操作被**静默跳过**，对局继续（本地历史默认值）。
# 旧版本本文件写死 True ⇒ 任何"非法操作"类 bug 在本地**永远测不出来**，只在正式比赛里变判负。
# 现在默认 False（对齐官方）；`--cold-illegal`（或环境变量 `AZAI_COLD_ILLEGAL=1`）= 旧行为。
#
# ⚠️ 两个**必须先说清楚的事实**（否则会以为"改个参数就对齐了"）：
#   ① `GameStateFacade.apply_operation_list` 返回 **None**（`game_state_facade.py:167-170`
#      把 binding 的 accepted 计数丢掉了）⇒ 本文件原来那行 `il0 = state.apply_operation_list(...)`
#      + `if il0 or il1:` **永远不成立**；`_tmp_ladder.py:192` 靠数 "ILLEGAL" 行得到的
#      "48 局 0 次非法操作"是**结构性 0**（检查根本不会触发），不是实测 0。
#   ② pybind 的 `apply_operation_list` **恒走冷路径**（`code/cpp_engine/binding.cpp:46-59` 无条件
#      调 `apply_operation_list_cold`），而 `cold_handle_rule_illegal` 只在 `round_read_from_judger`
#      里被读、binding 从不调用它 ⇒ **只把 cold=False 传进去不改变任何行为**。
#      ⇒ 严格模式必须由**本文件**实现：下面的影子校验负责"发现"，`run_match` 负责
#      "按官方语义就地终局（肇事方判负）"。
# ══════════════════════════════════════════════════════════════════════════════════════

# 严格模式（= 官方默认）。`--cold-illegal` / `AZAI_COLD_ILLEGAL=1` ⇒ True = 旧行为（静默跳过）。
COLD_ILLEGAL = ("--cold-illegal" in sys.argv
                or os.environ.get("AZAI_COLD_ILLEGAL", "") not in ("", "0"))

_OP_NAMES = {
    11: "BUILD_TOWER", 12: "UPGRADE_TOWER", 13: "DOWNGRADE_TOWER",
    21: "LIGHTNING_STORM", 22: "EMP_BLASTER", 23: "DEFLECTOR", 24: "EMERGENCY_EVASION",
    31: "UPGRADE_GENERATION_SPEED", 32: "UPGRADE_GENERATED_ANT",
}


def _log_stderr(msg: str) -> None:
    """非法操作相关的输出**一律走 stderr**：stdout 是与 AI 之间的二进制协议通道
    （`az_bridge_ai.py` 里 `sys.stdout = sys.stderr` 是同一个理由）。"""
    sys.stderr.write(f"[bridge] {msg}\n")
    sys.stderr.flush()


class IllegalStats:
    """一局的非法操作计数。**只读统计**，不参与任何决策。"""

    __slots__ = ("checked", "total", "holds", "by_side", "by_kind", "mismatch",
                 "seconds", "first")

    def __init__(self) -> None:
        self.checked = 0                            # 逐条校验过的 op 数（HOLD 不计）
        self.total = 0                              # 非法 op 数
        self.holds = {0: 0, 1: 0}                   # 空操作（HOLD）回合数
        self.by_side = {0: 0, 1: 0}
        self.by_kind: dict[tuple[int, str], int] = {}
        self.mismatch = 0                           # can_apply(dry) 与 can_apply_cold 不一致次数
        self.seconds = 0.0                          # 影子校验累计墙钟（成本可见）
        self.first: list[str] = []                  # 前几条非法 op 的摘要（进汇总行）

    def summary(self, seed: int, cold: bool) -> str:
        """一局的汇总行。⚠️ 不含大写 "ILLEGAL"：`_tmp_ladder.py:192` 数的是含该词的**行数**，
        汇总行若含它就会把 0 非法也数成 1。"""
        kinds = ",".join(f"({t},{r}):{n}" for (t, r), n in
                         sorted(self.by_kind.items(), key=lambda kv: (-kv[1], kv[0])))
        txt = (f"illegal-summary seed={seed} cold={cold} illegal_total={self.total} "
               f"checked_ops={self.checked} "
               f"holds=p0:{self.holds[0]},p1:{self.holds[1]} "
               f"by_side={{0:{self.by_side[0]},1:{self.by_side[1]}}} "
               f"by_kind={{{kinds}}} mismatch={self.mismatch} "
               f"shadow_ms={self.seconds * 1000.0:.1f}")
        if self.first:
            txt += " first=" + "|".join(self.first)
        return txt


def _op_tuple(op) -> tuple[int, int, int]:
    return (int(op.op_type), int(op.arg0), int(op.arg1))


def _op_text(op) -> str:
    t, a0, a1 = _op_tuple(op)
    return f"({t},{a0},{a1})"


def _tower_by_id(state, tower_id: int):
    for t in state.towers:
        if int(t.tower_id) == int(tower_id):
            return t
    return None


def op_reason(state, player: int, op, pending=()) -> str:
    """**尽力而为**的非法原因分类（引擎的 `can_apply` 只回 bool，不回原因）。

    检查次序照抄官方 `Game::apply_operation`（game.cpp:1940+）；只用于日志归类，
    **不参与任何判定** —— 判定由引擎的 `can_apply_*` 给。查不出的返回 `*_other` / `unspecified`。
    """
    t = int(op.op_type)
    a0, a1 = int(op.arg0), int(op.arg1)
    try:
        coins = int(state.coins[int(player)])
    except Exception:  # noqa: BLE001
        coins = -1
    pend_ids = {int(p.arg0) for p in pending if int(p.op_type) in (12, 13)}
    pend_base = any(int(p.op_type) in (31, 32) for p in pending)
    try:
        if t == 11:  # BUILD_TOWER：(x, y)
            x, y = a0, a1
            if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
                return "build_pos_out_of_range"
            if (x, y) in (tuple(PLAYER_BASES[0]), tuple(PLAYER_BASES[1])):
                return "build_on_base"
            if state.tower_at(x, y) is not None:
                return "build_cell_occupied"
            cost = int(state.build_tower_cost(state.tower_count(int(player))))
            if coins < cost:
                return f"build_no_gold(cost={cost},coins={coins})"
            return "build_cell_not_owned_or_emp_shielded"
        if t in (12, 13):  # UPGRADE_TOWER：(tower_id, target_type) / DOWNGRADE：(tower_id)
            tw = _tower_by_id(state, a0)
            if tw is None:
                return f"tower_id_not_found(id={a0})"
            if int(tw.player) != int(player):
                return f"tower_not_owned(id={a0},owner={int(tw.player)})"
            if a0 in pend_ids:
                return f"tower_already_used_in_round(id={a0})"
            if t == 13:
                return f"downgrade_other(id={a0})"
            if int(getattr(tw, "level", 0)) >= 2:
                return f"tower_max_level(id={a0})"
            # 官方 DefenseTower::upgrade_type_check：TowerType(target/10) == 当前类型
            if int(a1) < 0 or int(a1) // 10 != int(tw.tower_type):
                return (f"upgrade_target_type_mismatch(target={a1},"
                        f"tower_type={int(tw.tower_type)})")
            cost = -1
            try:
                cost = int(state.upgrade_tower_cost(TowerType(a1)))
            except Exception:  # noqa: BLE001
                pass
            if coins < cost:
                return f"upgrade_no_gold(cost={cost},coins={coins})"
            return f"upgrade_other(id={a0},target={a1})"
        if t in (21, 22, 23, 24):  # 超武：(x, y)
            x, y = a0, a1
            if not (0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE):
                return "weapon_pos_out_of_range"
            w = SuperWeaponType(t - 20)
            cd = int(state.weapon_cooldowns[int(player), w])
            if cd > 0:
                return f"weapon_in_cooldown(cd={cd})"
            cost = int(state.weapon_cost(w))
            if coins < cost:
                return f"weapon_no_gold(cost={cost},coins={coins})"
            return "weapon_other"
        if t in (31, 32):  # 基地升级（一回合只能升一次）
            if pend_base:
                return "base_upgrade_twice_in_round"
            base = state.bases[int(player)]
            lvl = int(base.generation_level if t == 31 else base.ant_level)
            if lvl >= 2:
                return f"base_max_level(lvl={lvl})"
            cost = -1
            try:
                cost = int(state.upgrade_base_cost(lvl))
            except Exception:  # noqa: BLE001
                pass
            if coins < cost:
                return f"base_no_gold(cost={cost},coins={coins})"
            return "base_upgrade_other"
        return f"unknown_op_type({t})"
    except Exception as exc:  # noqa: BLE001
        return f"reason_probe_error({type(exc).__name__})"


def _can_apply_cold(state, player: int, op, pending=()):
    """引擎的**真**路径检查：clone 后把 `pending+op` 交给 `apply_operation_list_cold`
    （内部逐条走官方 `apply_operation`）。返回 True/False；接口不可用时返回 None。

    比 `facade.can_apply_operation`（= `can_apply_dry`，clone-free 的**复刻**实现）更权威：
    后者是搜索用的快速版，长期有与引擎漂移的风险 ⇒ 这里两者都跑，不一致就记 MISMATCH。
    """
    g = getattr(state, "_g", None)
    if g is None or not hasattr(g, "can_apply_cold"):
        return None
    try:
        return bool(g.can_apply_cold(
            int(player), [int(op.op_type), int(op.arg0), int(op.arg1)],
            [[int(p.op_type), int(p.arg0), int(p.arg1)] for p in pending]))
    except Exception:  # noqa: BLE001
        return None


def check_ops(state, player: int, ops, stats: IllegalStats, *, seed: int = 0,
              round_index: int = -1, tag: str = "apply", cold: bool | None = None,
              log=None) -> list[dict]:
    """**只读**逐条合法性校验 + 计数 + 逐条日志。返回非法记录列表（空 = 全合法）。

    为什么不能靠 `apply_operation_list` 的返回值：它返回 None，且内部走冷路径会
    **静默跳过**非法 op（见文件头那段注释 ①）。所以这里对每个 op 先问引擎
    `can_apply_dry`（搜索同款，快）与 `can_apply_cold`（clone 版，权威）：
      · 权威判定 = cold（拿不到才退回 dry）；
      · 两者不一致 ⇒ 记 `MISMATCH`（这类漂移本身就是要抓的 bug）；
      · 两个检查都抛异常 ⇒ 记 `CHECK-ERROR` 并**放行**（绝不能因为检查器坏了判人负）。
    `pending=` 传已判合法的前缀：官方在同一回合内会累计 used_tower / 已升级基地标志。
    """
    if log is None:
        log = _log_stderr
    pl = int(player)
    if not ops:
        stats.holds[pl] = stats.holds.get(pl, 0) + 1       # HOLD / 空操作列表（合法）
        return []
    accepted: list = []
    bad: list[dict] = []
    for idx, op in enumerate(ops):
        t0 = time.perf_counter()
        ok_dry = None
        err = ""
        try:
            ok_dry = bool(state.can_apply_operation(pl, op, pending=accepted))
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        ok_cold = _can_apply_cold(state, pl, op, accepted)
        stats.seconds += time.perf_counter() - t0
        stats.checked += 1
        if ok_dry is not None and ok_cold is not None and ok_dry != ok_cold:
            stats.mismatch += 1
            log(f"MISMATCH(legality) seed={seed} round={round_index} side={pl} idx={idx} "
                f"op={_op_text(op)} dry={ok_dry} cold={ok_cold} "
                f"reason={op_reason(state, pl, op, accepted)}")
        ok = ok_dry if ok_cold is None else ok_cold
        if ok is None:
            log(f"CHECK-ERROR seed={seed} round={round_index} side={pl} idx={idx} "
                f"op={_op_text(op)} err={err or 'both_checkers_unavailable'}")
            accepted.append(op)
            continue
        if ok:
            accepted.append(op)
            continue
        reason = op_reason(state, pl, op, accepted)
        t_, a0_, a1_ = _op_tuple(op)
        bad.append({"idx": idx, "side": pl, "round": round_index,
                    "op": (t_, a0_, a1_), "reason": reason, "tag": tag,
                    "dry": ok_dry, "cold": ok_cold})
        stats.total += 1
        stats.by_side[pl] = stats.by_side.get(pl, 0) + 1
        stats.by_kind[(t_, reason)] = stats.by_kind.get((t_, reason), 0) + 1
        if len(stats.first) < 12:
            stats.first.append(f"r{round_index}s{pl}({t_},{a0_},{a1_})/{reason}")
        # ⚠️ 大写 ILLEGAL：`_tmp_ladder.py:192` 靠它数非法行，一条非法 = 一行
        log(f"ILLEGAL seed={seed} round={round_index} side={pl} tag={tag} idx={idx} "
            f"op={_op_text(op)} name={_OP_NAMES.get(t_, '?')} reason={reason} "
            f"dry={ok_dry} cold={ok_cold} cold_mode={cold} engine=reject")
    return bad


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
    # 2026-10-28：magica 的启动要求（PATH 里的 mingw DLL）**集中在这里**，
    #   这样**所有**调用点自动覆盖：真实对局（`run_match`）、注入会话（`runnerup_oracle`
    #   也是用本函数起进程的）、以及任何探针。详见 `magica_env()` 的 docstring。
    if "magica" in exe.name.lower():
        env.update(magica_env())
    if exe.suffix == ".py":
        # 协议 AI 的 Python 壳：用 sys.executable 起，并把 Ant-Game 根放进 PYTHONPATH。
        # 抄自 code/test_match/run_cpp_ai_match.py:65-70（code/test_match/rv4_pkg/main.py
        # 就是靠这个跑的）——本文件原来只支持可执行文件，所以 rule_v4 这类 Python AI 进不来。
        env["PYTHONPATH"] = str(REPO_ROOT / "Ant-Game") + os.pathsep + env.get("PYTHONPATH", "")
        # 诊断开关（2026-10-06，默认关）：给 Python AI 加 `-X faulthandler`，
        # 这样**原生崩溃**（segfault/访问违例）时会把 Python 调用栈写进该 AI 的 stderr。
        # 用途：定位 `pos_pin=mc_light` 在实战里"无 traceback 直接死"的那类故障。见 docs/engine_crash_probe.md
        _xflags = []
        if os.environ.get("MVS_FAULTHANDLER", "") not in ("", "0"):
            _xflags = ["-X", "faulthandler"]
        cmd = [sys.executable, *_xflags, str(exe), *argv]
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


def _dump_raw_state(fh, state, seed: int, round_idx: int, ents_fh=None) -> None:
    """可选的 raw-state dump（`MVS_RAW_STATE_DIR` 非空才调用）。

    两条流（按 round 一一对齐）：
      - `.bin`（`raw_seed*.bin`，两个玩家各一条）：board(28*19*19 f16)+stats(42 f16)+player(u8)+round(i16)+value(f32)
      - `.jsonl`（`ents_seed*.jsonl`，每回合一条）：原始**实体特征**（塔/蚁/金币/基地），供实体 token 模
        型吃——board 里没有塔 hp / 蚁 kind，所以必须单独存。
    默认关闭 ⇒ 不 import numpy/distill_top2、也不建文件 ⇒ 现有实验行为逐位不变。
    """
    try:
        import numpy as np
        from SDK.utils.features import FeatureExtractor
        from my_ai.distill_top2.lure_value_eval import evaluate_terminal

        fe = FeatureExtractor()
        for p in (0, 1):
            obs = fe.encode_observation(state, p, np.zeros(MAX_ACTIONS, dtype=np.int8))
            value = float(evaluate_terminal(state, p))
            fh.write(np.asarray(obs["board"], dtype=np.float16).tobytes())
            fh.write(np.asarray(obs["stats"], dtype=np.float16).tobytes())
            fh.write(bytes([int(p)]))
            fh.write(np.int16(int(round_idx)).tobytes())
            fh.write(np.float32(value).tobytes())
        fh.flush()
        if ents_fh is not None:
            import json as _json
            ents_fh.write(_json.dumps({
                "round": int(round_idx),
                "towers": [[int(t.tower_id), int(t.player), int(t.x), int(t.y),
                            int(t.tower_type), int(t.hp), float(t.cooldown_clock)]
                           for t in state.towers],
                "ants": [[int(a.ant_id), int(a.player), int(a.x), int(a.y), int(a.hp),
                          int(a.max_hp), int(a.kind), int(a.age), int(a.level),
                          int(a.behavior), int(a.status), int(bool(a.frozen))]
                         for a in state.ants],
                "coins": [int(c) for c in state.coins],
                "bases": [[int(b.hp), int(b.generation_level), int(b.ant_level)]
                          for b in state.bases],
                "weapon_cooldowns": [[int(state.weapon_cooldowns[p, w]) for w in range(5)]
                                     for p in (0, 1)],
                "super_weapon_usage": [int(state.super_weapon_usage[p]) for p in (0, 1)],
                "effects": [[int(e.weapon_type), int(e.player), int(e.x), int(e.y),
                             int(e.remaining_turns)] for e in state.active_effects],
            }, ensure_ascii=False) + "\n")
            ents_fh.flush()
    except Exception as exc:  # noqa: BLE001
        print(f"  !! [raw-dump] round {round_idx} 失败: {type(exc).__name__}: {exc}", flush=True)


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

    # 2026-10-23（用户要求）：逐回合进度文件，实时可看。tag 由 ladder 用 DSH_PROGRESS_TAG 传入。
    global _TRACE
    try:
        import os as _os
        from progress import GameTrace as _GT
        _TRACE = _GT(_os.environ.get("DSH_PROGRESS_TAG", "match"),
                      globals().get("_SEED_TOKEN") or seed,
                      _os.environ.get("DSH_PROGRESS_SUFFIX", ""))
    except Exception:
        _TRACE = None

    # 2026-10-28（opt-in，默认关闭）：`MVS_STDERR_SUFFIX=1` ⇒ 把驱动已按局设置好的
    # `DSH_PROGRESS_SUFFIX`（`_p0`/`_p1`）拼进 stderr 文件名。
    # 动机：`ab_search_modes --paired-sides` 同一个 seed 的两局**并发**，而这里原来的名字只含
    # (seat, label, seed) ⇒ 两局用 "wb" 打开**同一个** `ai?_<label>_seed<N>.stderr.log`，互相覆盖
    # （`docs/oracle_inject_design.md` §4.3 记录过：注入统计只能读最后存活的那局）。
    # 不设该环境变量 ⇒ 文件名与改动前**逐字相同**（`_tmp_ladder.py` / `analyze_paired.py` 等老消费者不受影响）。
    _suf = os.environ.get("DSH_PROGRESS_SUFFIX", "") if os.environ.get("MVS_STDERR_SUFFIX") == "1" else ""
    stderr0 = keep_dir / f"ai0_{p0_label}_seed{seed}{_suf}.stderr.log"
    stderr1 = keep_dir / f"ai1_{p1_label}_seed{seed}{_suf}.stderr.log"

    def ai_env(is_magica: bool, seed: int | None = None) -> dict:
        if is_magica:
            return {"ANTWAR_MAGICA_IO_TRACE": "1",
                    "ANTWAR_MAGICA_RUNTIME_ARTIFACT_DIR": r"e:\magica_runtime"}
        # 亚军（非 magica）的 debug 级别：默认 summary；要采"候选+打分"（先验蒸馏用）就设
        # MVS_RUNNERUP_DEBUG=plans（写进它的 stderr，由 _archive_stderr 归档）。
        env = {"ANTGAME_CPP_BASELINE_DEBUG": os.environ.get("MVS_RUNNERUP_DEBUG", "summary")}
        # 采集"rollout 终局态 + 价值"（数据放大；需用 -DLURE_DUMP_VALUE 编译的亚军 exe）。
        # MVS_RUNNERUP_VALUE_DUMP=<目录>，MVS_RUNNERUP_VALUE_STRIDE=<抽样步长，默认500>。
        dump_dir = os.environ.get("MVS_RUNNERUP_VALUE_DUMP", "")
        if dump_dir and seed is not None:
            # ⚠️ 必须绝对路径：AI 的 cwd 是 exe 所在目录，相对路径会解析到别处 ⇒ 文件打不开
            dump_path = (REPO_ROOT / Path(dump_dir) if not Path(dump_dir).is_absolute()
                         else Path(dump_dir))
            dump_path.mkdir(parents=True, exist_ok=True)
            env["LURE_VALUE_DUMP"] = str(dump_path / f"value_seed{seed:05d}.jsonl")
            env["LURE_VALUE_DUMP_STRIDE"] = os.environ.get("MVS_RUNNERUP_VALUE_STRIDE", "500")
        return env

    # AI 的 argv/env/cwd 跟着 **AI 的身份**（ai0/ai1）走，不跟着"谁执先手"走
    s0 = AI_SPEC[0] if not runnerup_first else AI_SPEC[1]
    s1 = AI_SPEC[1] if not runnerup_first else AI_SPEC[0]
    env0 = {**ai_env(p0_magica, seed), **s0["env"]}
    env1 = {**ai_env(p1_magica, seed), **s1["env"]}
    proc0, io0 = start_ai(p0_exe, 0, seed, stderr0, extra_env=env0,
                          strip_cr=not p0_magica, argv=s0["args"], cwd=s0["cwd"])
    proc1, io1 = start_ai(p1_exe, 1, seed, stderr1, extra_env=env1,
                          strip_cr=not p1_magica, argv=s1["args"], cwd=s1["cwd"])

    # 严格/冷模式见文件头那段注释：默认 False = **官方平台判题口径**（非法操作 ⇒ 判负）。
    # ⚠️ 该 flag 在 pybind 路径上是**惰性**的（binding 恒走冷路径），严格语义由下面的
    #    影子校验 + 就地终局实现；这里传对值只是为了语义自洽、以及不被后人误读成"已经是官方的了"。
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=COLD_ILLEGAL)
    # raw-state dump（可选，默认关）：每回合两个玩家各写一条（见 _dump_raw_state）
    raw_dir = os.environ.get("MVS_RAW_STATE_DIR", "")
    raw_fh = None
    ents_fh = None
    if raw_dir:
        _rd = Path(raw_dir)
        if not _rd.is_absolute():
            _rd = REPO_ROOT / _rd
        _rd.mkdir(parents=True, exist_ok=True)
        raw_fh = open(_rd / f"raw_seed{seed:05d}.bin", "wb")
        ents_fh = open(_rd / f"ents_seed{seed:05d}.jsonl", "w", encoding="utf-8")
        print(f"  [raw-dump] -> {raw_fh.name} + {ents_fh.name}", flush=True)
    result: dict = {"seed": seed, "champion_first": p0_magica,
                    "p0_label": p0_label, "p1_label": p1_label}
    stats = IllegalStats()          # 本局非法操作计数（汇总行在 finally 里打）
    _OPS_HISTORY: list[list[list[int]]] = []
    # DUMP_ROUND==0 时快照初始状态（整局重放用），否则等待对局推进到该回合
    saved_snapshot = copy.deepcopy(state) if DUMP_ROUND == 0 else None
    rounds = 0
    try:
        while not state.terminal and rounds < max_rounds:
            if raw_fh is not None:
                _dump_raw_state(raw_fh, state, seed, rounds, ents_fh)
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

            # 2026-10-23（用户要求）：**逐回合**写进度文件，便于实时观察
            #   格式与 az_progress_seed*.txt 一致：round=N P0=[(t,x,y)] P1=[...]
            if _TRACE is not None:
                try:
                    # 决策时刻的双方金币/血量（本回合操作**执行前**）
                    _coins = [int(c) for c in state.coins]
                    _hp = [int(b.hp) for b in state.bases]
                    _TRACE.round(rounds,
                                 [(int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops0],
                                 [(int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops1],
                                 coins=_coins, hp=_hp)
                except Exception:
                    pass
            _OPS_HISTORY.append([op.to_protocol_tokens() for op in ops0] + [None] + [op.to_protocol_tokens() for op in ops1])
            # ── ① 只读影子校验（**先做**，绝不改状态）──────────────────────────────
            # 逐条问引擎"这个 op 能不能落"（dry + clone 版冷路径），非法就逐条打 stderr、
            # 计数并按 (op_type, reason) 归类。这一步与是否严格模式**无关**：冷模式下也照打。
            # ⚠️ **顺序必须与官方一致：p0 校验→应用，再 p1 校验→应用**（main.cpp:41/52 的
            #    round_read_from_judger(0) → (1)）。若两侧都先查再一起应用，就会漏掉
            #    "p0 本回合放 EMP ⇒ p1 的建塔落在护盾格上" 这类**同回合跨侧**影响
            #    （官方 apply_operation 里 TowerBuild 的 `EMPBlaster is active` 检查）。
            bad0 = check_ops(state, 0, ops0, stats, seed=seed, round_index=rounds,
                             tag="apply", cold=COLD_ILLEGAL)
            if bad0 and not COLD_ILLEGAL:
                result["strict_illegal_side"] = 0
                result["strict_illegal_ops"] = bad0
                print(f"  !!! 严格模式(cold=False)：p0 在第 {rounds} 回合发非法操作 "
                      f"⇒ 按官方规则判负（winner=p1）", file=sys.stderr, flush=True)
                break
            # ── ② 应用（冷模式与"全合法"的严格模式都走这里，行为与历史逐位一致）────────
            state.apply_operation_list(0, ops0)
            bad1 = check_ops(state, 1, ops1, stats, seed=seed, round_index=rounds,
                             tag="apply", cold=COLD_ILLEGAL)
            if bad1 and not COLD_ILLEGAL:
                # 严格模式 = 官方判题口径：`set_AI_state_IO(player)` ⇒ 立刻终局、肇事方判负。
                # 官方按 p0 → p1 的顺序读操作块 ⇒ p0 先非法就先判 p0。
                # 这里**不再应用 p1 的 op**（对局已结束；官方那半步的前缀副作用不影响胜负）。
                result["strict_illegal_side"] = 1
                result["strict_illegal_ops"] = bad1
                print(f"  !!! 严格模式(cold=False)：p1 在第 {rounds} 回合发非法操作 "
                      f"⇒ 按官方规则判负（winner=p0）", file=sys.stderr, flush=True)
                break
            state.apply_operation_list(1, ops1)
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
        # 严格模式判负（`set_AI_state_IO` 语义）：引擎对象本身没有"外部置终局"的接口
        # （binding 只暴露 can_apply/apply/advance），所以终局**在结果层**表达：
        # winner = 对手、terminal=True；`verdict=illegal_strict` 让它不被当成正常对局计入。
        if result.get("strict_illegal_side") is not None:
            _side = int(result["strict_illegal_side"])
            result["winner"] = 1 - _side
            result["terminal"] = True
            result["strict"] = True
        # 2026-10-23：进度 trace 收尾行（格式与 az_progress_seed*.txt 一致）
        try:
            if _TRACE is not None:
                _TRACE.terminal(result.get("winner"), [int(b.hp) for b in state.bases], rounds,
                                coins=[int(c) for c in state.coins])
        except Exception:
            pass
        result["base_hp"] = [int(b.hp) for b in state.bases]
        result["coins"] = [int(c) for c in state.coins]
        return result
    except Exception as exc:
        import traceback
        result["exception"] = f"{type(exc).__name__}: {exc}"
        # 真实异常类型单独存一份：RESULT 行原来打的是「字符串的类型」（恒为 str），会把归因带偏
        result["exception_type"] = type(exc).__name__
        result["timed_out"] = isinstance(exc, TimeoutError)
        result["traceback"] = traceback.format_exc()
        result["rounds"] = rounds
        result["base_hp"] = [int(b.hp) for b in state.bases]
        print(f"  EXC {exc}", flush=True)
        print(f"  proc0({p0_label}).poll={proc0.poll()}  proc1({p1_label}).poll={proc1.poll()}", flush=True)
        return result
    finally:
        # ── 非法操作汇总（**每局一行、走 stderr**；`illegal_total=N` 是本次改动的判据）──
        # 放在 finally 里 ⇒ 超时/异常局也有汇总行（否则"0 次"可能只是没打出来）。
        try:
            result["illegal_stats"] = stats.summary(seed, COLD_ILLEGAL)
            result["illegal_total"] = stats.total
            _log_stderr(stats.summary(seed, COLD_ILLEGAL) +
                        f" p0={p0_label} p1={p1_label} rounds={rounds}")
        except Exception as exc:  # noqa: BLE001
            _log_stderr(f"illegal-summary FAILED ({type(exc).__name__}: {exc})")
        # 2026-10-23：进度 trace 的收尾行**放在 finally**里（幂等）。
        # 之前只写在 try 的成功路径上 ⇒ 超时/异常局不会写 terminal 行，文件看起来"没结束"。
        try:
            if _TRACE is not None:
                _TRACE.terminal(state.winner, [int(b.hp) for b in state.bases], rounds,
                                coins=[int(c) for c in state.coins])
        except Exception:
            pass
        if ents_fh is not None:
            try:
                ents_fh.flush()
                ents_fh.close()
            except Exception:  # noqa: BLE001
                pass
        if raw_fh is not None:
            try:
                raw_fh.flush()
                raw_fh.close()
            except Exception:  # noqa: BLE001
                pass
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


USAGE = """\
用法: python code/test_match/match_via_sdk.py [选项] <seed> [<seed>r ...]

  seed        整数 = 该图；后缀 `r` = 第二个 AI(AI1) 执先手（如 11r）

对局规则相关（2026-10-27 新增）:
  --cold-illegal     回到**旧行为**：非法操作被引擎**静默跳过**（cold_handle_rule_illegal=True）。
                     默认**关** ⇒ 与官方平台判题口径一致：一旦某方发出非法操作，按官方
                     `set_AI_state_IO` 语义**立刻终局、肇事方判负**（RESULT 行标
                     `verdict=illegal_strict`，且不计入正常胜负）。
                     等价环境变量：AZAI_COLD_ILLEGAL=1（用于不方便加参数的老脚本/梯子）。
  ⚠️ 无论哪种模式，每局都会逐条 `can_apply` 校验并在 **stderr** 打
     `ILLEGAL seed=.. round=.. side=.. op=(t,a0,a1) reason=..`（每次非法一行）与
     收尾一行 `illegal-summary ... illegal_total=N ...`；stdout 绝不写这些（二进制协议通道）。

其他选项:
  --rounds=N         回合上限（默认 MAX_ROUND）
  --ai0=/--ai1=      AI 可执行文件或 .py 桥接（默认 = 冠军 magica / 亚军 lure_v4）
  --ai0-args=/--ai1-args=   给该 AI 的额外命令行参数（shell 分词）
  --ai0-env=/--ai1-env=     给该 AI 的环境变量，`K=V;K2=V2`
  --ai0-cwd=/--ai1-cwd=     该 AI 的工作目录
  --trace-ops        逐回合打印双方原始操作文本与解析结果
  --dump-round=R[:path]     在第 R 回合存快照（对拍用）
  -h/--help          本帮助
"""


def main() -> int:
    global AI0_EXE, AI1_EXE
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print(USAGE, flush=True)
        return 0
    # 种子格式：整数 或 整数+r（如 11r 表示第二个 AI(AI1) 执先手）
    # 另有 --rounds=N（回合上限）、--ai0=路径、--ai1=路径（默认 = 冠军/亚军）
    seeds: list[tuple[int, bool]] = []
    max_rounds = MAX_ROUND
    for arg in sys.argv[1:]:
        if arg.startswith("--rounds="):
            max_rounds = int(arg.split("=")[1])
        elif arg == "--cold-illegal":
            pass          # 已在模块级读进 COLD_ILLEGAL（见文件头），这里只是"合法的已知参数"
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
            # 2026-10-23：带上**原始 token** —— 进度文件名必须用它。
            #   否则 7 与 7r 都解析成 seed=7 ⇒ 两局写同一个进度文件（实测行被切断）。
            seeds.append((int(arg), False, arg))
        elif arg.endswith("r") and arg[:-1].isdigit():
            seeds.append((int(arg[:-1]), True, arg))
    if not seeds:
        seeds = [(7, False, "7")]
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
    # 规则口径必须显式可见（2026-10-27）：默认 False = 官方平台判题口径（非法操作 ⇒ 判负）。
    print(f"[bridge] cold_illegal={COLD_ILLEGAL} "
          f"({'旧行为: 非法操作静默跳过' if COLD_ILLEGAL else '严格模式=官方默认: 非法操作判负'})",
          flush=True)
    print(f"[bridge] max_rounds={max_rounds}  seeds={seeds}", flush=True)

    wins: dict[str, int] = {}
    for _seed_tuple in seeds:
        seed, runnerup_first = _seed_tuple[0], _seed_tuple[1]
        _seed_token = _seed_tuple[2] if len(_seed_tuple) > 2 else str(seed)
        globals()["_SEED_TOKEN"] = _seed_token
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
            _et = r.get("exception_type") or (str(r.get("exception", "")).split(":", 1)[0] or "?")
            # 「太慢」与「无效」分开记：超时 ⇒ verdict=timeout（下游按慢归因，不并入坏）
            _verdict = "timeout" if r.get("timed_out") else "aborted"
            # ⚠️ 字段必须**补齐**：下游 `_tmp_ladder.RESULT_RE` 要求 `engine_winner=` 存在，
            #    以前这一行没有它 ⇒ **无效局永远解析不出来**（分析器只看到「winner 缺失/rounds=0」），
            #    把「第 55 回合超时」这类关键信息埋掉了（2026-10-06 发现）。
            # ⚠️ `exc=` 必须放在**行尾**：下游 RESULT_RE 要求 `engine_winner=... rounds=` 相邻，
            #    夹在中间会让整行解析失败（2026-10-06 第二次踩到）。
            # 2026-10-27：`illegal=N` 加在 coins 之后、exc 之前（RESULT_RE 只认到 coins，末尾追加安全）。
            print(f"  RESULT seed={seed} p0={r.get('p0_label', 'p0')} p1={r.get('p1_label', 'p1')} "
                  f"winner=INVALID winner_side=None verdict={_verdict} engine_winner=None "
                  f"rounds={r.get('rounds')} terminal={bool(r.get('terminal'))} "
                  f"base_hp={_hp[0]},{_hp[1]} coins={_cn[0]},{_cn[1]} "
                  f"illegal={int(r.get('illegal_total') or 0)} exc={_et}", flush=True)
            continue

        winner = r.get("winner")           # 引擎自己的 winner；None = 还没判
        hp = r.get("base_hp") or [0, 0]
        p0_lab = r.get("p0_label", "p0")
        p1_lab = r.get("p1_label", "p1")
        # ⚠️ 判词来源必须标出来。引擎在终局时已算好完整级联（HP→击杀数→超武次数少者→总时间少者→判 P0），
        # 我们只在"循环提前停下、引擎还没判"时退回"只比 HP"——那条**不是官方规则**（官方也从不判平局）。
        if r.get("strict_illegal_side") is not None:
            # 严格模式判负（2026-10-27）：胜负由**规则**决定而不是由引擎终局条件决定，
            # 所以单独一个 verdict ⇒ 下游"有效性门槛(verdict=engine)"会把它排除，不会被当成战果。
            verdict = "illegal_strict"
            _s = int(r["strict_illegal_side"])
            winner_side = "p0" if _s == 1 else "p1"
            winner_ai = p0_lab if _s == 1 else p1_lab
        elif winner is not None and winner in (0, 1):
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
              f"coins={(r.get('coins') or [0, 0])[0]},{(r.get('coins') or [0, 0])[1]} "
              f"illegal={int(r.get('illegal_total') or 0)}", flush=True)

    print(f"\n{'='*70}\n汇总: {json.dumps(wins, ensure_ascii=False)}\n{'='*70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
