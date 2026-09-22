"""把我们自己的 AI（az_intent 的 Bundle-MCTS）接进 bridge 的协议外壳。

背景（`docs/champ_ladder_plan.md` §3.2 / §7.6）：我们**从来没有跟冠军/亚军打过**——
我们的 AI 是 in-process 的，不会讲官方 stdin/stdout 协议。`code/test_match/rv4_pkg/`
是 rule_v4 的壳（官方 ProtocolSession），本文件是同一件事，但把决策换成我们的搜索。

**设计要点（关键）**：
  * **影子环境就是 bridge 的裁判本身**：`GameStateFacade.initial(seed=...)`。
    只要按 bridge 的确切顺序重放（init → 每回合 p0 先动 → 收对手操作 → advance），
    两边状态**按构造同步**，不需要从文本反猜局面。
  * **每回合用收到的局面文本对拍**（`state_to_text(facade)` vs 收到的字节），
    分歧**检出并计数**——而不是像当年两个 C++ AI 那样靠内嵌模拟悄悄分歧、然后静默弃权
    （见 `memory/project_champion_runnerup_cpp_match.md`）。
  * 文本序列化**从 match_via_sdk 导入**，绝不在这里重打：重打就等于对拍失效。

**协议**（由 `rv4_pkg/protocol.py` 核准，它是能跑通的 known-good control）：
  * AI 收：**裸 UTF-8 文本行**。init = `"<player> <seed>\\n"`；
    之后每回合：局面文本（结构化行） + 对手操作文本（首行是操作数量）。
  * AI 发：`struct.pack(">I", len(data)) + data`（4 字节大端长度前缀）。

**bridge 的确切时序**（`match_via_sdk.run_match` 的循环）：
  p0 先写本回合操作 → p1 收到 → p1 写 → p0 收到 p1 的操作 → 双方操作各自 apply
  → `advance_round()` → 把**推进后**的局面文本发给双方。

**配置走环境变量**（bridge 用 `Popen([sys.executable, path])` 起 .py AI，不传参数）：
  AZAI_CKPT=<path>          必填
  AZAI_ITERS AZAI_DEPTH AZAI_TCLASS AZAI_TPOS AZAI_K AZAI_TEMP
  AZAI_MODE(joint|class-only|pos-only) AZAI_POSPIN(argmax|playable)
  AZAI_SKIP1(0/1) AZAI_VALUE_TANH(0/1) AZAI_REL2ABS(float) AZAI_TRACE(0/1) AZAI_DEVICE(cpu|auto)
"""
from __future__ import annotations

import importlib.util
import os
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ⚠️ 协议流保护：stdout 是"长度前缀"的二进制通道，任何漏出去的 print（torch / SDK / warning）
# 都会把后续所有包的长度前缀冲掉 ⇒ 直接腐坏整局。所以先把真正的 stdout 存下来，
# 再把 sys.stdout 指到 stderr：漏出的输出全部去 stderr（bridge 会收进日志）。
_PROTO_OUT = sys.stdout.buffer
sys.stdout = sys.stderr


def _load_bridge_module():
    """加载 match_via_sdk，拿到**同一份** state_to_text/ops_to_text/parse_ops_text/GameState。

    必须用它、不能重写：对拍要求逐字节一致，而"重打一遍等价逻辑"正是记忆里点名过的坑
    （feedback_known_good_control_for_harness 第四条）。
    """
    spec = importlib.util.spec_from_file_location("mvs_for_azai", _HERE / "match_via_sdk.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MVS = _load_bridge_module()
GameState = MVS.GameState
state_to_text = MVS.state_to_text
ops_to_text = MVS.ops_to_text
parse_ops_text = MVS.parse_ops_text


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _log(msg: str) -> None:
    sys.stderr.write(f"[azai] {msg}\n")
    sys.stderr.flush()


# 统计（模块级：收尾报告要在被 bridge kill 之前打出来）
STATS = {"rounds": 0, "mismatch": 0, "illegal": 0}


def _report(*_args) -> None:
    """收尾报告。bridge 在 finally 里会 kill 我们，所以 atexit/SIGTERM 都要挂上，
    否则"对拍次数"这个**状态同步的证据**会丢。"""
    _log(f"退出报告: rounds={STATS['rounds']} mismatch={STATS['mismatch']} "
         f"illegal={STATS['illegal']}")
    sys.stderr.flush()


def _install_reporters() -> None:
    import atexit
    import signal
    atexit.register(_report)
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, lambda *_a: (_report(), sys.exit(0)))
        except (ValueError, OSError):
            pass


class ProtocolIO:
    """AI 侧协议：收裸文本行、发 4 字节长度前缀的包。"""

    def __init__(self) -> None:
        self._in = sys.stdin.buffer
        self._out = _PROTO_OUT

    def recv_line(self) -> str | None:
        raw = self._in.readline()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")

    def recv_ops(self) -> list | None:
        """读一段操作文本（首行数量 + 若干行 token）。EOF 返回 None。"""
        head = self.recv_line()
        if head is None:
            return None
        try:
            count = int(head.strip())
        except ValueError:
            _log(f"!! 操作文本首行不是数量: {head!r}")
            return []
        lines = [head]
        for _ in range(count):
            ln = self.recv_line()
            if ln is None:
                return None
            lines.append(ln)
        return parse_ops_text("\n".join(lines))

    def send_ops(self, ops: list) -> None:
        data = ops_to_text(ops).encode("utf-8")
        self._out.write(struct.pack(">I", len(data)))
        self._out.write(data)
        self._out.flush()


def build_engine(seed: int, player: int):
    """按环境变量构造 (mcts, net_fn, model, feat)。"""
    import numpy as np  # noqa: F401  (net_fn 的调用方需要)
    import torch

    torch.set_num_threads(int(_env("AZAI_TORCH_THREADS", "1")))
    ckpt = _env("AZAI_CKPT", "")
    if not ckpt:
        raise RuntimeError("必须设 AZAI_CKPT 环境变量指向 checkpoint")
    # ⚠️ bridge 用 cwd=exe.parent（= code/test_match）起我们，所以相对路径要按仓库根解析
    _c = Path(ckpt)
    ckpt = str(_c if _c.is_absolute() else (Path(MVS.REPO_ROOT) / _c).resolve())

    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    model, net_fn = make_net_fn_from_ckpt(
        ckpt, feat,
        value_tanh=bool(int(_env("AZAI_VALUE_TANH", "1"))),
        value_rel_to_abs=float(_env("AZAI_REL2ABS", "0")),
    )
    mcts = BundleMCTS(
        net_fn,
        iterations=int(_env("AZAI_ITERS", "256")),
        max_depth_rounds=int(_env("AZAI_DEPTH", "4")),
        k=int(_env("AZAI_K", "1")),
        sample_mult=int(_env("AZAI_SAMPLE_MULT", "1")),
        t_class=float(_env("AZAI_TCLASS", "0.5")),
        t_pos=float(_env("AZAI_TPOS", "1.0")),
        c_puct=float(_env("AZAI_C_PUCT", "1.25")),
        seed=seed,
        search_mode=_env("AZAI_MODE", "joint"),
        pos_pin=_env("AZAI_POSPIN", "argmax"),
        skip_single_candidate=bool(int(_env("AZAI_SKIP1", "0"))),
    )
    _log(f"engine built: ckpt={ckpt} iters={_env('AZAI_ITERS','256')} "
         f"depth={_env('AZAI_DEPTH','4')} mode={_env('AZAI_MODE','joint')} "
         f"pospin={_env('AZAI_POSPIN','argmax')} t_class={_env('AZAI_TCLASS','0.5')} "
         f"t_pos={_env('AZAI_TPOS','1.0')} temp={_env('AZAI_TEMP','1e-6')}")
    return mcts, model, feat


def main() -> int:
    _install_reporters()
    io = ProtocolIO()
    trace = bool(int(_env("AZAI_TRACE", "1")))
    verify = bool(int(_env("AZAI_VERIFY", "1")))
    temperature = float(_env("AZAI_TEMP", "1e-6"))

    init = io.recv_line()
    if init is None:
        _log("!! 没有收到 init 行就 EOF 了")
        return 1
    player, seed = (int(x) for x in init.split())
    _log(f"init: player={player} seed={seed} pid={os.getpid()}")

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    mcts, model, feat = build_engine(seed, player)

    facade = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    def decide() -> list:
        res = mcts.search(facade, player, temperature=temperature)
        chosen = res.chosen_bundle or ()
        return [Operation(OperationType(int(k[0])), int(k[1]), int(k[2])) for k in chosen]

    def _apply(pl: int, ops: list) -> None:
        """把操作打进影子状态，并统计非法数（影子与裁判的分歧会在这里先露头）。"""
        bad = facade.apply_operation_list(pl, ops)
        n = len(bad) if bad else 0
        if n:
            STATS["illegal"] += n
            _log(f"!! 第 {STATS['rounds']} 回合 p{pl} 有 {n} 个非法操作: {bad}")

    try:
        while not facade.terminal:
            if player == 0:
                self_ops = decide()
                io.send_ops(self_ops)
                _apply(0, self_ops)
                opp = io.recv_ops()
                if opp is None:
                    break
                _apply(1, opp)
            else:
                opp = io.recv_ops()
                if opp is None:
                    break
                _apply(0, opp)
                self_ops = decide()
                io.send_ops(self_ops)
                _apply(1, self_ops)

            # ---- 本回合结束：收"推进后"的局面文本，并和影子状态对拍 ----
            facade.advance_round()
            expected = state_to_text(facade)
            n_expect = len(expected.split("\n")) - 1   # 末尾的 "" 不算一行
            got: list[str] = []
            eof = False
            for _ in range(n_expect):
                ln = io.recv_line()
                if ln is None:
                    eof = True
                    break
                got.append(ln)
            if eof:
                break
            got_text = "\n".join(got) + "\n"
            if verify and got_text != expected:
                STATS["mismatch"] += 1
                if trace or STATS["mismatch"] <= 3:
                    _log(f"!! 第 {STATS['rounds']} 回合局面文本对拍不一致"
                         f"（mismatch #{STATS['mismatch']}）")
                    gl, el = got_text.split("\n"), expected.split("\n")
                    for i in range(max(len(gl), len(el))):
                        a = gl[i] if i < len(gl) else "<缺>"
                        b = el[i] if i < len(el) else "<缺>"
                        if a != b:
                            _log(f"   line{i}: 收到={a!r}  影子={b!r}")
            STATS["rounds"] += 1
            if trace:
                # 每回合一行轨迹（bridge 结束时会硬杀我们，收尾报告打不出来，
                # 所以累计计数必须随每回合落盘；最后一行就是终局状态）。
                _log(f"round={STATS['rounds']} hp={[b.hp for b in facade.bases]} "
                     f"coins={list(facade.coins)} towers={len(facade.towers)} "
                     f"mismatch={STATS['mismatch']} illegal={STATS['illegal']}")
    except BrokenPipeError:
        _log("bridge 关掉了管道（正常收尾）")
    except Exception as exc:  # noqa: BLE001
        import traceback
        _log(f"!! EXC {type(exc).__name__}: {exc}")
        sys.stderr.write(traceback.format_exc())
        return 1

    _log(f"结束: rounds={STATS['rounds']} terminal={facade.terminal} winner={facade.winner} "
         f"hp={[b.hp for b in facade.bases]} mismatch={STATS['mismatch']} "
         f"illegal={STATS['illegal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
