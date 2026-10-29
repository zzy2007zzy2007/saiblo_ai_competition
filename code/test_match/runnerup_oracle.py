"""亚军（cpp_lure_v4）出招查询器 —— 把「裁判↔AI」协议封装成一个可复用会话（2026-10-26）。

## 它解决什么问题

想知道"亚军在这个局面会怎么走"（候选注入 / 蒸馏标签 / 行为对照）⇒ 需要一个**常驻**的
亚军进程，按官方 stdio 协议的**确切顺序**逐回合喂局面。

## 协议的确切顺序（这是本文件的全部要害）

亚军是 C++ SDK，主循环见
`其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/ai_cpp_lure_v4.cpp:169-189`：

```
player == 0:  perform_self_turn → receive_opponent → finalize → sync_round
player == 1:  receive_opponent → perform_self_turn → finalize → sync_round
```

两边的**读**顺序其实一样，都是「先对手操作块、后局面文本」：

```
读 <对手操作块> → 写 <自己的操作> → 读 <下一回合的局面文本>
```

所以 P1 的 stdin 序列是（第 0 回合**只有一条**消息）：

```
"1 <seed>\\n"        ① init
<对手操作块>          ② 第 0 回合 —— 此时它**还没有任何局面**
[写自己的操作]        ③ 4 字节大端长度前缀包
<局面文本 r1>         ④ 第 1 回合开始时的局面
<对手操作块>          ⑤ 第 1 回合
[写自己的操作]
<局面文本 r2>
...
```

**第 0 回合没有局面文本**：协议规定 player 1 先读 player 0 的出招，此时它还没收到任何局面
（`docs/protocol_ai_spec.md:32-33`、`其他版本ai/.../ai_cpp_protocol.py:132-171`）。
旧探针 `probe_runnerup_oracle.py` 正是错在这里——它在第 0 回合先发了一条局面文本，
这条文本的首行"0"（回合号）被 AI 当成「对手操作块的操作数」吃掉了，于是整条流从第 0 回合起
就错位一整条消息；第 1 回合我们发的局面文本被当成上一回合还没读完的蚂蚁行，撞到
`sdk.hpp:1137` 的 `invalid ant row`。逐行推演见 `docs/runnerup_oracle_protocol.md`。

## 用法（P1：一手 ask 一个回合）

```python
oracle = RunnerUpOracle(seed=10, player=1)      # 起进程 + 发 init
st = GameStateFacade.initial(seed=10, cold_handle_rule_illegal=True)
for _ in range(N):
    ops = oracle.ask(st)                        # r0：它按 seed 自建局面出招
    oracle.set_opponent_ops(st.round_index, opponent_ops_of_this_round)   # 可选
    st.apply_operation_list(1, ops); st.advance_round()
oracle.close()
```

`player=0` 也支持，但注意 P0 **先出招、后收对手行**（官方协议如此），所以它的对手操作块
是「上一回合」的：在 `ask` 之后再 `set_opponent_ops(round, ops)`，oracle 会在下一次 `ask`
时投递。两边的调用方式因此完全一样。

⚠️ 必须**从 round 0 开始、逐回合连续**调用 `ask`（`state.round_index` 必须严格 +1）。
跳回合会让 AI 的读端错位——这不是可恢复错误，`ask` 会直接抛 `OracleDesync`。
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (REPO_ROOT / "code", REPO_ROOT / "code" / "cpp_engine", REPO_ROOT / "Ant-Game",
           REPO_ROOT / "Ant-Game" / "SDK", REPO_ROOT / "code" / "test_match"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ⚠️ 照抄桥接的启动方式（DLL 目录 + env + cwd + init 行），不自己 Popen：
#    亚军 exe 旁边需要 libstdc++-6.dll / libgcc_s_seh-1.dll / libwinpthread-1.dll，
#    缺了会在 0xC0000139 直接死（见 docs/runnerup_build_note.md）。
from test_match.match_via_sdk import (  # noqa: E402
    RUNNERUP_EXE,
    ops_to_text,
    parse_ops_text,
    send_line,
    start_ai,
    state_to_text,
)

# 一个包的最大合法长度。超过它 = 流已经错位（把局面文本当长度前缀读了），
# 立刻报错比等超时好：正常回包只有几十字节。
MAX_PACKET_BYTES = 1 << 20


class OracleError(RuntimeError):
    """查询器的所有失败的基类。"""


class OracleTimeout(OracleError):
    """在 timeout 秒内没读到包（进程卡住 / 被 CPU 抢死）。会话已不可继续。"""


class OracleDesync(OracleError):
    """调用方跳了回合、或上一回合失败 ⇒ 读端已错位，这个会话不能再用。"""


class OracleDead(OracleError):
    """亚军进程已经退出（EOF）。"""


class RunnerUpOracle:
    """常驻的亚军查询会话。

    参数
    ----
    seed : 与裁判一致的对局 seed（**必须**一致：第 0 回合它按 seed 自己重建初始局面）。
    player : 我们查询的那一方是 0 还是 1。它决定「自己的操作」与「对手操作块」谁先谁后。
    exe : 亚军可执行文件，默认 = 桥接用的那个 cpp_lure_v4。
    timeout : 单次读包超时（秒）。超时 ⇒ 抛 OracleTimeout 且会话作废，绝不无限阻塞。
              默认 5.0：实测**它自己的搜索**早期 ~0.1s，自对弈打到第 90 回合时单侧 ~2.0s
              （`match_results/selfplay_90.log`），所以 2.0 在长局会误杀；给 2 倍余量。
    stderr_path : 亚军 stderr（`{"kind":"decision",...}` 日志）落到哪个文件。
    verbose : 往**我们的 stderr** 打每回合收发摘要（stdout 留给调用方）。
    """

    def __init__(self, seed: int, player: int = 1, exe: str | None = None,
                 timeout: float = 5.0, stderr_path: str | os.PathLike | None = None,
                 verbose: bool = False, keep_history: bool = True,
                 extra_env: dict | None = None):
        self.seed = int(seed)
        self.player = int(player)
        if self.player not in (0, 1):
            raise ValueError(f"player 只能是 0 或 1，收到 {player}")
        self.timeout = float(timeout)
        self.verbose = bool(verbose)
        self.exe = Path(exe) if exe else Path(RUNNERUP_EXE)
        # 2026-10-29：**重建 + 回放追赶**（`docs/us_vs_champion_20261029.md` §4.2.1 的坑）。
        #   `keep_history=True` ⇒ 每次成功 ask 把"真正发出去的那几个帧"存下来（状态文本 + 对手操作块），
        #   会话死掉时新进程可以从 round 0 把这些帧原样重放一遍，追上当前回合。
        #   代价：每回合多存 ~1-5 KB 字符串（512 回合 ≈ 1-2 MB/会话）⇒ 默认开，但可关。
        self.keep_history = bool(keep_history)
        self.history: list[tuple[int, tuple[tuple[str, str], ...]]] = []   # (回合, ((标签, 文本), ...))
        self._next_round = 0           # 下一个 ask 期望的 state.round_index
        self._closed = False
        self._dead: str | None = None
        self._opp_ops: dict[int, str] = {}    # 回合号 -> 对手操作块原文
        self.stats = {"asks": 0, "ok": 0, "timeouts": 0, "total_ms": 0.0, "last_ms": 0.0,
                      "replayed": 0, "replay_s": 0.0}


        if stderr_path is None:
            log_dir = REPO_ROOT / "match_results" / "oracle_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stderr_path = log_dir / f"oracle_seed{self.seed}_p{self.player}_{os.getpid()}.stderr.log"
        self.stderr_path = Path(stderr_path)
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)

        # 与桥接完全相同的 env（ANTGAME_CPP_BASELINE_DEBUG 改它的 stderr 详细度，
        # 默认 summary = 真实对局用的值；MVS_RUNNERUP_DEBUG=plans 可拿候选+打分）
        env = {"ANTGAME_CPP_BASELINE_DEBUG": os.environ.get("MVS_RUNNERUP_DEBUG", "summary")}
        if extra_env:
            # 2026-10-29：给"多套配置"场景用（同一个 exe 用不同 AZAI_* 跑两个会话）。
            env.update({str(k): str(v) for k, v in extra_env.items()})
        self._proc, self._io = start_ai(
            self.exe, self.player, self.seed, self.stderr_path,
            extra_env=env, strip_cr=True, cwd=None,
        )
        if self.verbose:
            self._log(f"启动 {self.exe.name} player={self.player} seed={self.seed} "
                      f"pid={self._proc.pid} timeout={self.timeout}s stderr={self.stderr_path}")

    # ── 生命周期 ────────────────────────────────────────────────────────────
    def __enter__(self) -> "RunnerUpOracle":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        """关 stdin（让它读到 EOF 正常退出）→ terminate → kill。幂等。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self.verbose:
            self._log(f"关闭 pid={self._proc.pid} 统计={self.stats}")

    # ── 对手操作块 ──────────────────────────────────────────────────────────
    def set_opponent_ops(self, round_index: int, operations) -> None:
        """登记「对手在第 round_index 回合的操作」（str 原文 或 Operation 列表）。

        P1：在 `ask(round_index)` **之前**登记（那一回合它就要读）。
        P0：在 `ask(round_index)` **之后**登记（下一回合开头它才读）。
        没登记的回合一律按「对手空过」处理（等价于发 `ops_to_text([])` = `"0\\n"`）。
        """
        if isinstance(operations, str):
            text = operations
        else:
            text = ops_to_text(list(operations))
        if not text.endswith("\n"):
            text += "\n"
        self._opp_ops[int(round_index)] = text

    def _opp_text(self, round_index: int) -> str:
        return self._opp_ops.get(int(round_index)) or ops_to_text([])

    # ── 核心：问一回合 ──────────────────────────────────────────────────────
    def ask(self, state, opponent_ops: str | None = None) -> list:
        """喂这个局面，返回亚军本回合的操作（`SDK.backend.model.Operation` 列表）。

        state : `GameStateFacade`（`state_to_text` 只需要 round_index / towers / ants /
                coins / bases / weapon_cooldowns / active_effects 这些属性）。
        opponent_ops : 只给 P1 的便利参数（= 对手**本回合**的操作块）；P0 请用
                `set_opponent_ops`（因为 P0 先出招后收对手行，语义是上一回合）。

        失败一律抛 OracleError 子类（会话随之作废），**不会**阻塞调用方。
        """
        self._ensure_alive()
        rnd = int(state.round_index)
        if rnd != self._next_round:
            raise OracleDesync(
                f"必须从 round 0 起连续喂：期望 round_index={self._next_round}，收到 {rnd}"
                f"（跳回合会让亚军读端错位，无法恢复）")
        if opponent_ops is not None:
            if self.player != 1:
                raise ValueError("player=0 请用 set_opponent_ops(round, ops) 登记对手操作块")
            self.set_opponent_ops(rnd, opponent_ops)

        t0 = time.monotonic()
        # 2026-10-29：先把"本回合要发的那几个帧"算出来（顺序 = 协议的确切读序），
        # 再逐个发出去 —— 这样 `history` 里存的就是**逐字发出去的原文**，重建后能原样重放。
        frames: list[tuple[str, str]] = []
        if self.player == 1:
            # 读序：对手操作块(N) → 自己写 → 局面(N+1)
            if rnd >= 1:
                frames.append((f"state(r{rnd})", state_to_text(state)))
            frames.append((f"opp_ops(r{rnd})", self._opp_text(rnd)))
        else:
            # 读序（P0 也一样是「先对手后局面」）：对手操作块(N-1) → 局面(N) → 自己写
            if rnd >= 1:
                frames.append((f"opp_ops(r{rnd - 1})", self._opp_text(rnd - 1)))
                frames.append((f"state(r{rnd})", state_to_text(state)))
        try:
            for what, text in frames:
                self._send(text, what)
            payload = self._read_packet()
        except OracleError:
            self.stats["asks"] += 1
            self._dead = "上一回合失败，读端已错位"
            raise

        self.stats["asks"] += 1
        self.stats["last_ms"] = (time.monotonic() - t0) * 1000.0
        self.stats["total_ms"] += self.stats["last_ms"]
        ops = parse_ops_text(payload.decode("utf-8", errors="replace"))
        self.stats["ok"] += 1
        self._next_round = rnd + 1
        if self.keep_history:
            # 只记"成功消费掉的"回合：失败那一回合的帧没被它读走，记进来会污染回放。
            self.history.append((rnd, tuple(frames)))
        if self.verbose:
            self._log(f"r{rnd} {self.stats['last_ms']:.1f}ms -> {len(ops)} ops "
                      f"{[(int(o.op_type), o.arg0, o.arg1) for o in ops]}")
        return ops

    # ── 重建 + 回放追赶（2026-10-29）────────────────────────────────────────
    def replay(self, history=None, upto_round: int | None = None,
               deadline_s: float | None = None) -> tuple[int, float]:
        """把 `history` 里 **0..upto_round-1** 回合的帧逐字重放进**本会话**（新进程），丢弃它的输出。

        为什么只能重放不能"直接告诉它现在几回合"：协议要求从 round 0 起连续喂（跳回合 = 读端错位），
        而且它的内部影子局面是靠**逐回合推进**攒出来的，没有"从当前局面初始化"的入口。

        返回 `(重放的回合数, 墙钟秒)`；中途失败抛 `OracleError`（会话照样作废，由调用方决定是否再试）。
        `deadline_s` 给回放本身一个墙钟上限（冠军 O0 一局 512 回合回放 ~40s，给 180s 足够）。
        """
        hist = self.history if history is None else history
        t0 = time.monotonic()
        n = 0
        for rnd, frames in hist:
            if upto_round is not None and int(rnd) >= int(upto_round):
                break
            if deadline_s is not None and (time.monotonic() - t0) > float(deadline_s):
                raise OracleTimeout(f"回放超过 {deadline_s:g}s（已回放 {n} 回合）")
            for what, text in frames:
                self._send(text, f"replay.{what}")
            self._read_packet()                      # ⇐ 丢弃它的答案（回放阶段不需要）
            self._next_round = int(rnd) + 1
            if self.keep_history:
                self.history.append((int(rnd), frames))
            n += 1
        dt = time.monotonic() - t0
        self.stats["replayed"] += n
        self.stats["replay_s"] += dt
        if self.verbose:
            self._log(f"回放 {n} 回合完成（{dt:.1f}s，next_round={self._next_round}）")
        return n, dt


    def try_ask(self, state, opponent_ops: str | None = None):
        """失败返回 None 而不是抛（会话照样作废）。"""
        try:
            return self.ask(state, opponent_ops)
        except OracleError as exc:
            print(f"[oracle] try_ask 失败: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return None

    # ── 内部 ────────────────────────────────────────────────────────────────
    def _send(self, text: str, what: str) -> None:
        if self.verbose:
            head = text.split("\n", 1)[0]
            self._log(f"→ {what}: {len(text.encode('utf-8'))}B 首行={head!r}")
        try:
            send_line(self._io, text)
        except Exception as exc:  # noqa: BLE001  BrokenPipe 等
            raise OracleDead(f"写 {what} 失败（进程可能已退出）：{type(exc).__name__}: {exc}") from exc

    def _read_packet(self) -> bytes:
        """4 字节大端长度前缀 + payload（`sdk.hpp:1216-1239` 的对称实现）。"""
        header = self._read(4, "长度前缀")
        size = struct.unpack(">I", header)[0]
        if size == 0:
            return b""
        if size > MAX_PACKET_BYTES:
            raise OracleDesync(f"包长度 {size}B 不合理（流已错位），前缀={header.hex()}")
        return self._read(size, f"payload({size}B)")

    def _read(self, n: int, what: str) -> bytes:
        try:
            return self._io.read(n, timeout=self.timeout)
        except TimeoutError as exc:
            self.stats["timeouts"] += 1
            raise OracleTimeout(f"读{what} 超过 {self.timeout}s：{exc}") from exc
        except EOFError as exc:
            raise OracleDead(f"读{what} 时亚军已退出：{exc}") from exc

    def _ensure_alive(self) -> None:
        if self._closed:
            raise OracleDead("会话已 close()")
        if self._dead:
            raise OracleDesync(f"会话已作废：{self._dead}")
        if self._proc.poll() is not None:
            raise OracleDead(f"亚军进程已退出 (exit code {self._proc.poll()})，"
                             f"stderr 见 {self.stderr_path}")

    def _log(self, message: str) -> None:
        print(f"[oracle] {message}", file=sys.stderr, flush=True)

    def __repr__(self) -> str:
        return (f"<RunnerUpOracle seed={self.seed} player={self.player} "
                f"next_round={self._next_round} stats={self.stats}>")


def rebuild_session(old: "RunnerUpOracle", upto_round: int | None = None,
                    stderr_path: str | os.PathLike | None = None,
                    deadline_s: float = 180.0) -> tuple["RunnerUpOracle", int, float]:
    """**会话重建 + 回放追赶**：起一个新进程，把 `old` 记下的 0..upto_round-1 历史原样喂给它。

    返回 `(新会话, 回放的回合数, 回放墙钟秒)`。失败时抛 `OracleError`（新进程由调用方关掉）。

    ⚠️ 2026-10-29 实测（`_champ_oracle_rebuild_probe.py`）：**冠军可以回放到它崩的那一回合之前**，
    但一到崩点重放又会崩（`apply_operations.called_after_terminal` 由它影子模型里的"终局"决定，
    与进程活了多久无关）⇒ 这条路对冠军只能救"崩点之前"的会话。细则见
    `docs/e4_only_extra_20261029.md` §1。`stderr_path=None` ⇒ 默认名含 pid ⇒ 每次重建一个新文件
    （**旧会话的崩溃日志不会被覆盖**，这是有意为之）。
    """
    new = RunnerUpOracle(seed=old.seed, player=old.player, exe=str(old.exe),
                         timeout=old.timeout, stderr_path=stderr_path,
                         verbose=old.verbose, keep_history=old.keep_history)
    new._opp_ops.update(old._opp_ops)      # P0 的 ask(r) 要 r-1 的块（回放后继续用）
    n, dt = new.replay(old.history, upto_round=upto_round, deadline_s=deadline_s)
    return new, n, dt


def main() -> int:
    """自测：连喂几个回合，打印它的出招（对手默认空过）。"""
    from my_ai.az_intent.game_state_facade import GameStateFacade

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
    with RunnerUpOracle(seed=seed, player=1, verbose=True) as oracle:
        for _ in range(rounds):
            if st.terminal:
                break
            ops = oracle.ask(st)
            print(f"r{st.round_index} ops={[(int(o.op_type), o.arg0, o.arg1) for o in ops]}",
                  flush=True)
            st.apply_operation_list(1, ops)
            st.apply_operation_list(0, [])
            st.advance_round()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
