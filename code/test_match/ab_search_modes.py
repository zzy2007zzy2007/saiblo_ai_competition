"""A/B：**扁平 rollout 搜索 vs 原版 M6c（BundleMCTS）**，同一 ckpt、同一候选生成。

用法：
    python code/test_match/ab_search_modes.py --seeds 7 8 9 10 --jobs 4 --out-dir training_history/ab_flat

设计：
  · 两个 AI 都是 `az_bridge_ai.py`（部署桥接），只有 `AZAI_SEARCH` 不同：
        flat 侧 = AZAI_SEARCH=flat + AZAI_FLAT_*
        mcts 侧 = 默认（AZAI_SEARCH=mcts）
  · **交替执先**：seed 偶数 → flat 当 ai0；seed 奇数 → flat 当 ai1（抵消先手偏差）
  · 复用 `_tmp_ladder.read_game` 解析胜负（和既有实验同一套判据）

────────────────────────────────────────────────────────────────────────────
2026-10-28：**异常局自动重试 + 卡死看门狗 + 断电断点续跑**（docs/match_retry_resume_20261028.md）

    # 一晚上跑 32 局，卡死 90min 就看门狗杀掉，跑完自动把无效局重试 2 轮；
    # 第二天断电重启后同一条命令加 --resume 只补没打完的那些。
    python code/test_match/ab_search_modes.py --seeds 7-70 --jobs 8 --paired-sides \
        --out-dir match_results/rng_ab_ctl/efs4303 \
        --resume --retry-invalid 2 --game-timeout-s 5400

  · `--retry-invalid N`：一批跑完后扫 out-dir，**凡是没有 `verdict=engine` 的**（口径抄
    `analyze_paired.py` §4）重跑，最多 N 轮；**只重跑失败的**，成功局的日志绝不覆盖
    （重跑前还会把上一轮失败日志复制到 `<out-dir>/_attempts/` 留证据）。
  · `--game-timeout-s S`：单局墙钟超过 S 秒 ⇒ 用 Popen 句柄拿到的那一个 PID 精确 kill
    （Windows `taskkill /PID <pid> /T /F`），记 `timeout_watchdog`，再按上面的规则重试。
    ⚠️ **绝不按镜像名/通配符批量杀**（2026-10 刚因为误杀别人的进程踩过）。
  · `--resume`：启动时跳过已有 `verdict=engine` 的 seed/side，只跑缺失/无效的；
    并把旧结果**采纳**进本轮汇总（否则续跑那轮的胜率只覆盖重跑的几局，读数会骗人）。
  · `<out-dir>/_progress.json`：每局结束就更新（`.tmp` + `os.replace` 原子写），
    记 status/verdict/ts/attempts。**纯元数据**：判有效一律以日志为准，状态文件只是一份
    断电后能看出"跑到哪了"的线索（所以删日志后 `--resume` 仍会重跑那一局）。
  · 三个新参数默认全关 ⇒ 行为与改动前一致；全关时 stdout 除了"某局无效"的告警行之外
    与旧版一致（不打印重试/续跑/有效率区块）。
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in [REPO, REPO / "code", REPO / "Ant-Game"]:
    sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location("_tmp_ladder", REPO / "_tmp_ladder.py")
_ladder = importlib.util.module_from_spec(_spec)
sys.modules["_tmp_ladder"] = _ladder
_spec.loader.exec_module(_ladder)

BRIDGE = REPO / "code/test_match/az_bridge_ai.py"
# 部署口径：**逐字照抄 `M6cROOT_128`**（docs/experiment_log.md 行 6332；该臂 = 对 rule_v4 0.7344，
# 并被 docs/experiment_log.md 行 11337 明确标为"M6c-ROOT 基线"）。
#   ⚠️ 注意：不是 `grafted_sdk_A.pt` + MC_EVERY=1 + MC_TOPK=3 + MC_UPG=1 那套（那是后来的嫁接/加宽实验臂）。
DEFAULT_CANON_ENV = (
    "AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;"
    "AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;"
    "AZAI_MC_EVERY=5;AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;"
    "AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off"
)
# flat 侧：同一套配置 + 只把"评估器"换成扁平 rollout
DEFAULT_FLAT_ENV = (DEFAULT_CANON_ENV + ";AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;"
                    "AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000")
DEFAULT_MCTS_ENV = (DEFAULT_CANON_ENV + ";AZAI_SEARCH=mcts")

# 有效性口径（2026-10-28）：照抄 `analyze_paired.py` §4 —— 只有 `verdict=engine` 算有效局。
# 其余（timeout / aborted / illegal_strict / hp_tiebreak / 无 RESULT / 无日志）一律判无效 ⇒ 重跑。
VALID_VERDICT = "engine"


def _log_name(seed: int, flat_side: int | None) -> str:
    """该局日志文件名。`flat_side=None` = 非配对模式（不带座位后缀）。"""
    return f"seed{seed}.log" if flat_side is None else f"seed{seed}_p{flat_side}.log"


def _label(seed: int, flat_side: int | None) -> str:
    """人读标签 = 日志名去掉 .log（`seed900` / `seed900_p0`）。"""
    return _log_name(seed, flat_side)[:-4]


def parse_log(log_path: Path, seed: int) -> dict | None:
    """读日志 → `_tmp_ladder.read_game` 的 dict；读不到（不存在/解析崩）返回 None。

    ⚠️ 解析一律复用 `_tmp_ladder.read_game`，不在这里重写一遍 —— 两处逻辑慢慢分叉正是
       历史事故（"重打判定把败记成平"）的来源。
    """
    try:
        return _ladder.read_game(log_path, str(seed))
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001  解析失败不能把整批带崩
        print(f"[judge] ⚠️ 解析 {log_path.name} 失败: {type(exc).__name__}: {exc}", flush=True)
        return None


class ProgressFile:
    """`<out-dir>/_progress.json`：`{label: {status, verdict, ts, attempts, ...}}`。

    设计原则（抄 `code/test_match/progress.py` 的写法）：
      · **永不抛异常**（进度文件绝不能影响对局）；
      · `.tmp` + `os.replace` 原子替换 ⇒ 断电只会丢掉最后一次更新，不会留下半个 JSON；
      · **不是判据**：判"这局有没有效"永远读日志（用户在验证③里会手动删/改日志）。
    """

    def __init__(self, out_dir: Path) -> None:
        self.path = out_dir / "_progress.json"
        self.tmp = out_dir / "_progress.json.tmp"
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[progress] ⚠️ {self.path.name} 读不动（忽略并重建）: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return
        games = raw.get("games") if isinstance(raw, dict) else None
        if isinstance(games, dict):
            self.data = {str(k): v for k, v in games.items() if isinstance(v, dict)}

    def get(self, label: str) -> dict:
        with self.lock:
            return dict(self.data.get(label) or {})

    def update(self, label: str, **fields) -> None:
        with self.lock:
            rec = dict(self.data.get(label) or {})
            rec.update(fields)
            rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.data[label] = rec
            self._flush_locked()

    def _flush_locked(self) -> None:
        try:
            payload = {
                "schema": 1,
                "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "out_dir": self.path.parent.as_posix(),
                "note": "纯元数据；判有效性一律以日志里的 verdict=engine 为准",
                "games": self.data,
            }
            self.tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                                encoding="utf-8")
            os.replace(self.tmp, self.path)
        except Exception as exc:  # noqa: BLE001
            print(f"[progress] ⚠️ 写 {self.path.name} 失败（不影响对局）: "
                  f"{type(exc).__name__}: {exc}", flush=True)


def kill_game_tree(proc: subprocess.Popen, label: str, why: str) -> None:
    """只杀**这一局**的进程树（Popen 句柄给的 PID + 它的子进程）。

    为什么按 PID 而不是按镜像名/通配符（2026-10-28 用户明确要求）：
      本机同时可能有别的批次在跑（比如 `rng_ab_ctl` 的 8 局）。`taskkill /IM python.exe`
      这类写法会连别人的对局一起杀 —— 我们刚因为这种写法误杀过别人的进程。
      Popen 句柄持有的 PID 在进程活着期间不会被回收（Windows 上句柄未关就不回收）
      ⇒ 按 PID 杀是精确且安全的。
    """
    if proc.poll() is not None:      # 已经自己退了 ⇒ 什么都不用做
        return
    pid = proc.pid
    print(f"[watchdog] {label}: {why} ⇒ 精确 kill PID {pid} 及其子进程"
          f"（taskkill /PID {pid} /T /F，不碰其它进程）", flush=True)
    try:
        if os.name == "nt":
            out = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                                 capture_output=True, timeout=120)
            msg = ((out.stdout or b"") + (out.stderr or b"")).decode("gbk", "replace").strip()
            print(f"[watchdog] taskkill rc={out.returncode}: {msg}", flush=True)
        else:
            import signal as _signal
            try:
                os.killpg(os.getpgid(pid), _signal.SIGKILL)
            except Exception:        # noqa: BLE001  没有独立进程组就退化为杀自己
                proc.kill()
    except Exception as exc:  # noqa: BLE001
        print(f"[watchdog] ⚠️ kill 失败，退化为 proc.kill(): {type(exc).__name__}: {exc}",
              flush=True)
        try:
            proc.kill()
        except Exception:            # noqa: BLE001
            pass
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print(f"[watchdog] ⚠️ PID {pid} 60s 仍未退出，再补一刀", flush=True)
        try:
            proc.kill()
        except Exception:            # noqa: BLE001
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out-dir", default="training_history/ab_flat")
    ap.add_argument("--flat-env", default=DEFAULT_FLAT_ENV)
    ap.add_argument("--mcts-env", default=DEFAULT_MCTS_ENV)
    ap.add_argument("--timeout", type=float, default=4 * 3600.0)
    # 2026-10-22：同图交换座位（配对）。实测 P1 系统性占优 +19~31pp，
    # 旧逻辑是"不同地图各打一次" ⇒ 座位方差全进噪声。
    ap.add_argument("--paired-sides", action="store_true",
                    help="每个 seed 打两局，A 侧分别坐 P0/P1（同一张地图）")
    # ── 2026-10-28 健壮性三件套（默认全关 ⇒ 行为与旧版一致）────────────────────────
    ap.add_argument("--retry-invalid", type=int, default=0, metavar="N",
                    help="一批跑完后，把没有 verdict=engine 的局重跑，最多 N 轮（默认 0=关闭；"
                         "只重跑失败的，成功局日志不会被覆盖）")
    ap.add_argument("--game-timeout-s", type=float, default=0.0, metavar="S",
                    help="单局墙钟看门狗：超过 S 秒仍未结束 ⇒ 精确 kill 该局 PID 树、"
                         "记 verdict=timeout_watchdog，并按 --retry-invalid 重试（默认 0=关闭）")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：启动时跳过已有 verdict=engine 的 seed/side，只跑缺失/无效的")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    env_base = os.environ.copy()
    env_base["PYTHONIOENCODING"] = "utf-8"
    # 2026-10-23（用户要求）：逐回合进度文件用这个 tag（match_via_sdk 会写
    #   training_history/progress/<tag>_seed<seed>.txt）
    TAG = out_dir.name
    env_base["DSH_PROGRESS_TAG"] = TAG
    try:
        import progress as _progress
    except Exception:            # 进度输出绝不影响对局
        _progress = None
    T_START = time.time()

    # 新参数（任一开启）⇒ 打印重试/续跑/有效率区块；全关 ⇒ stdout 与旧版一致
    FEATURES_ON = bool(args.resume or args.retry_invalid > 0 or args.game_timeout_s > 0)

    sem = threading.Semaphore(args.jobs)
    results: dict = {}
    state: dict[str, dict] = {}       # label → {valid, verdict, score, attempts, ...}
    # 2026-10-28：用 RLock 而不是 Lock —— `write_status()`（持锁调用）里要读 `state`，
    #   如果那些读又去抢同一把非重入锁，整批会静默死锁（实测踩到：一局跑完后再也不打印、
    #   后续局全部卡在 lock 上）。RLock 让"持锁调用的小工具函数"也安全。
    lock = threading.RLock()
    # 2026-10-22：无结果/异常局计数（原来直接丢，不可见）
    # 2026-10-28：改成"任务标签集合"（同一个局不管重试几次只算一条）。
    #   默认路径（每个 task 只跑一次）⇒ len(INVALID) 与旧的计数器逐字相同。
    INVALID: set[str] = set()

    if args.paired_sides:
        tasks = [(s, side) for s in args.seeds for side in (0, 1)]
    else:
        tasks = [(s, None) for s in args.seeds]

    def task_label(task: tuple) -> str:
        return _label(task[0], task[1])

    def task_log(task: tuple) -> Path:
        return out_dir / _log_name(task[0], task[1])

    prog = ProgressFile(out_dir)

    # ───────────────────────── 判有效性（日志是唯一权威）─────────────────────────
    def judge(task: tuple) -> tuple[bool, str]:
        """(是否有效, verdict)。日志优先；日志里没判词时用本进程记下的杀因。"""
        seed, _side = task
        label = task_label(task)
        log_path = task_log(task)
        if not log_path.is_file():
            return False, "no_log"
        parsed = parse_log(log_path, seed)
        if parsed is None:
            return False, "unreadable"
        verdict = parsed.get("verdict")
        if verdict is None:
            with lock:
                prev = (state.get(label) or {}).get("verdict")
            return False, (prev or "no_result")
        return (verdict == VALID_VERDICT), verdict

    def winner_side_of(parsed: dict, label: str) -> int | None:
        """RESULT 行里的 winner_side → 0/1（判不出/无结果 → None，并计入 INVALID）。"""
        winner_side = parsed.get("winner_side")
        # 新格式的 winner_side 是 'p0'/'p1' 字符串；旧格式是 None/缺失 ⇒ 按平局记
        # 2026-10-22 修：`_ladder.read_game` 在"没打完/无结果"时会返回**字符串 'None'**，
        # 而这里原来直接 int() ⇒ ValueError ⇒ 该局线程崩掉、结果丢失（实测一次丢 6/32 局）。
        if isinstance(winner_side, str):
            _ws = winner_side.strip().lower()
            if _ws in ("p0", "p1"):
                return int(_ws[1])
            if _ws in ("none", "null", ""):
                INVALID.add(label)
                return None
            try:
                return int(_ws)
            except ValueError:
                INVALID.add(label)
                return None
        if winner_side is not None:
            return int(winner_side)
        return None

    def score_of(task: tuple, parsed: dict) -> tuple[float, bool]:
        """(flat 侧得分, flat 是否执 P0)。得分口径与改动前逐字相同。"""
        seed, flat_side_in = task
        label = task_label(task)
        flat_is_p0 = (seed % 2 == 0) if flat_side_in is None else (flat_side_in == 0)
        flat_side = 0 if flat_is_p0 else 1
        ws = winner_side_of(parsed, label)
        flat_score = 0.5 if ws is None else (1.0 if int(ws) == flat_side else 0.0)
        return flat_score, flat_is_p0

    def invalid_count_for_summary() -> int:
        """汇总行里"无效局"的数。

        口径 = 旧口径（`winner_side` 判不出来，见 `winner_side_of`）∪ 新能识别的
        **"根本没有 RESULT 行"**（0 字节日志 / 被看门狗杀掉 / 日志缺失）。
        后者是 2026-10-28 顺手补的：旧代码对"没有 RESULT 行"的局**既不计无效也不计负**
        ⇒ 直接按 0.5 平局混进胜率（断电那批局正好是这种情况，看起来像"一半平局"）。
        有效局（verdict=engine）在两套口径下都不受影响 ⇒ 正常批次打印的数字与旧版逐字相同。

        ⚠️ 这个函数会同时被"持锁调用"（`write_status`）和"无锁调用"（收尾汇总）用到，
           所以**不在这里抢 lock**（只做 dict 只读访问，CPython 下是原子的）。
        """
        bad = set(INVALID)
        for task in tasks:
            label = task_label(task)
            v = (state.get(label) or {}).get("verdict")
            if v in ("no_result", "no_log", "unreadable",
                     "timeout_watchdog", "timeout_hard", "driver_error"):
                bad.add(label)
        return len(bad)

    def write_status(total: int, done: int) -> None:
        """整批进度文件（每完成一局覆盖重写，用户可随时打开看）。永不抛异常。"""
        if _progress is None:
            return
        try:
            _w = sum(1 for v in results.values() if v["score"] >= 1.0)
            _l = sum(1 for v in results.values() if v["score"] == 0.0)
            _recent = [
                f"seed{k} {'P0' if v['flat_is_p0'] else 'P1'} "
                f"{'胜' if v['score'] >= 1.0 else ('负' if v['score'] == 0.0 else '平')} "
                f"hp={v.get('hp')} ({v.get('rounds')} 回合)"
                for k, v in list(results.items())[-6:]]
            _inflight = []
            for f in (REPO / "training_history/progress" / TAG).glob("az_progress_seed*.txt"):
                try:
                    last = f.read_text(encoding="utf-8").strip().splitlines()[-1]
                    if last.startswith("round="):
                        _inflight.append(int(last.split("=")[1].split()[0]))
                except Exception:      # noqa: BLE001
                    pass
            _progress.write_status(
                TAG, total=total, done=done, wins=_w, losses=_l,
                invalid=invalid_count_for_summary(),
                started=T_START, inflight_rounds=_inflight, recent=_recent)
        except Exception:              # noqa: BLE001
            pass

    # ───────────────────────── 单局执行（Popen + 看门狗）─────────────────────────
    attempts: dict[str, int] = {task_label(t): int((prog.get(task_label(t)) or {}).get("attempts") or 0)
                                for t in tasks}
    # 本轮开始时各局的累计尝试次数（`_progress.json` 可能是前几次运行留下的）——
    # 收尾的"本批重跑过几局"要靠它减出来，否则前几轮跑过的局会被算成"本次重试的"。
    attempts_at_start = dict(attempts)

    def next_attempt(label: str) -> int:
        with lock:
            attempts[label] = attempts.get(label, 0) + 1
            return attempts[label]

    def archive_previous_log(log_path: Path, label: str, prev_attempt: int) -> None:
        """重跑前把上一轮失败的日志复制一份留证据（只复制，不删/不改原日志）。"""
        try:
            if not log_path.is_file():
                return
            dst_dir = out_dir / "_attempts"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{label}.attempt{prev_attempt}.log"
            shutil.copy2(log_path, dst)
            print(f"[retry] 上一轮失败日志已留档: {dst.relative_to(REPO).as_posix()}", flush=True)
        except Exception as exc:      # noqa: BLE001
            print(f"[retry] ⚠️ 归档旧日志失败（不影响重跑）: {type(exc).__name__}: {exc}",
                  flush=True)

    def run_one(task: tuple, attempt: int) -> None:
        seed, flat_side_in = task
        label = task_label(task)
        log_path = task_log(task)
        flat_is_p0 = (seed % 2 == 0) if flat_side_in is None else (flat_side_in == 0)
        env0 = args.flat_env if flat_is_p0 else args.mcts_env
        env1 = args.mcts_env if flat_is_p0 else args.flat_env
        cmd = [sys.executable, "-u", str(REPO / "code/test_match/match_via_sdk.py"),
               "--trace-ops",
               f"--ai0={BRIDGE.as_posix()}",
               f"--ai1={BRIDGE.as_posix()}",
               f"--ai0-env={env0}",
               f"--ai1-env={env1}",
               str(seed)]
        # 2026-10-23：配对模式同一 seed 打两局 ⇒ 进度文件必须带座位后缀，否则互相覆盖
        env_game = dict(env_base)
        env_game["DSH_PROGRESS_SUFFIX"] = "" if flat_side_in is None else f"_p{0 if flat_is_p0 else 1}"
        t0 = time.time()
        wd = {"fired": False, "hard": False}
        with sem:
            with open(log_path, "wb") as handle:
                popen_kw = {} if os.name == "nt" else {"start_new_session": True}
                proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                        env=env_game, cwd=str(REPO), **popen_kw)
                timer = None
                if args.game_timeout_s > 0:
                    def _on_watchdog(_p=proc, _l=label) -> None:
                        wd["fired"] = True
                        kill_game_tree(_p, _l, f"墙钟超过 {args.game_timeout_s:g}s 仍未结束")
                    timer = threading.Timer(args.game_timeout_s, _on_watchdog)
                    timer.daemon = True
                    timer.start()
                try:
                    rc = proc.wait(timeout=args.timeout if args.timeout > 0 else None)
                except subprocess.TimeoutExpired:
                    # 旧实现这里会让 TimeoutExpired 直接把线程炸掉 ⇒ 结果丢失（既不计胜率也不计无效）
                    wd["hard"] = True
                    kill_game_tree(proc, label, f"--timeout={args.timeout:g}s 兜底超时")
                    rc = proc.wait(timeout=120)
                finally:
                    if timer is not None:
                        timer.cancel()
        elapsed = time.time() - t0
        parsed = parse_log(log_path, seed) or {}
        verdict = parsed.get("verdict")
        if verdict == VALID_VERDICT:
            pass                              # 日志优先：进程写完 RESULT 才算数
        elif wd["fired"]:
            verdict = "timeout_watchdog"
        elif wd["hard"]:
            verdict = "timeout_hard"
        elif verdict is None:
            verdict = "no_result"
        valid = (verdict == VALID_VERDICT)
        flat_score, flat_is_p0 = score_of(task, parsed)
        flat_side = 0 if flat_is_p0 else 1
        with lock:
            key = seed if flat_side_in is None else (seed, flat_side)
            results[key] = {"flat_is_p0": flat_is_p0, "score": flat_score,
                            "rc": rc, "minutes": elapsed / 60,
                            "rounds": parsed.get("rounds"), "hp": parsed.get("hp"),
                            "terminal": parsed.get("terminal")}
            state[label] = {"seed": seed, "side": None if flat_side_in is None else flat_side,
                            "flat_is_p0": flat_is_p0, "score": flat_score, "valid": valid,
                            "verdict": verdict, "attempts": attempt, "rc": rc,
                            "minutes": elapsed / 60, "rounds": parsed.get("rounds")}
            print(f"[seed {seed}] flat={'P0' if flat_is_p0 else 'P1'} "
                  f"score={flat_score} rc={rc} "
                  f"rounds={parsed.get('rounds')} hp={parsed.get('hp')} "
                  f"({elapsed/60:.1f} min)", flush=True)
            if not valid:
                print(f"  ⚠️ {label} 判无效：verdict={verdict}"
                      f"{'（--retry-invalid / --resume 会重跑）' if not FEATURES_ON else ''}",
                      flush=True)
            prog.update(label, status=("ok" if valid else "invalid"), verdict=verdict,
                        seed=seed, side=(None if flat_side_in is None else flat_side),
                        attempts=attempt, score=flat_score,
                        minutes=round(elapsed / 60, 2), rounds=parsed.get("rounds"))
            write_status(len(tasks), len(results))

    def run_one_safe(task: tuple, is_retry: bool = False) -> None:
        """线程入口：**驱动层异常也不能丢局**（旧实现会让该局结果彻底消失）。

        `is_retry=True` 只用于 `--retry-invalid` 的重跑轮：
          · 重跑前再确认一次日志（万一人工改过/别的进程补上了）⇒ 有效就跳过，**绝不覆盖成功局**；
          · 把上一轮失败的日志复制到 `_attempts/` 留证据。
        第一批（含 `--resume` 的补跑）走 `is_retry=False`，行为与旧版一致：直接覆盖日志。
        """
        label = task_label(task)
        attempt = next_attempt(label)
        if is_retry:
            ok, v = judge(task)
            if ok:
                # 防御：成功局的结果文件绝不能被重跑覆盖
                print(f"[retry] {label} 已有 verdict={v}（有效）⇒ 跳过，不覆盖", flush=True)
                with lock:
                    state[label] = {**(state.get(label) or {}), "valid": True, "verdict": v}
                return
            archive_previous_log(task_log(task), label, attempt - 1)
        try:
            run_one(task, attempt)
        except Exception as exc:      # noqa: BLE001
            traceback.print_exc()
            print(f"[!] {label} 驱动层异常: {type(exc).__name__}: {exc}", flush=True)
            with lock:
                state[label] = {"seed": task[0], "side": task[1], "valid": False,
                                "verdict": "driver_error", "attempts": attempt}
                prog.update(label, status="invalid", verdict="driver_error",
                            seed=task[0], side=task[1], attempts=attempt)

    def run_batch(task_list: list, is_retry: bool = False) -> None:
        threads = [threading.Thread(target=run_one_safe, args=(t, is_retry))
                   for t in task_list]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    # ───────────────────────── ③ 断点续跑：先算计划 ─────────────────────────
    def adopt(task: tuple) -> None:
        """把 out-dir 里已有的有效局直接采纳进本轮汇总（否则续跑那轮的胜率会骗人）。"""
        seed, flat_side_in = task
        label = task_label(task)
        parsed = parse_log(task_log(task), seed) or {}
        flat_score, flat_is_p0 = score_of(task, parsed)
        flat_side = 0 if flat_is_p0 else 1
        with lock:
            key = seed if flat_side_in is None else (seed, flat_side)
            results[key] = {"flat_is_p0": flat_is_p0, "score": flat_score, "rc": 0,
                            "minutes": 0.0, "rounds": parsed.get("rounds"),
                            "hp": parsed.get("hp"), "terminal": parsed.get("terminal")}
            state[label] = {"seed": seed, "side": None if flat_side_in is None else flat_side,
                            "flat_is_p0": flat_is_p0, "score": flat_score, "valid": True,
                            "verdict": parsed.get("verdict"), "attempts": attempts[label],
                            "minutes": 0.0, "rounds": parsed.get("rounds"), "resumed": True}

    pending = list(tasks)
    if args.resume:
        done: list[tuple] = []
        pend: list[tuple[tuple, str]] = []
        for task in tasks:
            ok, v = judge(task)
            (done if ok else pend).append(task if ok else (task, v))
        print(f"[resume] 续跑计划：已完成 {len(done)} 个、待跑 {len(pend)} 个", flush=True)
        if done:
            print(f"[resume] 跳过（日志 verdict=engine）: "
                  f"{', '.join(task_label(t) for t in done)}", flush=True)
            for task in done:
                adopt(task)
        if pend:
            print("[resume] 待跑明细（原因）:", flush=True)
            for task, v in pend:
                # _progress.json 里可能记着"上次是 ok"，但日志没了 ⇒ 以日志为准，照样重跑
                prev = (prog.get(task_label(task)) or {}).get("status")
                extra = f"，但 {prog.path.name} 记为 {prev} ⇒ 以日志为准重跑" \
                    if prev and prev != "invalid" else ""
                print(f"    {task_label(task)}: {v}{extra}", flush=True)
        pending = [t for t, _v in pend]

    # ───────────────────────── 第一批 + ① 重试 ─────────────────────────
    t0 = time.time()          # 汇总里的 wall 只算"真正跑对局"的时间（resume 的计划阶段不算）
    run_batch(pending)

    def invalid_tasks() -> list[tuple[tuple, str]]:
        out: list[tuple[tuple, str]] = []
        for task in tasks:
            label = task_label(task)
            with lock:
                rec = dict(state.get(label) or {})
            if rec.get("verdict") is not None:
                ok, v = bool(rec.get("valid")), rec["verdict"]
            else:
                ok, v = judge(task)
            if not ok:
                out.append((task, v))
        return out

    if args.retry_invalid > 0:
        for rnd in range(1, args.retry_invalid + 1):
            bad = invalid_tasks()
            if not bad:
                print(f"\n[retry] 第 {rnd}/{args.retry_invalid} 轮：没有无效局了，停止重试", flush=True)
                break
            cnt = collections.Counter(v for _t, v in bad)
            print(f"\n=== 重试第 {rnd}/{args.retry_invalid} 轮：{len(bad)} 局没有 verdict=engine ===",
                  flush=True)
            print(f"    重跑前失败原因分布（verdict 计数）: {dict(cnt)}", flush=True)
            print(f"    重跑清单: {', '.join(task_label(t) for t, _ in bad)}", flush=True)
            run_batch([t for t, _ in bad], is_retry=True)
        else:
            left = invalid_tasks()
            if left:
                print(f"\n[retry] ⚠️ 已用满 {args.retry_invalid} 轮，仍有 {len(left)} 局无效: "
                      f"{', '.join(f'{task_label(t)}({v})' for t, v in left)}", flush=True)

    # ───────────────────────── 汇总 ─────────────────────────
    scores = [v["score"] for v in results.values()]
    if scores:
        wins = sum(1 for x in scores if x == 1.0)
        draws = sum(1 for x in scores if x == 0.5)
        losses = sum(1 for x in scores if x == 0.0)
        rate = sum(scores) / len(scores)
        se = (0.25 / len(scores)) ** 0.5
        print(f"\n=== flat vs mcts：{len(scores)} 局  "
              f"flat 胜率 {rate:.1%}  ({wins}W/{draws}D/{losses}L)  "
              f"±{2*se:.1%}(2SE)  wall={(time.time()-t0)/60:.1f} min  "
              f"无效局={invalid_count_for_summary()} ===", flush=True)
        # 2026-10-22：先/后手拆分（座位诊断）。P1 系统性占优时，两条臂若不配对就会互相污染。
        for _side, _name in ((0, "P0"), (1, "P1")):
            _s = [v["score"] for v in results.values() if v["flat_is_p0"] == (_side == 0)]
            if _s:
                _w = sum(1 for x in _s if x == 1.0)
                print(f"    A 在 {_name}: {_w}W-{len(_s)-_w}L = {sum(_s)/len(_s):.1%} (n={len(_s)})",
                      flush=True)

    # 2026-10-28：健壮性区块（三个新参数全关时不打印 ⇒ 默认 stdout 与旧版一致）
    if FEATURES_ON:
        n_total = len(tasks)
        final = {task_label(t): (state.get(task_label(t)) or {}) for t in tasks}
        n_valid = sum(1 for r in final.values() if r.get("valid"))
        still_bad = {k: r.get("verdict") for k, r in final.items() if not r.get("valid")}
        retried = [k for k in final
                   if attempts.get(k, 0) - attempts_at_start.get(k, 0) > 1]   # 只算**本轮**重跑过的
        recovered = [k for k in retried if final[k].get("valid")]
        print(f"\n=== 最终有效率 {n_valid}/{n_total} "
              f"({(n_valid / n_total if n_total else 0):.1%}) ===", flush=True)
        if retried:
            print(f"    重跑过的局 {len(retried)} 个：{', '.join(retried)}"
                  f"（其中转有效 {len(recovered)} 个）", flush=True)
        if still_bad:
            print(f"    仍然无效 {len(still_bad)} 个: "
                  f"{', '.join(f'{k}({v})' for k, v in sorted(still_bad.items()))}", flush=True)
        # 有效局专用胜率：分母只算 verdict=engine 的局（口径同 analyze_paired §4）
        vs = [r["score"] for k, r in final.items() if r.get("valid")]
        if vs:
            _w = sum(1 for x in vs if x == 1.0)
            _d = sum(1 for x in vs if x == 0.5)
            _l = sum(1 for x in vs if x == 0.0)
            print(f"    有效局胜率（口径=verdict=engine，分母 {len(vs)}）: {sum(vs)/len(vs):.1%}"
                  f"  ({_w}W/{_d}D/{_l}L)", flush=True)
        print("    说明：上面的汇总按**最后一次尝试**计分；成功局不会被重跑，"
              "所以重试过的局用的就是最后一次有效结果。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
