#!/usr/bin/env python
"""探针：冠军 oracle「崩溃后重建」到底能不能救回来（2026-10-29，E4 前置）。

## 要回答的两个问题（决定 `az_bridge_ai._oracle_step` 的修法）

1. **重建 + 从 r0 全量回放**：冠军在第 ~239 次 ask 自杀（`apply_operations.called_after_terminal`）
   之后，起一个新会话、把 0..R 的历史原样再喂一遍，会不会**在同一个回合再崩**？
   * 若**不再崩** ⇒ 用户要的"重建 + 回放追赶"直接可用（每局 6 次够覆盖 512 回合）；
   * 若**同一回合又崩** ⇒ 回放本身会踩同一个雷，必须换方案（或只在崩点之前回放）。
2. **"轮次重标"重启**（把当前局面当成它的第 1 回合喂进去）：它会不会接受？
   这条是备选方案：代价 0.1s（不用回放），但它的影子局面从"全字段错位"起步 ⇒ 出招质量可能变差。

## 机制（已从 `其他版本ai/ant-war2-magica-v3/main.cpp` 源码确认，2026-10-29）

```
Game::step_round →  judge_base_camp():  shadow 里有一方 base HP ≤ 0  ⇒ is_end = 1
Env::ReplacePendingOperations  →  if (game_.is_ended()) throw ... "apply_operations.called_after_terminal"
```
`IsFatalDiagnostic()` 只认 `env.desynced / env.init_error / step_round.desynced`，而
`OnRoundState` 的 `MarkDesynced("round_sync.desync")` **不在**其中 ⇒ 逐回合的 `runtime mismatch`
（`field=coins[P]`）**不会**让它停手，它继续用自己那份影子局面推演，直到影子里的基地血量归零
⇒ 下一次 `OnOpponentOperations`（= 我们喂对手操作块）在 `is_ended()` 的 env 上 apply ⇒ 抛 ⇒
未捕获 ⇒ abort（`0xC0000409`）。⇒ **崩点是"影子局面自认终局"，不是"读端错位"**。

## 用法

```bash
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/_champ_oracle_rebuild_probe.py \
    --log match_results/ladder_logs/US_VS_CHAMPION_BASE/11.log --max-rounds 512
```
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

OP_RE = re.compile(r"Operation\(op_type=<OperationType\.(\w+): (-?\d+)>, arg0=(-?\d+), arg1=(-?\d+)\)")
ROUND_RE = re.compile(r"^\s*\[round (\d+)\] (p0|p1)\(([^)]*)\) .*?-> \[(.*)\]\s*$")
RESULT_RE = re.compile(r"RESULT seed=(\d+) p0=(\S+) p1=(\S+)")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_rounds(log: Path):
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = RESULT_RE.search(txt)
    if not m:
        raise SystemExit(f"!! {log} 里没有 RESULT 行")
    seed = int(m.group(1))
    labels = {0: m.group(2), 1: m.group(3)}
    champ_seat = 0 if labels[0] == "champion" else (1 if labels[1] == "champion" else None)
    if champ_seat is None:
        raise SystemExit(f"!! 两侧都不是 champion: {labels}")
    rounds: dict[int, dict[int, list]] = {}
    for line in txt.splitlines():
        mm = ROUND_RE.match(line)
        if not mm:
            continue
        r, who, ops_txt = int(mm.group(1)), int(mm.group(2)[1:]), mm.group(4)
        rounds.setdefault(r, {})[who] = [(n, int(a0), int(a1)) for (n, _v, a0, a1) in OP_RE.findall(ops_txt)]
    return seed, champ_seat, rounds


class Relabel:
    """把真局面的**回合号**换掉（其余属性透传）⇒ `state_to_text` 会吐出一个"第 k 回合"的局面。

    只用于问题 2 的备选方案：新会话的读端从第 0 回合起，第一条状态就是"第 1 回合"。
    """

    def __init__(self, inner, round_index: int):
        self._inner = inner
        self.round_index = int(round_index)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--attempts", type=int, default=3, help="重建尝试次数（问题 1）")
    ap.add_argument("--relabel-rounds", type=int, default=40, help="轮次重标后继续问几回合（问题 2）")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    log = (REPO / args.log) if not Path(args.log).is_absolute() else Path(args.log)
    seed, champ_seat, rounds = parse_rounds(log)
    from my_ai.az_intent.game_state_facade import GameStateFacade
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    ro = _load("runnerup_oracle", REPO / "code" / "test_match" / "runnerup_oracle.py")
    champ = REPO / "其他版本ai" / "ant-war2-magica-v3" / "magica_v3_O0.exe"

    def new_session(tag: str):
        p = REPO / "match_results" / "oracle_logs" / f"probe_{tag}_seed{seed}_p{champ_seat}.stderr.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        return ro.RunnerUpOracle(seed=seed, player=champ_seat, exe=str(champ),
                                 timeout=float(args.timeout), stderr_path=p)

    def drive(st, r: int, orc, want_ops: bool):
        """r 回合：冠军的决策点 → ask；然后双方操作落进影子、回合前进。返回它的答案。"""
        ops_r = rounds.get(r, {})
        if champ_seat == 1:      # 先落对手(p0)的操作，再问它
            p0 = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(0, [])]
            if p0:
                st.apply_operation_list(0, [Operation(t, a0, a1) for (t, a0, a1) in p0])
        got = orc.ask(st)
        if champ_seat == 0:
            p0 = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(0, [])]
            if p0:
                st.apply_operation_list(0, [Operation(t, a0, a1) for (t, a0, a1) in p0])
        p1 = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(1, [])]
        if p1:
            st.apply_operation_list(1, [Operation(t, a0, a1) for (t, a0, a1) in p1])
        st.advance_round()
        return got

    def replay(upto: int, tag: str):
        """从 r0 开始逐回合喂到 upto-1 回合（upto 之前），返回 (orc, st, 已喂的回合数, 崩溃信息)。"""
        st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
        orc = new_session(tag)
        for r in range(upto):
            if st.terminal:
                return orc, st, r, None
            try:
                drive(st, r, orc, True)
            except Exception as exc:  # noqa: BLE001
                return orc, st, r, f"{type(exc).__name__}: {exc}"
        return orc, st, upto, None

    # ── 问题 0：先复现崩溃点 ────────────────────────────────────────────────
    print(f"[probe] log={log.name} seed={seed} 冠军座位=P{champ_seat} "
          f"max_rounds={args.max_rounds}", flush=True)
    t0 = time.time()
    orc, st, fed, crash = replay(args.max_rounds, "p0_base")
    R = fed                                # 崩溃发生在"第 R 回合的 ask"
    print(f"[probe] 基线会话: 喂成功 {fed} 回合，崩于 r{R}: {crash}  "
          f"({time.time() - t0:.1f}s)", flush=True)
    orc.close()
    if crash is None:
        print("[probe] 没崩 ⇒ 这份日志不适合当探针（换 --log 或加大 --max-rounds）")
        return 0

    # ── 问题 1：重建 + 从 r0 全量回放，会不会在同一个回合再崩？ ──────────────
    print(f"\n[probe] === 问题 1：重建 + 全量回放 0..{R}（{args.attempts} 次）===", flush=True)
    results = []
    for k in range(1, args.attempts + 1):
        t1 = time.time()
        orc, st, fed, crash2 = replay(R, f"p1_rebuild{k}")
        replay_s = time.time() - t1
        if crash2 is not None:
            print(f"[probe] 重建#{k}: !! 回放到 r{fed} 就崩了（{crash2}）  {replay_s:.1f}s", flush=True)
            results.append(("replay_crash", fed, replay_s))
            orc.close()
            continue
        # 回放到 R 之前的 R 个回合都成功了 ⇒ 现在在 r{R} 上问它（这才是"追上"）
        t2 = time.time()
        try:
            got = drive(st, R, orc, True)
            ask_s = time.time() - t2
            want = [(n, a0, a1) for (n, a0, a1) in rounds.get(R, {}).get(champ_seat, [])]
            got_t = [(OperationType(int(o.op_type)).name, int(o.arg0), int(o.arg1)) for o in got]
            print(f"[probe] 重建#{k}: ✅ 回放 {R} 回合成功（{replay_s:.1f}s）+ r{R} 回答成功"
                  f"（{ask_s:.2f}s）与真实动作{'相同' if got_t == want else '不同'} "
                  f"{got_t} vs {want}", flush=True)
            # 再往后问 30 回合，看这个会话是不是**可用**（不只是"活着"）
            far = 0
            try:
                for r in range(R + 1, min(R + 31, max(rounds) + 1)):
                    if st.terminal:
                        break
                    drive(st, r, orc, True)
                    far += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[probe] 重建#{k}: 之后第 {far} 回合又崩: {type(exc).__name__}: {exc}",
                      flush=True)
            print(f"[probe] 重建#{k}: 追上之后又连续问了 {far} 回合"
                  f"{'（未再崩）' if far >= 30 else '（中途又崩）'}", flush=True)
            results.append(("ok", R, replay_s, far))
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] 重建#{k}: !! 回放成功但 r{R} 再崩: {type(exc).__name__}: {exc}",
                  flush=True)
            results.append(("ask_crash", R, replay_s))
        orc.close()

    # ── 问题 2：轮次重标（当前局面当"第 1 回合"喂） ──────────────────────────
    n_ok, n_agree = 0, 0
    print(f"\n[probe] === 问题 2：轮次重标（从 r{R} 的局面起步，标成第 1 回合）===", flush=True)
    # 先重建一个临时会话把局面重建到 R（只为了拿 facade；会话马上关掉）
    orc, st, fed, crash3 = replay(R, "p2_state")
    orc.close()
    if crash3 is not None or fed != R:
        print(f"[probe] !! 拿不到 r{R} 的局面（{crash3}）⇒ 跳过问题 2")
    else:
        print(f"[probe] r{R} 真实局面: round={st.round_index} hp={[b.hp for b in st.bases]} "
              f"coins={list(st.coins)}", flush=True)
        orc2 = new_session("p2_relabel")
        n_ok, n_agree, off = 0, 0, st.round_index
        t3 = time.time()
        crash4 = None
        try:
            for r in range(st.round_index, min(st.round_index + args.relabel_rounds,
                                               max(rounds) + 1)):
                if st.terminal:
                    break
                ops_r = rounds.get(r, {})
                if champ_seat == 1:
                    p0 = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(0, [])]
                    if p0:
                        st.apply_operation_list(0, [Operation(t, a0, a1) for (t, a0, a1) in p0])
                shim = Relabel(st, n_ok + 1)          # ← 关键：回合号重标为 1,2,3,...
                got = orc2.ask(shim)
                n_ok += 1
                want = [(n, a0, a1) for (n, a0, a1) in ops_r.get(champ_seat, [])]
                got_t = [(OperationType(int(o.op_type)).name, int(o.arg0), int(o.arg1)) for o in got]
                n_agree += int(got_t == want)
                if champ_seat == 0:
                    p0 = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(0, [])]
                    if p0:
                        st.apply_operation_list(0, [Operation(t, a0, a1) for (t, a0, a1) in p0])
                p1 = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(1, [])]
                if p1:
                    st.apply_operation_list(1, [Operation(t, a0, a1) for (t, a0, a1) in p1])
                st.advance_round()
        except Exception as exc:  # noqa: BLE001
            crash4 = f"{type(exc).__name__}: {exc}"
        print(f"[probe] 重标会话: 成功 {n_ok}/{args.relabel_rounds} 回合（offset={off}）"
              f"与真实动作一致 {n_agree}/{max(1, n_ok)}  {time.time() - t3:.1f}s  "
              f"crash={crash4 or 'None'}", flush=True)
        orc2.close()

    # ── 汇总 ────────────────────────────────────────────────────────────────
    print(f"\n[probe-sum] 基线崩于 r{R}  "
          f"重建尝试={[f'{a[0]}:r{a[1]}:{a[2]:.1f}s' for a in results]}  "
          f"relabel_ok={n_ok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
