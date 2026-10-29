#!/usr/bin/env python
"""复现探针：把**真实对局的局面序列**逐回合喂给冠军当 oracle，看它为什么在第 ~250 回合自杀。

## 为什么需要这个（2026-10-29 的实测发现）

E2（注入冠军动作）16 局的 `注入统计` 里 `dead=14/16`，`match_results/oracle_logs/*.stderr.log` 里
18 个会话都是同一条链：

```
[MagicaV0Policy] runtime mismatch seed=<N> player=<P> round=<1..11> field=coins[P]
[PolicyEnv:ERROR] where=ReplacePendingOperations.called_after_terminal round=<250..310> ...
terminate called after throwing an instance of 'std::runtime_error'
  what():  apply_operations.called_after_terminal      （Windows 侧 exit 0xC0000409）
```

⇒ 注入覆盖率只有 ~54%。这直接影响 E4（候选集 = {冠军那一手, 亚军那一手}）：
冠军会话一死，后半局候选集就塌成"只剩亚军那一手"。

## 这个探针要区分两个解释

| 解释 | 预测 |
|---|---|
| A. **冠军的影子模型**只按自己的操作推演，被我们**真实**的对手操作顶出分歧 ⇒ 迟早自己判"终局" | **单进程**复现真实局面序列，也会在 ~250 回合崩 |
| B. 同一局里**两个** magica 实例（对手 + oracle）共享 `ANTWAR_MAGICA_RUNTIME_ARTIFACT_DIR` 或资源冲突 | 单进程复现**不会**崩 |

做法：从 `--log`（`--trace-ops` 的 ladder 日志）里按回合重放**双方**的真实操作，
每回合在该冠军的决策点调一次 `oracle.ask(state)`（**只读**它的答案，不采用），
局面严格跟随真实对局 ⇒ 与 E2 里它的处境逐回合等价（唯一差别是只有一个 magica 进程）。

输出：每 25 回合一行进度 + 结束时的 `[probe-sum]`；崩了就打印异常与退出码。

用法：
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/_champ_oracle_replay_probe.py \
      --log match_results/ladder_logs/US_VS_CHAMPION_BASE/11.log --max-rounds 512
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

OUR_LABEL = "az_bridge_ai"
# [round 12] p0(champion) 原文='..' -> [Operation(op_type=<OperationType.X: 11>, arg0=1, arg1=2), ...]
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


def parse_rounds(log: Path) -> tuple[int, str, dict[int, dict[int, list]]]:
    """日志 → (seed, 冠军座位, {round: {player: [(op_name, a0, a1), ...]}})。"""
    txt = log.read_text(encoding="utf-8", errors="replace")
    m = RESULT_RE.search(txt)
    if not m:
        raise SystemExit(f"!! {log} 里没有 RESULT 行（没有对局结果就不知道该重放谁）")
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
        ops = [(name, int(a0), int(a1)) for (name, _v, a0, a1) in OP_RE.findall(ops_txt)]
        rounds.setdefault(r, {})[who] = ops
    return seed, champ_seat, rounds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="--trace-ops 的 ladder 日志")
    ap.add_argument("--max-rounds", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    log = (REPO / args.log) if not Path(args.log).is_absolute() else Path(args.log)
    seed, champ_seat, rounds = parse_rounds(log)
    print(f"[probe] log={log.name} seed={seed} 冠军座位=P{champ_seat} 重放到 {min(args.max_rounds, max(rounds) + 1)} 回合")
    print(f"[probe] 局面序列 = **真实对局**（双方真实操作都重放）⇒ 与 E2 里 oracle 的处境逐回合等价，"
          f"但**只有一个** magica 进程")

    from my_ai.az_intent.game_state_facade import GameStateFacade
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    ro = _load("runnerup_oracle", REPO / "code" / "test_match" / "runnerup_oracle.py")
    champ = REPO / "其他版本ai" / "ant-war2-magica-v3" / "magica_v3_O0.exe"
    st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)

    n_ask = 0
    n_agree = 0
    t0 = time.time()
    ms: list[float] = []
    crash = None
    try:
        with ro.RunnerUpOracle(seed=seed, player=champ_seat, exe=str(champ),
                               timeout=float(args.timeout)) as orc:
            for r in range(min(args.max_rounds, max(rounds) + 1)):
                if st.terminal:
                    print(f"[probe] r{r} 真实局面已终局，停止")
                    break
                ops_r = rounds.get(r, {})
                # 冠军在 p1 时：先落对手(p0)的操作，再问它（这才是它真实的决策点）
                if champ_seat == 1:
                    p0_ops = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(0, [])]
                    if p0_ops:
                        st.apply_operation_list(0, [Operation(t, a0, a1) for (t, a0, a1) in p0_ops])
                ta = time.monotonic()
                got = orc.ask(st)
                ms.append((time.monotonic() - ta) * 1000.0)
                n_ask += 1
                want = [(n, a0, a1) for (n, a0, a1) in ops_r.get(champ_seat, [])]
                got_t = [(OperationType(int(o.op_type)).name, int(o.arg0), int(o.arg1)) for o in got]
                if got_t == want:
                    n_agree += 1
                if champ_seat == 0:
                    p0_ops = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(0, [])]
                    if p0_ops:
                        st.apply_operation_list(0, [Operation(t, a0, a1) for (t, a0, a1) in p0_ops])
                p1_ops = [(OperationType[n], a0, a1) for (n, a0, a1) in ops_r.get(1, [])]
                if p1_ops:
                    st.apply_operation_list(1, [Operation(t, a0, a1) for (t, a0, a1) in p1_ops])
                if champ_seat == 0:
                    pass
                st.advance_round()
                if r % 25 == 0:
                    print(f"[probe] r{r:<4} ask#{n_ask} 与真实动作一致={n_agree} "
                          f"最近一次 {ms[-1]:.0f}ms hp={[b.hp for b in st.bases]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        crash = f"{type(exc).__name__}: {exc}"
        print(f"[probe] !! 第 {n_ask} 次 ask 后崩溃: {crash}", flush=True)

    dt = time.time() - t0
    agree_pct = 100.0 * n_agree / max(1, n_ask)
    print(f"\n[probe-sum] asks={n_ask} agree_with_real_champion={n_agree}({agree_pct:.1f}%) "
          f"wall={dt:.1f}s avg_ms={sum(ms) / max(1, len(ms)):.0f} max_ms={max(ms) if ms else -1:.0f} "
          f"crash={crash or 'None'}")
    return 0 if crash is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
