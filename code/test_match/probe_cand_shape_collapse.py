"""验证 A：**候选塌缩的真因**到底是"多重集索引"还是"经济限制（金币不足）"（2026-10-29）。

背景：上一版实现把"按多重集推进游标"改成"按类推进"，理由是"按多重集实测塌成 7 条/426 次重复"。
但那个实验是在**穷局**（金币 13–56，建塔/升级/超武全买不起）上做的 ⇒ 无法区分两个解释。
本脚本按最新口径（游标=带重数规范多重集 + 缓存命中复用分数并推进游标）在**同一局的富局/穷局**
上各跑一遍候选生成，直接读：

* `n_pool` / `n_cand`（去重后）
* 缓存命中次数、位置推进总步数、形状数、已穷尽的形状数
* 长度分布、含重复类的形状条数、同 op 类型多 op 的候选（`[build,build]` 的证据）
* 该局面的**金币/合法类/各通道合法格数**（经济侧的量）

判据（写在报告里）：
* 富局不塌 + 穷局塌 ⇒ 塌缩主因是**经济限制**，多重集索引没问题；
* 富局也塌 ⇒ 是**多重集索引**问题，如实报告。

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/probe_cand_shape_collapse.py \
    --tag UNIFIED_CAND_FULL --seed 8 --n-rich 6 --n-poor 4 \
    --out match_results/unified_cand/collapse_probe.log
```
（`--tag/--seed` 指向 `match_results/ladder_logs/<tag>/<seed>.log` 里的 `--trace-ops` 行。）
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_REAL_STDOUT = sys.stdout

BASE_SCRIPT = REPO / "code" / "test_match" / "_inject_vs_runnerup.sh"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _base_env() -> dict[str, str]:
    txt = BASE_SCRIPT.read_text(encoding="utf-8")
    m = re.search(r'^BASE="(.*)"$', txt, re.M)
    if not m:
        raise SystemExit("读不到 _inject_vs_runnerup.sh 里的 BASE")
    out = {}
    for kv in m.group(1).split(";"):
        if not kv:
            continue
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="UNIFIED_CAND_FULL")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--player", type=int, default=0, help="给哪个座位算候选（默认 p0 = 我方）")
    ap.add_argument("--n-rich", type=int, default=6, help="取金币最高的前 N 个回合")
    ap.add_argument("--n-poor", type=int, default=4, help="取金币最低的 N 个回合")
    ap.add_argument("--rounds", type=int, nargs="*", default=None,
                    help="显式指定回合（给了就不自动挑富/穷）")
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--inflate", type=int, default=0,
                    help="**单变量操纵金币**：把快照 clone 后双方都不出招、只 advance_round()"
                         "直到该座位金币 ≥ 该值（默认 0 = 不做）⇒ 同一牌面、只变金币")
    ap.add_argument("--oracle", action="store_true",
                    help="独立于生成器的\"可达空间下界\"：枚举全部 (通道, 合法格) 单 op + can_apply，"
                         "数**不同单 op bundle** 的条数")
    ap.add_argument("--max-idle", type=int, default=3000, help="--inflate 最多推几回合（默认 3000）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg, file=_REAL_STDOUT, flush=True)
        lines.append(msg)

    env = _base_env()
    for k, v in env.items():
        os.environ[k] = v
    os.environ["AZAI_TORCH_THREADS"] = "1"
    os.environ["AZAI_CAND_MODE"] = "unified"
    os.environ["AZAI_K"] = str(int(args.k))

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.game_state_facade import GameStateFacade

    rsc = _load("_repro_search_crash_cs", REPO / "code" / "test_match" / "repro_search_crash.py")
    rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
    if not rounds:
        emit(f"!! {args.tag}/{args.seed}.log 里没有 --trace-ops 行")
        return 2
    max_r = max(rounds) if rounds else 0
    st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
    # 逐回合记录：给指定座位决策时的局面 + 该座位的金币（= 经济侧的量）
    snaps: list = []
    for r in range(0, max_r + 1):
        for pl in (0, 1):
            if pl == args.player:
                snaps.append((r, int(st.coins[pl]), st.clone()))
            ops = rounds.get(r, {}).get(pl, [])
            if ops:
                st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                             for (t, a0, a1) in ops])
        st.advance_round()
    if not snaps:
        emit("!! 没采到快照")
        return 2

    if args.rounds:
        want = set(int(x) for x in args.rounds)
        sel = [s for s in snaps if s[0] in want]
    else:
        by_coin = sorted(snaps, key=lambda x: -x[1])
        rich = by_coin[: max(0, int(args.n_rich))]
        poor = sorted(snaps, key=lambda x: x[1])[: max(0, int(args.n_poor))]
        seen = set()
        sel = []
        for tag, group in (("富", rich), ("穷", poor)):
            for item in group:
                if item[0] in seen:
                    continue
                seen.add(item[0])
                sel.append((tag,) + item)
        sel = [(s[0], s[1], s[2], s[3]) if len(s) == 4 else ("?", s[0], s[1], s[2])
               for s in sel]
    # 统一成 (tag, round, coins, snap)
    fixed = []
    for item in sel:
        if len(item) == 4:
            fixed.append(item)
        else:
            fixed.append(("?", item[0], item[1], item[2]))
    sel = fixed

    bridge = _load("az_bridge_ai_cs", REPO / "code" / "test_match" / "az_bridge_ai.py")
    searcher, _m, _f = bridge.build_engine(args.seed, 0)
    gen = searcher.cand_gen
    emit(f"[cs] 生成器 = {type(gen).__name__} k={gen.k} tau={gen.tau} "
         f"steps_per_shape={gen.steps_per_shape} steps_total={gen.steps_total} "
         f"beta={gen.beta} ucb_c={gen.ucb_c} force_hold={gen.force_hold}")
    emit(f"[cs] 局面来源 tag={args.tag} seed={args.seed} player={args.player} "
         f"共 {len(snaps)} 个决策点，选 {len(sel)} 个来跑")
    emit("")
    hdr = (f"{'':2} {'round':>5} {'coins':>5} {'推后':>5} {'推N':>5} {'hp':>4} {'合法类':>5} "
           f"{'oracle':>6} {'空格':>4} {'塔格':>4} "
           f"{'n_pool':>6} {'n_cand':>6} {'命中':>5} {'步数':>5} {'形状':>5} {'穷尽':>5} "
           f"{'重复形状':>8} {'同类型多op':>9} {'Q条目':>6} {'ms':>7}  长度分布")
    emit(hdr)

    rows = []
    for (tag, rnd, coins, snap) in sel:
        # ── 可选：单变量操纵金币（同一牌面、只变金币）─────────────────────────────
        snap_use = snap.clone()
        coins_after = int(coins)
        n_idle = 0
        if args.inflate and int(coins) < int(args.inflate):
            while (coins_after < int(args.inflate) and n_idle < int(args.max_idle)
                   and not snap_use.terminal):
                snap_use.advance_round()
                n_idle += 1
                coins_after = int(snap_use.coins[int(args.player)])
        net_out = searcher.net_fn(snap_use.clone(), int(args.player))
        res = gen.generate(snap_use.clone(), int(args.player), net_out, k=int(args.k))
        stt = dict(res["stats"])
        e = gen  # 生成器内部状态可读：合法类数 / 各通道格数
        # 空格/塔格数（塔通道的合法格，来自通道 0 的位置掩码）
        cc0 = e.cells.get(0)
        n_cell0 = len(cc0) if cc0 is not None else 0
        n_tw = sum(1 for t in snap_use.towers_of(int(args.player))) \
            if hasattr(snap_use, "towers_of") else 0
        n_hp = int(snap_use.bases[int(args.player)].hp) if hasattr(snap_use, "bases") else -1
        oracle = -1
        if args.oracle:
            oracle = _single_op_space(snap_use, int(args.player), e)
        row = dict(tag=tag, round=int(rnd), coins=int(coins), coins_after=int(coins_after),
                   n_idle=int(n_idle), hp=n_hp, oracle=int(oracle),
                   n_legal=len(e.legal),
                   n_cell0=int(n_cell0), n_towers=int(n_tw),
                   n_pool=int(stt["n_pool"]), n_cand=int(stt["n_cand"]),
                   cache_hit=int(stt["n_dup"]), steps=int(stt.get("steps", 0)),
                   shapes=int(stt.get("shapes", 0)), dead=int(stt.get("dead_shapes", 0)),
                   n_rep=int(stt.get("n_rep", 0)), n_multi_same=int(stt.get("n_multi_same", 0)),
                   q_entries=int(stt.get("q_entries", 0)),
                   p1_ms=float(stt["p1_ms"]), sizes=stt["sizes"],
                   rep_examples=stt.get("rep_examples", []),
                   multi_same=stt.get("multi_same_examples", []),
                   bundles=[tuple(tuple(int(x) for x in op) for op in b) for b in res["bundles"]])
        rows.append(row)
        emit(f"{tag:2} {rnd:>5} {coins:>5} {coins_after:>5} {n_idle:>5} {row['hp']:>4} "
             f"{row['n_legal']:>5} {row['oracle']:>6} {row['n_cell0']:>4} "
             f"{row['n_towers']:>4} {row['n_pool']:>6} {row['n_cand']:>6} {row['cache_hit']:>5} "
             f"{row['steps']:>5} {row['shapes']:>5} {row['dead']:>5} {row['n_rep']:>8} "
             f"{row['n_multi_same']:>9} {row['q_entries']:>6} {row['p1_ms']:>7.0f}  {row['sizes']}")

    emit("")
    emit("## 小结")
    rich_rows = [r for r in rows if r["tag"] == "富"]
    poor_rows = [r for r in rows if r["tag"] == "穷"]
    for name, group in (("富局（金币最高）", rich_rows), ("穷局（金币最低）", poor_rows)):
        if not group:
            continue
        nc = [r["n_cand"] for r in group]
        emit(f"  {name}: n={len(group)} 金币={[r['coins'] for r in group]} "
             f"n_cand={nc} 中位={sorted(nc)[len(nc)//2]} "
             f"同类型多op 合计={sum(r['n_multi_same'] for r in group)} "
             f"重复类形状 合计={sum(r['n_rep'] for r in group)}")
    emit("")
    emit("判据：富局不塌、穷局塌 ⇒ 塌缩主因 = **经济限制**（金币买不起 ⇒ 合法动作本来就少）；")
    emit("      富局也塌 ⇒ 塌缩主因 = **多重集索引**（要另行决策）。")
    tot_multi = sum(r["n_multi_same"] for r in rows)
    emit(f"`[build,build]`/`[downgrade,downgrade]` 类（同 op 类型 ≥2 个 op）的候选合计 = {tot_multi}")
    for r in rows:
        if r["multi_same"]:
            emit(f"  例 r{r['round']}（金币 {r['coins']}）: {r['multi_same'][:3]}")
    for r in rows:
        if r["rep_examples"]:
            emit(f"  重复类**形状**例 r{r['round']}: {r['rep_examples'][:4]}")

    # 顺手验证：LogSumExp 对"重复同值"不敏感
    emit("")
    emit("## 附带验证：LogSumExp 对重复同值不敏感（⇒ 缓存命中重复计分不会撑虚 Q[c]）")
    for beta in (1.0, 3.0, 8.0):
        xs = [0.30, -0.10, 0.05]
        base = _lse_of(xs, beta)
        dup = _lse_of(xs + [0.30, 0.30], beta)
        emit(f"  beta={beta:>3}: LSE{xs} = {base:+.6f}；补两个重复值 0.30 后 = {dup:+.6f} "
             f"（差 {dup - base:+.2e}）")

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[cs] 已写入 {out}", file=_REAL_STDOUT, flush=True)
    return 0


def _single_op_space(snap, player: int, gen) -> int:
    """**独立于生成器**的可达空间下界：枚举全部 (合法类, 合法格) 的单 op，`can_apply` 后数**不同**条数。

    为什么要有它：`n_cand` 小有两种解释 ——（a）生成器枚举不全，（b）该局面**真的**只有这么多个
    能打的动作。这个 oracle 只看游戏规则（`make_position_masks` + `can_apply`），
    与"游标/多重集/缓存"全都无关 ⇒ `n_cand ≈ oracle` 就说明生成器**贴着空间上界**在走。
    """
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.unified_cand import HOLD, NO_POSITION, decode_channel_op

    seen: set = set()
    for ch in gen.legal:
        if int(ch) == HOLD:
            seen.add(())                     # HOLD = 空 bundle，也是一个可达动作
            continue
        cc = gen.cells.get(int(ch))
        if cc is None:                       # 无位置的类（21/22）：只有一个动作
            op = decode_channel_op(snap, player, int(ch), *NO_POSITION)
            if op is not None:
                t = (int(op.op_type), int(op.arg0), int(op.arg1))
                if snap.can_apply_operation(player, op, ()):
                    seen.add((t,))
            continue
        for i in range(len(cc)):
            cell = cc.kth_cell(i)
            op = decode_channel_op(snap, player, int(ch), cell[0], cell[1])
            if op is None:
                continue
            try:
                if not snap.can_apply_operation(player, op, ()):
                    continue
            except Exception:                # noqa: BLE001
                continue
            seen.add(((int(op.op_type), int(op.arg0), int(op.arg1)),))
    return len(seen)


def _lse_of(xs, beta: float) -> float:
    """与 `UnifiedCandidateGen._lse` 完全同一个公式（(1/β)log(mean exp(βx))）。"""
    import math
    arr = np.asarray(xs, dtype=np.float64)
    m = float(arr.max())
    return float(m + math.log(float(np.mean(np.exp(beta * (arr - m))))) / beta)


if __name__ == "__main__":
    raise SystemExit(main())
