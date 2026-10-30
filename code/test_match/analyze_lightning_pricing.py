"""闪电频率 + **临界金币定价**（2026-10-31，用户假说验证：值网是不是"乱花钱"）。

数据源有两块，**都不需要再跑对局**：

**A. 对局日志**（`[candq]` / `[act]` / `round=` 行，`AZAI_CAND_VERBOSE=1` 时才有）：
* `[azai] round=N … coins=[a,b] …` —— 每回合结束时的金币（= 下个决策前的金币）
* `[azai] [act] round=N n=… ops=[…]` —— 每回合**真的发出去的** op
* `[candq] topk … 构成=dict({...}) 前8=[(bundle),+分] …` —— top-k 构成 + 前 8 条明细
* `[candq] chosen (bundle) 类别=… 阶段2=… | 非HOLD最高=…(类别) … | HOLD阶段2=… | gap=…`

**B. 离线复算候选**（同一局的 `--trace-ops` 重放 + 同一个生成器）：
  逐决策重跑 `UnifiedCandidateGen.generate`（确定性、状态指纹种子化）⇒ **精确**数出
  "候选里含 `LIGHTNING`(op 21) 的决策/条数"。日志的 `前8` 只有前 8 条，不够。

统计：
1. **闪电频率**：我方真的发 LIGHTNING 的回合 / 决策数（并给历史 pin 参考值）
2. **闪电进阶段 2 候选的频率**（每决策 top-k 里含 op 21 的占比；区分"候选没提出"vs"提出没选"）
3. **临界金币（70–89，差一点够 90 的闪电）的定价行为**：
   * 这些回合里 top-k 有没有 建塔/升级 候选？HOLD 有没有？
   * 阶段 2 分：建塔/升级 与 HOLD 谁高？最终选了什么？
   * 「评估器把建塔/升级排在 HOLD 前面」的回合占比 ⇒ 支持/反对"值网乱花钱"。

用法：
```bash
P="D:/anaconda3/envs/pytorch-gpu/python.exe"
PYTHONIOENCODING=utf-8 $P code/test_match/analyze_lightning_pricing.py \
    --stderr match_results/ai0_az_bridge_ai_seed8_UCV.stderr.log \
    --tag UNIFIED_SPLIT_VERBOSE --seed 8 --player 0 \
    --out match_results/unified_cand/lightning_pricing.txt
```
`--no-offline` 可跳过 B 段（只读日志，秒级）。
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RE_ROUND = re.compile(r"\[azai\] round=(?P<r>\d+) hp=\[(?P<h0>-?\d+), (?P<h1>-?\d+)\] "
                      r"coins=\[(?P<c0>-?\d+), (?P<c1>-?\d+)\]")
RE_ACT = re.compile(r"\[act\] round=(?P<r>\d+) n=(?P<n>\d+) ops=\[(?P<ops>.*?)\]")
RE_OP = re.compile(r"\((\d+), (-?\d+), (-?\d+)\)")
RE_TOPK = re.compile(r"\[candq\] topk n=(?P<n>\d+) 构成=dict\((?P<comp>\{.*?\})\) 前8=(?P<top>.*)$")
RE_P1 = re.compile(r"\[(?P<b>\(.*?\)),(?P<s>[+-][\d.]+)\]")
RE_CH = re.compile(r"\[candq\] chosen (?P<b>.*?) 类别=(?P<cat>\S+) 阶段2=(?P<s2>[+-]?[\d.nan]+) "
                   r"回退=(?P<deg>\S+) \| 非HOLD最高=(?P<bn>[-+\d.na]+)\((?P<bnc>[^)]*)\) "
                   r"(?P<bnb>\(.*?\)|\(\)) \| HOLD阶段2=(?P<sh>[-+\d.na]+) "
                   r"\| 非HOLD最高−HOLD=(?P<gap>[-+\d.na]+)")
LIGHTNING = 21


def _num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def pct(k: int, n: int) -> str:
    return f"{100.0*k/max(1,n):5.1f}%"


def _f(x) -> str:
    return "n/a" if x is None or x != x else f"{x:+.4f}"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _base_env() -> dict:
    txt = (REPO / "code" / "test_match" / "_inject_vs_runnerup.sh").read_text(encoding="utf-8")
    m = re.search(r'^BASE="(.*)"$', txt, re.M)
    out = {}
    for kv in m.group(1).split(";"):
        if kv:
            k, v = kv.split("=", 1)
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stderr", required=True)
    ap.add_argument("--tag", default="UNIFIED_SPLIT_VERBOSE")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--player", type=int, default=0)
    ap.add_argument("--decode", default="split",
                    help="离线复算用的 AZAI_CAND_DECODE（本局是 split）")
    ap.add_argument("--no-offline", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    sp = Path(args.stderr)
    if not sp.is_absolute():
        sp = Path.cwd() / sp
    txt = sp.read_text(encoding="utf-8", errors="replace")
    out: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    # ── A. 解析日志 ────────────────────────────────────────────────────────
    decs: list[dict] = []
    acts: dict[int, list] = {}
    last_coins = None
    cur = None
    for line in txt.splitlines():
        m = RE_ROUND.search(line)
        if m:
            last_coins = int(m.group("c0" if args.player == 0 else "c1"))
            continue
        m = RE_ACT.search(line)
        if m:
            acts[int(m.group("r"))] = [(int(a), int(b), int(c)) for a, b, c in
                                       RE_OP.findall(m.group("ops"))]
            continue
        m = RE_TOPK.search(line)
        if m and cur is not None:
            cur["topk"] = int(m.group("n"))
            cur["comp"] = m.group("comp")
            cur["top8"] = [(x.group("b"), float(x.group("s")))
                           for x in RE_P1.finditer(m.group("top"))]
            continue
        m = RE_CH.search(line)
        if m and cur is not None:
            cur["chosen"] = dict(bundle=m.group("b"), cat=m.group("cat"),
                                 s2=_num(m.group("s2")), deg=m.group("deg"),
                                 best_nh=_num(m.group("bn")), best_nh_cat=m.group("bnc"),
                                 best_nh_b=m.group("bnb"), s_hold=_num(m.group("sh")),
                                 gap=_num(m.group("gap")))
            decs.append(cur)
            cur = None
            continue
        if "[candq] 阶段1结束" in line:
            if cur is not None:
                decs.append(cur)
            cur = {"coins": last_coins, "topk": None, "comp": None, "top8": None,
                   "chosen": None}
    if cur is not None:
        decs.append(cur)

    n_rounds = len(acts)
    n_emit_light = sum(1 for r, ops in acts.items() if any(t == LIGHTNING for (t, _a, _b) in ops))
    n_emit_light_ops = sum(1 for ops in acts.values() for (t, _a, _b) in ops if t == LIGHTNING)
    emit(f"[lp] 决策/回合数 = {len(decs)}（有 [act] 的回合 {n_rounds}）；"
         f"日志 {sp.name}")
    emit("")
    emit("## 1. 我方**真的发出**的闪电（op 21）")
    emit(f"  发过闪电的回合：{n_emit_light}/{n_rounds} = {pct(n_emit_light, n_rounds)}")
    emit(f"  闪电 op 总数：{n_emit_light_ops}")
    emit("  （历史 pin 参考：`INJECT_VS_RU_B` 那批 14 局里我方 USE_LIGHTNING_STORM 8–14 次/局，"
         "512 回合 ⇒ 约 1.6–2.7% 的回合；本局口径是「回合数/回合数」，可直接对比。）")
    # 金币分布，看有多少回合"放得起闪电"
    coins_all = [d["coins"] for d in decs if d["coins"] is not None]
    if coins_all:
        emit(f"  金币（决策前）：中位 {stats.median(coins_all):.0f} min {min(coins_all)} "
             f"max {max(coins_all)}")
        n90 = sum(1 for c in coins_all if c >= 90)
        emit(f"  金币 ≥90（**买得起 90 金币闪电**）的决策：{n90}/{len(coins_all)} = "
             f"{pct(n90, len(coins_all))}")

    # ── B. 离线复算：候选里含 LIGHTNING 的决策 ────────────────────────────────
    cand_light = None
    if not args.no_offline:
        try:
            rsc = _load("_rsc_lp", REPO / "code" / "test_match" / "repro_search_crash.py")
            rounds = rsc.read_ops(args.tag, f"{args.seed}.log")
            if not rounds:
                emit(f"  ⚠️ 离线复算跳过：`{args.tag}/{args.seed}.log` 没有 --trace-ops 行")
            else:
                env = _base_env()
                import os
                for k, v in env.items():
                    os.environ[k] = v
                os.environ["AZAI_TORCH_THREADS"] = "1"
                os.environ["AZAI_CAND_MODE"] = "unified"
                os.environ["AZAI_CAND_DECODE"] = str(args.decode)
                from SDK.backend.model import Operation
                from SDK.utils.constants import OperationType
                from my_ai.az_intent.game_state_facade import GameStateFacade
                bridge = _load("az_bridge_ai_lp", REPO / "code" / "test_match" / "az_bridge_ai.py")
                searcher, _mdl, _feat = bridge.build_engine(args.seed, 0)
                gen = searcher.cand_gen
                gen.rollout_fn = searcher.idle_rollout_values
                st = GameStateFacade.initial(seed=args.seed, cold_handle_rule_illegal=True)
                me = int(args.player)
                cand_light = []      # 每决策：(coins, n_cand, n_light_cand)
                for r in range(0, max(rounds) + 1):
                    coins = int(st.coins[me])
                    net_out = searcher.net_fn(st.clone(), me)
                    res = gen.generate(st.clone(), me, net_out, k=gen.k)
                    nb = res["bundles"]
                    nl = sum(1 for b in nb if any(int(o[0]) == LIGHTNING for o in b))
                    cand_light.append((coins, len(nb), nl))
                    for pl in (0, 1):
                        ops = rounds.get(r, {}).get(pl, [])
                        if ops:
                            st.apply_operation_list(pl, [Operation(OperationType(t), a0, a1)
                                                         for (t, a0, a1) in ops])
                    st.advance_round()
                tot = len(cand_light)
                withl = sum(1 for (_c, _n, nl) in cand_light if nl > 0)
                emit("")
                emit("## 2. **候选里含闪电**（离线复算同一局的生成器，精确值）")
                emit(f"  含至少 1 条闪电候选的决策：{withl}/{tot} = {pct(withl, tot)}")
                emit(f"  闪电候选条数合计：{sum(nl for (_c, _n, nl) in cand_light)} / "
                     f"候选总条数 {sum(n for (_c, n, _nl) in cand_light)} = "
                     f"{pct(sum(nl for (_c, _n, nl) in cand_light), sum(n for (_c, n, _nl) in cand_light))}")
                c90 = [(c, nl) for (c, _n, nl) in cand_light if c >= 90]
                emit(f"  金币 ≥90 的决策里，含闪电候选的："
                     f"{sum(1 for (_c, nl) in c90 if nl > 0)}/{len(c90)}")
        except Exception as exc:      # noqa: BLE001
            emit(f"  ⚠️ 离线复算失败（不影响日志部分）：{type(exc).__name__}: {exc}")

    # ── C. 临界金币定价（70–89）─────────────────────────────────────────────
    emit("")
    emit("## 3. 临界金币（70–89，差一点够 90 的闪电）的定价行为")
    near = [d for d in decs if d["coins"] is not None and 70 <= d["coins"] < 90]
    emit(f"  临界金币的决策数：{len(near)}")
    if near:
        def _has(tag: str, d) -> bool:
            c = d.get("comp") or ""
            m = re.search(re.escape("'" + tag + "'") + r": (\d+)", c)
            return bool(m and int(m.group(1)) > 0)
        n_build = sum(1 for d in near if _has("建塔", d))
        n_upg = sum(1 for d in near if _has("升级", d))
        n_hold = sum(1 for d in near if _has("HOLD", d))
        n_other = sum(1 for d in near if _has("其它", d))
        emit(f"  候选里有 建塔 / 升级 / HOLD / 其它(含闪电) 的回合："
             f"{n_build} / {n_upg} / {n_hold} / {n_other}")
        chosen_cat = Counter(d["chosen"]["cat"] for d in near if d.get("chosen"))
        emit(f"  最终选中了什么：{dict(chosen_cat)}")
        # 建塔/升级 是否被评估器排在 HOLD 前面
        wins = []
        for d in near:
            c = d.get("chosen")
            if not c:
                continue
            bh = c["best_nh_cat"]
            if ("建塔" in bh or "升级" in bh) and c["gap"] == c["gap"]:
                wins.append(c["gap"])
        n_beat = sum(1 for g in wins if g > 0)
        emit(f"  临界回合里『**最好的非 HOLD 是建塔/升级**』的决策：{len(wins)}")
        if wins:
            emit(f"    其中 建塔/升级 的阶段 2 分**高于 HOLD**（gap>0，= 评估器宁可花钱）的："
                 f"{n_beat}/{len(wins)} = {pct(n_beat, len(wins))}")
            emit(f"    gap（非HOLD最高−HOLD）分布：中位 {_f(stats.median(wins))} "
                 f"min {_f(min(wins))} max {_f(max(wins))}")
    # 对照：金币 ≥90（真放得起闪电）的回合怎么定价
    rich = [d for d in decs if d["coins"] is not None and d["coins"] >= 90]
    if rich:
        emit("")
        emit(f"## 3b. 对照 —— 金币 ≥90（放得起闪电）的 {len(rich)} 个决策")
        emit(f"  选中类别：{dict(Counter(d['chosen']['cat'] for d in rich if d.get('chosen')))}")
        n_light_c = sum(1 for d in rich if (d.get("comp") or "").find("'其它'") >= 0)
        emit(f"  候选里含『其它』(闪电/超武/基地升级) 的：{n_light_c}/{len(rich)}")

    if args.out:
        p = Path(args.out)
        if not p.is_absolute():
            p = REPO / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"[lp] 已写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
