"""**只读探针**：类轴（class axis）层面到底能不能表达 `UPGRADE_TOWER`？

回答三个问题，全部在**真实对局重放**出来的状态上算（不跑对局）：

  Q1 解码器的 16 个「塔类」通道里，**有几个能把升级表达出来**？
     （`decoder.CHANNEL_TO_TOWER_TYPE` 把 0-15 号类编码成"**目标塔类型**"；
      格子上已有我们的塔时 `_decode_tower_action` 走 `upgrade_step` ⇒ 出 `UPGRADE_TOWER`）
  Q2 `pos_pin=mc_quiet` 实际**钉住的是哪个类**？（复刻 `bundle_mcts.py:700-712` 的
     `_econ_rank`：`argmax_c max_{合法格} action_map[c]`，c ∈ 0..16）
     ⇒ 菜单 = {17?, 23} ∪ {(经济类, 格)}，而 `pinned = [_ccls]*3`（`bundle_mcts.py:870`）
     ⇒ **整束 48 个候选全部同属一个类**；只要那个类不是 1..15，束里就**不可能**有升级。
  Q3 真局面上「引擎认定合法」的升级 op 有多少？（`facade.can_apply_operation` 逐条问）

用法：
    python code/test_match/probe_upgrade_class_axis.py --tag INJECT_VS_RU_B --games 7,12
    python code/test_match/probe_upgrade_class_axis.py --tag INJECT_VS_RU_B --games 7 --json out.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO, REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402

from SDK.backend.model import Operation                      # noqa: E402
from SDK.utils.constants import OperationType, TowerType, TOWER_UPGRADE_TREE  # noqa: E402
from my_ai.decoder import (                                  # noqa: E402
    CHANNEL_TO_TOWER_TYPE,
    make_class_mask,
    make_position_masks,
    upgrade_step,
)
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

UPG = int(OperationType.UPGRADE_TOWER)

OPLINE_RE = re.compile(r"\[round (\d+)\] (p[01])\(\S+?\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OPOBJ_RE = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")
TOWERS_RE = re.compile(r"^\s*\[round (\d+)\] towers=\[(.*)\]\s*$")


def parse_ladder(path: Path):
    """→ (ops_by_round, towers_by_round)"""
    txt = path.read_text(encoding="utf-8", errors="replace")
    ops: dict[int, list[list[tuple[int, int, int]]]] = {}
    for m in OPLINE_RE.finditer(txt):
        rnd, side = int(m.group(1)), (0 if m.group(2) == "p0" else 1)
        row = ops.setdefault(rnd, [[], []])
        row[side] = [(int(o.group(2)), int(o.group(3)), int(o.group(4)))
                     for o in OPOBJ_RE.finditer(m.group(4))]
    tws: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
    for line in txt.splitlines():
        m = TOWERS_RE.match(line)
        if not m:
            continue
        rows = []
        for t in re.finditer(r"\(([^()]*)\)", m.group(2)):
            parts = [p.strip() for p in t.group(1).split(",")]
            if len(parts) >= 6:
                rows.append(tuple(int(float(p)) for p in parts[:6]))
        tws.setdefault(int(m.group(1)), []).extend(rows)
    return ops, tws


def our_side(token: str) -> int:
    return 1 if token.strip().endswith("r") else 0


def alive_channels(pos_mask: np.ndarray, state, player: int) -> dict[int, int]:
    """每个类通道里，**会把 op 解成 UPGRADE_TOWER** 的格子数（其余格子是建塔/None）。"""
    out: dict[int, int] = {}
    towers = {(int(t.x), int(t.y)): t for t in state.towers_of(player)}
    for cid in range(16):
        target = CHANNEL_TO_TOWER_TYPE[cid]
        n = 0
        xi, yi = np.where(pos_mask[cid])
        for x, y in zip(xi.tolist(), yi.tolist()):
            tw = towers.get((int(x), int(y)))
            if tw is None:
                continue
            if upgrade_step(tw.tower_type, target) is not None:
                n += 1
        out[cid] = n
    return out


def op_types_for_class(state, player: int, cid: int, pos_mask: np.ndarray) -> dict[int, int]:
    """**候选束的可达 op 类型分布**：把类 `cid` 的合法格逐个按解码器口径解一遍。

    ⚠️ 这是**支撑集**分析：`_sample_position`（`decoder.py:231-274`）在 t_pos>0 时对
    **全部合法格**做 z-score softmax 抽样 ⇒ 候选束里可能出现的 op = 这里枚举出的集合。
    """
    from my_ai.decoder import _decode_tower_action

    out: dict[int, int] = {}
    xi, yi = np.where(pos_mask[cid])
    for x, y in zip(xi.tolist(), yi.tolist()):
        if 0 <= cid <= 15:
            op = _decode_tower_action(state, player, cid, int(x), int(y))
            t = -1 if op is None else int(op.op_type)
        elif cid == 16:
            t = int(OperationType.DOWNGRADE_TOWER)
        elif 17 <= cid <= 20:
            t = {17: int(OperationType.USE_LIGHTNING_STORM),
                 18: int(OperationType.USE_EMP_BLASTER),
                 19: int(OperationType.USE_DEFLECTOR),
                 20: int(OperationType.USE_EMERGENCY_EVASION)}[cid]
        else:
            t = -2      # 21/22/23：无位置 / HOLD
        out[t] = out.get(t, 0) + 1
    return out


def econ_rank(action_map, cls_mask, pos_mask, no_downgrade: bool = False):
    """复刻 `bundle_mcts.py:700-712`（pos_pin=mc_quiet 的经济类排序）。"""
    hi = 16 if no_downgrade else 17
    rank = []
    for c in range(0, hi):
        if cls_mask[c] and pos_mask[c].any():
            v = float(np.where(pos_mask[c], action_map[c], -np.inf).max())
            if np.isfinite(v):
                rank.append((v, int(c)))
    rank.sort(key=lambda x: -x[0])
    return rank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--games", default="", help="逗号分隔 token；空 = 前 2 局")
    ap.add_argument("--ckpt", default="checkpoints/posnet_A_k5_m32.pt")
    ap.add_argument("--every", type=int, default=1, help="每 N 回合抽一次（省算力）")
    ap.add_argument("--json", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ladder = REPO / "match_results" / "ladder_logs" / args.tag
    if not ladder.is_dir():
        print(f"未找到 {ladder}")
        return 1
    tokens = [t.strip() for t in args.games.split(",") if t.strip()] or ["7", "12"]

    import torch
    torch.set_num_threads(4)
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt

    feat = FeatureExtractor(max_actions=96)
    ckpt = Path(args.ckpt)
    ckpt = ckpt if ckpt.is_absolute() else (REPO / ckpt)
    model, net_fn = make_net_fn_from_ckpt(str(ckpt), feat, value_tanh=True)
    print(f"[net] ckpt={ckpt.name} device={args.device}")

    # ── 静态事实：哪些通道能把 BASIC 塔升上去 ────────────────────────────────
    print("=" * 100)
    print("§0 类通道 → 目标塔类型（`decoder.CHANNEL_TO_TOWER_TYPE`，0-15）与"
          "「从 BASIC 出发能否一步升到」")
    print("-" * 100)
    from_basic = []
    for cid in range(16):
        tgt = CHANNEL_TO_TOWER_TYPE[cid]
        step = upgrade_step(TowerType.BASIC, tgt)
        ok = step is not None
        if ok:
            from_basic.append(cid)
        mark = "✅ 可一步升到" if ok else "❌ 从 BASIC 不可达（要先有别的二级塔）"
        print(f"  类 {cid:>2} → {tgt.name:<14} value={int(tgt):<3} {mark}")
    print(f"  ⇒ 从 **BASIC（一级）** 出发，能表达升级的类 = {from_basic}"
          f"（{len(from_basic)}/16；唯一不行的是类 0 = 目标就是 BASIC 本身 ⇒ `upgrade_step` 返回 None）")
    print("  ⇒ 机制说明：`upgrade_step(BASIC, 三级目标)` 返回的是**中间步**（例如类 2 → 升到 HEAVY）"
          "⇒ 任何 1-15 类在 BASIC 塔格里都会产出 `UPGRADE_TOWER`")

    # ── 重放 + 每回合算类轴 ────────────────────────────────────────────────
    agg = {
        "rounds": 0, "games": {},
        "econ_top": {}, "econ_in_upg_class": 0, "head_argmax": {},
        "alive_chan_any": {}, "upg_legal_ops": 0, "upg_legal_rounds": 0,
        "pin_support_tot": 0, "pin_support_has_upg": 0, "pin_cells": 0,
        "pin_in_upg_range": 0, "if_pin_upg_cells": 0, "if_pin_upg_states": 0,
        "pinned_possible": {},
    }
    for tok in tokens:
        side = our_side(tok)
        opp = 1 - side
        logp = ladder / f"{tok}.log"
        if not logp.is_file():
            print(f"[!] 缺 {logp}")
            continue
        ops, tws = parse_ladder(logp)
        if not ops:
            print(f"[!] {logp.name} 没有 --trace-ops 行")
            continue
        seed = int(re.sub(r"[^0-9]", "", tok) or "0")
        st = GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)
        g = {"rounds": 0, "econ": {}, "alive": {}, "legal_upg": 0, "alive_rounds": 0,
             "mismatch": 0}
        for rnd in sorted(ops):
            if st.terminal:
                break
            # 校验重放：塔态是否与对局日志一致（round R 的行 = 本回合操作**前**）
            if rnd in tws:
                want = sorted((t[0], t[1], t[2], t[3], int(t[4])) for t in tws[rnd])
                got = sorted((int(t.tower_id), int(t.player), int(t.x), int(t.y),
                              int(t.tower_type)) for t in st.towers)
                if want != got:
                    g["mismatch"] += 1
            if rnd % max(1, args.every) == 0:
                _class_axis(st, side, net_fn, feat, agg, g)
            # 本回合双方操作 → 推进
            for pl in (0, 1):
                pl_ops = ops[rnd][pl]
                if pl_ops:
                    st.apply_operation_list(pl, [Operation(OperationType(int(t)), int(a0), int(a1))
                                                 for (t, a0, a1) in pl_ops])
            st.advance_round()
        agg["games"][tok] = g
        print(f"[game {tok}] 重放回合={len(ops)} 塔态不符={g['mismatch']} "
              f"合法升级回合={g['alive_rounds']} 合法升级 op={g['legal_upg']}")

    # ── 汇总 ───────────────────────────────────────────────────────────────
    n = max(1, agg["rounds"])
    print("\n" + "=" * 100)
    print(f"§1 mc_quiet 的『经济类』（= 整束候选被钉住的类）分布，n={agg['rounds']} 个决策点")
    print("-" * 100)
    tot = sum(agg["econ_top"].values())
    for c, k in sorted(agg["econ_top"].items(), key=lambda x: -x[1]):
        tag_c = " ← **能表达升级**" if 1 <= int(c) <= 15 else ""
        print(f"  类 {c:>2}：{k:>6}  {100.0*k/max(1,tot):>6.1f}%{tag_c}")
    print(f"  ⇒ 经济类落在 1..15（唯一能表达升级的区间）的比例："
          f"{agg['econ_in_upg_class']}/{tot} = {100.0*agg['econ_in_upg_class']/max(1,tot):.2f}%"
          f"（另有 {agg['rounds']-tot} 个决策点连一个合法经济类都没有）")

    print("\n" + "=" * 100)
    print("§2 类头 argmax（3 个 head 各自）")
    print("-" * 100)
    for h, d in sorted(agg["head_argmax"].items()):
        top = sorted(d.items(), key=lambda x: -x[1])[:6]
        s = "  ".join(f"类{c}:{k}({100.0*k/max(1,tot):.1f}%)" for c, k in top)
        print(f"  head{h}: {s}")

    print("\n" + "=" * 100)
    print("§3 引擎判定的『合法升级 op』（can_apply_dry）")
    print("-" * 100)
    print(f"  有 >=1 个引擎认定合法的升级 op 的决策点：{agg['upg_legal_rounds']}/{tot} "
          f"= {100.0*agg['upg_legal_rounds']/max(1,tot):.1f}%")
    print(f"  这些决策点上累计合法升级 op 数：{agg['upg_legal_ops']}")

    print("\n" + "=" * 100)
    print("§4 每一类通道『能解成升级的格数』（在本批真实状态上）")
    print("-" * 100)
    for c, k in sorted(agg["alive_chan_any"].items(), key=lambda x: -x[1]):
        print(f"  类 {c:>2} ({CHANNEL_TO_TOWER_TYPE[int(c)].name:<14})："
              f"{k} 个状态里有 >=1 个这样的格")

    print("\n" + "=" * 100)
    print("§5 候选束的**可达 op 类型**（钉住的类 = `_c0`，位置从该类的合法格里抽样）")
    print("-" * 100)
    n_sup = max(1, agg["pin_support_tot"])
    print(f"  决策点 {n_sup} 个，累计合法格 {agg['pin_cells']} 个")
    print(f"  ★ 支撑集里**含** UPGRADE_TOWER 的决策点：{agg['pin_support_has_upg']}/{n_sup} "
          f"= {100.0*agg['pin_support_has_upg']/n_sup:.2f}%"
          f"  ⇒ 现役配置下整束 48 个候选里**出现升级的概率 = 0**")
    _ns = max(1, agg.get("if_pin_samples", 0))
    print(f"  （对照）若整束被钉在类 1..15 上：抽样 {_ns} 个决策点 × 15 个类 = "
          f"{15*_ns} 个 (类,决策点) 组合里，{agg['if_pin_upg_states']} 个组合有 >=1 个能解成升级的格"
          f"（{100.0*agg['if_pin_upg_states']/max(1,15*_ns):.1f}%，与 §3 的『有升级机会』同量级）"
          f"，累计 {agg['if_pin_upg_cells']} 个 (类,格) 组合")
    print(f"  钉住类落在 1..15 的决策点：{agg['pin_in_upg_range']}/{n_sup}")

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({k: v for k, v in agg.items() if k != "games"},
                                ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[json] -> {p}")
    return 0


def _class_axis(st, side: int, net_fn, feat, agg: dict, g: dict) -> None:
    """在**我方决策时刻** t 上算类轴：经济类 / 类头 argmax / 可用升级通道。"""
    out = net_fn(st, side)
    am = np.asarray(out["action_map"], dtype=np.float32)
    hls = [np.asarray(h, dtype=np.float32) for h in out["head_logits"]]
    pos_mask = make_position_masks(st, side, intent_decoding=True)
    cls_mask = make_class_mask(st, side, position_mask=pos_mask, intent_decoding=True)

    rank = econ_rank(am, cls_mask, pos_mask)
    if rank:
        c0 = int(rank[0][1])
        agg["econ_top"][c0] = agg["econ_top"].get(c0, 0) + 1
        g["econ"][c0] = g["econ"].get(c0, 0) + 1
        if 1 <= c0 <= 15:
            agg["econ_in_upg_class"] += 1
    for h, hl in enumerate(hls):
        a = int(np.argmax(hl))
        d = agg["head_argmax"].setdefault(h, {})
        d[a] = d.get(a, 0) + 1

    al = alive_channels(pos_mask, st, side)
    g["alive"] = g["alive"] or {}
    for c, k in al.items():
        if k > 0:
            agg["alive_chan_any"][c] = agg["alive_chan_any"].get(c, 0) + 1
            g["alive"][c] = g["alive"].get(c, 0) + 1
    agg["rounds"] += 1
    g["rounds"] += 1

    # ── 候选束的**可达 op 类型**：钉住的类 = _c0（`bundle_mcts.py:870` 里 pinned=[_ccls]*3）──
    if rank:
        c_star = int(rank[0][1])
        sup = op_types_for_class(st, side, c_star, pos_mask)
        agg["pin_support_tot"] += 1
        agg["pin_cells"] += sum(sup.values())
        if sup.get(UPG, 0) > 0:
            agg["pin_support_has_upg"] += 1
        # 如果整束被钉在 1..15 的某个类上，会怎样（= 修复后的上界）。抽样算（每 8 个决策点一次）
        if 1 <= c_star <= 15:
            agg["pin_in_upg_range"] += 1
        if agg["rounds"] % 8 == 1:
            for _c in range(1, 16):
                _s = op_types_for_class(st, side, _c, pos_mask)
                _n_up = _s.get(UPG, 0)
                agg["if_pin_upg_cells"] += _n_up
                if _n_up > 0:
                    agg["if_pin_upg_states"] += 1
            agg["if_pin_samples"] = agg.get("if_pin_samples", 0) + 1

    # 引擎合法升级 op（逐条 can_apply_dry）
    coins = int(st.coins[side])
    n_legal = 0
    for tw in st.towers_of(side):
        for tgt in TOWER_UPGRADE_TREE.get(tw.tower_type, ()):
            if coins < st.upgrade_tower_cost(tgt):
                continue
            op = Operation(OperationType(UPG), int(tw.tower_id), int(tgt))
            if st.can_apply_operation(side, op):
                n_legal += 1
    if n_legal:
        agg["upg_legal_rounds"] += 1
        agg["upg_legal_ops"] += n_legal
        g["legal_upg"] += n_legal
        g["alive_rounds"] += 1


if __name__ == "__main__":
    raise SystemExit(main())
