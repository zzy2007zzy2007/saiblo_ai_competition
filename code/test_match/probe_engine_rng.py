"""引擎 RNG 只读探针（不改任何现役代码，只读 native_game.pyd）。

回答三个问题：
  1. `determinism`：同 seed + 同操作序列 ⇒ 逐回合状态是否逐位相同？
     同时也测"一个进程里先后/交错跑两局"是否互相污染（引擎的 pheromone LCG 是
     文件级全局变量，见 Ant-Game/game/src/map.cpp:3）。
  2. `census`：每回合到底消费了几次引擎 RNG？
     引擎 RNG 是 48-bit LCG（game.cpp:174-189），状态可由 `rng_state_now()` 读到，
     所以可以从"回合前后的状态"**反解**出这一回合推进了几步（乘子是奇数 ⇒ 双射，
     k 在 2^48 周期内唯一）。
  3. `clone`：从同一状态 clone 出的两条分支，跑同一串操作是否逐位相同
     （⇒ 引擎 RNG 在 flat rollout 内提供 0 多样性，因为 rng_state 被一起 clone 了）。

用法：
    python code/test_match/probe_engine_rng.py determinism --log <log>
    python code/test_match/probe_engine_rng.py census     --log <log>
    python code/test_match/probe_engine_rng.py interleave --log <log> --cut 120
    python code/test_match/probe_engine_rng.py clone      --log <log> --cut 120
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game",
           REPO / "code" / "test_match"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import probe_replay_engines  # noqa: E402
from probe_replay_engines import parse_log, parse_ops_text  # noqa: E402

PROBE_DIR = Path(r"C:\Temp\rng_probe\out")

# ── 一个进程只加载**一个**引擎模块 ──
# pybind11 会报 `type "NativeGame" is already registered`，所以"同一进程里同时
# import native_game 与 native_game_rng"是做不到的 ⇒ 引擎选择放在进程级；
# 需要跨引擎比较时由 `parity` 模式起两个子进程各自算摘要再比。
MODULE = "live"
ENGINE_DIR = ""      # 可选：把另一个目录里的 native_game.pyd 顶到 sys.path 最前面
_ENG = None


def ENG():
    global _ENG
    if _ENG is None:
        if MODULE == "probe":
            if str(PROBE_DIR) not in sys.path:
                sys.path.insert(0, str(PROBE_DIR))
            import native_game_rng as _m
        else:
            if ENGINE_DIR:
                sys.path.insert(0, ENGINE_DIR)
            import native_game as _m
        _ENG = _m
    return _ENG


# ── 引擎 RNG 常量（Ant-Game/game/src/game.cpp:25-27 逐字复刻）──
RNG_MASK = (1 << 48) - 1
RNG_MULT = 25214903917
RNG_INC = 11


def lcg_step(s: int) -> int:
    return (RNG_MULT * s + RNG_INC) & RNG_MASK


def draws_between(s0: int, s1: int, limit: int = 4_000_000) -> int | None:
    """从 s0 前推多少步能到 s1（含 0 步）；找不到返回 None。"""
    if s0 == s1:
        return 0
    s = s0
    for k in range(1, limit):
        s = lcg_step(s)
        if s == s1:
            return k
    return None


# ── 状态摘要：把"会不会影响未来"的所有可观测量都灌进 sha256 ──
def digest(g) -> str:
    h = hashlib.sha256()
    h.update(f"r={g.round()} hp={g.base_hp(0)},{g.base_hp(1)} "
             f"coin={g.coin(0)},{g.coin(1)} w={g.winner()} "
             f"end={int(g.is_ended())} rng={g.rng_state_now()}".encode())
    for t in g.tower_details():
        h.update(b"T" + repr(t).encode())
    for a in g.ant_details():
        h.update(b"A" + repr(a).encode())
    for p in (0, 1):
        h.update(b"C" + repr(list(g.weapon_cds(p))).encode())
        h.update(b"L" + repr(list(g.base_levels(p))).encode())
    h.update(b"E" + repr(g.active_effects()).encode())
    ph = g.pheromone_flat()
    h.update(struct.pack(f"<{len(ph)}d", *ph))
    return h.hexdigest()[:16]


def load_ops(log: Path):
    info = parse_log(log)
    if info["seed"] is None:
        raise SystemExit(f"[probe] 无法从日志解析 seed: {log}")
    # parse_ops_text 需要 python 引擎的 Operation 类（只做协议解析，不跑规则）
    from SDK.backend.engine import Operation, OperationType
    rows = {}
    for rnd, per_side in info["ops_text"].items():
        rows[rnd] = {p: parse_ops_text(per_side.get(p, "0\n"), Operation, OperationType)
                     for p in (0, 1)}
    n = max(rows) + 1 if rows else 0
    full = {r: rows.get(r, {0: [], 1: []}) for r in range(n)}
    for r in full:
        for p in (0, 1):
            full[r].setdefault(p, [])
    return info, full, n


def as_raw(ops):
    return [(int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops]


# ─────────────────────────── modes ───────────────────────────
def mode_determinism(args) -> int:
    info, rows, n = load_ops(args.log)
    print(f"[log] {args.log} seed={info['seed']} 回合数={n} "
          f"记录终局={info['result']}")

    def play():
        g = ENG().NativeGame()
        g.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
        digs, rngs = [], []
        for r in range(n):
            for p in (0, 1):
                raw = as_raw(rows[r][p])
                if raw:
                    g.apply_operation_list(p, raw)
            g.advance_round()
            digs.append(digest(g))
            rngs.append(g.rng_state_now())
            if g.is_ended():
                break
        return g, digs, rngs

    g1, d1, r1 = play()
    g2, d2, r2 = play()
    print(f"[run1] {len(d1)} 回合 终局 hp={g1.base_hp(0)},{g1.base_hp(1)} "
          f"rng={r1[-1]}")
    print(f"[run2] {len(d2)} 回合 终局 hp={g2.base_hp(0)},{g2.base_hp(1)} "
          f"rng={r2[-1]}")
    same = d1 == d2
    first = next((i for i, (a, b) in enumerate(zip(d1, d2)) if a != b), None)
    print(f"[同进程 二次运行] 逐回合粗摘要完全一致: {same}"
          + ("" if same else f"  首个分歧回合={first}"))
    print(f"[同进程 二次运行] rng_state 序列完全一致: {r1 == r2}")

    # 交错跑第二局：检验 map.cpp 的文件级全局 lcg_seed 是否造成跨局污染
    if n >= 2:
        ga = ENG().NativeGame()
        ga.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
        gb = ENG().NativeGame()
        gb.init_game(seed=12345, movement_policy="enhanced", cold=True)
        digs_a = []
        for r in range(n):
            for p in (0, 1):
                raw = as_raw(rows[r][p])
                if raw:
                    ga.apply_operation_list(p, raw)
            gb.advance_round()          # 另一局穿插推进（用另一个 seed）
            ga.advance_round()
            digs_a.append(digest(ga))
            if ga.is_ended():
                break
        print(f"[交错另一局] 与单独运行逐回合摘要一致: {digs_a == d1[:len(digs_a)]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"digests": d1, "rng": r1}, indent=1), encoding="utf-8")
    print(f"[out] {out}")
    return 0 if same else 1


def mode_census(args) -> int:
    info, rows, n = load_ops(args.log)
    g = ENG().NativeGame()
    g.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
    per_round, ants_per_round = [], []
    fails = 0
    for r in range(n):
        for p in (0, 1):
            raw = as_raw(rows[r][p])
            if raw:
                g.apply_operation_list(p, raw)
        s0 = g.rng_state_now()
        pre_ants = len(g.ant_details())
        g.advance_round()
        s1 = g.rng_state_now()
        k = draws_between(s0, s1)
        if k is None:
            fails += 1
        per_round.append(k)
        ants_per_round.append(pre_ants)
        if g.is_ended():
            break
    ok = [k for k in per_round if k is not None]
    total = sum(ok)
    nonzero = [k for k in ok if k > 0]
    print(f"[census] {args.log.name if hasattr(args.log, 'name') else args.log}")
    print(f"  回合数                 : {len(per_round)}（反解失败 {fails}）")
    print(f"  RNG 总消费次数         : {total}")
    print(f"  平均/回合              : {total / max(1, len(ok)):.2f}")
    print(f"  消耗>0 的回合          : {len(nonzero)}/{len(ok)} "
          f"({100.0 * len(nonzero) / max(1, len(ok)):.1f}%)")
    if ok:
        srt = sorted(ok)
        print(f"  单回合最少/中位/最多   : {srt[0]} / {srt[len(srt) // 2]} / {srt[-1]}")
    print(f"  终局 hp={g.base_hp(0)},{g.base_hp(1)} 记录={info['result']}")
    print("  逐回合消费次数（前 40）: " + ",".join(str(k) for k in per_round[:40]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_round": per_round,
                               "ants_before_round": ants_per_round}, indent=1),
                   encoding="utf-8")
    print(f"[out] {out}")
    return 0


def mode_clone(args) -> int:
    info, rows, n = load_ops(args.log)
    cut = min(args.cut, n - 2)
    g = ENG().NativeGame()
    g.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
    for r in range(cut):
        for p in (0, 1):
            raw = as_raw(rows[r][p])
            if raw:
                g.apply_operation_list(p, raw)
        g.advance_round()
        if g.is_ended():
            print("[clone] 对局在 cut 之前结束，无法测试")
            return 1
    print(f"[clone] 在第 {cut} 回合切片：rng_state={g.rng_state_now()} "
          f"hp={g.base_hp(0)},{g.base_hp(1)} ants={len(g.ant_details())}")
    a, b = g.clone(), g.clone()
    print(f"[clone] 两个 clone 的初始摘要相同: {digest(a) == digest(b)}")
    horizon = min(args.horizon, n - cut)
    da, db = [], []
    for i in range(horizon):
        r = cut + i
        for p in (0, 1):
            raw = as_raw(rows[r][p]) if r < n else []
            if raw:
                a.apply_operation_list(p, raw)
                b.apply_operation_list(p, raw)
        a.advance_round()
        b.advance_round()
        da.append(digest(a))
        db.append(digest(b))
        if a.is_ended():
            break
    identical = da == db
    first = next((i for i, (x, y) in enumerate(zip(da, db)) if x != y), None)
    print(f"[clone] 同一串操作推 {len(da)} 回合，逐回合摘要一致: {identical}"
          + ("" if identical else f"  首个分歧={first}"))
    print(f"[clone] 终局 hp: A={a.base_hp(0)},{a.base_hp(1)}  "
          f"B={b.base_hp(0)},{b.base_hp(1)}  rng A={a.rng_state_now()} "
          f"B={b.rng_state_now()}")
    return 0 if identical else 1


def mode_interleave(args) -> int:
    """同 seed 两局交错推进 vs 单独推进：只看前 cut 回合是否逐步相同。"""
    info, rows, n = load_ops(args.log)
    cut = min(args.cut, n)

    def play(after_each=None):
        g = ENG().NativeGame()
        g.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
        out = []
        for r in range(cut):
            for p in (0, 1):
                raw = as_raw(rows[r][p])
                if raw:
                    g.apply_operation_list(p, raw)
            g.advance_round()
            out.append(digest(g))
            if after_each is not None:
                after_each(r)
            if g.is_ended():
                break
        return out

    other = ENG().NativeGame()
    other.init_game(seed=424242, movement_policy="enhanced", cold=True)
    seq_alone = play()
    # 交错版：每回合先推进"别的局"，再推进本局
    g = ENG().NativeGame()
    g.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
    seq_mixed = []
    for r in range(cut):
        for p in (0, 1):
            raw = as_raw(rows[r][p])
            if raw:
                g.apply_operation_list(p, raw)
        other.advance_round()
        g.advance_round()
        seq_mixed.append(digest(g))
        if g.is_ended():
            break
    print(f"[interleave] 前 {len(seq_alone)} 回合：两局交错 vs 单独 "
          f"逐步一致: {seq_alone == seq_mixed}")
    print(f"[interleave] （对照：另一局自己的 rng={other.rng_state_now()}，"
          f"互不干扰则本局摘要不应变化）")
    return 0 if seq_alone == seq_mixed else 1


# ───────────── 探针版引擎（含 get/set/consume_rng）相关模式 ─────────────
SITE_NAMES = ["OTHER(consume_rng 等)", "choose_random_legal_move:236",
              "random_bewitch_target:815", "choose_ant_move_legacy:1025",
              "choose_ant_move_enhanced:1217", "Randomized_move_legacy:934",
              "Randomized_move_enhanced:1041", "resolve_random_move_steps:1290",
              "teleport_ants:1312", "drift_items:1333",
              "generate_ants:spawn_profile:1549", "generate_ants:siege:1684"]


def mode_digest(args) -> int:
    """跑一遍日志并把逐回合强摘要落盘（供 parity 跨进程比对）。"""
    info, rows, n = load_ops(args.log)
    g = ENG().NativeGame()
    g.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
    digs, rngs = [], []
    for r in range(n):
        for p in (0, 1):
            raw = as_raw(rows[r][p])
            if raw:
                g.apply_operation_list(p, raw)
        g.advance_round()
        digs.append(digest(g))
        rngs.append(g.rng_state_now())
        if g.is_ended():
            break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"module": MODULE, "digests": digs, "rng": rngs,
                               "hp": [int(g.base_hp(0)), int(g.base_hp(1))]},
                              indent=1), encoding="utf-8")
    print(f"[digest] module={MODULE} 回合={len(digs)} "
          f"终局hp={g.base_hp(0)},{g.base_hp(1)} rng={rngs[-1] if rngs else '-'}")
    print(f"[digest] out={out}")
    return 0


def mode_parity(args) -> int:
    """回归验证：探针版引擎在**不消费** RNG 时，是否与现役 pyd 逐位一致。

    必须分两个子进程跑（同一进程 import 两个 pyd 会被 pybind11 的
    `type already registered` 拒绝）。`--logs` 可传多条日志做批量对拍。
    """
    logs = [Path(x) for x in (args.logs or [])] or [args.log]
    engine_dir = args.engine_dir or str(REPO / "code/cpp_engine/rng_build")
    if not (Path(engine_dir) / "native_game.pyd").exists():
        print(f"[parity] ✗ {engine_dir} 里没有 native_game.pyd；"
              f"先跑 python code/cpp_engine/build_rng_variant.py")
        return 1
    print(f"[parity] 候选引擎 = {Path(engine_dir) / 'native_game.pyd'}")
    all_ok = True
    for log in logs:
        if not log.exists():
            print(f"[parity] ✗ 找不到日志 {log}")
            return 1
        tmp = Path(args.out).with_name(f"parity_live_{log.stem}.json")
        tmpp = Path(args.out).with_name(f"parity_cand_{log.stem}.json")
        for tag, extra, outp in (("live", [], tmp),
                                 ("cand", ["--engine-dir", str(engine_dir)], tmpp)):
            cmd = [sys.executable, __file__, "digest", "--log", str(log),
                   "--out", str(outp), "--module", "live", *extra]
            r = subprocess.run(cmd, capture_output=True, encoding="utf-8",
                               errors="replace")
            if r.returncode != 0:
                print(r.stdout.strip())
                print(r.stderr[-3000:])
                raise SystemExit(f"[parity] ✗ 子进程失败 {tag} log={log.name}")
        a = json.loads(tmp.read_text(encoding="utf-8"))
        b = json.loads(tmpp.read_text(encoding="utf-8"))
        same = a["digests"] == b["digests"]
        first = next((i for i, (x, y) in enumerate(zip(a["digests"], b["digests"]))
                      if x != y), None)
        rng_same = a["rng"] == b["rng"]
        print(f"[parity] {log.name:10s} 回合={len(a['digests']):4d} "
              f"hp={tuple(a['hp'])} vs {tuple(b['hp'])}  "
              f"逐回合强摘要一致={same}  rng_state 序列一致={rng_same}"
              + ("" if same else f"  首个分歧回合={first}"), flush=True)
        all_ok = all_ok and same and rng_same
    print(f"[parity] 汇总: {'✅ 全部逐位一致' if all_ok else '❌ 存在差异'} "
          f"（{len(logs)} 条日志）")
    return 0 if all_ok else 1


def mode_sites(args) -> int:
    """每个 RNG 消费点的调用次数（探针版引擎，逐回合差分）。"""
    info, rows, n = load_ops(args.log)
    g = ENG().NativeGame()
    g.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
    prev = list(g.rng_site_counts())
    rows_out = []
    for r in range(n):
        for p in (0, 1):
            raw = as_raw(rows[r][p])
            if raw:
                g.apply_operation_list(p, raw)
        g.advance_round()
        cur = list(g.rng_site_counts())
        delta = [c - p for c, p in zip(cur, prev)]
        prev = cur
        rows_out.append(delta)
        if g.is_ended():
            break
    tot = [sum(row[i] for row in rows_out) for i in range(len(SITE_NAMES))]
    grand = sum(tot)
    print(f"[sites] {args.log.name}  回合数={len(rows_out)}  总消费={grand}")
    print(f"  {'site':38s}{'次数':>8s}{'占比':>8s}")
    for i, name in enumerate(SITE_NAMES):
        if tot[i] or i == 0:
            print(f"  {name:38s}{tot[i]:8d}{100.0 * tot[i] / max(1, grand):7.1f}%")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_round": rows_out, "totals": tot,
                               "names": SITE_NAMES}, indent=1), encoding="utf-8")
    print(f"[out] {out}")
    return 0


HIGHLAND_CAND = None


def own_cells(player: int):
    """候选建塔格（等同 flat_rollout `_decide_actions` 用的 HIGHLAND_CELLS）。"""
    global HIGHLAND_CAND
    if HIGHLAND_CAND is None:
        from SDK.utils.constants import HIGHLAND_CELLS
        HIGHLAND_CAND = {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}
    return HIGHLAND_CAND[player]


def random_ops(g, player: int, rng, build_ratio: float = 0.7):
    """逐字复刻 flat_rollout_greedy._decide_actions 的单侧决策（只取一侧）。"""
    did = False
    out = []
    if rng.random() < build_ratio:
        tws = g.tower_details()
        n_tower = sum(1 for t in tws if t[3] == player)
        cost = g.tower_build_cost(n_tower)
        if g.coin(player) >= cost:
            occ = {(t[1], t[2]) for t in tws if t[3] == player}
            cand = [c for c in own_cells(player) if (c[0], c[1]) not in occ]
            if cand:
                cell = cand[int(rng.integers(len(cand)))]
                out.append((11, int(cell[0]), int(cell[1])))
                did = True
    if not did:
        mine = [t for t in g.tower_details() if t[3] == player]
        if mine:
            t = mine[int(rng.integers(len(mine)))]
            out.append((13, int(t[0]), -1))
    return out


def mode_perturb(args) -> int:
    """从真实对局切片出发，比较三种"同预算 N 分支"的叶子多样性。

    A engine : 只在**起点**多消费 k 次 RNG（k=分支号），之后双方都按日志原操作走
    B action : RNG 不动，对手（p1）每回合以概率 p 出随机招（≈ 现役配置的机制）
    C both   : A + B
    """
    ng2 = ENG()
    info, rows, n = load_ops(args.log)
    cut = min(args.cut, n - 2)
    base = ng2.NativeGame()
    base.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
    for r in range(cut):
        for p in (0, 1):
            raw = as_raw(rows[r][p])
            if raw:
                base.apply_operation_list(p, raw)
        base.advance_round()
    H = min(args.horizon, n - cut - 1)
    n_br = args.branches
    print(f"[perturb] 切片回合={cut} horizon={H} 分支数={n_br} "
          f"p={args.p} 起点 hp={base.base_hp(0)},{base.base_hp(1)} "
          f"ants={len(base.ant_details())} rng={base.get_rng_state()}")

    def np_random(seed):
        import numpy as _np
        return _np.random.default_rng(int(seed))

    def prob(round_idx: int) -> float:
        """对手出招概率：复刻 flat_rollout 的 const / decay / first 三种模式。"""
        import math as _math
        if args.prob_mode == "first":
            return args.decay_c if round_idx == 0 else 0.0
        if args.prob_mode == "decay":
            return args.decay_c * _math.exp(-round_idx / max(1e-6, args.decay_k))
        return args.p

    def run(variant: str, idx: int):
        rng = np_random(idx * 7919 + 13) if variant in ("action", "both",
                                                        "engine_pr", "reroll") else None
        sim = base.clone()
        if variant in ("engine", "both"):
            sim.consume_rng(idx)
        digs = []
        for i in range(H):
            r = cut + i
            ops = {0: [], 1: []}
            if variant == "engine_pr" and rng.random() < prob(i):
                sim.consume_rng(1)
            if variant == "reroll" and rng.random() < prob(i):
                # 把 RNG 状态直接重掷成随机值（= 换一条流，而不是"延后 k 步"）
                sim.set_rng_state(int(rng.integers(0, 1 << 48)))
            for p in (0, 1):
                if p == 1 and rng is not None and variant in ("action", "both"):
                    if rng.random() < prob(i):
                        ops[p] = random_ops(sim, p, rng)
                else:
                    ops[p] = as_raw(rows[r][p]) if r < n else []
            for p in (0, 1):
                if ops[p]:
                    sim.apply_operation_list(p, ops[p])
            sim.advance_round()
            digs.append(digest(sim))
            if sim.is_ended():
                break
        d_hp = int(sim.base_hp(1)) - int(sim.base_hp(0))
        return digs, d_hp, int(sim.base_hp(0)), int(sim.base_hp(1))

    for variant in ("engine", "engine_pr", "reroll", "action", "both"):
        sigs, dhps, hps, all_digs = [], [], [], []
        for k in range(n_br):
            digs, d_hp, h0, h1 = run(variant, k)
            sigs.append(tuple(digs))
            all_digs.append(digs)
            dhps.append(d_hp)
            hps.append((h0, h1))
        ref = all_digs[0]
        firsts = []
        for digs in all_digs[1:]:
            fd = next((i for i, (x, y) in enumerate(zip(ref, digs)) if x != y), None)
            firsts.append(len(ref) if fd is None else fd)
        import statistics as st
        k0 = dhps[0]                     # 分支 0 = 完全不扰动（就是引擎"真实"会走的流）
        mean = st.fmean(dhps)
        sd = st.pstdev(dhps)
        se = sd / max(1.0, len(dhps) ** 0.5)
        print(f"  [{variant:6s}] 不同叶子轨迹 {len(set(sigs))}/{n_br}"
              f"   不同(hp0,hp1) {len(set(hps))}/{n_br}"
              f"   Δhp(1-0) mean={mean:+7.2f} sd={sd:6.2f}"
              f"   k=0 分支={k0:+7.2f}  |mean-k0|/se={abs(mean - k0) / max(1e-9, se):5.2f}"
              f"   与分支0首次分歧回合 min/中位/最大="
              f"{min(firsts) if firsts else '-'}/"
              f"{(sorted(firsts)[len(firsts) // 2] if firsts else '-')}/"
              f"{max(firsts) if firsts else '-'}")
    return 0


def mode_crn(args) -> int:
    """**配对（common random numbers）实验**：同一个 RNG 偏移下比较两个候选。

    这正是"引擎 RNG 可控"最该被用到的场景：候选之间的**差**才是搜索要的东西，
    而 RNG 抖动是共模噪声 ⇒ 给所有候选喂同一串 RNG 偏移，差值上的噪声会被抵消。
    这里直接量：配对 vs 不配对时，Δ=score(B)-score(A) 的 sd / |t| 差多少。
    """
    info, rows, n = load_ops(args.log)
    cut = min(args.cut, n - 2)
    H = min(args.horizon, n - cut - 1)
    N = args.branches
    base = ENG().NativeGame()
    base.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
    for r in range(cut):
        for p in (0, 1):
            raw = as_raw(rows[r][p])
            if raw:
                base.apply_operation_list(p, raw)
        base.advance_round()

    # 选两个"都能过 can_apply_dry"的候选操作（一个真实的两选一决策）
    def find_two(g):
        out = []
        # 1) 日志里 p0 本回合真实出的招
        for op in as_raw(rows[cut][0]) if cut < n else []:
            if g.can_apply_dry(0, op, []):
                out.append(op)
        # 2) 建塔（自己半场空位）
        for (x, y) in own_cells(0):
            if g.can_apply_dry(0, [11, int(x), int(y)], []):
                out.append([11, int(x), int(y)])
        # 3) 降级自家塔
        for t in g.tower_details():
            if t[3] == 0 and g.can_apply_dry(0, [13, int(t[0]), -1], []):
                out.append([13, int(t[0]), -1])
        uniq, seen = [], set()
        for op in out:
            key = tuple(op)
            if key not in seen:
                seen.add(key)
                uniq.append(op)
        return uniq[:2]

    pair = find_two(base)
    if len(pair) < 2 and len(pair) == 1:
        pair = [[], pair[0]]     # 退化时用 "HOLD vs 唯一合法操作"
    if len(pair) < 2:
        print("[crn] ✗ 找不到两个合法候选操作")
        return 1
    A, B = pair
    print(f"[crn] 切片回合={cut} H={H} N={N}  候选A={A} 候选B={B}")

    def score(op, offset: int) -> int:
        sim = base.clone()
        if op:
            sim.apply_operation_list(0, [tuple(op)])
        sim.consume_rng(offset)
        for i in range(H):
            r = cut + i
            for p in (0, 1):
                raw = as_raw(rows[r][p]) if r < n else []
                if raw:
                    sim.apply_operation_list(p, raw)
            sim.advance_round()
            if sim.is_ended():
                break
        return int(sim.base_hp(1)) - int(sim.base_hp(0))

    import statistics as st
    sa = [score(A, j) for j in range(N)]
    sb = [score(B, j) for j in range(N)]
    paired = [b - a for a, b in zip(sa, sb)]
    shift = N // 2
    indep = [sb[(j + shift) % N] - sa[j] for j in range(N)]

    def report(tag, d):
        mean, sd = st.fmean(d), st.pstdev(d)
        se = sd / len(d) ** 0.5
        print(f"  [{tag:9s}] Δ mean={mean:+7.3f} sd={sd:6.3f} se={se:6.3f} "
              f"|t|={abs(mean) / max(1e-9, se):5.2f}  A/B 独立均值="
              f"{st.fmean(sa):+.2f}/{st.fmean(sb):+.2f}")

    print(f"  [单点 k=0] A={sa[0]:+d} B={sb[0]:+d} ⇒ 用'真实流'判 B-A="
          f"{sb[0] - sa[0]:+d}")
    report("配对 CRN", paired)
    report("不配对", indep)
    return 0


def mode_official(args) -> int:
    """官方 C++ 判题器（judge_O0.exe）vs 我们的 pyd：**信息素**逐元素比对。

    为什么比信息素：`update_pheromone` 每回合对每只蚂蚁写一次，值由蚂蚁位置决定，
    而蚂蚁位置由 RNG 抽签决定 ⇒ 722 个格子的信息素在第 440 回合仍逐元素相同，
    等价于"两边消费的 RNG 序列完全相同"（比只比 hp/coins 强得多）。
    """
    from probe_replay_engines import run_judge
    info, rows, n = load_ops(args.log)
    replay = Path(r"C:\Temp\rng_probe\out\replay_official.json")   # 必须 ASCII 路径
    exe = REPO / args.judge
    if not exe.exists():
        print(f"[official] ✗ 找不到官方判题器 {exe}")
        return 1
    out, proc, dt = run_judge(exe, rows, n, info["seed"], replay)
    data = json.loads(replay.read_text(encoding="utf-8"))
    last = data[-1]["round_state"]
    camps = last.get("camps")
    print(f"[official] judge rc={proc.returncode} {len(data)} 条 replay "
          f"终局 camps={camps}")

    g = ENG().NativeGame()
    g.init_game(seed=info["seed"], movement_policy="enhanced", cold=True)
    our_ph, our_hp = [], []
    for r in range(n):
        for p in (0, 1):
            raw = as_raw(rows[r][p])
            if raw:
                g.apply_operation_list(p, raw)
        g.advance_round()
        our_ph.append([float(v) for v in g.pheromone_flat()])
        our_hp.append((int(g.base_hp(0)), int(g.base_hp(1))))
        if g.is_ended():
            break
    print(f"[official] 我们的引擎 终局 hp={g.base_hp(0)},{g.base_hp(1)} "
          f"（{len(our_ph)} 回合）")

    ph_len = None
    nbad_total, maxd_total, bad_rounds = 0, 0.0, []
    for i, entry in enumerate(data):
        if i >= len(our_ph):
            break
        rs = entry.get("round_state") or {}
        ph = rs.get("pheromone")
        if not ph:
            continue
        flat_j = [float(ph[p][a][b]) for p in (0, 1) for a in range(len(ph[0]))
                  for b in range(len(ph[0][0]))]
        flat_o = our_ph[i]
        if ph_len is None:
            ph_len = len(flat_j)
        bad = sum(1 for a, b in zip(flat_j, flat_o) if abs(a - b) > 0)
        md = max((abs(a - b) for a, b in zip(flat_j, flat_o)), default=0.0)
        if bad:
            bad_rounds.append((i + 1, bad, md))
        nbad_total += bad
        maxd_total = max(maxd_total, md)
    print(f"[official] 信息素逐回合比对：{len(our_ph)} 回合 × {ph_len} 元素，"
          f"不相等元素总数={nbad_total}  max|diff|={maxd_total:.3e}"
          + ("" if not bad_rounds else f"  首个不一致回合={bad_rounds[:3]}"))
    print("[official] 终局 ants 记录是**增量**（不是全量快照）⇒ 不作逐只比对；"
          "信息素每回合由全部蚂蚁位置决定，它逐元素相同已蕴含'每回合每只蚂蚁位置都相同'")

    hp_j = [tuple(r["hp"]) if r["hp"] else None for r in out]
    n_hp = min(len(hp_j), len(our_hp))
    hp_bad = next((i for i in range(n_hp) if hp_j[i] != our_hp[i]), None)
    print(f"[official] 每回合 base_hp 一致: {hp_bad is None}"
          + ("" if hp_bad is None else f"  首个分歧回合={hp_bad + 1}"))
    ok = nbad_total == 0 and hp_bad is None
    print(f"[official] 结论: "
          + ("✅ 引擎 RNG 消费序列与官方判题器逐位一致" if ok else "❌ 有差异"))
    return 0 if ok else 1


def main() -> int:
    global MODULE, ENGINE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["determinism", "census", "clone", "interleave",
                                     "parity", "sites", "perturb", "digest", "crn",
                                     "official"])
    ap.add_argument("--log", type=Path,
                    default=REPO / "match_results/ladder_logs/VS_RUNNERUP_cells4/10.log")
    ap.add_argument("--logs", type=str, nargs="+", default=None,
                    help="parity 批量对拍用：多条日志路径")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--cut", type=int, default=120)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--branches", type=int, default=32)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--prob-mode", choices=["const", "decay", "first"], default="const")
    ap.add_argument("--decay-c", type=float, default=0.3)
    ap.add_argument("--decay-k", type=float, default=1.0)
    ap.add_argument("--probe-dir", type=str, default=str(PROBE_DIR))
    ap.add_argument("--module", choices=["live", "probe"], default="live")
    ap.add_argument("--engine-dir", type=str, default="",
                    help="module=live 时优先用这个目录里的 native_game.pyd"
                         "（parity 默认 = code/cpp_engine/rng_build 那套 RNG 变体）")
    ap.add_argument("--judge", type=str,
                    default="code/cpp_engine/build_probe/judge_O0.exe")
    args = ap.parse_args()
    MODULE = args.module
    ENGINE_DIR = args.engine_dir
    if args.out is None:
        stem = f"{args.log.stem}_{args.mode}"
        args.out = str(REPO / "code/cpp_engine/build_probe/rng_probe" / f"{stem}.json")
    return {"determinism": mode_determinism, "census": mode_census,
            "clone": mode_clone, "interleave": mode_interleave,
            "parity": mode_parity, "sites": mode_sites,
            "perturb": mode_perturb, "digest": mode_digest,
            "crn": mode_crn, "official": mode_official}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
