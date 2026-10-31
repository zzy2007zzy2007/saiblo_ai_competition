"""极快的**进程内**对局竞技场（C++ 官方引擎），用来做"策略层"实验。

动机：项目历史上的 A/B 都走 bridge + 进程通信，单局 10-20 分钟；因此所有实验的分辨率
都很低（8/16 局），只能看见大效应。C++ 引擎本身极快（ms/回合），所以只要策略是
"便宜的"（规则策略 / 网络 argmax，不带 rollout 搜索），一局可以做到几十毫秒。

于是可以做到：单臂 128 局（64 seed × 2 座位）用秒级时间，从而：
  - 系统地扫"策略轴"（塔数 / 升级 vs 造塔 / 目标塔型 / 武器 / 基地升级 / 迁塔 / 塔位）
  - 用经验数据给 33 个可建格点打分（塔位价值表）

设计：
  - 唯一权威引擎：`GameStateFacade`（pybind C++）。
  - Policy = callable(state, player) -> list[Operation]（一回合的 bundle）
  - 座位镜像：seed 偶数 A 当 P0，奇数 A 当 P1 ⇒ 消掉先手偏差。
  - 指标：终局血量差 margin = hp[A] - hp[B]（A 视角）+ 胜负。

用法：
  python code/test_match/fast_arena.py --speed
  python code/test_match/fast_arena.py --duel refbest default --seeds 32
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "Ant-Game", REPO / "code", REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

MAX_ROUNDS = 512

OP_BUILD, OP_UPG, OP_DOWN = 11, 12, 13
OP_LIGHT, OP_EMP, OP_DEFL, OP_EVAS = 21, 22, 23, 24
OP_BASE_SPEED, OP_BASE_HP = 31, 32

WEAPON_COST = {"light": 90, "emp": 135, "defl": 60, "evas": 60}


# ───────────────────────────── 引擎封装 ─────────────────────────────

def _new_state(seed: int):
    from my_ai.az_intent.game_state_facade import GameStateFacade
    return GameStateFacade.initial(seed=seed, cold_handle_rule_illegal=True)


def _op(t, a0=-1, a1=-1):
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    return Operation(OperationType(int(t)), int(a0), int(a1))


def _hexd(a, b):
    from SDK.utils.geometry import hex_distance
    return hex_distance(a[0], a[1], b[0], b[1])


# ───────────────────────────── 规则策略 ─────────────────────────────

# 目标塔型的升级链（Basic 0 → 二级 → 三级）
TARGET_CHAIN = {
    "basic": [],
    "heavy": [1],
    "heavy+": [1, 11],
    "quick": [2],
    "quick+": [2, 21],
    "double": [2, 22],
    "sniper": [2, 23],
    "mortar": [3],
    "mortar+": [3, 31],
    "pulse": [3, 32],
    "missile": [3, 33],
    "producer": [4],
    "producer+": [4, 41],
    "siege": [4, 42],
    "medic": [4, 43],
}


def default_slots(player: int) -> list[tuple[int, int]]:
    """默认建塔顺序：官方 SDK 的 STRATEGIC_BUILD_ORDER（规则 AI 用的那个）。"""
    from SDK.utils.constants import STRATEGIC_BUILD_ORDER
    return list(STRATEGIC_BUILD_ORDER[player])


class RulePolicy:
    """一个可调参数的规则策略。所有决策都是"查表 + 简单几何"，无网络、无搜索。"""

    def __init__(self,
                 n_towers: int = 3,
                 target: str = "sniper",
                 upg: str = "always",       # always | lvl2 | never | damaged
                 upg_hp_thr: float = 0.6,   # upg="damaged" 时：只在塔血量比 <= 这个阈值时升级（=修理）
                 repair_cycle: bool = False,  # 升级回满血后**下一回合降级退回**（净花 ~6 币修一次）
                 slots: list | None = None,  # 建塔优先顺序（None = 官方顺序）
                 weapons: tuple = ("light",),
                 base_upg: tuple = (),       # () | ("speed",) | ("hp",) | ("speed","hp")
                 relocate: bool = False,     # 塔"够不到蚂蚁"时拆掉挪位
                 reserve: int = 0,           # 建塔前保留的金币（= build_reserve）
                 build_reserve: int = None,
                 upg_reserve: int = 0,
                 light_min_ants: int = 3,    # 闪电最少覆盖蚂蚁数
                 light_max_dist: int = 99,   # 只打离我方基地 <= 这个距离的集群
                 light_mode: str = "cluster",  # cluster(防) | towers(拆塔) | nearbase(就近)
                 slot_mode: str = "fixed",   # fixed | nearest_front
                 max_ops: int = 3):
        self.p = dict(n_towers=n_towers, target=target, upg=upg, slots=slots,
                      weapons=tuple(weapons), base_upg=tuple(base_upg),
                      relocate=relocate,
                      build_reserve=(reserve if build_reserve is None else build_reserve),
                      upg_reserve=upg_reserve, upg_hp_thr=upg_hp_thr,
                      repair_cycle=bool(repair_cycle),
                      light_min_ants=light_min_ants, light_max_dist=light_max_dist,
                      light_mode=light_mode,
                      slot_mode=slot_mode, max_ops=max_ops)

    # ---- 小工具 ----
    def _cells(self, state, player):
        """建塔优先顺序。`slots` 里若给的是**整数**，表示 STRATEGIC_BUILD_ORDER[player]
        的下标（座位无关，两个座位都能用）；若是元组则按坐标原样用（只对 P0 有效）。"""
        sl = self.p["slots"]
        base = default_slots(player)
        if sl is None:
            return base
        if sl and isinstance(sl[0], int):
            return [base[i] for i in sl if 0 <= i < len(base)]
        return [tuple(c) for c in sl]

    def _apply(self, state, player, ops, out):
        from SDK.backend.model import Operation
        for t, a0, a1 in ops:
            op = _op(t, a0, a1)
            if state.can_apply_operation(player, op, []) and len(out) < self.p["max_ops"]:
                state.apply_operation_list(player, [op])
                out.append(op)

    def _weapon(self, state, player, kind):
        """返回 (op_type, x, y) 或 None。"""
        cds = state.weapon_cooldowns
        idx = {"light": 1, "emp": 2, "defl": 3, "evas": 4}[kind]
        if cds[player][idx] != 0 or state.coins[player] < WEAPON_COST[kind]:
            return None
        ants = state.ants
        enemy_ants = [(a.x, a.y) for a in ants if a.player != player and a.is_alive()]
        own_ants = [(a.x, a.y) for a in ants if a.player == player and a.is_alive()]
        if kind == "light":
            mode = self.p.get("light_mode", "cluster")
            from SDK.utils.constants import PLAYER_BASES
            myb = PLAYER_BASES[player]
            if mode == "towers":
                # 进攻型：风暴中心放在敌方塔最密处（第 5/10/15 回合各对范围内塔 3 点伤害）
                e_towers = [(t.x, t.y) for t in state.towers if t.player != player]
                if not e_towers:
                    return None
                best, cnt = None, 0
                for (x, y) in e_towers:
                    c = sum(1 for (tx, ty) in e_towers if _hexd((x, y), (tx, ty)) <= 3)
                    if c > cnt:
                        best, cnt = (x, y), c
                if best is None or cnt < self.p["light_min_ants"]:
                    return None
                return (OP_LIGHT, best[0], best[1])
            if len(enemy_ants) < self.p["light_min_ants"]:
                return None
            best_cell, best_key = None, None
            for (x, y) in enemy_ants:
                d = _hexd((x, y), myb)
                if d > self.p["light_max_dist"]:
                    continue
                c = sum(1 for (ax, ay) in enemy_ants if _hexd((x, y), (ax, ay)) <= 3)
                if c < self.p["light_min_ants"]:
                    continue
                key = (-d, c) if mode == "nearbase" else (c, -d)
                if best_key is None or key > best_key:
                    best_key, best_cell = key, (x, y)
            if best_cell is None:
                return None
            return (OP_LIGHT, best_cell[0], best_cell[1])
        if kind == "emp":
            enemy_towers = [(t.x, t.y) for t in state.towers if t.player != player]
            if not enemy_towers or not own_ants:
                return None
            best, score = None, 0
            for (x, y) in enemy_towers:
                c = sum(1 for (tx, ty) in enemy_towers if _hexd((x, y), (tx, ty)) <= 3)
                # 只有我方蚂蚁推到附近才值得
                near = min((_hexd((x, y), p) for p in own_ants), default=99)
                if near > 6:
                    continue
                if c * 2 > score:
                    best, score = (x, y), c * 2
            if best is None:
                return None
            return (OP_EMP, best[0], best[1])
        # defl / evas：给我方蚂蚁上护盾，要求正在推进（离敌方基地不远）
        if len(own_ants) < 2:
            return None
        from SDK.utils.constants import PLAYER_BASES
        eb = PLAYER_BASES[1 - player]
        pushed = [p for p in own_ants if _hexd(p, eb) <= 10]
        if len(pushed) < 2:
            return None
        best, cnt = None, 0
        for (x, y) in pushed:
            c = sum(1 for p in own_ants if _hexd((x, y), p) <= 3)
            if c > cnt:
                best, cnt = (x, y), c
        if best is None:
            return None
        return (OP_EMP + (1 if kind == "defl" else 2), best[0], best[1])

    def _coverage(self, state, player, x, y, rng_):
        """该格在半径 rng_ 内的敌方蚂蚁数（用当下局面近似"这条路上有没有蚂蚁"）。"""
        return sum(1 for a in state.ants
                   if a.player != player and a.is_alive() and _hexd((x, y), (a.x, a.y)) <= rng_)

    def _pick_build_cell(self, state, player, cells):
        for c in cells:
            if state.can_apply_operation(player, _op(OP_BUILD, c[0], c[1]), []):
                return c
        return None

    def act(self, state, player):
        out = []
        p = self.p
        # 0. 修理循环的"退回"半步：上回合为修理而升级的塔，这回合血量已满 ⇒ 降级把钱拿回来
        if p.get("repair_cycle"):
            for tid, orig in list(getattr(self, "_rep", {}).items()):
                tw = state.tower_by_id(tid)
                if tw is None or int(tw.tower_type) == orig:
                    self._rep.pop(tid, None)
                    continue
                if tw.hp >= tw.max_hp and tw.max_hp > 0:
                    self._apply(state, player, [(OP_DOWN, tid, -1)], out)
                    self._rep.pop(tid, None)
        # 1. 武器
        for kind in p["weapons"]:
            w = self._weapon(state, player, kind)
            if w is not None:
                self._apply(state, player, [w], out)
                break
        # 2. 基地升级
        if p["base_upg"] and state.coins[player] >= 200:
            lv = state.bases[player].generation_level
            alv = state.bases[player].ant_level
            if "speed" in p["base_upg"] and lv < 2:
                self._apply(state, player, [(OP_BASE_SPEED, -1, -1)], out)
            if "hp" in p["base_upg"] and alv < 2:
                self._apply(state, player, [(OP_BASE_HP, -1, -1)], out)
        # 3. 造塔（**先造塔再升级**：否则升级会吃掉建塔的钱，把塔数锁死在 1）
        cells = self._cells(state, player)
        for _ in range(p["max_ops"]):
            if len(out) >= p["max_ops"]:
                break
            n = len(state.towers_of(player))
            if n >= p["n_towers"]:
                break
            cost = state.build_tower_cost(n)
            if state.coins[player] < cost + p["build_reserve"]:
                break
            cell = self._pick_build_cell(state, player, cells)
            if cell is None:
                break
            self._apply(state, player, [(OP_BUILD, cell[0], cell[1])], out)
        # 4. 升级已有塔（朝目标塔型）
        chain = TARGET_CHAIN.get(p["target"], [])
        if p["upg"] != "never" and chain:
            for tower in sorted(state.towers_of(player), key=lambda t: t.tower_id):
                if len(out) >= p["max_ops"]:
                    break
                cur = int(tower.tower_type)
                nxt = None
                if cur == 0:
                    nxt = chain[0]
                elif cur in chain[:-1]:
                    nxt = chain[chain.index(cur) + 1]
                if nxt is None:
                    continue
                if p["upg"] == "lvl2" and nxt >= 10:
                    continue
                if p["upg"] == "damaged":
                    # 只在塔被打残时升级（利用"升级回满血"= 廉价修理）
                    if tower.max_hp <= 0 or tower.hp / tower.max_hp > p["upg_hp_thr"]:
                        continue
                cost = 60 if nxt < 10 else 200
                if state.coins[player] < cost + p["upg_reserve"]:
                    continue
                before_type = cur
                n0 = len(out)
                self._apply(state, player, [(OP_UPG, tower.tower_id, nxt)], out)
                if p.get("repair_cycle") and len(out) > n0:
                    if not hasattr(self, "_rep"):
                        self._rep = {}
                    self._rep[tower.tower_id] = before_type
        # 5. 迁塔（把完全没用的塔拆掉，下一回合会重新建）
        if p["relocate"]:
            for tower in state.towers_of(player):
                r = tower.attack_range
                if self._coverage(state, player, tower.x, tower.y, max(r, 1) + 2) == 0:
                    # 只在"最近 12 回合没有敌人经过"这种无法判断时保守：直接拆
                    self._apply(state, player, [(OP_DOWN, tower.tower_id, -1)], out)
                    break
        return out


class NoopPolicy:
    def act(self, state, player):
        return []


def make_policy(spec: str):
    """把字符串名映射成策略实例（CLI 用）。"""
    if spec == "noop":
        return NoopPolicy()
    if spec.startswith("{"):
        return RulePolicy(**json.loads(spec))
    # 预设
    if spec == "ref":
        return RulePolicy()
    if spec == "basic":
        return RulePolicy(target="basic", upg="never")
    if spec == "best":
        # 策略轴扫描出来的最优规则配置（见 docs/free_exploration_strategy_arena_20261102.md §4）
        return RulePolicy(n_towers=2, target="sniper", upg="always", weapons=("light",),
                          build_reserve=300, upg_reserve=0, light_min_ants=3)
    if spec == "bestv2":
        # 联合网格最优（§5b）：修理循环 + 留 100 + 2 塔  ⇒ N=192 时 +5.88 / 75.0%
        return RulePolicy(n_towers=2, target="sniper", upg="damaged", upg_hp_thr=0.6,
                          repair_cycle=True, weapons=("light",),
                          build_reserve=100, upg_reserve=0, light_min_ants=3)
    if spec == "ref2":
        return RulePolicy(n_towers=2, target="sniper", upg="always", weapons=("light",),
                          build_reserve=100, upg_reserve=0, light_min_ants=3)
    raise SystemExit(f"unknown policy spec: {spec}")


# ───────────────────────────── 对局 ─────────────────────────────

@dataclass
class GameOut:
    seed: int
    a_seat: int
    rounds: int
    hp: tuple
    winner: int | None
    margin_a: int
    ms: float = 0.0
    coins: tuple = (0, 0)
    towers: tuple = (0, 0)
    opmix_a: dict = field(default_factory=dict)
    opmix_b: dict = field(default_factory=dict)


def play_game(spec_a: str, spec_b: str, seed: int) -> GameOut:
    t0 = time.perf_counter()
    a_seat = 0 if seed % 2 == 0 else 1
    pa, pb = make_policy(spec_a), make_policy(spec_b)
    pol = {a_seat: pa, 1 - a_seat: pb}
    st = _new_state(seed)
    mix = {0: {}, 1: {}}
    rnd = 0
    for rnd in range(1, MAX_ROUNDS + 1):
        if st.terminal:
            break
        for player in (0, 1):
            if st.terminal:
                break
            ops = pol[player].act(st, player)
            if ops:
                st.apply_operation_list(player, ops)
                m = mix[player]
                for op in ops:
                    k = int(op.op_type)
                    m[k] = m.get(k, 0) + 1
        if st.terminal:
            break
        st.advance_round()
    hp = (int(st.bases[0].hp), int(st.bases[1].hp))
    return GameOut(seed=seed, a_seat=a_seat, rounds=int(st.round_index), hp=hp,
                   winner=(None if st.winner is None else int(st.winner)),
                   margin_a=hp[a_seat] - hp[1 - a_seat],
                   ms=(time.perf_counter() - t0) * 1000.0,
                   coins=(int(st.coins[0]), int(st.coins[1])),
                   towers=(len(st.towers_of(0)), len(st.towers_of(1))),
                   opmix_a=mix[a_seat], opmix_b=mix[1 - a_seat])


def _worker(args):
    return play_game(*args)


def duel(spec_a, spec_b, seeds: int, workers: int = 8, start_seed: int = 1000):
    jobs = [(spec_a, spec_b, start_seed + s) for s in range(seeds)]
    if workers <= 1:
        outs = [_worker(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(_worker, jobs, chunksize=4))
    mgs = [o.margin_a for o in outs]
    wins = sum(1 for o in outs if o.margin_a > 0)
    ties = sum(1 for o in outs if o.margin_a == 0)
    se = (statistics.pstdev(mgs) / max(len(mgs) ** 0.5, 1)) if len(mgs) > 1 else 0.0
    return dict(
        a=spec_a if len(str(spec_a)) < 60 else "<spec>", b=spec_b if len(str(spec_b)) < 60 else "<spec>",
        n=len(outs), margin=statistics.mean(mgs), se=se,
        win=wins, tie=ties, loss=len(outs) - wins - ties,
        rounds=statistics.mean(o.rounds for o in outs),
        ms_per_game=statistics.mean(o.ms for o in outs),
        margin_p0=statistics.mean(o.margin_a for o in outs if o.a_seat == 0) if any(o.a_seat == 0 for o in outs) else None,
        margin_p1=statistics.mean(o.margin_a for o in outs if o.a_seat == 1) if any(o.a_seat == 1 for o in outs) else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--duel", nargs=2)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.speed:
        for spec_a, spec_b in (("noop", "noop"), ("ref", "ref"), ("basic", "ref")):
            r = duel(spec_a, spec_b, seeds=8, workers=8)
            print(json.dumps(r, ensure_ascii=False), flush=True)
        return
    if a.duel:
        r = duel(a.duel[0], a.duel[1], seeds=a.seeds, workers=a.workers)
        print(json.dumps(r, ensure_ascii=False, indent=1))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
