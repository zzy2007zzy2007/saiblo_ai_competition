"""统一候选生成（方案 D）：**类 UCB bandit + 位置游标**（2026-10-29）。

设计（冻结，逐条照做）：`docs/unified_cand_design_20261029.md`
实现与验证读数：`docs/unified_cand_impl_20261029.md`

## 这个文件做什么

把"候选束"的来源从 **先钉一个类**（`bundle_mcts._expand` 里 `pinned=[_ccls]*3`）
换成 **从 `action_map` 全图跨类 × 位置取候选**：

1. **类选择**：UCB。利用项 = LogSumExp 软最大
   `Q[c] = (1/β)·log( (1/n)·Σ exp(β·x) )`（β=3，`AZAI_CAND_BETA`）；
   探索项 = `c·√(log N / n)`（c=1.0，`AZAI_CAND_UCBC`）。`Q[c]` 初始全相等（均匀），
   **类网先验接口预留但暂不用**（类网现在基本不可用，要用得先重训）。
2. **位置游标**：`cursor[规范多重集]` = 该组已提议过的位置组合列表。
   空 ⇒ 各通道位置图 argmax；否则看**最后一个动作在该通道位置图里排第 k** ⇒ 取第 k+1 名
   （`np.argpartition` 只用来定"第 k 大"的边界 + 并列按 (x,y) 字典序，**不做全排序**）。
3. **游标键 = 动作类的规范多重集**：去掉 HOLD(23)、排序。
   `(hold,build,lightning)` 与 `(build,hold,lightning)` 归一到 `(build,lightning)`
   —— 它们产出**同一个** bundle，不该拆成两个游标。
4. **无位置的类**（HOLD=23、基地升级=21/22）是**单位置臂**，游标恒指向那一个动作。
5. **通道 → op 精确映射**：照 `code/my_ai/decoder.py`（`_decode_tower_action` /
   `CHANNEL_TO_SUPER_WEAPON` / `decoder.py:369-373` 的基地升级），**不自己发明**。
6. **每个候选逐条 `can_apply` 过滤**（严格模式一条非法 = 判负，绝不能漏）。
7. **阶段 1 = 确定性 idle rollout（双方不出招）×1**，分数用于 **top-k 选择**（λ=0：位置图
   分只参与"游标提议顺序"，不参与"选择权重"）。**阶段 2 = 现役 flat 随机化 rollout 完全不动**
   （在 `UnifiedCandFlatSearch` 里由父类 `search()` 原样执行）。
8. **缓存/两阶段复用**：缓存键 = 规范化 bundle（`(op_type,arg0,arg1)` 排序元组，去 HOLD）。

## 为什么位置由**静态位置图**提议、类预算由**学到的 Q[c]** 分配

设计 §2「关键设计判断」：避免"类 × 位置"笛卡尔积。bandit 只用**一小部分预算**
生成候选（≈ K 条），大部分预算交给现役 flat 强评估器正式评估 —— 当年 0/42 就是
"让弱评估器重新主导选类"输掉的。
"""
from __future__ import annotations

import math
import sys

import numpy as np

from my_ai.decoder import (
    CHANNEL_TO_SUPER_WEAPON,
    SUPER_WEAPON_TO_OP_TYPE,
    _decode_tower_action,
)

# 通道语义（照 `decoder.py`：0-15 塔型 / 16 降级 / 17-20 超武 / 21-22 基地升级 / 23 HOLD）
HOLD = 23
TOWER_CHANNELS = tuple(range(16))
DOWNGRADE = 16
SUPER_CHANNELS = (17, 18, 19, 20)
# `decoder.py:369-373`：class 21 ⇒ UPGRADE_GENERATION_SPEED(31)，class 22 ⇒ UPGRADE_GENERATED_ANT(32)
BASE_UPGRADE_OP = {21: 31, 22: 32}
N_CHANNELS = 24
NO_POSITION = (-1, -1)


# ─── 规范化 bundle 键 ────────────────────────────────────────────────────────
def canon_ops(ops) -> tuple:
    """规范化 bundle：`(op_type,arg0,arg1)` 三元组的**排序**元组（HOLD=空束 ⇒ `()`）。

    排序而不是保序：不同通道/不同类序列可能解出**同一个** bundle（例：多个塔通道在空地都
    解成同一个 `BUILD_TOWER(x,y)`）⇒ 缓存/去重必须把它们认成同一条，否则白烧 rollout 预算。
    ⚠️ 真实下发的**顺序**不被排序影响：`canon_ops` 只当键用，bundle 本身保持生成顺序
    （`DOWNGRADE` 退钱在前、`BUILD` 在后这类顺序是**语义**，不能动）。
    """
    return tuple(sorted((int(o[0]), int(o[1]), int(o[2])) for o in (ops or ())))


def decode_channel_op(state, player: int, ch: int, x: int, y: int):
    """**通道 + 位置 → Operation**（逐条照 `decoder.py`，不发明语义）。

    * `0-15`：`_decode_tower_action`（`decoder.py:433-456`）—— 空格 ⇒ `BUILD_TOWER(x,y)`
      （**建 Basic，忽略 target_type**）；我方塔 ⇒ `UPGRADE_TOWER(tower_id, step)`
      （step 由 `upgrade_step`，`decoder.py:82-94`）；敌方塔 ⇒ `None`。
    * `16`：我方塔格 ⇒ `DOWNGRADE_TOWER(tower_id)`（`decoder.py:426-428`）。
    * `17-20`：`CHANNEL_TO_SUPER_WEAPON` → `SUPER_WEAPON_TO_OP_TYPE`，位置即 (x,y)。
    * `21/22`：`UPGRADE_GENERATION_SPEED(31)` / `UPGRADE_GENERATED_ANT(32)`（无位置）。
    * `23`：HOLD ⇒ `None`（不产生 op；空 bundle）。
    """
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    ch = int(ch)
    if 0 <= ch <= 15:
        return _decode_tower_action(state, int(player), ch, int(x), int(y))
    if ch == DOWNGRADE:
        tower = state.tower_at(int(x), int(y))
        if tower is not None and int(tower.player) == int(player):
            return Operation(OperationType.DOWNGRADE_TOWER, int(tower.tower_id))
        return None
    if ch in SUPER_CHANNELS:
        op_type = SUPER_WEAPON_TO_OP_TYPE[CHANNEL_TO_SUPER_WEAPON[ch]]
        return Operation(op_type, int(x), int(y))
    op_type = BASE_UPGRADE_OP.get(ch)
    if op_type is not None:
        return Operation(OperationType(int(op_type)))
    return None    # 23 = HOLD（该 head 什么都不出）


# ─── 位置表（每类：合法格 + 该通道位置图分数）────────────────────────────────
class _ClassCells:
    """某一通道在**当前局面**下的合法格与位置图分数（只建一次，决策内不变）。"""

    __slots__ = ("xs", "ys", "vals")

    def __init__(self, xs: np.ndarray, ys: np.ndarray, vals: np.ndarray):
        self.xs, self.ys, self.vals = xs, ys, vals

    def __len__(self) -> int:
        return int(self.vals.size)

    def argmax_cell(self) -> tuple:
        i = int(np.argmax(self.vals))
        return (int(self.xs[i]), int(self.ys[i]))

    def value_at(self, cell: tuple):
        px, py = int(cell[0]), int(cell[1])
        hit = np.nonzero((self.xs == px) & (self.ys == py))[0]
        if not hit.size:
            return None
        return float(self.vals[hit[0]])

    def rank_of(self, cell: tuple) -> int:
        """`cell` 在这张位置图里的名次（0-based；并列按 (x,y) 字典序）。

        只用计数（`vals > v` + 并列里 (x,y) 更小的），**不做全排序**。
        """
        v = self.value_at(cell)
        if v is None:
            return -1
        px, py = int(cell[0]), int(cell[1])
        rank = int(np.count_nonzero(self.vals > v))
        tie = self.vals == v
        if int(np.count_nonzero(tie)) > 1:
            rank += int(np.count_nonzero(tie & ((self.xs < px) | ((self.xs == px) & (self.ys < py)))))
        return rank

    def kth_cell(self, k: int) -> tuple:
        """名次 k（0-based）对应的格子。**用 `np.argpartition` 取第 k 大，不全排序。**

        并列处理：先用 `np.partition` 定"第 k 大的**值**"（O(n)），值**严格更大**的全排在
        它前面，剩下的第 `k - #greater` 个名额在"值等于第 k 大"的那一撮里按 (x,y) 字典序取
        （那一撮通常很短 ⇒ 不做全排序）。
        """
        n = len(self)
        if n <= 0:
            return NO_POSITION
        k = 0 if k < 0 else (n - 1 if k >= n else int(k))
        kth_val = float(-np.partition(-self.vals, k)[k])
        greater = np.nonzero(self.vals > kth_val)[0]
        ties = np.nonzero(self.vals == kth_val)[0]
        left = int(k) - int(greater.size)
        if 0 <= left < int(ties.size):
            order = np.lexsort((self.ys[ties], self.xs[ties]))   # ← 主键最后：先 x 再 y
            i = int(ties[order[left]])
            return (int(self.xs[i]), int(self.ys[i]))
        if greater.size:
            i = int(greater[0])
            return (int(self.xs[i]), int(self.ys[i]))
        i = int(np.argmax(self.vals))
        return (int(self.xs[i]), int(self.ys[i]))


class UnifiedCandidateGen:
    """方案 D 的候选生成器（**每条候选各跑一次确定性 idle rollout ⇒ 按分取 top-k**）。"""

    def __init__(self,
                 k: int = 48,
                 beta: float = 3.0,
                 ucb_c: float = 1.0,
                 n_heads: int = 3,
                 pool_mult: float = 1.5,
                 attempts_mult: int = 6,
                 force_hold: bool = True,
                 verbose: bool = False) -> None:
        self.k = max(1, int(k))
        self.beta = float(beta)
        self.ucb_c = float(ucb_c)
        # 一个 bundle 最多几个 op：与网络头数一致（3）。HOLD 占一个头 ⇒ 自然产生 1/2 op 的束。
        self.n_heads = max(1, int(n_heads))
        # 生成的**候选池** = k × pool_mult（多生成一些，再按阶段 1 分数取 top-k）
        self.pool_mult = max(1.0, float(pool_mult))
        # 去重后仍凑不满 ⇒ 最多再试 k×attempts_mult 次
        self.attempts_mult = max(1, int(attempts_mult))
        self.force_hold = bool(force_hold)
        self.verbose = bool(verbose)
        # 阶段 1 的 rollout 函数（由搜索器注入：`(state, player, bundles) -> np.ndarray`）
        self.rollout_fn = None
        # 缓存：规范化 bundle → [value]（同一 bundle 只算一次；也给阶段 2 复用）
        self.cache: dict = {}
        # 最近一次决策的读数（给日志/报告用）
        self.last_stats: dict = {}
        self._reset_state()

    # ── 决策内状态 ────────────────────────────────────────────────────────
    def _reset_state(self) -> None:
        self.state = None
        self.player = 0
        self.AM = None
        self.pmask = None
        self.cmask = None
        self.cells: dict = {}
        self.legal: list = []
        self.Q: dict = {}          # 类 → 已观测分数列表（LogSumExp 的原料）
        self.N: dict = {}          # 类 → 被选中次数（UCB 的 n）
        self.N_total = 0
        self.cursor: dict = {}     # 规范多重集 → [位置组合, ...]（设计里的"游标"，留档/日志）
        self.last_cell: dict = {}  # **类 → 该类上一次被提议的格子**（"该类最后那个动作"，见 §偏差）
        self.cache = {}

    def _reset_decision(self, state, player: int, net_out: dict) -> None:
        from my_ai.decoder import make_class_mask, make_position_masks

        self._reset_state()
        self.state = state
        self.player = int(player)
        self.AM = np.asarray(net_out["action_map"], dtype=np.float32)
        # `intent_decoding=False`：**连金币一起查**（建塔/升级/超武买不起就不进位置掩码）
        #   ⇒ 位置枚举天然"买得起"，后面的 `can_apply` 只是**兜底**，不是唯一防线。
        self.pmask = make_position_masks(state, self.player, intent_decoding=False)
        self.cmask = make_class_mask(state, self.player, position_mask=self.pmask,
                                     intent_decoding=False)
        for ch in range(min(N_CHANNELS, int(self.AM.shape[0]))):
            if not bool(self.pmask[ch].any()):
                continue
            xs, ys = np.nonzero(self.pmask[ch])
            self.cells[ch] = _ClassCells(xs, ys, self.AM[ch][xs, ys].astype(np.float64))
        # 类池：`class_mask` 为真的类（HOLD=23 恒真 ⇒ 天然在里面）
        self.legal = [c for c in range(min(N_CHANNELS, int(self.cmask.shape[0])))
                      if bool(self.cmask[c])]
        self.Q = {c: [] for c in self.legal}
        self.N = {c: 0 for c in self.legal}
        self.N_total = 0

    # ── ① 类选择：UCB（LogSumExp 利用项 + 标准探索项）───────────────────────
    def _lse(self, ch: int) -> float:
        """利用项：`(1/β)·log((1/n)Σ exp(β·x))`（软最大；对负数成立，β→0 退化为均值）。"""
        xs = self.Q.get(int(ch)) or []
        if not xs:
            return 0.0                      # 无观测 ⇒ 均匀先验（Q 全相等）
        arr = np.asarray(xs, dtype=np.float64)
        m = float(arr.max())
        b = self.beta
        return float(m + math.log(float(np.mean(np.exp(b * (arr - m))))) / b)

    def _ucb(self, ch: int) -> float:
        n = int(self.N.get(int(ch), 0))
        if n <= 0:
            return math.inf                 # 未试过的臂优先（标准 UCB1 初始化）
        log_n = math.log(max(1, int(self.N_total)))
        return self._lse(ch) + self.ucb_c * math.sqrt(log_n / n)

    def _ucb_pick(self, exclude=()) -> int | None:
        best, best_v = None, -math.inf
        for c in self.legal:
            if c in exclude:
                continue
            v = self._ucb(c)
            if v > best_v:
                best, best_v = int(c), v
        return best

    # ── ② 位置游标 ────────────────────────────────────────────────────────
    def _next_cell(self, ch: int) -> tuple:
        """**该类**的下一个提议位置：空 ⇒ 位置图 argmax；否则看该类**最后一个动作**排第 k ⇒ 第 k+1 名。

        ⚠️ 设计里同时写了"每类一个 `cursor[c]`"（单动作版）和"游标按**规范多重集**索引"
        （多动作组版），两者对多 op 束是**冲突**的：若按多重集重启，则每个新键都从 argmax 开始，
        而**塔通道在空格上解码结果与类无关**（都解成同一个 `BUILD_TOWER(x,y)`）⇒ 大量键
        解出同一个 bundle ⇒ 去重后候选数塌到个位数（实测 7 条 / 426 次重复）。
        ⇒ 本实现取**按类**跟踪"最后那个位置"（= 单动作版的规则，设计原话"该类最后那个动作在
        位置图里排第 k ⇒ 选第 k+1 名"），多重集 `cursor[key]` 仍保留作提议记录/日志。
        """
        cc = self.cells.get(ch)
        if cc is None:
            return NO_POSITION                     # 21/22/23：**单位置臂**（位置无意义）
        prev = self.last_cell.get(ch)
        if prev is None:
            cell = cc.argmax_cell()
        else:
            r = cc.rank_of(prev)
            cell = cc.kth_cell(r + 1) if r >= 0 else cc.argmax_cell()
        self.last_cell[ch] = cell
        return cell

    def _positions_for(self, key: tuple) -> tuple:
        """给规范多重集 `key` 提议下一组位置（各类**独立**推进；多重集游标留档）。"""
        pos = tuple(self._next_cell(c) for c in key)
        self.cursor.setdefault(tuple(key), []).append(pos)
        return pos

    # ── ③ 解码 + 逐条 can_apply ───────────────────────────────────────────
    def _decode_group(self, group: tuple, positions: tuple) -> tuple:
        """把 (类, 位置) 组解成 op 序列；**逐条 `can_apply` 过滤**（含与已 pending 的冲突）。"""
        ops: list = []
        pending: list = []
        seen: set = set()
        for ch, cell in zip(group, positions):
            op = decode_channel_op(self.state, self.player, ch, cell[0], cell[1])
            if op is None:
                continue
            t = (int(op.op_type), int(op.arg0), int(op.arg1))
            if t in seen:
                continue
            try:
                ok = bool(self.state.can_apply_operation(self.player, op, pending))
            except Exception as exc:                      # noqa: BLE001
                # 判负开关：`can_apply_dry` 抛异常 = 我们**不知道**它合不合法 ⇒ 一律丢弃。
                if self.verbose:
                    print(f"[cand] ⚠️ can_apply 异常（丢弃该 op）: {type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)
                ok = False
            if not ok:
                continue
            ops.append(t)
            seen.add(t)
            pending.append(op)
        return tuple(ops)

    # ── ④ 主流程 ──────────────────────────────────────────────────────────
    def generate(self, state, player: int, net_out: dict, k: int | None = None) -> dict:
        """生成候选并**按阶段 1 确定性 rollout 分排序**，返回 dict（含 `bundles`/统计）。"""
        k = int(self.k if k is None else k)
        self._reset_decision(state, player, net_out)
        if not self.legal or self.rollout_fn is None:
            self.last_stats = {"n_pool": 0, "n_cand": 0, "n_dup": 0, "attempts": 0,
                               "has_hold": 0, "hold_forced": 0, "n_upg": 0,
                               "sizes": {}, "p1_ms": 0.0}
            return {"bundles": [], "stats": dict(self.last_stats)}

        import time as _time
        t0 = _time.perf_counter()
        pool_target = max(k, int(round(k * self.pool_mult)))
        max_attempts = max(pool_target * self.attempts_mult, 32)
        picked_out: list = []          # [(key, bundle, score, pos_score_text)]
        seen: set = set()
        n_dup = 0
        n_attempts = 0
        hold_forced = 0
        score_of: dict = {}

        def _emit(group: tuple, positions: tuple, pos_text: str = "") -> bool:
            """生成一条候选（去重 + 阶段 1 rollout + 回写 Q）。返回是否**新增**。"""
            nonlocal n_dup
            bundle = self._decode_group(group, positions)
            key = canon_ops(bundle)
            if key in seen:
                n_dup += 1
                return False
            seen.add(key)
            vals = self.cache.get(key)
            if vals is None:
                vals = list(np.asarray(self.rollout_fn(self.state, self.player, [bundle]),
                                       dtype=np.float64).tolist())
                self.cache[key] = vals
            score = float(np.mean(vals)) if vals else 0.0
            score_of[key] = score
            # 回写：**贡献了 op 的类**（HOLD 只在空束时算它的功劳）
            for c in group:
                if c in self.Q:
                    self.Q[c].append(score)
            if not group and HOLD in self.Q:
                self.Q[HOLD].append(score)
            picked_out.append((key, bundle, score, pos_text))
            return True

        # ① HOLD（类 23 = 单位置臂）：**恒含 1 条**（设计 §6 第 3 问未拍板 ⇒ 取保守选项，
        #    与"注入 HOLD ⇒ +29.2pp"的证据口径一致）。bandit 之后仍会自然再抽到它（会被去重）。
        if self.force_hold and HOLD in self.legal:
            if _emit((), ()):
                hold_forced = 1
            self.N[HOLD] = int(self.N.get(HOLD, 0)) + 1
            self.N_total += 1
            self.cursor.setdefault((), []).append(())
        # ② bandit 主循环：按 UCB 选类（**同一束内类不重复**；HOLD 可重复 ⇒ 决定束的长度）
        #    "dry" 早停：连续 dry_limit 次尝试都只解出**已见过**的 bundle ⇒ 说明该局面的可达
        #    候选空间已被枚举完（例：金币=0 时只剩降级 + HOLD）⇒ 立刻停，别空转到 attempts 上限。
        dry_limit = max(32, 2 * k)
        n_dry = 0
        while len(picked_out) < pool_target and n_attempts < max_attempts:
            n_attempts += 1
            picked: list = []
            for _h in range(self.n_heads):
                c = self._ucb_pick(exclude={x for x in picked if x != HOLD})
                if c is None:
                    break
                picked.append(int(c))
            if not picked:
                break
            for c in picked:                       # N 记账：被选中就算（含 HOLD 当"填充位"）
                self.N[c] = int(self.N.get(c, 0)) + 1
                self.N_total += 1
            group = tuple(sorted({c for c in picked if c != HOLD}))
            positions = self._positions_for(group)
            # 位置分 = 各类位置图分数之积（只用于日志/读数；λ=0 ⇒ 不参与选择权重）
            _pv = 1.0
            for c, cell in zip(group, positions):
                cc = self.cells.get(c)
                _pv *= (cc.value_at(cell) if (cc is not None and cell != NO_POSITION) else 1.0)
            if _emit(group, positions, f"{_pv:.4g}"):
                n_dry = 0
            else:
                n_dry += 1
                if n_dry >= dry_limit:
                    # 连续 dry_limit 次都只解出"已见过"的 bundle ⇒ 该局面的可达候选空间
                    # 已枚举完（早期金币不足以建塔/升级、或只剩降级）⇒ 立刻停，别空转。
                    if self.verbose:
                        print(f"[cand] 早停：连续 {n_dry} 次无新候选（可达空间已枚举完）",
                              file=sys.stderr, flush=True)
                    break

        # ③ 按阶段 1 分数取 top-k（设计 §2 阶段 1：确定性 idle rollout 分）
        picked_out.sort(key=lambda x: (-x[2], x[0]))
        bundles = [b for (_k, b, _s, _p) in picked_out[:k]]
        keys = [kk for (kk, _b, _s, _p) in picked_out[:k]]
        n_upg = sum(1 for b in bundles if any(int(o[0]) == 12 for o in b))
        has_hold = int(any(len(b) == 0 for b in bundles))
        sizes: dict = {}
        for b in bundles:
            sizes[len(b)] = sizes.get(len(b), 0) + 1
        p1_ms = (_time.perf_counter() - t0) * 1000.0
        self.last_stats = {
            "n_pool": len(picked_out), "n_cand": len(bundles), "n_dup": int(n_dup),
            "attempts": int(n_attempts), "has_hold": int(has_hold),
            "hold_forced": int(hold_forced), "n_upg": int(n_upg),
            "sizes": sizes, "p1_ms": p1_ms,
            "keys": keys, "scores": [s for (_k, _b, s, _p) in picked_out[:k]],
        }
        if self.verbose:
            print(f"[cand] gen 池={len(picked_out)} 取={len(bundles)} 重复={n_dup} "
                  f"HOLD={has_hold}(兜底{hold_forced}) 升级={n_upg} 长度分布={sizes} "
                  f"{p1_ms:.0f} ms", file=sys.stderr, flush=True)
        return {"bundles": bundles, "stats": dict(self.last_stats)}


# ─── 搜索器：候选来源换成统一生成器；**阶段 2 = 现役 flat rollout 一字不动** ─────
def make_unified_searcher_class():
    """延迟导入：避免 `unified_cand` 在 `flat_rollout_greedy` 的导入链上形成环。"""
    from my_ai.az_intent.bundle_mcts import BundleNode
    from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy

    class UnifiedCandFlatSearch(FlatRolloutGreedy):
        """`AZAI_CAND_MODE=unified`：候选 = `UnifiedCandidateGen`，评估 = 父类 `search()`。

        ⚠️ 只覆盖 `_expand`（候选来源）+ 在 `search()` 外面包一层日志：
        `FlatRolloutGreedy.search()` 的预算公式 / rollout / 叶值网 / 选择规则**全部继承**，
        一行都不改 ⇒ "阶段 2 = 现役 flat 随机化 rollout" 是**结构上**成立的，不是靠对齐参数。
        """

        def __init__(self, *args, cand_mode: str = "unified", cand_gen=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.cand_mode = str(cand_mode)
            self.last_cand_stats: dict = {}
            self.cand_gen = cand_gen if cand_gen is not None else UnifiedCandidateGen(
                k=self.k,
                beta=float(_env("AZAI_CAND_BETA", "3.0")),
                ucb_c=float(_env("AZAI_CAND_UCBC", "1.0")),
                pool_mult=float(_env("AZAI_CAND_POOL", "1.5")),
                attempts_mult=int(_env("AZAI_CAND_ATTEMPTS", "6")),
                force_hold=_env("AZAI_CAND_FORCE_HOLD", "1") == "1",
                verbose=self.cand_verbose,
            )
            self.cand_gen.k = int(self.k)
            self.cand_gen.verbose = self.cand_verbose
            # 阶段 1 的 rollout 函数 = 本类的**确定性 idle** 实现（现役 idle 语义）
            self.cand_gen.rollout_fn = self.idle_rollout_values
            self._p1_cache = None          # 阶段 1 的 {规范化 bundle: [value]}（给阶段 2 复用）
            self._phase2_cache = None      # 只在 P1/P2 估计器一致时非 None

        # ── 候选来源：统一生成器（`bundle_mcts._expand` 的 pin 路径原样保留、不经过这里）──
        def _expand(self, node, k: int) -> float:
            from SDK.backend.model import Operation
            from SDK.utils.constants import OperationType

            if node.state.terminal:
                node.expanded = True
                return self._terminal_value(node.state, node.player)
            net_out = self.net_fn(node.state, node.player)
            node.net_value = float(net_out["value"])
            res = self.cand_gen.generate(node.state, node.player, net_out, k=int(k))
            bundles = list(res["bundles"])
            self.last_cand_stats = dict(res["stats"])
            self._p1_cache = dict(self.cand_gen.cache)
            # 阶段 2 复用：**只有两个估计器完全一致时**才允许（默认 idle vs random 不一致）
            self._phase2_cache = (self._p1_cache
                                  if self._p1_signature() == self._p2_signature() else None)
            if not bundles:
                node.expanded = True
                return node.net_value
            next_player = 1 - int(node.player)
            pri = 1.0 / float(len(bundles))
            for key in bundles:
                ops = [Operation(OperationType(int(x[0])), int(x[1]), int(x[2])) for x in key]
                child_state = node.state.clone()
                if ops:
                    child_state.apply_operation_list(node.player, ops)
                if node.player == 1:
                    child_state.advance_round()
                node.children.append(BundleNode(state=child_state, player=next_player,
                                                prior=pri))
                node.bundles.append(tuple(tuple(int(x) for x in op) for op in key))
                node.intent_counts.append([dict(), dict(), dict()])
            node.expanded = True
            return node.net_value

        # ── 估计器签名（判"阶段 1 的结果能不能直接当阶段 2 的样本"）──────────────
        def _p1_signature(self) -> tuple:
            return ("idle", int(self.horizon), bool(self._rng_on))

        def _p2_signature(self) -> tuple:
            return (str(self.rollout_mode), int(self.horizon), bool(self._rng_on))

        # ── search：父类一字不动，只在外面包日志 ──────────────────────────────
        def search(self, state, player: int, *, temperature: float = 0.0,
                   extra_bundles=None):
            import time as _time
            t0 = _time.perf_counter()
            res = super().search(state, player, temperature=temperature,
                                 extra_bundles=extra_bundles)
            dt_ms = (_time.perf_counter() - t0) * 1000.0
            st = self.last_cand_stats
            n_cand = len(res.bundles)
            has_hold = int(any(len(b or ()) == 0 for b in res.bundles))
            n_upg = sum(1 for b in res.bundles if any(int(o[0]) == 12 for o in (b or ())))
            # 选中项的性质（升级被选中多少次，靠这行统计）
            _ch = res.chosen_bundle or ()
            chosen_upg = int(any(int(o[0]) == 12 for o in _ch))
            chosen_hold = int(len(_ch) == 0)
            self.last_cand_stats.update({"n_cand": n_cand, "has_hold": has_hold,
                                         "n_upg": n_upg, "chosen_upg": chosen_upg,
                                         "chosen_hold": chosen_hold,
                                         "decide_ms": dt_ms})
            if self.cand_verbose:
                print(f"[cand] mode=unified n_cand={n_cand} 含HOLD={has_hold} 含升级={n_upg} "
                      f"池={st.get('n_pool', 0)} 重复={st.get('n_dup', 0)} "
                      f"长度分布={st.get('sizes', {})} "
                      f"p1_ms={st.get('p1_ms', 0.0):.1f} 决策_ms={dt_ms:.1f} "
                      f"选中={'HOLD' if chosen_hold else ('升级' if chosen_upg else '其它')}",
                      file=sys.stderr, flush=True)
            return res

    return UnifiedCandFlatSearch


def _env(name: str, default: str) -> str:
    import os
    return os.environ.get(name, default)
