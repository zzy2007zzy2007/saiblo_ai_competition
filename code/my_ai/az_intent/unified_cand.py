"""统一候选生成（方案 D）：**类 UCB bandit + 位置游标**（2026-10-29）。

设计（冻结，逐条照做）：`docs/unified_cand_design_20261029.md`
实现与验证读数：`docs/unified_cand_impl_20261029.md`

## 这个文件做什么

把"候选束"的来源从 **先钉一个类**（`bundle_mcts._expand` 里 `pinned=[_ccls]*3`）
换成 **从 `action_map` 全图跨类 × 位置取候选**：

1. **类选择**：UCB。利用项 = **玻尔兹曼加权平均**
   `Q[c] = Σᵢ wᵢ·xᵢ`，`wᵢ = exp(β·xᵢ)/Σⱼexp(β·xⱼ)`（β=3，`AZAI_CAND_BETA`）；
   探索项 = `c·√(log N / n)`（c=1.0，`AZAI_CAND_UCBC`）。`Q[c]` 初始全相等（均匀），
   **类网先验接口预留但暂不用**（类网现在基本不可用，要用得先重训）。
   （2026-10-30 从 mellowmax/LogSumExp 换过来：后者带 `log(n)/β` 折扣，且折扣最大恰在
   "一个神手 + 一堆垃圾"，与"突出神手"的意图相反；见 `_q_util` 的注释与调研
   `docs/prior_art_unified_search_20261030.md`。）
   信用分配：束分**完整**回写束内每个类（不按 |束长| 折扣 —— 语义 = "含 c 的组最好能有多好"）。
2. **位置游标 = 按"带重数的规范多重集（= bundle 形状）"索引，每个形状独立推进**：
   `cursor[形状] = {"t": 已推进步数, "pos": [已提议过的位置组合, ...]}`；第 t 步的位置 =
   各槽位名次按**混合进制（槽 0 变化最快）**展开 `rank_i = (t // Π_{j<i} n_j) % n_i`
   ⇒ 步 0 = 全体 argmax，之后逐格推进、穷尽全部组合（`np.argpartition` 只取"第 k 名"，
   并列按 (x,y) 字典序，**不做全排序**）。
   ⚠️ **不能按类共享游标**：那会让"一个位置被某一个 bundle 形状独占" —— `[build]` 把 build
   游标从 A 推到 C 之后，`[build(A), lightning(B)]` 这种组合就**永远构造不出**。
3. **候选键 = 带重数的规范多重集**：去掉 HOLD(23)、排序、**保留重复**。
   `[build] ≠ [build,build]`；`[build,lightning] == [lightning,build]`。
   ⇒ **同一束内允许重复同一个非 HOLD 类**（能生成 `[build,build]`、`[downgrade,downgrade]`）。
4. **无位置的类**（HOLD=23、基地升级=21/22）是**单位置臂**：槽位恒为 `NO_POSITION`，
   多重集里出现几次就占几个槽（名次数 = 1 ⇒ 该槽不随步数变化）。
5. **通道 → op 精确映射**：照 `code/my_ai/decoder.py`（`_decode_tower_action` /
   `CHANNEL_TO_SUPER_WEAPON` / `decoder.py:369-373` 的基地升级），**不自己发明**。
6. **每个候选逐条 `can_apply` 过滤**（严格模式一条非法 = 判负，绝不能漏）。
7. **阶段 1 = 确定性 idle rollout（双方不出招）×1**，分数用于 **top-k 选择**（λ=0：位置图
   分只参与"游标提议顺序"，不参与"选择权重"）。**阶段 2 = 现役 flat 随机化 rollout 完全不动**
   （在 `UnifiedCandFlatSearch` 里由父类 `search()` 原样执行）。
8. **缓存命中 ≠ 跳过**：缓存键 = 规范化 bundle（`(op_type,arg0,arg1)` 排序元组，去 HOLD），
   **跨形状共享**。命中时**复用缓存分数并计入对应类的 `Q[c]`**（等于已经 rollout 过），
   再**把该形状游标推进一格**继续试下一个位置，直到解出新 bundle 或该形状位置穷尽。
   玻尔兹曼加权对"重复同值"同样不敏感（`Σwᵢxᵢ` 里两个同值只把权重挪给该值）
   ⇒ 重复计分不会撑虚 `Q[c]`。

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
    CHANNEL_TO_TOWER_TYPE,
    SUPER_WEAPON_TO_OP_TYPE,
    _decode_tower_action,
    upgrade_step,
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


def decode_channel_op(state, player: int, ch: int, x: int, y: int,
                      decode_mode: str = "legacy"):
    """**通道 + 位置 → Operation**（逐条照 `decoder.py`，不发明语义）。

    * `0-15`（`decode_mode="legacy"`，默认）：`_decode_tower_action`（`decoder.py:433-456`）——
      空格 ⇒ `BUILD_TOWER(x,y)`（**建 Basic，忽略 target_type**）；我方塔 ⇒ `UPGRADE_TOWER(tower_id, step)`
      （step 由 `upgrade_step`，`decoder.py:82-94`）；敌方塔 ⇒ `None`。
    * `0-15`（`decode_mode="split"`，2026-10-31 新增，`AZAI_CAND_DECODE=split`）：
      **通道 0 = 只在空格建 Basic**（我方塔/敌方塔 ⇒ `None`）；
      **通道 1-15 = 只在"我方塔格"上升级**（空格 ⇒ `None`），目标类型照旧由
      `upgrade_step(tower_type, CHANNEL_TO_TOWER_TYPE[ch])` 决定。
      动机：16 个塔通道在空格**都建 Basic** ⇒ 16→1 的别名冗余，且升级只能靠"游标碰巧落到已占格"，
      位置网又把空格排前面 ⇒ 升级几乎搜不到（`3b306b5` 冒烟：含升级只有 5.6%）。
      split 后：① 别名塌缩消失 ② 升级变成 bandit 的**正常臂**（合法格 = 我方塔格）。
    * `16`：我方塔格 ⇒ `DOWNGRADE_TOWER(tower_id)`（`decoder.py:426-428`）。
    * `17-20`：`CHANNEL_TO_SUPER_WEAPON` → `SUPER_WEAPON_TO_OP_TYPE`，位置即 (x,y)。
    * `21/22`：`UPGRADE_GENERATION_SPEED(31)` / `UPGRADE_GENERATED_ANT(32)`（无位置）。
    * `23`：HOLD ⇒ `None`（不产生 op；空 bundle）。
    """
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    ch = int(ch)
    if 0 <= ch <= 15:
        if decode_mode != "split":
            return _decode_tower_action(state, int(player), ch, int(x), int(y))
        # ── split：通道 0 只建 Basic；通道 1-15 只升级 ─────────────────────────
        tower = state.tower_at(int(x), int(y))
        if ch == 0:
            # 空格才建（敌方占格/我方塔格都不建）
            if tower is None:
                return Operation(OperationType.BUILD_TOWER, int(x), int(y))
            return None
        if tower is None or int(tower.player) != int(player):
            return None                      # 升级只能在**我方塔格**
        step = upgrade_step(tower.tower_type, CHANNEL_TO_TOWER_TYPE[ch])
        if step is None:
            return None                      # 已到目标 / 到不了
        return Operation(OperationType.UPGRADE_TOWER, int(tower.tower_id), int(step))
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
                 tau: float = 1.0,
                 steps_per_shape: int = 12,
                 steps_total: int = 0,
                 seed: int = 0,
                 decode_mode: str = "legacy",
                 mode: str = "unified",
                 verbose: bool = False) -> None:
        # 2026-11-02（shape_pin）：候选生成模式。
        #   `unified`（默认）= 阶段 1 按 s1 取 top-k（现役统一候选，行为逐位不变）；
        #   `shape_pin` = 阶段 1 结束后**锁定 s1 第一名候选的有序类序列**，只在该序列内按
        #     位置游标顺序（网络先验 argmax→rank k+1）取前 k 个位置进阶段 2（介于 pin 与
        #     unified 之间）。默认关 ⇒ 不影响现役 unified/pin。见 docs/shape_pin_mode_20261102.md。
        assert mode in ("unified", "shape_pin"), mode
        self.mode = str(mode)
        self.k = max(1, int(k))
        self.beta = float(beta)
        self.ucb_c = float(ucb_c)
        # 2026-10-31（`AZAI_CAND_DECODE`）：塔通道的解码模式。
        #   `legacy`（默认）= 0-15 全走 `_decode_tower_action`（空格建 Basic / 我方塔升级）；
        #   `split`  = 通道 0 只建 Basic、通道 1-15 只在我方塔格升级（消掉 16→1 的别名冗余，
        #              并让"升级"成为 bandit 的正常臂）。**默认 legacy ⇒ 行为与 3b306b5 逐位相同**。
        assert decode_mode in ("legacy", "split"), decode_mode
        self.decode_mode = str(decode_mode)
        # 一个 bundle 最多几个 op：与网络头数一致（3）。HOLD 占一个头 ⇒ 自然产生 1/2 op 的束。
        self.n_heads = max(1, int(n_heads))
        # 生成的**候选池** = k × pool_mult（多生成一些，再按阶段 1 分数取 top-k）
        self.pool_mult = max(1.0, float(pool_mult))
        # 去重后仍凑不满 ⇒ 最多再试 k×attempts_mult 次
        self.attempts_mult = max(1, int(attempts_mult))
        self.force_hold = bool(force_hold)
        # 2026-10-29 修正 ①：**束内允许重复类**（`[build,build]` 要能生成）⇒ 头 0 用 UCB argmax
        #   （exploit），头 1..n-1 用 `softmax(UCB/τ)` **采样**（设计 §方案 D 步骤 1 原话：
        #   "按 Q[c] 的分布随机选类 c（UCB1 或 softmax）"）。τ→∞ 均匀；τ→0 退化成全取 argmax
        #   （= 每束 `[c,c,c]`，会把 `[build,lightning]` 这类互异组合挤掉）。
        self.tau = max(1e-6, float(tau))
        # 2026-10-29 修正 ④：命中缓存 ⇒ 复用分数 + **推进游标一格**继续试 ⇒ 每个形状一次最多走几格
        self.steps_per_shape = max(1, int(steps_per_shape))
        # 每次决策的**位置推进总步数**上限（0 ⇒ max(1024, 16k)）；防止"形状多 × 步数多"拖慢决策。
        #   实测代价 ≈ 0.3 ms/步（纯解码+can_apply，无 rollout）⇒ 1024 步 ≈ 0.3 s，相对阶段 2 可忽略；
        #   而步数不足会让"可达空间没枚举完"被误读成"候选塌缩"（`collapse_probe_p0_bigbudget.log`）。
        self.steps_total = (max(1024, 16 * self.k) if int(steps_total) <= 0
                            else max(1, int(steps_total)))
        # 类采样的随机流种子（默认 0；实际种子 = 本值 ⊕ **状态指纹** ⇒ 同局面结果可复现）
        self.cand_seed = int(seed)
        self.verbose = bool(verbose)
        # 阶段 1 的 rollout 函数（由搜索器注入：`(state, player, bundles) -> np.ndarray`）
        self.rollout_fn = None
        # 缓存：规范化 bundle → [value]（同一 bundle 只算一次；也给阶段 2 复用；**跨形状共享**）
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
        self.Q: dict = {}          # 类 → 已观测分数列表（玻尔兹曼加权利用项的原料）
        self.N: dict = {}          # 类 → 被选中次数（UCB 的 n）
        self.N_total = 0
        # 2026-10-29 修正 ③：游标**按带重数的规范多重集（= bundle 形状）索引，每形状独立推进**。
        #   `cursor[形状] = {"t": 已推进步数, "pos": [已提议过的位置组合, ...]}`
        self.cursor: dict = {}
        self.dead: set = set()     # 该局面下位置组合已穷尽的形状（避免重复空转）
        self._rng = None           # 决策内类采样随机流（由状态指纹种子化）
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
        if self.decode_mode == "split":
            # 2026-10-31（`AZAI_CAND_DECODE=split`）：把塔通道的合法位置**按角色切开** ——
            #   通道 0（建 Basic）只在**空格**；通道 1-15（高级塔）只在**我方塔格**。
            #   复用 `make_position_masks` 的代价逻辑（空格=建塔费、我方塔=升级费），只是加掩码约束
            #   ⇒ 金币门槛语义不变，变的只是"哪个通道能在哪类格子上出手"。
            my_cell = np.zeros_like(self.pmask[0], dtype=bool)
            for _t in state.towers_of(self.player):
                my_cell[int(_t.x), int(_t.y)] = True
            self.pmask[0] = self.pmask[0] & (~my_cell)      # 只建（空格）
            for _ch in range(1, 16):
                self.pmask[_ch] = self.pmask[_ch] & my_cell  # 只升级（我方塔格）
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

    def _seed_of(self, net_out: dict) -> int:
        """类采样随机流的种子 = `AZAI_CAND_SEED` ⊕ **状态指纹**（action_map + 玩家号）。

        有意**不含**"决策序号 / 时间 / 调用顺序" ⇒ 同一局面 ⇒ 同一候选集合（可复现），
        也让"改一个超参前后对比"是干净的同分布比较。
        """
        import hashlib
        fp = (np.ascontiguousarray(self.AM, dtype=np.float32).tobytes()
              + bytes([int(self.player) & 0xFF]))
        h = int.from_bytes(hashlib.blake2b(fp, digest_size=8).digest(), "little")
        return int(h ^ (int(self.cand_seed) * 1000003)) & ((1 << 63) - 1)

    # ── ① 类选择：UCB（玻尔兹曼加权利用项 + 标准探索项）──────────────────────
    def _q_util(self, ch: int) -> float:
        """利用项 = **玻尔兹曼加权平均**：`Q[c] = Σᵢ wᵢ·xᵢ`，`wᵢ = exp(βxᵢ)/Σⱼexp(βxⱼ)`。

        2026-10-30 改（设计 §"Q[c] 利用项"）：原来是 `(1/β)log((1/n)Σexp(βx))`（mellowmax），
        它带折扣 `log(n)/β`，而折扣**最大恰在"一个神手 + 一堆垃圾"**的情形
        （β=3、n=10、x=[1,−1×9] 时 mellowmax≈0.244 vs 真值 1.0），与"想突出神手"的意图相反。
        玻尔兹曼加权没有 `log(n)` 项，同一组数 β=3 时 ≈0.957（天然贴近最大值）。
        β→0 ⇒ 退化成均值；β→+∞ ⇒ 退化成 max；权重恒正 ⇒ 对负数成立。
        数值稳定：先减 max 再 exp（`w ∝ exp(β(x−m))`），归一化后再加权求和。
        """
        xs = self.Q.get(int(ch)) or []
        if not xs:
            return 0.0                      # 无观测 ⇒ 均匀先验（Q 全相等）
        arr = np.asarray(xs, dtype=np.float64)
        z = self.beta * (arr - float(arr.max()))
        w = np.exp(z)
        tot = float(w.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            return float(arr.mean())        # 安全阀（理论上到不了）
        w /= tot
        return float(np.dot(w, arr))

    def _ucb(self, ch: int) -> float:
        n = int(self.N.get(int(ch), 0))
        if n <= 0:
            return math.inf                 # 未试过的臂优先（标准 UCB1 初始化）
        log_n = math.log(max(1, int(self.N_total)))
        return self._q_util(ch) + self.ucb_c * math.sqrt(log_n / n)

    def _credit(self, shape: tuple, score: float) -> None:
        """回写 `Q[c]`：束分**完整**写给束内每个类（**不打 `1/|束长|` 折扣**）。

        语义（用户定夺，2026-10-30）：`Q[c]` = "**含 c 的动作组最好能有多好**"。
        例：`[build,lightning]=1.2` ⇒ "含 build 的最好的组"和"含 lightning 的最好的组"**都**是 1.2；
        若按 `1/|束长|` 折扣成 0.6，就退化成"c 的平均边际贡献"，与这个语义矛盾，
        还会把"只靠组合才强的类"压下去、漏掉好组合 ⇒ **不做折扣**。
        （曾短暂实现过折扣，已按用户定夺回滚；`Q[c]` 只影响探索分配，最终 top-k 仍按阶段 1
        rollout 分选，软最大天然容忍少量虚高分 ⇒ 完整回写不会扭曲选择。）
        重复类记多次；玻尔兹曼加权对重复同值不敏感（不会超过 max）。
        """
        score = float(score)
        for c in shape:
            if c in self.Q:
                self.Q[c].append(score)
        if not shape and HOLD in self.Q:
            self.Q[HOLD].append(score)

    def _ucb_pick(self, exclude=()) -> int | None:
        best, best_v = None, -math.inf
        for c in self.legal:
            if c in exclude:
                continue
            v = self._ucb(c)
            if v > best_v:
                best, best_v = int(c), v
        return best

    def _sample_ucb(self, rng) -> int | None:
        """按 `softmax(UCB/τ)` **采样**一个类（允许与同束已选的类重复）。"""
        if not self.legal:
            return None
        vals = [self._ucb(c) for c in self.legal]
        inf_c = [c for c, v in zip(self.legal, vals) if v == math.inf]
        if inf_c:
            # 未试过的臂（UCB=+inf）优先，在它们之间均匀取（= UCB1 的"先各试一遍"）
            return int(inf_c[int(rng.integers(len(inf_c)))])
        arr = np.asarray(vals, dtype=np.float64)
        z = (arr - float(arr.max())) / self.tau
        p = np.exp(z)
        tot = float(p.sum())
        if not np.isfinite(tot) or tot <= 0.0:
            return int(self.legal[int(rng.integers(len(self.legal)))])
        return int(self.legal[int(rng.choice(len(self.legal), p=p / tot))])

    def _pick_group(self, rng) -> tuple:
        """选一束的类（**允许重复**）⇒ `(形状, 被选类列表)`，并做 `N[c]` 记账。

        形状 = 去 HOLD、排序、**保留重复**（带重数的规范多重集）：
        `[build,build]` 与 `[build]` 是两个不同形状；`[build,lightning] == [lightning,build]`。
        头 0 = UCB argmax（exploit 主导）；头 1..n-1 = `softmax(UCB/τ)` 采样 ⇒ 同束既能出
        `[build,lightning]`（互异）也能出 `[build,build]`（重复）。
        """
        picked: list = []
        for h in range(self.n_heads):
            c = self._ucb_pick() if h == 0 else self._sample_ucb(rng)
            if c is None:
                break
            picked.append(int(c))
        for c in picked:                       # N 记账：被选中就算（含 HOLD 当"填充位"）
            self.N[c] = int(self.N.get(c, 0)) + 1
            self.N_total += 1
        return tuple(sorted(c for c in picked if c != HOLD)), tuple(picked)

    # ── ② 位置游标（按**形状** = 带重数规范多重集索引，每形状独立推进）──────────
    def _positions_for(self, shape: tuple):
        """给形状 `shape` 提议**下一组**位置；该形状的位置组合已穷尽 ⇒ `None`。

        第 t 步：各槽位名次按**混合进制**展开（槽 0 变化最快）
        `rank_i = (t // Π_{j<i} n_j) % n_i`，`n_i` = 该类位置图合法格数（无位置类 = 1）。
        ⇒ 步 0 = 全体 argmax；之后逐格推进、**穷尽全部组合**。不同形状**互不干扰**（这正是
        "按多重集而不是按类"的理由：按类共享会让某位置被某个形状独占，`[build(A),lightning(B)]`
        永远构造不出）。同一形状里两个同类槽位（`[build,build]`）也会拿到**不同**格子。
        """
        shape = tuple(shape)
        st = self.cursor.get(shape)
        if st is None:
            st = {"t": 0, "pos": []}
            self.cursor[shape] = st
        t = int(st["t"])
        cells = [self.cells.get(c) for c in shape]
        ns = [len(cc) if cc is not None else 1 for cc in cells]
        total = 1
        for n in ns:
            total *= int(n)
        if t >= total:
            return None                        # 该形状位置已穷尽
        r = t
        pos: list = []
        for cc, n in zip(cells, ns):
            k = r % int(n)
            r //= int(n)
            pos.append(cc.kth_cell(k) if cc is not None else NO_POSITION)
        pos_t = tuple(pos)
        st["t"] = t + 1
        st["pos"].append(pos_t)
        return pos_t

    # ── ③ 解码 + 逐条 can_apply ───────────────────────────────────────────
    def _decode_group(self, group: tuple, positions: tuple):
        """把 (类, 位置) 组解成 op 序列；**逐条 `can_apply` 过滤**（含与已 pending 的冲突）。

        返回 `(ops, classes)`：`classes[i]` = 产出 `ops[i]` 的通道号（**按 decode 顺序**）。
        2026-11-02（shape_pin）：下游要锁定 s1 第一名候选的"有序类序列"，必须知道每个 op
        来自哪个类、且保持 decode 顺序 ⇒ 这里顺带把通道号带出来（unified 路径忽略 `classes`）。
        """
        ops: list = []
        classes: list = []
        pending: list = []
        seen: set = set()
        for ch, cell in zip(group, positions):
            op = decode_channel_op(self.state, self.player, ch, cell[0], cell[1],
                                   decode_mode=self.decode_mode)
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
            classes.append(int(ch))
            seen.add(t)
            pending.append(op)
        return tuple(ops), tuple(classes)

    # ── ③b shape_pin：锁定有序类序列 → 按位置游标顺序取前 k 个位置 ──────────────
    def _shape_pin_topk(self, seq: tuple, k: int):
        """锁定有序类序列 `seq`，按**位置游标顺序**（混合进制，槽 0 变化最快，步 0 = 全体
        argmax，即"网络先验 argmax→rank k+1"）枚举其位置组合，取前 k 个**合法**候选。

        与阶段 1 共用 cache：新位置未缓存 ⇒ 跑一次确定性 idle rollout 填 s1（与阶段 1 同款）；
        已缓存 ⇒ 复用（计入 `n_dup`）。解不出任何 op 的位置组合**跳过**（不把"该位置全非法"
        当 HOLD）；canon_ops 去重。枚举上限 = min(k, 位置组合总数) ⇒ "凑不满就少输出"。

        返回 `(bundles, keys, scores, classes_list, n_enum, n_dup)`。
        """
        seq = tuple(seq)
        cells = [self.cells.get(c) for c in seq]
        ns = [len(cc) if cc is not None else 1 for cc in cells]
        total = 1
        for n in ns:
            total *= int(n)
        limit = min(int(k), total)
        bundles: list = []
        keys: list = []
        scores: list = []
        cls: list = []
        n_enum = 0
        n_dup = 0
        seen: set = set()
        for t in range(limit):
            r = t
            pos: list = []
            for cc, n in zip(cells, ns):
                kk = r % int(n)
                r //= int(n)
                pos.append(cc.kth_cell(kk) if cc is not None else NO_POSITION)
            n_enum += 1
            ops, _cs = self._decode_group(seq, tuple(pos))
            if not ops:
                continue                       # 该位置组合解不出任何合法 op ⇒ 跳过
            key = canon_ops(ops)
            if key in seen:
                continue
            seen.add(key)
            vals = self.cache.get(key)
            if vals is None:
                if self.rollout_fn is not None:
                    vals = list(np.asarray(self.rollout_fn(self.state, self.player, [ops]),
                                           dtype=np.float64).tolist())
                    self.cache[key] = vals
                else:
                    vals = []
            else:
                n_dup += 1
            score = float(np.mean(vals)) if vals else 0.0
            bundles.append(ops)
            keys.append(key)
            scores.append(score)
            cls.append(list(_cs))
        return bundles, keys, scores, cls, n_enum, n_dup

    # ── ④ 主流程 ──────────────────────────────────────────────────────────
    def generate(self, state, player: int, net_out: dict, k: int | None = None) -> dict:
        """生成候选并**按阶段 1 确定性 rollout 分排序**，返回 dict（含 `bundles`/统计）。"""
        k = int(self.k if k is None else k)
        self._reset_decision(state, player, net_out)
        if not self.legal or self.rollout_fn is None:
            self.last_stats = {"n_pool": 0, "n_cand": 0, "n_dup": 0, "attempts": 0,
                               "has_hold": 0, "hold_forced": 0, "n_upg": 0,
                               "sizes": {}, "p1_ms": 0.0, "steps": 0, "shapes": 0,
                               "n_rep": 0, "rep_examples": [], "dead_shapes": 0}
            return {"bundles": [], "stats": dict(self.last_stats)}

        import time as _time
        t0 = _time.perf_counter()
        pool_target = max(k, int(round(k * self.pool_mult)))
        max_attempts = max(pool_target * self.attempts_mult, 32)
        steps_total = int(self.steps_total)
        # 类采样的随机流：**由状态指纹种子化** ⇒ 同局面同结果（可复现，且不依赖调用顺序/时间）
        self._rng = np.random.default_rng(self._seed_of(net_out))
        rng = self._rng
        picked_out: list = []          # [(key, bundle, score, classes)]（classes = 有序类序列）
        n_dup = 0                      # **缓存命中次数**（复用分数 + 推进游标，不 rollout）
        n_attempts = 0
        n_steps = 0                    # 位置组合推进总步数
        hold_forced = 0
        n_rep = 0                      # 形状**含重复类**的候选条数（如 [build,build]）
        rep_examples: list = []

        def _credit(shape: tuple, score: float) -> None:
            """（转调外层方法，便于单测；语义见 `UnifiedCandidateGen._credit`。）"""
            self._credit(shape, score)

        def _emit(bundle: tuple, shape: tuple, classes: tuple = ()) -> bool:
            """缓存未命中 ⇒ rollout 收下（True）；命中 ⇒ 复用分数 + 计入 Q（False）。

            2026-11-02（shape_pin）：`classes` = 该 bundle 的有序类序列（decode 顺序），
            供 shape_pin 锁定 s1 第一名候选的类序列用。unified 路径只把它存进 picked_out。
            """
            nonlocal n_dup
            key = canon_ops(bundle)
            vals = self.cache.get(key)
            if vals is not None:
                n_dup += 1
                _credit(shape, float(np.mean(vals)) if vals else 0.0)
                return False
            vals = list(np.asarray(self.rollout_fn(self.state, self.player, [bundle]),
                                   dtype=np.float64).tolist())
            self.cache[key] = vals
            score = float(np.mean(vals)) if vals else 0.0
            _credit(shape, score)
            picked_out.append((key, bundle, score, classes))
            return True

        def _advance_shape(shape: tuple) -> bool:
            """给形状推进游标，直到解出**新** bundle（True）或位置穷尽/步数用尽（False）。

            修正 ④：命中缓存时**不是跳过** —— 复用分数计入 Q[c]，再推进一格继续试下一个位置。
            """
            nonlocal n_steps
            for _ in range(self.steps_per_shape):
                if n_steps >= steps_total:
                    return False
                pos = self._positions_for(shape)
                if pos is None:
                    self.dead.add(shape)       # 该形状位置已穷尽
                    return False
                n_steps += 1
                bundle, classes = self._decode_group(shape, pos)
                if _emit(bundle, shape, classes):
                    return True
            return False

        # ① HOLD（类 23 = 单位置臂）：**恒含 1 条进池**（设计 §6 第 3 问未拍板 ⇒ 取保守选项，
        #    与"注入 HOLD ⇒ +29.2pp"的证据口径一致）。top-k 仍按阶段 1 分决定它留不留。
        if self.force_hold and HOLD in self.legal:
            if _emit((), ()):
                hold_forced = 1
            self.N[HOLD] = int(self.N.get(HOLD, 0)) + 1
            self.N_total += 1
        # ② bandit 主循环：选一束的类（**允许同束内重复**）⇒ 该形状游标推进
        #    "dry" 早停：连续 dry_limit 次尝试都没解出新 bundle ⇒ 可达候选空间已枚举完
        #    （例：金币不足时只剩降级 + HOLD）⇒ 立刻停，别空转到 attempts 上限。
        dry_limit = max(32, k)
        n_dry = 0
        while (len(picked_out) < pool_target and n_attempts < max_attempts
               and n_steps < steps_total):
            n_attempts += 1
            shape, picked = self._pick_group(rng)
            if not picked:
                break
            if shape in self.dead:             # 该形状位置已穷尽 ⇒ 换形状（记一次 dry）
                n_dry += 1
                if n_dry >= dry_limit:
                    break
                continue
            got = _advance_shape(shape)
            if got:
                n_dry = 0
                if len(set(shape)) < len(shape):      # 形状含重复类（含重数的多重集）
                    n_rep += 1
                    if len(rep_examples) < 8:
                        rep_examples.append(list(shape))
            else:
                n_dry += 1
                if n_dry >= dry_limit:
                    if self.verbose:
                        print(f"[cand] 早停：连续 {n_dry} 次尝试无新候选"
                              f"（可达空间已枚举完：steps={n_steps} 形状={len(self.cursor)}"
                              f" 穷尽={len(self.dead)}）", file=sys.stderr, flush=True)
                    break

        # ── ③ 阶段 1 → 阶段 2 的候选选择 ───────────────────────────────────────
        #   unified（默认）：按阶段 1 分数取 top-k（设计 §2：确定性 idle rollout 分）。
        #   shape_pin（2026-11-02）：**锁定 s1 第一名候选的有序类序列**，再按位置游标顺序
        #     （网络先验 argmax→rank k+1）取该序列的前 k 个位置，**不按 s1 排序**。
        locked_seq: tuple = ()
        locked_score = 0.0
        n_locked_enum = 0
        n_locked_dup = 0
        if self.mode == "shape_pin":
            if picked_out:
                _best = max(picked_out, key=lambda x: x[2])   # (key, bundle, score, classes)
                locked_seq = tuple(int(c) for c in (_best[3] or ()))
                locked_score = float(_best[2])
            if not locked_seq:
                # s1 第一名是 HOLD（空束）⇒ 锁定"空序列" = pin 到 HOLD：候选 = 仅 HOLD 一条
                bundles = [()]
                keys = [()]
                scores = [float(max((x[2] for x in picked_out), default=0.0))]
            else:
                bundles, keys, scores, _cls, n_locked_enum, n_locked_dup = \
                    self._shape_pin_topk(locked_seq, k)
        else:
            picked_out.sort(key=lambda x: (-x[2], x[0]))
            bundles = [b for (_k, b, _s, _cs) in picked_out[:k]]
            keys = [kk for (kk, _b, _s, _cs) in picked_out[:k]]
            scores = [s for (_k, _b, s, _cs) in picked_out[:k]]
        n_upg = sum(1 for b in bundles if any(int(o[0]) == 12 for o in b))
        has_hold = int(any(len(b) == 0 for b in bundles))
        sizes: dict = {}
        for b in bundles:
            sizes[len(b)] = sizes.get(len(b), 0) + 1
        # 束内**同类重复**的候选（如 [build,build]）：按 op 三元组恢复不出类，改看"同 op 类型重复"
        n_multi_same = 0
        multi_same_examples: list = []
        for b in bundles:
            ts = [int(o[0]) for o in b]
            if len(ts) >= 2 and len(set(ts)) < len(ts):
                n_multi_same += 1
                if len(multi_same_examples) < 8:
                    multi_same_examples.append([tuple(int(x) for x in o) for o in b])
        p1_ms = (_time.perf_counter() - t0) * 1000.0
        self.last_stats = {
            "n_pool": len(picked_out), "n_cand": len(bundles), "n_dup": int(n_dup),
            "attempts": int(n_attempts), "has_hold": int(has_hold),
            "hold_forced": int(hold_forced), "n_upg": int(n_upg),
            "sizes": sizes, "p1_ms": p1_ms, "steps": int(n_steps),
            "shapes": len(self.cursor), "dead_shapes": len(self.dead),
            "n_rep": int(n_rep), "rep_examples": rep_examples,
            "n_multi_same": int(n_multi_same), "multi_same_examples": multi_same_examples,
            # `q_entries` = Σ_c |Q[c]|：缓存命中**也**计入 Q ⇒ 这个数远大于池大小（命中语义的读数）
            "q_entries": int(sum(len(v) for v in self.Q.values())),
            "keys": keys, "scores": scores,
            # shape_pin 专用读数（unified 下 locked_seq 恒空、n_locked_enum=0 ⇒ 零影响）
            "locked_seq": list(locked_seq),
            "locked_score": float(locked_score),
            "n_locked_enum": int(n_locked_enum),
            "n_locked_dup": int(n_locked_dup),
            # 2026-10-31（用户要求的可解释性埋点）：把进 top-k 的**真实 bundle**（下发顺序）
            #   与生成器内部状态一起交给 `search()` 打 `[candq]`。只在 verbose 时会被打印，
            #   但列表本身**总是**填（48 个小元组，零成本）—— 免得"verbose 与非 verbose 走不同分支"。
            "bundles": [tuple(tuple(int(x) for x in op) for op in b) for b in bundles],
            "Q": {int(c): float(self._q_util(c)) for c in self.legal},
            "Nmap": {int(c): int(self.N.get(c, 0)) for c in self.legal},
            "N_total": int(self.N_total),
            "beta": float(self.beta), "ucb_c": float(self.ucb_c),
        }
        if self.verbose:
            _sp = (f" 锁定序列={locked_seq}(枚举{n_locked_enum},缓存命中{n_locked_dup})"
                   if self.mode == "shape_pin" else "")
            print(f"[cand] gen mode={self.mode} 池={len(picked_out)} 取={len(bundles)} "
                  f"缓存命中={n_dup} 步数={n_steps} 形状={len(self.cursor)}(穷尽{len(self.dead)}) "
                  f"重复类形状={n_rep} 同类型多op={n_multi_same} Q条目={self.last_stats['q_entries']} "
                  f"HOLD={has_hold}(兜底{hold_forced}) 升级={n_upg} 长度分布={sizes} "
                  f"{p1_ms:.0f} ms{_sp}", file=sys.stderr, flush=True)
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
            # 2026-11-02：unified_noupg = unified + 阶段1→阶段2 之间删掉含升级(op12)的候选。
            #   默认 False（unified/shape_pin 逐字不变）；见 `_expand` 的过滤 + docs/noupg 文档。
            self.cand_no_upgrade = (str(cand_mode) == "unified_noupg")
            self.last_cand_stats: dict = {}
            # 2026-11-01（用户要求）：**游标推进深度**的跨决策累计（`AZAI_CAND_VERBOSE=1` 才填）。
            #   `cursor_agg[shape_str] = {"n_trials": 该形状累计被提议的步数,
            #                             "n_dec": 出现过该形状的决策数,
            #                             "t": [每决策的 t（推进步数）, ...]}`
            #   用途：回答"被选得多的形状，游标推到几格？是卡在 steps_per_shape 还是更浅？"
            self.cursor_agg: dict = {}
            # 2026-11-02：unified_noupg 的**阶段1 生成器**与 unified 完全相同（bandit + 游标），
            #   差异只在 `_expand` 里"阶段1→阶段2 之间删升级候选" ⇒ 生成器 mode 用 "unified"。
            _gen_mode = "unified" if self.cand_mode == "unified_noupg" else self.cand_mode
            self.cand_gen = cand_gen if cand_gen is not None else UnifiedCandidateGen(
                k=self.k,
                beta=float(_env("AZAI_CAND_BETA", "3.0")),
                ucb_c=float(_env("AZAI_CAND_UCBC", "1.0")),
                pool_mult=float(_env("AZAI_CAND_POOL", "1.5")),
                attempts_mult=int(_env("AZAI_CAND_ATTEMPTS", "6")),
                force_hold=_env("AZAI_CAND_FORCE_HOLD", "1") == "1",
                tau=float(_env("AZAI_CAND_TAU", "1.0")),
                steps_per_shape=int(_env("AZAI_CAND_STEPS", "12")),
                steps_total=int(_env("AZAI_CAND_STEPS_TOTAL", "0")),
                seed=int(_env("AZAI_CAND_SEED", "0")),
                decode_mode=_env("AZAI_CAND_DECODE", "legacy").strip().lower(),
                mode=_gen_mode,
                verbose=self.cand_verbose,
            )
            self.cand_gen.k = int(self.k)
            self.cand_gen.verbose = self.cand_verbose
            # 阶段 1 的 rollout 函数 = 本类的**确定性 idle** 实现（现役 idle 语义）
            self.cand_gen.rollout_fn = self.idle_rollout_values
            self._p1_cache = None          # 阶段 1 的 {规范化 bundle: [value]}（给阶段 2 复用）
            self._phase2_cache = None      # 只在 P1/P2 估计器一致时非 None
            # 2026-11-02（父任务硬约束）：shape_pin 只在"阶段 1 → 阶段 2 选候选"这一处动刀，
            #   阶段 2 仍是父类 FlatRolloutGreedy.search()。这里**无条件**打一行，证明阶段 2
            #   参数 = 部署 BASE 的 flat 参数（rollouts/horizon/batch/mode/side/prob/rounds/seed/
            #   share_prefix/eval/k 全原样从 self 读），便于与 pin/unified 的 stderr 对表。
            if self.cand_mode in ("unified", "shape_pin", "unified_noupg"):
                # 2026-11-02：unified 也打这一行（原来只 shape_pin 打），证明阶段 2 的
                #   flat 参数（rollout_mode 等）确实 = 部署 BASE / 本实验的 lightning 臂。
                #   ⚠️ lightning 模式忽略 rollout_random_side、固定随机侧=我方 ⇒ 侧这里
                #   要显示成"我方"，否则 stderr 会打出误导性的 `side=opponent`。
                _side_disp = ("ours(root)" if self.rollout_mode == "lightning"
                              else self.rollout_random_side)
                print(f"[{self.cand_mode}] engine=flat rollouts={self.rollouts_per_candidate} "
                      f"horizon={self.horizon} batch={self.value_batch} "
                      f"rollout_mode={self.rollout_mode} side={_side_disp} "
                      f"prob={self.rollout_prob_mode}({self.rollout_decay_c},"
                      f"{self.rollout_decay_k}) rounds={self.max_total_rounds} "
                      f"seed={self.rollout_seed} share_prefix={int(self.rollout_share_prefix)} "
                      f"eval={self.eval_mode} k={self.k} decode={self.cand_gen.decode_mode}",
                      file=sys.stderr, flush=True)

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
            # 2026-11-02：unified_noupg —— **只在"阶段1 结束、top-k 已生成、阶段2 之前"** 删掉
            #   含升级(op12)的候选；阶段1 bandit/cursor 与阶段2 flat rollout 都一字不动。
            #   兜底：全被删 ⇒ 退化为 HOLD（空束）。理论上有 force_hold=1 兜底所以罕见。
            if self.cand_no_upgrade:
                _n_before = len(bundles)
                bundles = [b for b in bundles
                           if not any(int(o[0]) == 12 for o in (b or ()))]
                if not bundles:
                    bundles = [()]
                self.last_cand_stats["n_upg_removed"] = _n_before - len(bundles)
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

        # ── 决策内可解释性日志（`AZAI_CAND_VERBOSE=1` 才打；用户 2026-10-31 要求）──────
        @staticmethod
        def _cat(bundle) -> str:
            """候选/选中项的类别标签（可多标，用 `+` 连）。"""
            ts = [int(o[0]) for o in (bundle or ())]
            if not ts:
                return "HOLD"
            tags = []
            if 12 in ts:
                tags.append("升级")
            if 11 in ts:
                tags.append("建塔")
            if 13 in ts:
                tags.append("降级")
            if len(ts) >= 2:
                tags.append("多op")
            if not tags:
                tags.append("其它")
            return "+".join(tags)

        def _emit_candq(self, st: dict, res, n_cand: int) -> None:
            """打 `[candq]` 三行：阶段 1 的 Q[c]/N[c]、进 top-k 的候选+阶段 1 分、最终选中。

            ⚠️ 只读 `st`（生成器给的读数）与 `res`（父类 search 的结果）⇒ 不改任何决策。
            """
            beta, cc = st.get("beta"), st.get("ucb_c")
            Q = st.get("Q") or {}
            N = st.get("Nmap") or {}
            n_tot = int(st.get("N_total", 0))
            order = sorted(Q, key=lambda c: (-Q[c], -N.get(c, 0)))
            qs = " | ".join(f"{c}:Q{Q[c]:+.3f}/N{N.get(c, 0)}" for c in order)
            print(f"[candq] 阶段1结束 β={beta} c={cc} N_total={n_tot} "
                  f"合规类={len(Q)} | Q/N（按 Q 降序）: {qs}",
                  file=sys.stderr, flush=True)
            bundles = st.get("bundles") or []
            sc1 = st.get("scores") or []
            pairs = list(zip(bundles, sc1))[:n_cand]
            from collections import Counter
            cnt = Counter()
            for b, _s in pairs:
                for tag in self._cat(b).split("+"):
                    cnt[tag] += 1
            top = " ".join(f"[{b},{s:+.3f}]" for b, s in pairs[:8])
            print(f"[candq] topk n={len(pairs)} 构成=dict({dict(cnt)}) 前8={top}",
                  file=sys.stderr, flush=True)
            scores = getattr(self, "last_scores", None)
            ch = res.chosen_bundle or ()
            deg = False
            if scores is not None and len(scores) == len(res.bundles) and len(scores):
                mx, mn = float(np.nanmax(scores)), float(np.nanmin(scores))
                deg = (not np.isfinite(scores).any()) or (mx - mn < 1e-9)
            s_ch = (float(scores[res.chosen_index])
                    if scores is not None and len(scores) > res.chosen_index else float("nan"))
            # 非 HOLD 里阶段 2 分最高的那条（用来回答"选 HOLD 时是不是有更好的非 HOLD"）
            best_nh, best_nh_s, best_nh_cat = None, -math.inf, "-"
            s_hold = None
            if scores is not None:
                for b, s in zip(res.bundles, np.asarray(scores).tolist()):
                    if len(b or ()) == 0:
                        s_hold = float(s)
                        continue
                    if float(s) > best_nh_s:
                        best_nh, best_nh_s, best_nh_cat = b, float(s), self._cat(b)
            gap = (best_nh_s - (s_hold if s_hold is not None else float("nan"))
                   if best_nh is not None and s_hold is not None else float("nan"))
            print(f"[candq] chosen {tuple(ch)} 类别={self._cat(ch)} 阶段2={s_ch:+.4f} "
                  f"回退={'是(全等/全NaN)' if deg else '否'} | "
                  f"非HOLD最高={best_nh_s:+.4f}({best_nh_cat}) {tuple(best_nh) if best_nh else ()} "
                  f"| HOLD阶段2={'n/a' if s_hold is None else f'{s_hold:+.4f}'} "
                  f"| 非HOLD最高−HOLD={gap:+.4f}",
                  file=sys.stderr, flush=True)

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
                print(f"[cand] mode={self.cand_mode} n_cand={n_cand} 含HOLD={has_hold} 含升级={n_upg} "
                      f"删升级={st.get('n_upg_removed', 0)} decode={self.cand_gen.decode_mode} "
                      f"池={st.get('n_pool', 0)} 缓存命中={st.get('n_dup', 0)} "
                      f"步数={st.get('steps', 0)} 形状={st.get('shapes', 0)}"
                      f"(穷尽{st.get('dead_shapes', 0)}) 重复类形状={st.get('n_rep', 0)} "
                      f"同类型多op={st.get('n_multi_same', 0)} "
                      f"长度分布={st.get('sizes', {})} "
                      f"p1_ms={st.get('p1_ms', 0.0):.1f} 决策_ms={dt_ms:.1f} "
                      f"选中={'HOLD' if chosen_hold else ('升级' if chosen_upg else '其它')}",
                      file=sys.stderr, flush=True)
                if st.get("rep_examples"):
                    print(f"[cand] 重复类形状样例={st['rep_examples'][:4]} "
                          f"同类型多op样例={st.get('multi_same_examples', [])[:3]}",
                          file=sys.stderr, flush=True)
                self._emit_candq(st, res, n_cand)
                self._emit_candcursor()
                if self.cand_mode == "shape_pin":
                    # 2026-11-02（shape_pin 逐决策读数）：锁定的是哪个有序类序列 + 枚举了多少位置。
                    #   供离线聚合"shape_pin 的候选构成"（docs/shape_pin_mode_20261102.md）。
                    print(f"[shape_pin] locked_seq={self._shape_str(tuple(st.get('locked_seq', ())))} "
                          f"locked_score={float(st.get('locked_score', float('nan'))):+.4f} "
                          f"n_cand={n_cand} n_enum={st.get('n_locked_enum', 0)} "
                          f"cache_hit={st.get('n_locked_dup', 0)} n_pool={st.get('n_pool', 0)}",
                          file=sys.stderr, flush=True)
            return res

        # ── 游标推进深度（`AZAI_CAND_VERBOSE=1` 才打；只读 `cand_gen.cursor`，不改任何决策）──
        _CH_NAME = {0: "建塔", 12: "升级", 13: "降级", 16: "降级", 17: "闪电", 18: "EMP",
                    19: "偏转", 20: "闪避", 21: "升速", 22: "升蚁", 23: "HOLD"}

        @classmethod
        def _shape_str(cls, shape) -> str:
            return "+".join(cls._CH_NAME.get(int(c), f"c{int(c)}") for c in shape) or "空"

        def _emit_candcursor(self) -> None:
            """每决策一行 `[candcursor]`：本决策各形状的游标推进步数 t（按 t 降序取前 8）。

            并把 (形状 → 每决策 t) 累计进 `self.cursor_agg`，供整局结束后算"被选得多的形状推多深"。
            """
            cur = getattr(self.cand_gen, "cursor", None) or {}
            if not cur:
                return
            rows = []
            for shape, stt in cur.items():
                t = int(stt.get("t", 0))
                key = self._shape_str(shape)
                agg = self.cursor_agg.get(key)
                if agg is None:
                    agg = {"n_trials": 0, "n_dec": 0, "t": []}
                    self.cursor_agg[key] = agg
                agg["n_trials"] += t
                agg["n_dec"] += 1
                agg["t"].append(t)
                rows.append((t, key))
            rows.sort(reverse=True)
            top = " ".join(f"{k}:t={t}" for t, k in rows[:8])
            print(f"[candcursor] shapes={len(rows)} 本决策 t 合计={sum(r[0] for r in rows)} "
                  f"t={max(r[0] for r in rows)} 前8={top}", file=sys.stderr, flush=True)

    return UnifiedCandFlatSearch


def _env(name: str, default: str) -> str:
    import os
    return os.environ.get(name, default)
