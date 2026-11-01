"""把我们自己的 AI（az_intent 的 Bundle-MCTS）接进 bridge 的协议外壳。

背景（`docs/champ_ladder_plan.md` §3.2 / §7.6）：我们**从来没有跟冠军/亚军打过**——
我们的 AI 是 in-process 的，不会讲官方 stdin/stdout 协议。`code/test_match/rv4_pkg/`
是 rule_v4 的壳（官方 ProtocolSession），本文件是同一件事，但把决策换成我们的搜索。

**设计要点（关键）**：
  * **影子环境就是 bridge 的裁判本身**：`GameStateFacade.initial(seed=...)`。
    只要按 bridge 的确切顺序重放（init → 每回合 p0 先动 → 收对手操作 → advance），
    两边状态**按构造同步**，不需要从文本反猜局面。
  * **每回合用收到的局面文本对拍**（`state_to_text(facade)` vs 收到的字节），
    分歧**检出并计数**——而不是像当年两个 C++ AI 那样靠内嵌模拟悄悄分歧、然后静默弃权
    （见 `memory/project_champion_runnerup_cpp_match.md`）。
  * 文本序列化**从 match_via_sdk 导入**，绝不在这里重打：重打就等于对拍失效。

**协议**（由 `rv4_pkg/protocol.py` 核准，它是能跑通的 known-good control）：
  * AI 收：**裸 UTF-8 文本行**。init = `"<player> <seed>\\n"`；
    之后每回合：局面文本（结构化行） + 对手操作文本（首行是操作数量）。
  * AI 发：`struct.pack(">I", len(data)) + data`（4 字节大端长度前缀）。

**bridge 的确切时序**（`match_via_sdk.run_match` 的循环）：
  p0 先写本回合操作 → p1 收到 → p1 写 → p0 收到 p1 的操作 → 双方操作各自 apply
  → `advance_round()` → 把**推进后**的局面文本发给双方。

**配置走环境变量**（bridge 用 `Popen([sys.executable, path])` 起 .py AI，不传参数）：
  AZAI_CKPT=<path>          必填
  AZAI_ITERS AZAI_DEPTH AZAI_TCLASS AZAI_TPOS AZAI_K AZAI_TEMP
  AZAI_MODE(joint|class-only|pos-only) AZAI_POSPIN(argmax|playable)
  AZAI_SKIP1(0/1) AZAI_VALUE_TANH(0/1) AZAI_REL2ABS(float) AZAI_TRACE(0/1) AZAI_DEVICE(cpu|auto)
  AZAI_VALUE_CKPT=<path>    可选：把**叶子值网**换成该 ckpt 里的独立 `ValueNetwork`
                            （**空 = 关 = 完全走原逻辑**，不加载任何额外模型，行为逐位不变）。
                            见 `code/my_ai/az_intent/value_override.py`；产出该 ckpt 用
                            `code/test_match/fold_small_value_ckpt.py`（必须已折 y_std/y_mean）。
  AZAI_UPG_BUNDLES=<int>     **升级候选进束**（2026-10-28，默认 **0 = 完全关闭、行为逐位不变**）：
                             每回合把最多 N 条"合法且买得起"的单 op 升级 `((12, tower_id, target),)`
                             通过**已有的**额外候选通道 `set_extra_bundles`（注入/安慰剂同一条）塞进去，
                             与网络的 K 条候选**同台**被 flat rollout 评估器打分。**不改搜索算法**，
                             也**不是** `AZAI_MC_UPG`（那个是 MC 仲裁直接接管一手、绕过评估器）。
                             见 `docs/upgrade_bundles_20261028.md`；日志前缀 `[upg]`（stderr）。
  AZAI_ORACLE_INJECT_B=<path> **第二个外部查询会话**（2026-10-28，E4；默认空 = 完全关闭）：
                              A 会话仍是 `AZAI_ORACLE_INJECT`（那条路径**一行没改**）；B 是第二个
                              `RunnerUpOracle` 进程（E4 里 A=冠军、B=亚军）。两者每回合各问一次，
                              两条动作**一起**进 `set_extra_bundles([A, B])`。`AZAI_ORACLE_TIMEOUT_B`
                              是 B 的单次读包超时（默认取 `AZAI_ORACLE_TIMEOUT`；E4 脚本设 20s）。
                              见 `docs/us_vs_champion_20261028.md` §5；日志前缀 `[oracle-b]`。
  AZAI_ONLY_EXTRA=1          **候选集只保留额外候选**（2026-10-28，E4；默认 0 = 完全关闭）：
                              打开后 `FlatRolloutGreedy.set_only_extra(True)` ⇒ 候选集 = 额外塞进去
                              的 bundle（E4 里 = 冠军动作 + 亚军动作），网络那 K 条被类轴 pinned 的
                              候选**不进候选集**；预算公式/rollout/叶值网/选择规则一律不变。
  AZAI_ONLY_EXTRA_RANDOM=1   **随机挑一个**（2026-10-28，E4R；默认 0 = 关；隐含 `AZAI_ONLY_EXTRA=1`）：
                              候选集与 E4 相同，但**不看评估器分数**，用
                              `sha256("onlyextra|<seed>|<dec_n>")` 的独立随机流挑一个 ⇒ 固定种子可复现。
                              `AZAI_ONLY_EXTRA_SEED` 可覆盖随机流种子（默认 = `AZAI_FLAT_SEED`）。
  AZAI_PIN_K=<int>           **整束允许落在 K 个不同类上**（2026-10-28，默认 **0 = 完全关闭、行为逐位不变**）：
                              现役 `pinned=[_ccls]*3`（`bundle_mcts.py:870`）把 720 次采样钉在
                              **一个类**上，而该类在 78% 的决策里是 HOLD(23) ⇒ 空束 ⇒ `n_cand=1`
                              （= 搜索完全没有选择余地）。`=K>1` ⇒ 把 720 次采样轮流分给 K 个类
                              （MC 的赢家 + `_econ_rank` 剩下的前几名）。**只改"哪些类参与采样"**：
                              去重 / top-k 截断 / flat rollout 评估器 / 预算公式一行不动。
                              见 `docs/why_effective_candidates_collapse_20261028.md`。
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import struct
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ⚠️ 协议流保护：stdout 是"长度前缀"的二进制通道，任何漏出去的 print（torch / SDK / warning）
# 都会把后续所有包的长度前缀冲掉 ⇒ 直接腐坏整局。所以先把真正的 stdout 存下来，
# 再把 sys.stdout 指到 stderr：漏出的输出全部去 stderr（bridge 会收进日志）。
_PROTO_OUT = sys.stdout.buffer
sys.stdout = sys.stderr


def _load_bridge_module():
    """加载 match_via_sdk，拿到**同一份** state_to_text/ops_to_text/parse_ops_text/GameState。

    必须用它、不能重写：对拍要求逐字节一致，而"重打一遍等价逻辑"正是记忆里点名过的坑
    （feedback_known_good_control_for_harness 第四条）。
    """
    spec = importlib.util.spec_from_file_location("mvs_for_azai", _HERE / "match_via_sdk.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


MVS = _load_bridge_module()
GameState = MVS.GameState
state_to_text = MVS.state_to_text
ops_to_text = MVS.ops_to_text
parse_ops_text = MVS.parse_ops_text


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ── 可选：把"判据分布内的 (observation, 终局) 对"dump 出来当价值网训练数据（2026-10-05）
# 动机：我们跑的**对 rule_v4 的对局**正是判据的局面分布（对手每局建 ~16.5 座塔），
# 而自对弈采到的数据**一个塔局面都没有**（实测 `己方有塔` = 0.0%，见
# `code/test_match/check_tower_coverage.py`）。⇒ 让**测量批同时当采集批**。
#
# ⚠️ **为什么是"边打边追加的裸文件"而不是"结束时写 pkl"**：bridge 在收尾时会**硬杀**子进程，
# 实测**连 `atexit` 和自己函数末尾的代码都来不及跑完**（`[dump] 调用: samples=367` 成了日志最后一行、
# 文件没写出来）。所以这里改成：**每个决策就把 board/stats/player 追加进一个定长记录的 .bin**，
# 终局标签**不由我们写**，而是事后用 **ladder 日志里的 `RESULT ... base_hp=`** 反推
# （那个日志是 bridge 自己写的，不会被杀）⇒ 竞态彻底消失。
# 默认关闭（`AZAI_DUMP_RAW_DIR` 为空）⇒ 不影响任何既有读数。
_DUMP: dict = {"dir": "", "seed": 0, "player": 0, "fh": None, "n": 0}
# ── AZ 环数据（2026-10-19；预注册 `docs/prereg_20261019_goal2_az_loop.md`）────────────
# 把**搜索的软策略**（root 的候选束 + 访问分布 + intent_counts）dump 成 JSONL；观测（board/stats）
# 走既有的 `AZAI_DUMP_RAW_DIR` 路径，两者按 (seed, player, 决策序号 n) 对齐。**默认关闭**（`AZAI_AZ_DUMP` 空）。
_AZ: dict = {"dir": None, "seed": 0, "player": 0, "fh": None, "n": 0}


def _json_safe(o, depth: int = 0):
    """把任意嵌套结构（numpy / 元组 / 字典 / op 对象）转 JSON 安全纯类型。

    实测踩坑（2026-10-19 AZTEST，第二次）：`intent_counts` 是**字典的嵌套**（键本身还可下标），
    `chosen_index`/`visits` 的形状也不保证是标量 ⇒ 逐字段 int() 会 TypeError（整个 dump 变空）。
    """
    if depth > 4:
        return None
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if hasattr(o, "op_type"):
        return [int(o.op_type), int(getattr(o, "arg0", 0)), int(getattr(o, "arg1", 0))]
    if hasattr(o, "tolist"):
        try:
            return _json_safe(o.tolist(), depth + 1)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(o, dict):
        return {str(_json_safe(k, depth + 1)): _json_safe(v, depth + 1) for k, v in list(o.items())[:128]}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in list(o)[:256]]
    try:
        return int(o)
    except Exception:  # noqa: BLE001
        try:
            return float(o)
        except Exception:  # noqa: BLE001
            return str(o)[:64]


def _bundle_ops(b):
    """从一个候选束里取出 op 三元组列表（**稳健**）。

    实测踩坑（2026-10-19 AZTEST）：`res.bundles` 的元素**外层还包了一层**（不是直接的 op 序列），
    直接 `int(o[0])` 会 `TypeError: int() ... not 'list'`。这里递归抽取，兼容：
    ① `(op_type, arg0, arg1)` 三元组；② 带 `op_type/arg0/arg1` 属性的 op 对象；③ 上述两者的任意嵌套。
    """
    out: list = []

    def _is_scalar(v) -> bool:
        return isinstance(v, (int, float, bool)) or (hasattr(v, "__int__") and not isinstance(v, (list, tuple, dict)))

    def _walk(x) -> None:
        if x is None:
            return
        if hasattr(x, "op_type"):
            out.append((int(x.op_type), int(getattr(x, "arg0", 0)), int(getattr(x, "arg1", 0))))
            return
        if isinstance(x, (list, tuple)):
            if len(x) >= 3 and all(_is_scalar(v) for v in x[:3]):
                out.append((int(x[0]), int(x[1]), int(x[2])))
                return
            for v in x:
                _walk(v)

    _walk(b)
    return out


def _op_label(op_type: int, a0: int, a1: int, facade, player: int):
    """把 (op_type, arg0, arg1) 解成 (class, x, y)；解不出就 (23, -1, -1)。

    与 `code/distill/collect.py:_op_to_label` 同一口径（类号约定：0–15 = 目标塔型通道、16 = 降级、
    17 = 闪电、23 = HOLD）。**升级/降级需要局面**（塔 id → 格子）⇒ 所以必须在桥接端（有 facade）解，
    不能留给离线转换器（观测 dump 里只有类型网格，没有塔 id）。
    """
    try:
        from SDK.utils.constants import OperationType as _OT
        from my_ai.decoder import CHANNEL_TO_TOWER_TYPE, upgrade_step
        if int(op_type) == int(_OT.BUILD_TOWER):
            return (0, int(a0), int(a1))
        if int(op_type) == int(_OT.UPGRADE_TOWER):
            for _t in facade.towers_of(int(player)):
                if int(getattr(_t, "tower_id", -1)) == int(a0):
                    for _ch in range(1, 16):
                        _tt = CHANNEL_TO_TOWER_TYPE[_ch]
                        _step = upgrade_step(getattr(_t, "tower_type", None), _tt)
                        if _step is not None and int(_step) == int(a1):
                            return (_ch, int(_t.x), int(_t.y))
            return (23, -1, -1)
        if int(op_type) == int(_OT.DOWNGRADE_TOWER):
            for _t in facade.towers_of(int(player)):
                if int(getattr(_t, "tower_id", -1)) == int(a0):
                    return (16, int(_t.x), int(_t.y))
            return (23, -1, -1)
        if int(op_type) == int(_OT.USE_LIGHTNING_STORM):
            return (17, int(a0), int(a1))
    except Exception as _exc:  # noqa: BLE001
        # 2026-10-22（审计）：这里静默返回 (23,-1,-1)=HOLD ⇒ 升级/降级 op 会被标成"空过"，
        # **污染训练标签**。实测正常 op 不受影响（升级→(1,4,2)、降级→(16,4,2)），
        # 但一旦发生必须看得见。
        global _OP_LABEL_WARNED
        if not _OP_LABEL_WARNED:
            _OP_LABEL_WARNED = True
            print(f"[label] ⚠️ _op_label 失败（首次）：{type(_exc).__name__}: {_exc}", flush=True)
    return (23, -1, -1)


def _record_az(res, facade=None, player=None) -> None:
    """把一次决策的**搜索软策略**追加进 JSONL（`AZAI_AZ_DUMP=<dir>` 时才写）。

    字段：`n`(决策序号) `seed` `player` `cands`(候选束，每个束 ≤3 个 op) `visits`(访问分布，与 cands 同序)
    `chosen`(选中下标) `intent`(3 个头的 intent 计数) `root_value`。默认关闭 ⇒ 不影响任何既有读数。
    """
    if _AZ["dir"] is None:
        _AZ["dir"] = _env("AZAI_AZ_DUMP", "")
        if _AZ["dir"]:
            out = Path(_AZ["dir"])
            if not out.is_absolute():
                out = Path(MVS.REPO_ROOT) / out
            out.mkdir(parents=True, exist_ok=True)
            _AZ["path"] = str(out / f"az_seed{_AZ['seed']:05d}_p{_AZ['player']}.jsonl")
            _AZ["fh"] = open(_AZ["path"], "a", encoding="utf-8")
            _log(f"AZ 软策略 dump 已开启 -> {_AZ['path']}")
    if not _AZ["dir"]:
        return
    import json
    try:
        vp = _json_safe(getattr(res, "visit_policy", None)) or []
        bs = getattr(res, "bundles", None) or []
        cands = [[[int(o[0]), int(o[1]), int(o[2])] for o in _bundle_ops(b)] for b in bs]
        # 已解出的 (class, x, y) 标签（升级/降级要局面 ⇒ 在这里解；供离线转换器直接构目标）
        labels = [[list(_op_label(int(o[0]), int(o[1]), int(o[2]), facade, _AZ["player"]))
                   for o in _bundle_ops(b)] for b in bs] if facade is not None else []
        rec = {"n": int(_AZ["n"]), "seed": int(_AZ["seed"]), "player": int(_AZ["player"]),
               "cands": cands, "labels": labels, "visits": vp,
               # 附加字段（2026-10-11）：**原始访问计数** —— `visits` 在 TEMP=1e-6 下已被 argmax 成 one-hot
               # ⇒ 只有 `visits_raw` 才能用来做"蒸搜索偏好"的目标（见 docs/pos_topk_candidates_plan.md）
               "visits_raw": _json_safe(getattr(res, "visits_raw", None)),
               "chosen": _json_safe(getattr(res, "chosen_index", -1)),
               "intent": _json_safe(getattr(res, "intent_counts", None)),
               "root_value": _json_safe(getattr(res, "root_value", 0.0))}
        # 2026-10（蒸馏用）：把**原始状态**也 dump 进同一条 JSONL（`AZAI_AZ_DUMP_RAW=1`）。
        # 动机：编码后的 board/stats 缺 tower.hp / tower.type / ant.kind ⇒ 没法用亚军的
        # `evaluate_terminal` 精确打分；这里把塔/蚁/金币/基地原样存下来，离线即可算精确标签。
        # 与 .bin（`AZAI_DUMP_RAW_DIR`）按 n 一一对齐（两者都在 `_record_obs` 里同一拍写）。
        if _env("AZAI_AZ_DUMP_RAW", "0") == "1" and facade is not None:
            try:
                rec["state"] = {
                    "round": int(facade.round_index),
                    "coins": [int(c) for c in facade.coins],
                    "bases_hp": [int(facade.bases[p].hp) for p in (0, 1)],
                    "towers": [[int(t.tower_id), int(t.player), int(t.x), int(t.y),
                                int(t.tower_type), int(t.hp)] for t in facade.towers],
                    "ants": [[int(a.ant_id), int(a.player), int(a.x), int(a.y), int(a.hp),
                              int(a.max_hp), int(a.kind), int(a.age), int(a.behavior)]
                             for a in facade.ants],
                }
            except Exception as _se:  # noqa: BLE001
                rec["state"] = None
                _log(f"raw state dump 失败: {type(_se).__name__}: {_se}")
        # 2026-10-09（用户建议）：**与 MC 裁判角色完全匹配的标签** —— 裁判用的就是这个函数，
        # 所以用它当训练目标，才是"接在截断 rollout 后面补尾 / 当裁判"的合适价值网；
        # 同时记录 rollout **实际用掉多少回合**（判断 MC_HORIZON 上限卡不卡）。默认关：AZAI_DUMP_QUIET=1。
        if _env("AZAI_DUMP_QUIET", "0") == "1" and facade is not None:
            try:
                from my_ai.az_intent.mc_quiet import quiet_value
                _qv, _qr = quiet_value(facade, int(_AZ["player"]),
                                       max_rounds=int(_env("AZAI_MC_HORIZON", "256")), return_rounds=True)
                rec["quiet_value"] = float(_qv)
                rec["quiet_rounds"] = int(_qr)
            except Exception as _qe:  # noqa: BLE001
                rec["quiet_value"] = None
                rec["quiet_rounds"] = None
                _log(f"quiet 标签计算失败（该条留空）: {type(_qe).__name__}: {_qe}")
    except Exception as _e:  # noqa: BLE001
        _log(f"AZ dump 记录失败（跳过该条）: {type(_e).__name__}: {_e}")
        return
    _AZ["n"] += 1
    _AZ["fh"].write(json.dumps(rec, ensure_ascii=False) + "\n")
    _AZ["fh"].flush()


def _close_az() -> None:
    if _AZ.get("fh") is not None:
        _AZ["fh"].flush()
        _AZ["fh"].close()
        _AZ["fh"] = None
        _log(f"[az-dump] 已写入 {_AZ['n']} 条决策 -> {_AZ.get('path')}")


def _log(msg: str) -> None:
    sys.stderr.write(f"[azai] {msg}\n")
    sys.stderr.flush()


# 统计（模块级：收尾报告要在被 bridge kill 之前打出来）
_ENGINE = {}   # 2026-10-22：模块级持有 mcts/model，供 _report 读取
_OP_LABEL_WARNED = False   # 2026-10-22：op 标签解析失败只提示一次
STATS = {"rounds": 0, "mismatch": 0, "illegal": 0, "dec_max_s": 0.0, "dec_slow": 0, "dec_n": 0,
         "dec_times": []}   # 2026-10-26：每决策墙钟（算 p50/p95；只用于日志，不参与决策）
# 2026-10-28（规则内口径）：**固定阈值**的超时计数。`dec_slow` 的阈值来自 `AZAI_SLOW_DECISION_SEC`
# （默认 20s，可被实验配置改），而"每步 ≤ 10s"是**游戏硬约束**（比赛规则），必须有一个
# 不随环境变量漂移的读数，否则两臂配置一改就不可比。⇒ 这里写死三条线，日志里常驻
# `dec_ge10=… dec_ge20=… dec_ge30=…`（含义 = 该局**决策墙钟** ≥ 该秒数的决策条数）。
_DEC_THRESHOLDS = (10, 20, 30)
# 2026-10-31：`AZAI_CAND_VERBOSE=1` ⇒ 逐回合把**真的发出去的 op** 打一行 `[act] ...`
#   （统一候选 A/B 的"行为差异"读数靠它；本批 ladder 不透传 `--trace-ops`）。默认关 ⇒ 零日志。
_CAND_VERBOSE = [os.environ.get("AZAI_CAND_VERBOSE", "0") == "1"]

# ── 非法操作自检（2026-10-27）──────────────────────────────────────────────────────
# 为什么需要它：`GameStateFacade.apply_operation_list` 返回 **None**
# （`game_state_facade.py:167-170` 把 binding 的 accepted 计数丢掉了），而冷路径本身
# **静默跳过**非法 op ⇒ 本文件原来的 `bad = facade.apply_operation_list(...)` +
# `len(bad) if bad else 0` **恒为 0**（`STATS["illegal"]` 从来没真的数过任何东西，
# 和 bridge 侧 `if il0 or il1:` 是同一个坑）。
# 现在改用 `match_via_sdk.check_ops`（逐条 can_apply_dry + can_apply_cold 交叉验证、**只读**），
# 只记日志/计数，**不改变任何决策**。覆盖的高风险改写路径：
#   decide_final（出口，含强制升级 / mc 显式升级 op(12) / HOLD）
#   after_filters（**过滤闸门**：`AZAI_NO_DOWNGRADE`/`BAN_OPS`/`*_MIN_COINS` 删 op 之后复核整块；
#                  2026-10-28 起它取代了原来那条只记录不拦的 `after_no_downgrade` ——
#                  闸门**自己**用独立的 IllegalStats + `GATE-BLOCK` 日志词，
#                  所以 `illegal_total` 仍然只数"真的发出去了几个非法 op"）
#   shadow_apply（影子状态应用前，两侧都查）
#   oracle_bundle（AZAI_ORACLE_INJECT 注入的候选，见 `_oracle_bundle` 里原有的逐条检查）
ILLEGAL_STATS = MVS.IllegalStats()

# ── 事后过滤的「整块合法性闸门」（2026-10-28；修 docs/illegal_op_strict_mode_20261028.md §5.3）──
# 背景：`AZAI_NO_DOWNGRADE` / `AZAI_BAN_OPS` / `AZAI_LIGHT_MIN_COINS` / `AZAI_BUILD_MIN_COINS`
# 都是**事后**从一块（本来合法的）候选里删 op。删"降级退钱"会让后面的建塔变成"钱不够"：
#   oracle 注入 [(13,3,-1),(11,8,2)]（合法：先退钱再建塔）→ NO_DOWNGRADE 只删 (13,3,-1)
#   ⇒ 剩下 (11,8,2) 要 45 金币而手上只有 3 ⇒ 严格模式（=官方口径）当场判负。
# 实测 2/2 局（seed 8103 r61、8103r r244）。修法：过滤之后**复核剩余整块**，
# 只要有一条非法就**整块丢弃**（本回合 HOLD），**绝不发半块**。
def _gate_log(msg: str) -> None:
    """闸门的日志通道：`check_ops` 的 `ILLEGAL` 行改写为 `GATE-BLOCK`。

    为什么要改词：`_tmp_ladder.py:192` 数的是"含 `ILLEGAL` 的行数" = **真的发出去**的非法 op；
    闸门拦下的 op 没有发出去，混进去会让那份统计虚高（0 非法被数成 N）。
    """
    _log("🛡️ [gate] " + str(msg).replace("ILLEGAL ", "GATE-BLOCK "))


def _block_legal_gate(facade, pl: int, ops: list) -> list:
    """**整块**只读合法性复核（逐条 `can_apply`，带 `pending=` 前缀语义）。空返回 = 整块合法。

    * 检查器与 `decide_final` / `shadow_apply` 自检**同一个**（`MVS.check_ops`：dry + clone 冷路径）；
    * 用**独立**的 `IllegalStats`：`ILLEGAL_STATS` 的口径是"真发出去的非法 op"，被拦下的不算；
    * 闸门自身异常 / 校验器对某一条失效（`CHECK-ERROR`）⇒ 返回"非法"（**fail-closed**）：
      宁可 HOLD 也不能因为检查器坏了把非法 op 发出去（严格模式=官方口径，发出去就是判负；HOLD 一定安全）。
      这与 `decide_final` 自检的 fail-open 相反，因为那里只是记录、这里决定发不发。
    """
    seen: list[str] = []

    def _spy(msg: str) -> None:
        seen.append(str(msg))
        _gate_log(msg)

    try:
        st = MVS.IllegalStats()
        bad = list(MVS.check_ops(facade, int(pl), list(ops), st,
                                 seed=int(STATS.get("seed", 0)),
                                 round_index=int(STATS.get("rounds", 0)),
                                 tag="after_filters", cold=False, log=_spy))
    except Exception as exc:  # noqa: BLE001
        _log(f"🛡️ [gate] ⚠️ 闸门自身异常 ⇒ 按非法处理（丢弃整块，保守）："
             f"{type(exc).__name__}: {exc}")
        return [{"idx": -1, "op": None, "reason": f"gate_error({type(exc).__name__})"}]
    if not bad and any("CHECK-ERROR" in m for m in seen):
        # 校验器没能给出判定（不是"判成合法"）⇒ fail-closed
        _log("🛡️ [gate] ⚠️ 有 op 的合法性校验器失效（CHECK-ERROR）⇒ 按非法处理（丢弃整块，保守）")
        return [{"idx": -1, "op": None, "reason": "checker_unavailable"}]
    return bad

# ── 安慰剂候选（2026-10-28；`AZAI_ORACLE_PLACEBO`；见 docs/placebo_inject_ab_20261028.md）──────
# 目的：把"注入"的**信息量**抽掉、只留"多一个候选"，用来分离 A/B 里 +31pp 的两种解释：
#   ① **知识**（亚军那一手含我们的先验/评估器给不出的信息）；
#   ② **候选数 / 预算效应**（与"这一手好不好"无关，纯粹因为候选集里多了一项）。
# 做法：走**同一条**通道 `mcts.set_extra_bundles([bundle])`（同样 1 个额外候选、同样进前缀/预算、
#   同样的 new/dup/hit 统计与每决策一条日志），但候选 = 本地随机抽的一个**合法 op**，**不查询 oracle**。
# 默认（`AZAI_ORACLE_PLACEBO` 不设）⇒ 下面这些函数**一个都不会被调用**、常量一个都不 import。
_PLACEBO_OP_ORDER = ("build", "downgrade", "lightning", "upgrade")   # 固定顺序 ⇒ 可复现
_PLACEBO_LIGHT_TRIES = 8     # 闪电的随机格最多试几个（闪电合法性几乎与格子无关，通常第 1 个就合法）
_PLACEBO_IMPORTS: dict = {}


def _placebo_imports() -> dict:
    """懒加载 placebo 需要的 SDK 符号（只有 placebo 打开时才会执行 ⇒ 默认零成本）。"""
    if not _PLACEBO_IMPORTS:
        from SDK.backend.model import Operation as _Op
        from SDK.utils.constants import (HIGHLAND_CELLS as _HC, OperationType as _OT,
                                         TOWER_UPGRADE_TREE as _TREE)
        _PLACEBO_IMPORTS.update({"Op": _Op, "OT": _OT, "HC": _HC, "TREE": _TREE})
    return _PLACEBO_IMPORTS


def _placebo_rng(dec_n: int, flat_seed: int):
    """`seed = int(sha256("placebo|<FLAT_SEED>|<dec_n>").digest()[:8])`。

    为什么用哈希而不是线性式：rollout 侧用的是
    `(rollout_seed*1000003 + decision_idx)*100003 + index*101 + k`（`flat_rollout_greedy.py:540`），
    线性式在参数相近时会落进同一族种子；哈希把两条随机流**彻底隔开**
    （= 任务书要求的"与 rollout 种子无关的独立流"），同时"同一 (FLAT_SEED, 决策序号) ⇒ 同一个 Random"
    ⇒ 完全可复现（不依赖时间/哈希随机化）。
    """
    h = hashlib.sha256(f"placebo|{int(flat_seed)}|{int(dec_n)}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _placebo_ok(facade, player: int, op: tuple) -> bool:
    """**只读**合法性判定（引擎 `can_apply_dry`，与 `_oracle_bundle` 检查亚军候选**同一口径**）。

    异常 ⇒ 按非法处理（**fail-closed**）：宁可本回合退回 HOLD，也绝不把一个没验过的 op 塞进候选
    （严格模式 = 官方口径，发出去非法就是当场判负）。
    """
    _m = _placebo_imports()
    try:
        return bool(facade.can_apply_operation(
            int(player), _m["Op"](_m["OT"](int(op[0])), int(op[1]), int(op[2]))))
    except Exception:  # noqa: BLE001
        return False


def _placebo_bundle(facade, player: int, dec_n: int, flat_seed: int,
                    hold_p: float = 0.0) -> tuple:
    """生成一个**合法**的单 op 束（或 HOLD）：返回 `(bundle, kind)`。

    * bundle = `((op,),)` / `()`；kind ∈ `hold / nolegal / build / upgrade / downgrade / lightning`；
    * 规则（预注册，见 docs/placebo_inject_ab_20261028.md §3.2）：
        1. `rng.random() < hold_p` ⇒ HOLD（P 臂 hold_p=0 ⇒ **永不 HOLD**；P2 臂 = 0.78，对齐亚军 78% HOLD）；
        2. 否则**先均匀抽 op 类**（只抽"有合法目标"的那几类），再在该类的合法目标里均匀抽一个：
             11 建塔（我方高地的空格）/ 12 升级（我方塔 id + 下一级）/ 13 降级（我方塔 id）/
             21 闪电（随机格）；
        3. 合法性 = `_placebo_ok`（引擎 dry-run，只读）；
        4. 一类合法目标都没有 ⇒ HOLD（**fail-safe**：绝不发非法 op）。
    * 单 op ⇒ 不存在"半块非法"问题（P1 那个 bug 的前提是"多 op 块被事后删 op"）；
    * ⚠️ **先抽类再抽目标**（而不是"在全池上均匀"）：闪电对**任何**格都合法，全池均匀会让
      后期有钱时 361 个闪电格淹没别的类 ⇒ "随机"退化成"闪电发生器"。先抽类与策略头
      "先类后位"的结构一致，也让 4 类各占 1/4（报告里给出实测直方图）。
    """
    rng = _placebo_rng(int(dec_n), int(flat_seed))
    if rng.random() < float(hold_p):
        return (), "hold"
    _m = _placebo_imports()
    pl = int(player)
    towers = list(facade.towers_of(pl))
    occupied = {(int(t.x), int(t.y)) for t in facade.towers}     # 一次集合化，别对每个高地格线性找塔
    groups: dict = {
        "build": [(11, int(x), int(y)) for (x, y) in _m["HC"][pl]
                  if (int(x), int(y)) not in occupied],
        "upgrade": [(12, int(t.tower_id), int(tgt))
                    for t in towers
                    for tgt in (_m["TREE"].get(int(t.tower_type), ()) or ())],
        "downgrade": [(13, int(t.tower_id), -1) for t in towers],
    }
    legal: dict = {}
    for kind in _PLACEBO_OP_ORDER:          # 固定顺序遍历 ⇒ 同一 rng 状态下的结果可复现
        if kind == "lightning":
            for _ in range(_PLACEBO_LIGHT_TRIES):
                op = (21, int(rng.randrange(19)), int(rng.randrange(19)))
                if _placebo_ok(facade, pl, op):
                    legal[kind] = op
                    break
            continue
        cand = list(groups.get(kind) or ())
        if not cand:
            continue
        rng.shuffle(cand)
        for op in cand:
            if _placebo_ok(facade, pl, op):
                legal[kind] = op
                break
    if not legal:
        return (), "nolegal"
    kind = tuple(legal)[rng.randrange(len(legal))]    # dict 顺序 = 上面的固定顺序 ⇒ 可复现
    return (legal[kind],), kind

# ── 亚军候选注入（2026-10-26；**开关式，默认关**；见 docs/oracle_inject_design.md）─────────
# `AZAI_ORACLE_INJECT` 为空 ⇒ 每条分支第一行就返回：**不 import 查询器、不起子进程、
# 不加任何耗时**（零影响由 `code/test_match/probe_oracle_inject_zero_impact.py` 的哈希比对验证）。
# 启用 ⇒ 每局按 (seed, player) 建一个常驻亚军会话，**每回合**喂「当前局面 + 对手操作块」
# （会话有状态：跳一回合 = 读端永久错位），把它的 ops 变成一个额外候选 bundle 注入搜索。
_ORACLE: dict = {
    "off": True,          # True = 不启用（默认）
    "obj": None,          # RunnerUpOracle（作废后置 None）
    "seed": 0, "player": 0, "exe": "", "timeout": 2.5,
    "asks": 0,            # 实际 ask 的回合数
    "ok": 0,              # 成功拿到回包
    "checked": 0,         # 2026-10-27：注入通道**逐条 can_apply 复核过**的 op 数（都应合法）
    "inject": 0,          # 交给搜索的次数（含 HOLD / 去重后为空的）
    "new": 0,             # 去重后**真正追加为候选**的次数
    "dup": 0,             # 与已有候选重复而被去重掉的次数
    # 2026-10-28（每决策一条日志）：把 dup 按"亚军这一手是不是 HOLD"拆开 —— 这一刀决定
    # "注入的增量信息到底有多少"：dup_hold = 亚军也只是空过、而我们候选里本来就有 HOLD。
    "dup_hold": 0,        # dup 且 亚军动作 = ()（HOLD）
    "dup_nonhold": 0,     # dup 且 亚军动作非空
    # 2026-10-28：`hit` 的口径扩成**动作一致率**（chosen == 亚军这一手），
    # 老口径（"注入的额外候选被搜索选中"）保留为 `hit_sel`（数值与改动前逐位相同）。
    "hit": 0,             # 动作一致：我们最终发出去的 == 亚军这一手（含 dup 情形）
    "hit_new": 0,         # 动作一致 且 该候选是新增的（new=1）
    "hit_dup": 0,         # 动作一致 且 该候选与已有候选重复（dup=1）
    "hit_sel": 0,         # 老口径：额外候选被选中（chosen_index >= n_net）
    "illegal": 0,         # 因非法操作丢掉整条候选的次数
    "timeouts": 0, "dead": 0,
    "err": "",
    # ── 2026-10-29：**会话自动重建 + 回放追赶**（E4 的前置条件；默认开，`AZAI_ORACLE_REBUILD_MAX=0` 关）──
    #   背景（`docs/e4_only_extra_20261029.md` §1）：冠军当 oracle 会在第 ~239-310 回合**自己 abort**
    #   （`apply_operations.called_after_terminal`）⇒ 旧的"死一次就永久停用"让 E2 的注入覆盖只有 54.4%。
    #   现在：死 ⇒ 起新进程 + 把本局 0..当前回合的历史回放进去（`RunnerUpOracle.replay`）+ 重试本回合。
    "mod": None,          # `runnerup_oracle` 模块（重建时要 `RunnerUpOracle(...)`）
    "rebuild_max": 6,     # 每局最多重建几次（含失败的尝试）⇒ 防无限重启
    "rebuilds": 0,        # 成功的重建次数
    "rebuild_fail": 0,    # 失败的重建尝试次数
    "replay_rounds": 0,   # 累计回放的回合数
    "replay_s_sum": 0.0,
    "replay_s_max": 0.0,  # 单次重建的最大墙钟（验收判据：<60s）
    "hist": [],           # 上一次会话的 history 备份（重建失败时保住，供下一次重试）
    # 主动重启（`AZAI_ORACLE_RESTART_EVERY=N`，默认 0 = 关）：不等它崩，每 N 回合换一个新进程
    #   （新进程从 round 0 起、但喂的是**当前局面** ⇒ 读端不错位；它自己的影子局面则从头攒）。
    #   用途：冠军的影子局面**必然**会与真对局分叉（见 `docs/e4_only_extra_20261029.md` §1.2），
    #   靠它自己的影子根本撑不到 512 回合；主动重启是唯一能让会话"活着跑完整局"的办法。
    "restart_every": 0,   # 0 = 关（默认行为与改动前**逐位相同**）
    "restarts": 0,        # 主动重启次数
    "relabel_off": 0,     # 当前会话的"回合号偏移" = 该会话建立时的真回合号（回合号重标用）
    "hold": 0,            # 它给 `()` 的次数（= 空过）—— 判"会话是不是变成 HOLD 机器"的关键读数
    # 2026-10-29（父任务）：`AZAI_ORACLE_SKIP_HOLD=1`（默认 0）⇒ oracle 给 HOLD 时**不注入**、
    #   只计 `skipped_hold`。用途：给"**真外部动作**"定价——HOLD 的功劳已经由 `AZAI_ALWAYS_HOLD`
    #   单独控制住（两臂都开），所以两臂之差 = 非 HOLD 外部动作的净价值。
    "skip_hold": 0, "skipped_hold": 0,
    "ms": [],             # 每次 ask 的墙钟（毫秒）
    "pending_opp": None,  # P1 用：本回合的对手操作块原文
    "last_bundle": None,  # 上一回合注入的 bundle（命中日志用）
    "pending": False,     # 上一回合是否交过候选（用于统计 new/dup）
    # 2026-10-28：本回合待打印的明细（`_oracle_step` 填 action，`_oracle_after_search` 填 new/dup，
    # `_oracle_log_round` 在决定出口把它打出来并清空）
    "round_log": None,
    "round_new": None, "round_dup": None,
    # 2026-10-28：**安慰剂臂**（`AZAI_ORACLE_PLACEBO=1`）—— 同一条注入通道，候选换成
    # "本地随机抽的一个合法 op"（**不查 oracle**）。目的是分离"知识"与"多一个候选/预算效应"，
    # 见 docs/placebo_inject_ab_20261028.md。默认 False ⇒ 下面整条 placebo 路径一次都不进。
    "placebo": False,
    "placebo_hold_p": 0.0,   # 该决策走 HOLD 的概率（P 臂 0；P2 臂 0.78 = 对齐亚军 78% HOLD）
    "placebo_ms": [],        # 本地生成的墙钟（毫秒）—— 与 `ms`（oracle 查询）**分开记**，读数别混
    "placebo_kinds": {},     # 生成结果直方图：hold / nolegal / build / upgrade / downgrade / lightning
}


# ── 升级候选进束（2026-10-28；`AZAI_UPG_BUNDLES=N`，**默认 0 = 完全关闭**）──────────────
#   背景与设计见 `docs/upgrade_bundles_20261028.md`（上游只读诊断 `docs/why_no_upgrade_20261028.md`）。
#   一句话：现役配置的 48 束被钉在同一类上（`bundle_mcts.py:870`），该类 100% ∈ {类 0, 类 16}
#   ⇒ 升级**进不了候选**（0/7850 决策点）；本开关把最多 N 条单 op 升级走**已有的**额外候选通道
#   `set_extra_bundles` 塞进去，让它们与另外 48 条**同台**被 flat rollout 评估器打分。
#   N=0（默认）⇒ 三个钩子函数的**第一行**就 return：不枚举、不 dry-run、不碰通道、不打日志。
_UPG: dict = {
    "n": 0,            # 每回合最多塞几条（0 = 关；`AZAI_UPG_BUNDLES`）
    "decided": 0,      # 决策点数（"有升级候选的决策点占比"的**分母**）
    "rounds": 0,       # 真的塞进 ≥1 条升级候选的决策点数（分子）
    "offers": 0,       # 累计塞进去的条数（选中率的**分母**）
    "added": 0,        # `mcts.last_extra` 口径：去重后真正进束的条数（累计）
    "sel": 0,          # 被评估器选中的**条数**（每决策 ≤1 ⇒ 也是"命中决策数"）
    "sent": 0,         # 最终真的发出去的 `op=12` 数（执行层过滤之后）
    "no_legal": 0,     # 开了开关、但本回合一条都凑不出来的次数
    "gate_drop": 0,    # 被 `_block_legal_gate` 整块拦下的条数（应当恒 0）
    "err": "",
    "targets": {},     # 塞进去的目标塔型直方图
    "sel_targets": {}, # **被选中**的目标塔型直方图（评估器偏好哪个升级）
    "kinds": {},       # 每回合塞了几条 (1..N) 的直方图
    "budget": {},      # "n_cand/n_rollouts" 直方图 —— 预算稀释的**实测**
    "mine": None,      # 本回合塞进去的 bundle 元组（搜索后判命中用）
    "set_prev": (),    # 上一回合我写进通道的那份（下回合先摘掉，否则会逐回合累加）
}


# ── E4：候选集只保留额外候选 + 随机挑一个（2026-10-28；两个开关**默认关**）──────────────
#   目的（`docs/us_vs_champion_20261028.md` §5）：
#     E4  = 候选集**只有**「冠军这一手 + 亚军这一手」，由**同一个** flat 评估器裁决
#           ⇒ 直接量化"候选/先验"这一侧的上限（网络那 K 条被类轴 pinned 的候选不参与）；
#     E4R = 同一候选集，但**不走评估器**，随机挑一个（固定种子可复现）
#           ⇒ 对照臂：分离"动作来源的质量"与"评估器把两个里挑对的贡献"。
#   两个开关都是 `0`/不设 ⇒ 下面的函数第一行就 return，行为与改动前**逐位相同**
#   （证据：`_only_extra_zero_impact.sh` 的 `RESULT` 逐字段比较）。
_ONLY: dict = {
    "on": False,        # `AZAI_ONLY_EXTRA=1` ⇒ `mcts.set_only_extra(True)`
    "random": False,    # `AZAI_ONLY_EXTRA_RANDOM=1` ⇒ 选层不看分数，随机挑一个
    "seed": 0,          # 随机流种子（默认 = `AZAI_FLAT_SEED`；可用 `AZAI_ONLY_EXTRA_SEED` 覆盖）
    "decided": 0,       # 做过随机挑的决策数
    "pick": {},         # 被挑中的候选直方图（`repr(bundle) -> 次数`）
    "ncand": {},        # 每决策候选数直方图
    "err": "",
}

# ── 双 oracle：A/B 两个常驻查询会话（2026-10-28；`AZAI_ORACLE_INJECT_B` 非空才启用）────────
#   A 会话 = 原来的 `AZAI_ORACLE_INJECT`（走 `_ORACLE`，那条路径**一行都没改**）；
#   B 会话 = 这里新增的第二个进程（E4 用冠军做 A、亚军做 B）。
#   ⚠️ 两个会话都只是 `RunnerUpOracle` —— 它不绑定亚军，只是"按官方协议喂局面、收操作"的外壳。
#   ⚠️ B 的 stderr 必须**显式**给不同路径：`RunnerUpOracle` 的默认名只含 (seed, player, pid)，
#      两个会话会写同一个文件（历史踩过，`docs/oracle_inject_design.md` §4.3）。
_ORACLE_B: dict = {
    "off": True,          # True = 不启用（默认）
    "obj": None, "exe": "", "timeout": 20.0, "seed": 0, "player": 0,
    "asks": 0, "ok": 0, "inject": 0, "checked": 0, "illegal": 0,
    "timeouts": 0, "dead": 0, "hit": 0, "ms": [], "err": "",
    "pending_opp": None, "last_bundle": None, "round_log": None,
    # 2026-10-29：与 A 会话同一套"重建 + 回放"机制（字段含义见上面 `_ORACLE` 的注释）
    "mod": None, "rebuild_max": 6, "rebuilds": 0, "rebuild_fail": 0,
    "replay_rounds": 0, "replay_s_sum": 0.0, "replay_s_max": 0.0, "hist": [],
    "restart_every": 0, "restarts": 0, "relabel_off": 0, "hold": 0,
}


# ── 双 oracle 逐回合记录（2026-10-28；用户要求；**默认零日志**）──────────────────────────
#   一句话：E4/E4R（= 冠军 + 亚军两个会话）每回合记一行 `[xor]`：
#     · **冠亚军一致率** `agree`   —— 高 ⇒ 好动作集中（最优解唯一）；低 ⇒ 存在大量近似等价的好动作；
#     · **都 HOLD** `both_hold`    —— 亚军 77.9% 是 HOLD，冠军也常在攒钱 ⇒ 预期很高；
#     · **新信息占比** `champ_new` / `ru_new` —— 该动作**不在我们自己的网络候选集**里（= 真·增量）；
#     · **我们更接近谁** `chosen_eq_champ` / `chosen_eq_ru` / `chosen_eq_neither`（含"两者不一致时"）。
#   E2 只有一个会话（A=冠军）⇒ 同一行里 `ru=NA agree=NA dup_r=NA hit_r=NA`。
#   默认（两个会话都没开）⇒ `_xor_log_round()` 第一行就 return，一个字节都不写。
_XOR: dict = {
    "rounds": 0, "agree": 0, "both_hold": 0, "disagree": 0,
    "have_a": 0, "have_b": 0, "champ_hold": 0, "ru_hold": 0,
    "champ_new": 0, "ru_new": 0, "champ_ops": 0, "ru_ops": 0,
    "chosen_eq_champ": 0, "chosen_eq_ru": 0, "chosen_eq_neither": 0,
    "eq_champ_when_disagree": 0,
}

# ── 候选分数差（2026-10-29；用户要求的「检验力诊断」；**默认零日志**）─────────────────────
#   动机：E4（评估器从 2 个外部候选里挑）vs E4-rand（同样 2 个候选里随机挑）**只在"两个候选分数
#   不同"的那些决策上才可能不同**；`agree=1`（冠亚军同一手）或 `both_hold` 的决策两臂**按构造
#   完全一样**。不把这类决策剔掉，"E4≈E4-rand" 会被稀释成"评估器无用"的错误结论
#   ⇒ 逐决策记下 `|Δscore|`，事后才能按分差分层比较、并报出"有效样本量"。
_DS: dict = {"rounds": 0, "n2": 0, "n1": 0, "pick_hi": 0, "pick_lo": 0,
             "pick_net": 0, "pick_na": 0, "ds": [], "dvn": [], "err": ""}

# ── HOLD 恒为候选（2026-10-29；`AZAI_ALWAYS_HOLD=1`，**默认 0 = 关**）──────────────────
#   为什么（父任务的判断，2026-10-29 13:4x）：冠军 oracle 实际上恒给 `()`（`force_pass_` 锁死），
#   而亚军注入臂 8117 次注入里 **84.5% 也是 HOLD**、`hit_new` 里 70.5% 是 HOLD
#   ⇒ "注入有效"的 +37pp 很可能只是"**把 HOLD 补进候选集**"这一个效应
#   （我们自己的 48 条网络候选**并不总含 HOLD**：E2 实测 23.9%~31.7% 的回合不含）。
#   本开关做一个**不需要任何 oracle** 的消融：把空 bundle `()` 放进额外候选通道
#   （去重由 `flat_rollout_greedy._collect_extras()` 负责 ⇒ "已存在就不加"）。
#   两条臂：A = `AZAI_ALWAYS_HOLD=1`，B = 不设（现役）⇒ 差值 = "候选集缺 HOLD" 的效应。
_AHOLD: dict = {"on": False, "dec": 0, "hold": 0, "added": 0, "chose_after_add": 0,
                "present_kept": 0, "present_moved": 0, "last_added": 0, "err": ""}


def _xor_summary_text() -> str:
    """`[xor-sum]` 一行（字段含义见上面 `_XOR` 的注释；百分比的分母都是 `rounds`）。"""
    x = _XOR
    n = max(1, int(x["rounds"]))

    def _pc(k: str) -> str:
        return f"{int(x[k])}({100.0 * int(x[k]) / n:.1f}%)"

    txt = (f"rounds={int(x['rounds'])} agree={_pc('agree')} both_hold={_pc('both_hold')} "
           f"disagree={int(x['disagree'])} champ_new={int(x['champ_new'])} "
           f"ru_new={int(x['ru_new'])} chosen_eq_champ={int(x['chosen_eq_champ'])} "
           f"chosen_eq_ru={int(x['chosen_eq_ru'])} "
           f"chosen_eq_neither={int(x['chosen_eq_neither'])}")
    if int(x["have_b"]):
        txt += (f" eq_champ_when_disagree={int(x['eq_champ_when_disagree'])}/{int(x['disagree'])}"
                f" champ_hold={_pc('champ_hold')} ru_hold={_pc('ru_hold')}"
                f" champ_ops={int(x['champ_ops'])} ru_ops={int(x['ru_ops'])}")
    return txt


def _dscore_summary_text() -> str:
    """`[dscore-sum]` 一行：分差分布 + 三个"有效样本量"分母（口径见 `_DS` 的注释）。

    `d_*` = 两个额外候选在本评估器下的分数差 `|Δscore|` 的分位（只在 `n_extra≥2` 时才有值）；
    `dvn_*` = 最佳额外候选 − 最佳网络候选（E2 单会话时这就是"注入 vs 我们自己候选"的差）；
    `pick_hi/pick_lo` 的比例是**正向对照**：E4 应该 ≈ 全 hi（评估器按分数挑），
    E4-rand 应该 ≈ 一半（随机挑）——两边都偏离就说明臂没按预期跑。
    """
    d = _DS
    ds = sorted(float(v) for v in d["ds"])
    dv = sorted(float(v) for v in d["dvn"])

    def _q(xs: list, q: float) -> str:
        if not xs:
            return "NA"
        i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
        return f"{xs[i]:+.4f}"

    txt = (f"rounds={int(d['rounds'])} two_cand={int(d['n2'])} one_cand={int(d['n1'])} "
           f"pick_hi={int(d['pick_hi'])} pick_lo={int(d['pick_lo'])} "
           f"pick_net={int(d['pick_net'])} pick_na={int(d['pick_na'])} "
           f"d_p25={_q(ds, 0.25)} d_med={_q(ds, 0.50)} d_p75={_q(ds, 0.75)} d_p90={_q(ds, 0.90)} "
           f"dvn_p25={_q(dv, 0.25)} dvn_med={_q(dv, 0.50)} dvn_p75={_q(dv, 0.75)}")
    if d.get("err"):
        txt += f" err={d['err']}"
    return txt


def _ahold_stats_text() -> str:
    """`[ahold]` 一行（`AZAI_ALWAYS_HOLD=1` 的消融臂主读数；默认为 0 ⇒ 不打）。

    字段（父任务要的六项指标里的 2/4 两项）：
      * `dec` / `hold` —— 决策数 / 其中**最终发出空 op** 的回合数（臂 A 的 HOLD 率）；
      * `added` —— "候选集里本来没有 HOLD、被本开关补进去"的回合数
                   （≈"类仲裁把动/不动钉在非 HOLD 类"的那部分，父任务口径 ~22%）；
      * `chose_after_add` —— 上面这些回合里，强评估器**反而选了 HOLD** 的次数
                   ⇒ **"被架构剥夺的决策"被拿回来的比例**（★核心读数）；
      * `present_kept` / `present_moved` —— HOLD **本来就在**候选里的回合中，评估器选 HOLD / 选动。
    """
    a = _AHOLD
    n = max(1, int(a["dec"]))
    add = max(1, int(a["added"]))
    pres = int(a["present_kept"]) + int(a["present_moved"])
    txt = (f"dec={int(a['dec'])} hold={int(a['hold'])}({100.0 * int(a['hold']) / n:.1f}%) "
           f"added={int(a['added'])}({100.0 * int(a['added']) / n:.1f}%) "
           f"chose_after_add={int(a['chose_after_add'])}"
           f"({100.0 * int(a['chose_after_add']) / add:.1f}% of added) "
           f"present={pres} kept={int(a['present_kept'])}"
           f"({100.0 * int(a['present_kept']) / max(1, pres):.1f}%) "
           f"moved={int(a['present_moved'])}")
    if a.get("err"):
        txt += f" err={a['err']}"
    return txt


def _oracle_b_stats_text() -> str:
    """B 会话的统计一行（默认零日志；只有 `AZAI_ORACLE_INJECT_B` 才用得上）。"""
    st = _ORACLE_B
    txt = (f"asks={st['asks']} ok={st['ok']} checked={st['checked']} inject={st['inject']} "
           f"hit={st['hit']} illegal={st['illegal']} timeout={st['timeouts']} dead={st['dead']}")
    if int(st.get("rebuilds", 0)) or int(st.get("rebuild_fail", 0)) or int(st.get("restarts", 0)) \
            or int(st.get("hold", 0)):
        txt += (f" rebuild={int(st.get('rebuilds', 0))} rebuild_fail={int(st.get('rebuild_fail', 0))}"
                f" restart={int(st.get('restarts', 0))} replay_rounds={int(st.get('replay_rounds', 0))}"
                f" replay_s_max={float(st.get('replay_s_max', 0.0)):.1f}"
                f" hold={int(st.get('hold', 0))}")
    _ms = st.get("ms") or []
    if _ms:
        txt += (f" avg_ms={sum(_ms) / len(_ms):.0f} p50_ms={_pct(_ms, 0.50):.0f} "
                f"p95_ms={_pct(_ms, 0.95):.0f} max_ms={max(_ms):.0f}")
    if st.get("err"):
        txt += f" err={st['err']}"
    return txt


def _upg_stats_text() -> str:
    """`[upg] 统计:` 一行（对局结束前打；bridge 会硬杀 ⇒ 每决策那些日志才是主读数）。"""
    st = _UPG
    n_off = int(st.get("offers", 0))
    n_sel = int(st.get("sel", 0))
    n_dec = int(st.get("decided", 0))
    txt = (f"n_max={st.get('n', 0)} decisions={n_dec} rounds_with_cand={st.get('rounds', 0)}"
           f"({(100.0 * st.get('rounds', 0) / max(1, n_dec)):.1f}% of decisions)"
           f" offered={n_off} sel={n_sel}({100.0 * n_sel / max(1, n_off):.1f}% of offered)"
           f" added={st.get('added', 0)} sent_op12={st.get('sent', 0)}"
           f" no_legal={st.get('no_legal', 0)} gate_drop={st.get('gate_drop', 0)}")
    _k = st.get("kinds") or {}
    if _k:
        txt += " kinds={" + ",".join(f"{k}:{int(_k[k])}" for k in sorted(_k)) + "}"
    _t = st.get("targets") or {}
    if _t:
        txt += " targets={" + ",".join(f"{k}:{int(_t[k])}" for k in sorted(_t)) + "}"
    _st = st.get("sel_targets") or {}
    if _st:
        txt += " sel_targets={" + ",".join(f"{k}:{int(_st[k])}" for k in sorted(_st)) + "}"
    _b = st.get("budget") or {}
    if _b:
        txt += " cand/rollouts={" + ",".join(f"{k}:{int(_b[k])}" for k in sorted(_b)) + "}"
    if st.get("err"):
        txt += f" err={st['err']}"
    return txt


def _inj_tag() -> str:
    """注入日志的标签：`oracle`（真注入）/ `placebo`（安慰剂）。

    默认（`AZAI_ORACLE_PLACEBO` 不设）返回 `oracle` ⇒ A 臂/基准臂的日志**逐字不变**
    （`docs/oracle_inject_ab_20261028.md` 的读数口径可以继续跨版本比较）。
    """
    return "placebo" if _ORACLE.get("placebo") else "oracle"


def _fmt_act(act) -> str:
    """动作的日志格式：`()` = HOLD；`(t,a0,a1)` = 单个 op；`((t,a0,a1),(t,a0,a1))` = 多 op。

    ⚠️ 单 op 是"扁平三元组"、多 op 是"三元组的元组" —— 这样它与既有日志里
    `[(13,3,-1),(11,8,2)]` 这种 Python repr 的读法一致，同时 HOLD 仍是 `()`。
    """
    t = tuple(tuple(int(x) for x in o) for o in (act or ()))
    if not t:
        return "()"
    if len(t) == 1:
        return f"({t[0][0]},{t[0][1]},{t[0][2]})"
    return "(" + ",".join(f"({a},{b},{c})" for a, b, c in t) + ")"


def _pct(vals: list, q: float) -> float:
    """最近的百分位（样本少时退化到端点，够看）。"""
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    k = int(round(q * (len(s) - 1)))
    return float(s[max(0, min(len(s) - 1, k))])


def _dec_hist_text() -> str:
    """固定阈值超时计数（见 `_DEC_THRESHOLDS`）+ 占比。只读 `STATS`，不参与决策。"""
    _n = max(1, int(STATS.get("dec_n", 0) or 0))
    parts = []
    for _th in _DEC_THRESHOLDS:
        _c = int(STATS.get(f"dec_ge{_th}", 0) or 0)
        parts.append(f"dec_ge{_th}={_c}({100.0 * _c / _n:.2f}%)")
    return " ".join(parts)


def _oracle_stats_text() -> str:
    """注入统计的一行文本（本功能的**主读数**）。字段含义见 docs/oracle_inject_design.md §3.5。

    2026-10-28 起的口径（**三段分母，别混着看**）：
      * `hit` = **动作一致率**的分子（我们最终发出去的 op 序列 == 亚军这一手），含 dup 情形；
      * `hit_new` / `hit_dup` = 上面那个分子按"该候选是新增 / 与已有候选重复"拆开；
      * `hit_sel` = 老口径（注入的**额外候选被搜索选中**），与改动前逐位相同、可跨版本比较；
      * `hit_rate` = hit/new、`hit_rate_asks` = hit/asks、`hit_sel_rate` = hit_sel/new。
    ⚠️ **`hit_rate`(hit/new) 在新口径下不是比率、可以 >100%**：`hit` 含 `hit_dup`（亚军说"空过"、
       我们也空过 —— 这类"一致"不带来任何新动作），而分母 `new` 只数**新增候选**。
       实测一局 40 决策：`new=7 hit=25` ⇒ `hit/new=357%`。
       真正能当比率读的是：`hit/asks`（总体一致率）、`hit_new/new`、`hit_dup/dup`。
       保留 `hit_rate=…%(hit/new)` 只是为了与旧日志的字段名对齐，**分母已写在括号里**。
    """
    st = _ORACLE
    txt = (f"asks={st['asks']} ok={st['ok']} checked={st['checked']} inject={st['inject']} "
           f"new={st['new']} "
           f"dup={st['dup']} dup_hold={st['dup_hold']} dup_nonhold={st['dup_nonhold']} "
           f"hit={st['hit']} hit_new={st['hit_new']} hit_dup={st['hit_dup']} "
           f"hit_sel={st['hit_sel']} illegal={st['illegal']} "
           f"timeout={st['timeouts']} dead={st['dead']}")
    # 2026-10-29：重建/重启/空过 三个读数（默认全 0 ⇒ 老批次那一行**逐字不变**）
    if int(st.get("rebuilds", 0)) or int(st.get("rebuild_fail", 0)) or int(st.get("restarts", 0)) \
            or int(st.get("hold", 0)) or int(st.get("skipped_hold", 0)):
        txt += (f" rebuild={int(st.get('rebuilds', 0))} rebuild_fail={int(st.get('rebuild_fail', 0))}"
                f" restart={int(st.get('restarts', 0))} replay_rounds={int(st.get('replay_rounds', 0))}"
                f" replay_s_max={float(st.get('replay_s_max', 0.0)):.1f}"
                f" hold={int(st.get('hold', 0))}")
    if int(st.get("skipped_hold", 0)):
        # 2026-10-29：`AZAI_ORACLE_SKIP_HOLD=1` ⇒ 它给 HOLD 时**不注入**，只计数。
        #   `injected_nonhold` = 真注入出去的（= 非 HOLD 动作），是这一臂的"剂量"。
        txt += (f" skipped_hold={int(st['skipped_hold'])}"
                f" injected_nonhold={int(st.get('inject', 0))}")
    if st["ms"]:
        txt += (f" avg_ms={sum(st['ms']) / len(st['ms']):.0f} p50_ms={_pct(st['ms'], 0.50):.0f} "
                f"p95_ms={_pct(st['ms'], 0.95):.0f} max_ms={max(st['ms']):.0f}")
    txt += (f" hit_rate={(100.0 * st['hit'] / max(1, st['new'])):.1f}%(hit/new) "
            f"hit_rate_asks={(100.0 * st['hit'] / max(1, st['asks'])):.1f}%(hit/asks) "
            f"hit_new_rate={(100.0 * st['hit_new'] / max(1, st['new'])):.1f}%(hit_new/new) "
            f"hit_dup_rate={(100.0 * st['hit_dup'] / max(1, st['dup'])):.1f}%(hit_dup/dup) "
            f"hit_sel_rate={(100.0 * st['hit_sel'] / max(1, st['new'])):.1f}%(hit_sel/new)")
    if st.get("placebo"):
        # 安慰剂臂专有：HOLD 概率、本地生成耗时（**没有** oracle 查询 ⇒ 这个量级才是"注入成本"）
        # 与生成结果直方图（hold/nolegal/四类 op 各多少）。
        txt += f" hold_p={float(st.get('placebo_hold_p', 0.0)):g}"
        _gm = st.get("placebo_ms") or []
        if _gm:
            txt += f" gen_ms_avg={sum(_gm) / len(_gm):.2f} gen_ms_max={max(_gm):.2f}"
        _k = st.get("placebo_kinds") or {}
        if _k:
            txt += " kinds={" + ",".join(f"{k}:{int(_k[k])}" for k in sorted(_k)) + "}"
    if st["err"]:
        txt += f" err={st['err']}"
    return txt


# ── 2026-10-29：oracle 会话的「重建 + 回放追赶」与「主动重启」（E4 的前置条件）──────────
#   为什么要有这一层（完整证据见 `docs/e4_only_extra_20261029.md` §1）：
#     * 冠军当 oracle 会在第 ~239-310 回合 abort（`apply_operations.called_after_terminal`），
#       旧代码"死一次就永久停用" ⇒ E2 的注入覆盖只有 54.4%；
#     * 更糟的是：它的影子局面**从第 1 个我们不执行的决策起就分叉**（`force_pass_` 一置上就整局 HOLD），
#       实测 E2 16 局 **4177/4180 = 99.93% 的回合它给的是 `()`**。
#   ⇒ 重建只能救"会话活着"（覆盖率），救不了"它给出的动作是否是真动作"（内容）。两件事都要报。
class _RelabelState:
    """回合号重标壳：属性全透传，只把 `round_index` 换掉（主动重启后新会话的读端从 0 起）。"""

    __slots__ = ("_inner", "round_index")

    def __init__(self, inner, round_index: int):
        self._inner = inner
        self.round_index = int(round_index)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _oracle_stderr_path(tag: str, st: dict, kind: str):
    """重建/重启出来的会话写**新文件**（旧会话的崩溃日志是证据，绝不能被覆盖）。"""
    d = Path(MVS.REPO_ROOT) / "match_results" / "oracle_logs"
    d.mkdir(parents=True, exist_ok=True)
    return d / (f"{str(tag).replace('-', '_')}_{kind}_seed{int(st['seed'])}"
                f"_p{int(st['player'])}_{os.getpid()}.stderr.log")


def _oracle_state(st: dict, facade):
    """该会话该看到的"局面对象"：偏移 0 ⇒ 原样（默认，零开销）；否则回合号重标。"""
    off = int(st.get("relabel_off", 0) or 0)
    return facade if off == 0 else _RelabelState(facade, int(facade.round_index) - off)


def _oracle_rebuild_try(tag: str, st: dict, facade, why: str = "") -> bool:
    """会话死了 ⇒ 起新进程 + **把本局 0..当前回合的历史回放进去**（追平后调用方可立刻重试本回合）。

    返回 True = 已追平（`st["obj"]` 是新会话，`_next_round` = 当前回合）。
    次数上限 `AZAI_ORACLE_REBUILD_MAX`（默认 6；`0` = 关闭 = **旧行为"死了就永久停用"**）。
    """
    mod = st.get("mod")
    if st.get("off") or mod is None:
        return False
    old = st.get("obj")
    hist = list(getattr(old, "history", None) or st.get("hist") or [])
    opp_ops = dict(getattr(old, "_opp_ops", None) or {})
    off = int(st.get("relabel_off", 0) or 0)
    rnd = int(facade.round_index) - off          # 它眼里的回合号（重标后从 0 起）
    used = int(st.get("rebuilds", 0)) + int(st.get("rebuild_fail", 0))
    if int(st.get("rebuild_max", 0)) <= 0:
        return False
    if used >= int(st["rebuild_max"]):
        _log(f"[{tag}] ⚠️ 第 {int(facade.round_index)} 回合：重建次数已用尽"
             f"（{used}/{int(st['rebuild_max'])}）⇒ **本局停用注入**（决策照常）")
        return False
    if old is not None:
        try:
            old.close()
        except Exception:  # noqa: BLE001
            pass
    st["obj"] = None
    t0 = time.monotonic()
    new = None
    kid = used + 1
    try:
        new = mod.RunnerUpOracle(
            seed=int(st["seed"]), player=int(st["player"]), exe=str(st["exe"]),
            timeout=float(st["timeout"]),
            stderr_path=_oracle_stderr_path(tag, st, f"rebuild{kid}"),
            verbose=bool(int(_env("AZAI_ORACLE_DEBUG", "0"))))
        new._opp_ops.update(opp_ops)
        n, _dt = new.replay(hist, upto_round=rnd)
        st["obj"] = new
        st["hist"] = []
        st["rebuilds"] = int(st.get("rebuilds", 0)) + 1
        dt = time.monotonic() - t0
        st["replay_rounds"] = int(st.get("replay_rounds", 0)) + int(n)
        st["replay_s_sum"] = float(st.get("replay_s_sum", 0.0)) + dt
        st["replay_s_max"] = max(float(st.get("replay_s_max", 0.0)), dt)
        _log(f"[{tag}] 重建会话 #{int(st['rebuilds'])} at round={int(facade.round_index)}"
             f"（回放 {int(n)} 回合，耗时 {dt:.1f} s）{why}")
        return True
    except Exception as exc:  # noqa: BLE001
        st["rebuild_fail"] = int(st.get("rebuild_fail", 0)) + 1
        st["hist"] = hist
        st["err"] = f"rebuild: {type(exc).__name__}: {exc}"
        dt = time.monotonic() - t0
        _log(f"[{tag}] ⚠️ 重建会话失败（第 {int(st['rebuild_fail'])} 次，{dt:.1f}s 后放弃）"
             f"⇒ 本回合不注入: {type(exc).__name__}: {exc}")
        if new is not None:
            try:
                new.close()
            except Exception:  # noqa: BLE001
                pass
        return False


def _oracle_restart(tag: str, st: dict, facade) -> bool:
    """**主动重启**（`AZAI_ORACLE_RESTART_EVERY=N`）：不等它崩，每 N 回合换一个新进程。

    与 `_oracle_rebuild_try` 的差别 = **不重放历史**：新会话从回合 0 起、喂的是**当前局面**
    （回合号重标 ⇒ 读端不错位），它自己的影子局面则从头攒。
    为什么需要：冠军的影子局面必然与真对局分叉（它的自己执行的是**它挑的**操作），一旦分叉就整局
    HOLD、再往后影子里的基地血会归零 ⇒ abort。主动重启把"崩"变成"每 N 回合换一张白纸"。
    代价：新会话的影子和真局面差得远 ⇒ 它几乎一律 HOLD（冠军本来 99.9% 也是 HOLD，见报告 §1）。
    """
    mod = st.get("mod")
    if st.get("off") or mod is None:
        return False
    old = st.get("obj")
    if old is not None:
        try:
            old.close()
        except Exception:  # noqa: BLE001
            pass
    st["obj"] = None
    kid = int(st.get("restarts", 0)) + 1
    try:
        new = mod.RunnerUpOracle(
            seed=int(st["seed"]), player=int(st["player"]), exe=str(st["exe"]),
            timeout=float(st["timeout"]),
            stderr_path=_oracle_stderr_path(tag, st, f"restart{kid}"),
            verbose=bool(int(_env("AZAI_ORACLE_DEBUG", "0"))))
        st["obj"] = new
        st["restarts"] = kid
        st["relabel_off"] = int(facade.round_index)
        st["hist"] = []
        _log(f"[{tag}] 主动重启会话 #{kid} at round={int(facade.round_index)}"
             f"（回合号重标 off={int(facade.round_index)}；不重放历史）")
        return True
    except Exception as exc:  # noqa: BLE001
        st["err"] = f"restart: {type(exc).__name__}: {exc}"
        _log(f"[{tag}] ⚠️ 主动重启失败（第 {kid} 次）⇒ 本回合不注入: "
             f"{type(exc).__name__}: {exc}")
        return False


def _oracle_close_module() -> None:
    """释放注入会话（模块级：供 atexit / SIGTERM 兜底；`main()` 正常收尾也会显式调）。"""
    _ob = _ORACLE.get("obj")
    if _ob is not None:
        try:
            _ob.close()
        except Exception:  # noqa: BLE001
            pass
        _ORACLE["obj"] = None
    # 2026-10-28：B 会话（E4 双 oracle；默认 `off=True` ⇒ 这里什么都不做）
    _ob2 = _ORACLE_B.get("obj")
    if _ob2 is not None:
        try:
            _ob2.close()
        except Exception:  # noqa: BLE001
            pass
        _ORACLE_B["obj"] = None


def _report(*_args) -> None:
    """收尾报告。bridge 在 finally 里会 kill 我们，所以 atexit/SIGTERM 都要挂上，
    否则"对拍次数"这个**状态同步的证据**会丢。

    2026-10-22：把 **mc_fallback / mc_last_error** 无条件带进收尾报告。
    教训：`int('op')` 的异常被宽 except 吞掉后，只有计时日志才会输出 mc_fallback，
    而计时日志默认不开 ⇒ "整类动作静默失效"完全看不见。
    """
    _mc = ""
    try:
        _m = _ENGINE.get("mcts")
        _mc_err = getattr(_m, "mc_last_error", None) if _m is not None else None
        _mt = int(getattr(_m, "mc_taken", 0)) if _m is not None else 0
        _mf = int(getattr(_m, "mc_fallback", 0)) if _m is not None else 0
        if _mt or _mf:
            _mc = (f" mc_taken={_mt} mc_fallback={_mf}"
                   + (f" mc_err={_mc_err}" if _mc_err else ""))
    except Exception:  # noqa: BLE001
        pass
    _gate = ""
    if int(STATS.get("filter_gate_checks", 0)):
        # 2026-10-28：事后过滤闸门的关键读数（拦下多少次半块 / 丢成 HOLD 多少次）
        _gate = (f" gate_checks={STATS['filter_gate_checks']}"
                 f" gate_dropped={STATS.get('filter_gate_dropped', 0)}")
    _log(f"退出报告: rounds={STATS['rounds']} mismatch={STATS['mismatch']} "
         f"illegal={STATS['illegal']}{_gate}{_mc} dec_n={STATS.get('dec_n')} "
         f"dec_max={STATS.get('dec_max_s')}s dec_slow={STATS.get('dec_slow')} "
         f"{_dec_hist_text()}")
    _dt = STATS.get("dec_times") or []
    if _dt:
        _log(f"决策耗时: n={len(_dt)} p50={_pct(_dt, 0.50):.2f}s p95={_pct(_dt, 0.95):.2f}s "
             f"max={max(_dt):.2f}s")
    if _ORACLE.get("asks"):
        _log(f"[{_inj_tag()}] 注入统计: {_oracle_stats_text()}")
    if _ORACLE_B.get("asks"):
        # 2026-10-28：E4 的第二个会话（默认 0 ⇒ 不打）
        _log(f"[oracle-b] 注入统计: {_oracle_b_stats_text()}")
    if _XOR.get("rounds"):
        # 2026-10-28：双 oracle 的收尾汇总（E4/E4R；E2 只有 A 侧）。默认 0 ⇒ 不打。
        _log(f"[xor-sum] {_xor_summary_text()}")
    if _DS.get("rounds"):
        # 2026-10-29：候选分差汇总（E2/E4/E4R；默认 0 ⇒ 不打）
        _log(f"[dscore-sum] {_dscore_summary_text()}")
    if _ONLY.get("on"):
        _log(f"[onlyextra] 统计: random={int(_ONLY['random'])} decided={_ONLY['decided']} "
             f"ncand={_ONLY['ncand']} pick={_ONLY['pick']}"
             + (f" err={_ONLY['err']}" if _ONLY.get("err") else ""))
    if _UPG.get("n", 0) > 0:
        # 2026-10-28：升级候选臂的收尾读数（每 50 决策还会周期打一次，防 bridge 硬杀丢掉）
        _log(f"[upg] 统计: {_upg_stats_text()}")
    if _AHOLD.get("on"):
        # 2026-10-29：HOLD 恒为候选（消融臂）的收尾读数
        _log(f"[ahold] 统计: {_ahold_stats_text()}")
    sys.stderr.flush()


def _install_reporters() -> None:
    import atexit
    import signal
    atexit.register(_report)
    atexit.register(_dump_value_data)
    atexit.register(_close_az)  # AZ 软策略 dump（默认关闭）


    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, lambda *_a: (_report(), _dump_value_data(), sys.exit(0)))
        except (ValueError, OSError):
            pass


def _dump_value_data() -> None:
    """收尾：关掉裸 dump 文件并（尽力）写一个 sidecar（**标签主要靠 ladder 日志反推**）。

    ⚠️ 实测 bridge 会把我们**硬杀**在这里（连函数体都跑不完）⇒ 本函数只是"尽力而为"：
    数据的完整性由"边打边追加"保证，标签由 `ingest_bridge_dump.py` 从 ladder 日志补。
    """
    if _DUMP.get("fh") is not None:
        try:
            _DUMP["fh"].flush()
            _DUMP["fh"].close()
        except Exception:  # noqa: BLE001
            pass
        _DUMP["fh"] = None
        _log(f"[dump] 已写入 {_DUMP['n']} 条观测 -> {_DUMP['dir']}")


class ProtocolIO:
    """AI 侧协议：收裸文本行、发 4 字节长度前缀的包。"""

    def __init__(self) -> None:
        self._in = sys.stdin.buffer
        self._out = _PROTO_OUT

    def recv_line(self) -> str | None:
        raw = self._in.readline()
        if not raw:
            return None
        return raw.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")

    def recv_ops(self) -> list | None:
        """读一段操作文本（首行数量 + 若干行 token）。EOF 返回 None。"""
        head = self.recv_line()
        if head is None:
            return None
        try:
            count = int(head.strip())
        except ValueError:
            _log(f"!! 操作文本首行不是数量: {head!r}")
            return []
        lines = [head]
        for _ in range(count):
            ln = self.recv_line()
            if ln is None:
                return None
            lines.append(ln)
        return parse_ops_text("\n".join(lines))

    def send_ops(self, ops: list) -> None:
        data = ops_to_text(ops).encode("utf-8")
        self._out.write(struct.pack(">I", len(data)))
        self._out.write(data)
        self._out.flush()


def build_engine(seed: int, player: int):
    """按环境变量构造 (mcts, net_fn, model, feat)。"""
    import numpy as np  # noqa: F401  (net_fn 的调用方需要)
    import torch

    torch.set_num_threads(int(_env("AZAI_TORCH_THREADS", "1")))
    ckpt = _env("AZAI_CKPT", "")
    if not ckpt:
        raise RuntimeError("必须设 AZAI_CKPT 环境变量指向 checkpoint")
    # ⚠️ bridge 用 cwd=exe.parent（= code/test_match）起我们，所以相对路径要按仓库根解析
    _c = Path(ckpt)
    ckpt = str(_c if _c.is_absolute() else (Path(MVS.REPO_ROOT) / _c).resolve())
    # 同理：类条件价值头（M2）的路径也要按仓库根解析（实测踩过：相对路径 ⇒ FileNotFoundError ⇒ 整局崩）
    _q = _env("AZAI_Q_CKPT", "")
    if _q:
        _qp = Path(_q)
        _q = str(_qp if _qp.is_absolute() else (Path(MVS.REPO_ROOT) / _qp).resolve())
        os.environ["AZAI_Q_CKPT"] = _q

    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.az_intent.menu3_candidates import make_menu3_fn as _menu3_fn

    feat = FeatureExtractor(max_actions=96)
    model, net_fn = make_net_fn_from_ckpt(
        ckpt, feat,
        value_tanh=bool(int(_env("AZAI_VALUE_TANH", "1"))),
        value_rel_to_abs=float(_env("AZAI_REL2ABS", "0")),
        # 价值来源：net=自身 value 网（默认）；handcrafted=亚军的 evaluate_terminal
        # （用 AZAI_VALUE_OFFSET/AZAI_VALUE_SCALE 线性归一到搜索尺度）
        value_src=_env("AZAI_VALUE_SRC", "net"),
        value_offset=float(_env("AZAI_VALUE_OFFSET", "10000")),
        value_scale=float(_env("AZAI_VALUE_SCALE", "2000")),
        # 可选：类条件价值头（方法侧 M2，`AZAI_Q_CKPT` 指向 QHead.save 的文件）
        q_ckpt=_env("AZAI_Q_CKPT", ""),
    )
    # 可选：**3 候选菜单**（方法侧 M5，`AZAI_MENU3=1`）：把"出哪个类"交回搜索，
    # 候选 = {闪电, HOLD, 一个经济类}（见 code/my_ai/az_intent/menu3_candidates.py + 预注册）。
    _menu3_holder: dict = {}
    if int(_env("AZAI_MENU3", "0")):
        _inner = net_fn

        def net_fn(state, player, _inner=_inner, _h=_menu3_holder):   # noqa: F811
            out = _inner(state, player)
            _h["out"] = out
            return out

    # 可选：**根节点的位置先验**（`AZAI_POSPRIOR=value|uniform`，默认 off）
    #   `value`   = 用**我们自己的价值网**在"钉住的类的合法格"上算 1-ply z-score 当 action_map
    #               （这正是位置网当初被训练去模仿的那个先验；实现见
    #               `code/my_ai/az_intent/eval.py:_make_value_pos_prior`）
    #   `uniform` = 同一条枚举/可执行过滤，但权重全 0（**对照臂**，用来分离"价值信息"与"过滤"）
    # ⚠️ 默认 off 是为了让已钉死的配置（A0/A1）逐字节可复现；打开它属于**协议/配置变更**，要记档。
    ppf = None
    _pp = _env("AZAI_POSPRIOR", "off")
    if _pp in ("value", "uniform"):
        from my_ai.az_intent.az_selfplay import load_three_models
        from my_ai.az_intent.eval import _make_value_pos_prior
        _, _, _vm = load_three_models(ckpt)
        ppf = _make_value_pos_prior(feat, _vm, mode=_pp)
        _log(f"pos_prior={_pp}（根节点位置先验已启用）")
    _kw = dict(
        iterations=int(_env("AZAI_ITERS", "256")),
        max_depth_rounds=int(_env("AZAI_DEPTH", "4")),
        k=int(_env("AZAI_K", "1")),
        sample_mult=int(_env("AZAI_SAMPLE_MULT", "1")),
        # 候选确定化（docs/pos_topk_candidates_plan.md）：默认 sample ⇒ 既有读数逐位不变 ✓
        cands_mode=_env("AZAI_CANDS", "sample"),
        cands_k=int(_env("AZAI_CANDS_K", "0")),
        cands_combo=int(_env("AZAI_CANDS_COMBO", "1")),
        cands_prior=_env("AZAI_CANDS_PRIOR", "uniform"),
        seed_from_state=bool(int(_env("AZAI_SEED_STATE", "0"))),
        # 名次实验（用户 2026-10-12 设计）：取 visit 排序的第 r 名（1 = argmax = 旧行为 ✓）
        visit_rank=int(_env("AZAI_VISIT_RANK", "1")),
        t_class=float(_env("AZAI_TCLASS", "0.5")),
        t_pos=float(_env("AZAI_TPOS", "1.0")),
        c_puct=float(_env("AZAI_C_PUCT", "1.25")),
        seed=seed,
        search_mode=_env("AZAI_MODE", "joint"),
        pos_pin=_env("AZAI_POSPIN", "argmax"),
        skip_single_candidate=bool(int(_env("AZAI_SKIP1", "0"))),
        # 「类探索」（只用于**采集**，默认 0 = 关）：见 docs/class_head_outcome_plan.md
        class_pin_random_prob=float(_env("AZAI_CLASSPIN_PROB", "0")),
        # 候选 E（诊断臂）：pos_pin=reserve 时的闪电储备门槛：见 docs/prereg_20261005_reserve_economy.md
        reserve_coins=int(_env("AZAI_RESERVE", "90")),
        # M6（mc_light）的成本旋钮：rollout 截断长度 / 每 N 回合最多比一次
        mc_horizon=int(_env("AZAI_MC_HORIZON", "100")),
        # R1/R2：K 步双向 rollout（0 = 现有"安静"行为一字不变）
        mc_bidir=int(_env("AZAI_MC_BIDIR", "0")),
        mc_every=int(_env("AZAI_MC_EVERY", "1")),
        # M6b-ND：候选里排除类 16（降级）
        mc_no_downgrade=int(_env("AZAI_MC_NODOWN", "0")),
        # M6c-M4：菜单显式含类 16
        mc_m4=int(_env("AZAI_MC_M4", "0")),
        # M6c-ROOT：MC 只在根节点仲裁
        mc_root_only=int(_env("AZAI_MC_ROOTONLY", "0")),
        # M6c-2E：候选放两个不同的经济类
        mc_top2=int(_env("AZAI_MC_TOP2", "0")),
        # 菜单加宽：经济候选取前 k 个互不相同的类（0 = 现有行为不变）
        mc_topk=int(_env("AZAI_MC_TOPK", "0")),
        # 选类×选格：>0 ⇒ 菜单放「类+格」候选（含已有塔格 ⇒ 可表达升级）
        mc_cells=int(_env("AZAI_MC_CELLS", "0")),
        # 显式升级候选：>0 ⇒ 对我们前 n 座塔各放一个 UPGRADE_TOWER 候选
        mc_upg=int(_env("AZAI_MC_UPG", "0")),
        # 分差门控：分差 < τ ⇒ MC 不接管（0 = 现有行为不变）
        mc_margin=float(_env("AZAI_MC_MARGIN", "0")),
        # 评分方式："" = 安静 rollout（现状）；delta = rollout + V差分；value_only = 只用 V 差分
        mc_score=_env("AZAI_MC_SCORE", ""),
        # 单侧支出：>0 ⇒ rollout 里我方继续出招 K 回合（把候选花掉的钱换成塔 ⇒ 让评分看见代价）
        mc_selfspend=int(_env("AZAI_MC_SELFSPEND", "0")),
        mc_selfspend_mix=float(_env("AZAI_MC_SELFSPEND_MIX", "1.0")),
        # 控制实验（2026-10-21）：AZAI_MC_SCORE=net 时，用"rollout+值网"当类仲裁器
        mc_net_samples=int(_env("AZAI_MC_NET_SAMPLES", "4")),
        # 2026-10-22：显式升级奖励（亚军 c1_root_bonus 的类比）
        mc_upgrade_bonus=float(_env("AZAI_MC_UPG_BONUS", "0")),
        mc_upg_targets=int(_env("AZAI_MC_UPG_TARGETS", "1")),
        # Phase 0：1 ⇒ 闪电不进 MC 菜单（仍由默认行为放闪电）
        mc_no_light=_env("AZAI_MC_NO_LIGHT", "0") == "1",
        # 2026-10-28（诊断臂）：整束允许落在 K 个不同类上。0 = 旧的"单类"行为逐位不变。
        #   见 docs/why_effective_candidates_collapse_20261028.md（78% 的决策被钉在 HOLD ⇒ n_cand=1）。
        pin_k=int(_env("AZAI_PIN_K", "0")),
        # 方法侧 M5：3 候选菜单（AZAI_MENU3=1 时接管候选来源）
        candidate_fn=(_menu3_fn(_menu3_holder) if int(_env("AZAI_MENU3", "0")) else None),
        pos_prior_fn=ppf,
    )
    # ── 控制实验（2026-10-21）：MC_SCORE=net 时给仲裁挂**批量值函数** ────────────────
    #   不挂的话 mc_quiet_choose 会逐叶调用 net_fn（三网、未批量）⇒ 72 次 × 7.6 ms ≈ 0.55 s，
    #   占仲裁成本 80%。挂上后是"一次前向给全部叶子打分"（实测同分、耗时 0.82→0.33 s）。
    if _env("AZAI_MC_SCORE", "") == "net":
        from my_ai.az_intent.az_selfplay import load_three_models as _l3
        from my_ai.az_intent.value_batch import make_batched_value_fn as _mbvf
        from my_ai.az_intent.value_override import describe as _vdesc, maybe_override as _vovr
        _, _, _mc_vm = _l3(ckpt)
        # 可选：叶子值网替换（`AZAI_VALUE_CKPT` 空 ⇒ 原样返回同一对象，行为不变）
        _mc_vm = _vovr(_mc_vm)
        _mc_vm.eval()
        if _vdesc(_mc_vm):
            _log(_vdesc(_mc_vm))
        _kw["mc_value_fn"] = _mbvf(_mc_vm, feat,
                                   batch_size=int(_env("AZAI_FLAT_BATCH", "32")))
        _log("mc_score=net：已挂批量值函数（叶子批量评估）")
    # ── 搜索器选择（2026-10-21）：AZAI_SEARCH=flat ⇒ 扁平 rollout 评估器 ────────────
    #   flat：根节点一次动作网出候选 + 每候选跑 N 次 h 回合真实引擎 rollout + 叶值网。
    #   与默认 BundleMCTS（depth=4、iterations 次展开、每次展开一次 net_fn）共用
    #   **同一份候选集 / 类判定 / 值网**，只有"评估器"不同 ⇒ 干净的树 vs 扁平等预算消融。
    _search_kind = _env("AZAI_SEARCH", "mcts")
    # ── 统一候选生成（2026-10-29，`AZAI_CAND_MODE=pin|unified|shape_pin`，**默认 pin ⇒ 逐位不变**）──
    #   设计（冻结）：docs/unified_cand_design_20261029.md；实现/读数：docs/unified_cand_impl_20261029.md
    #   `unified` = 候选来源换成"类 UCB bandit + 位置游标"（`my_ai/az_intent/unified_cand.py`），
    #   **评估器（阶段 2）仍是 flat 随机化 rollout，一行不动** ⇒ 组合必须 `AZAI_SEARCH=flat`。
    #   `shape_pin`（2026-11-02）= 介于 pin 与 unified 之间：阶段 1 照跑 unified 的 bandit + 游标，
    #   但阶段 1 结束后**锁定 s1 第一名候选的有序类序列**，只在该序列内按位置游标顺序取前 k 个
    #   位置进阶段 2（见 docs/shape_pin_mode_20261102.md）。阶段 2 仍是 flat 一字不动。
    #   ⚠️ `pin` 时这里**只多读一个环境变量**，构造的仍是原来那行 `FlatRolloutGreedy(`。
    _cand_mode = _env("AZAI_CAND_MODE", "pin").strip().lower()
    if _cand_mode not in ("pin", "unified", "shape_pin", "unified_noupg"):
        _log(f"[cand] ⚠️ AZAI_CAND_MODE={_cand_mode!r} 不认识 ⇒ 按 pin（现役行为）处理")
        _cand_mode = "pin"
    _is_unified_cand = _cand_mode in ("unified", "shape_pin", "unified_noupg")
    if _is_unified_cand and _search_kind != "flat":
        _log(f"[cand] ⚠️ AZAI_CAND_MODE={_cand_mode} 只与 AZAI_SEARCH=flat 组合"
             f"（阶段 2 必须是现役 flat rollout）；当前 AZAI_SEARCH={_search_kind} ⇒ 忽略，按 pin 跑")
        _cand_mode = "pin"
        _is_unified_cand = False
    if _is_unified_cand:
        # 设计 §3：固定 `AZAI_K=48`，用 top-k 截断（不要动态扩 k）。
        # `AZAI_K` 不设时 pin 的默认是 1（BundleMCTS 的 k）⇒ unified/shape_pin 下改成 48，否则只有 1 条候选。
        if not _env("AZAI_K", "").strip():
            _kw["k"] = 48
            _log(f"[cand] AZAI_K 未设 ⇒ {_cand_mode} 模式按默认 k=48（设计 §3）")
        # 设计 §4.6：unified/shape_pin 直接接管候选生成，下面这些"候选补丁"开关不再生效（避免双重路径）。
        #   ⚠️ 注入类（`AZAI_ORACLE_INJECT*`）**不忽略**：设计 §3 要求"与 unified 候选合并后一起去重"，
        #   它们走的是 `set_extra_bundles` 那条独立通道。
        _ignored = [k for k in ("AZAI_PIN_K", "AZAI_ALWAYS_HOLD", "AZAI_MC_UPG",
                                "AZAI_MC_CELLS", "AZAI_UPG_BUNDLES")
                    if _env(k, "").strip() not in ("", "0")]
        if _ignored:
            _log(f"[cand] ⚠️ {_cand_mode} 模式下这些候选补丁开关**不再生效**：{_ignored}"
                 f"（候选由统一生成器独占）")
        if _env("AZAI_ONLY_EXTRA", "").strip() not in ("", "0"):
            _log(f"[cand] ⚠️ AZAI_ONLY_EXTRA 会**丢掉** {_cand_mode} 候选（候选集=只有额外候选）⇒ 两者别同时用")
        _log(f"[cand] mode={_cand_mode} k={_env('AZAI_K', '48')} beta={_env('AZAI_CAND_BETA','3.0')} "
             f"ucb_c={_env('AZAI_CAND_UCBC','1.0')} pool={_env('AZAI_CAND_POOL','1.5')} "
             f"tau={_env('AZAI_CAND_TAU','1.0')} steps={_env('AZAI_CAND_STEPS','12')} "
             f"steps_total={_env('AZAI_CAND_STEPS_TOTAL','0')} "
             f"decode={_env('AZAI_CAND_DECODE','legacy')} "
             f"force_hold={_env('AZAI_CAND_FORCE_HOLD','1')} verbose={_env('AZAI_CAND_VERBOSE','0')}")
    if _search_kind == "flat":
        from my_ai.az_intent.az_selfplay import load_three_models
        from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
        from my_ai.az_intent.value_override import describe as _vdesc, maybe_override as _vovr
        _, _, _value_model = load_three_models(ckpt)
        # 可选：叶子值网替换（`AZAI_VALUE_CKPT` 空 ⇒ 原样返回同一对象 ⇒ 行为逐位不变）
        _value_model = _vovr(_value_model)
        _value_model.eval()
        if _vdesc(_value_model):
            _log(_vdesc(_value_model))
        # ⚠️ 只有 `AZAI_CAND_MODE=unified|shape_pin` 才换类；pin 时**就是** `FlatRolloutGreedy`（同一个类）。
        _searcher_cls = FlatRolloutGreedy
        if _is_unified_cand:
            from my_ai.az_intent.unified_cand import make_unified_searcher_class
            _searcher_cls = make_unified_searcher_class()
        mcts = _searcher_cls(
            net_fn,
            rollouts_per_candidate=int(_env("AZAI_FLAT_ROLLOUTS", "16")),
            horizon=int(_env("AZAI_FLAT_HORIZON", "64")),
            value_model=_value_model,
            feature_extractor=feat,
            value_batch=int(_env("AZAI_FLAT_BATCH", "32")),
            eval_mode=_env("AZAI_FLAT_EVAL", "net"),
            max_total_rounds=int(_env("AZAI_FLAT_ROUNDS", "39000")),
            verbose=bool(int(_env("AZAI_FLAT_VERBOSE", "0"))),
            # 2026-10-23：rollout 里是否随机出招（idle=旧行为，双方不出招⇒确定性）
            rollout_mode=_env("AZAI_FLAT_ROLLOUT_MODE", "idle"),
            rollout_act_prob=float(_env("AZAI_FLAT_ACT_PROB", "0.5")),
            rollout_build_ratio=float(_env("AZAI_FLAT_BUILD_RATIO", "0.7")),
            rollout_seed=int(_env("AZAI_FLAT_SEED", "0")),
            rollout_random_side=_env("AZAI_FLAT_RANDOM_SIDE", "opponent"),
            rollout_prob_mode=_env("AZAI_FLAT_PROB_MODE", "const"),
            rollout_decay_c=float(_env("AZAI_FLAT_DECAY_C", "0.6")),
            rollout_decay_k=float(_env("AZAI_FLAT_DECAY_K", "3.0")),
            rollout_share_prefix=bool(int(_env("AZAI_FLAT_SHARE_PREFIX", "0"))),
            rollout_dedupe_weight=bool(int(_env("AZAI_FLAT_DEDUPE", "0"))),
            **({} if not _is_unified_cand else {"cand_mode": _cand_mode}),
            **_kw,
        )
        # ⚠️ pin 时这一行与改动前**逐字相同**（`cand_mode=` 只在 unified 下追加，免得日志也变）
        _log(f"engine=flat rollouts={_env('AZAI_FLAT_ROLLOUTS','16')} "
             f"horizon={_env('AZAI_FLAT_HORIZON','64')} "
             f"batch={_env('AZAI_FLAT_BATCH','32')} eval={_env('AZAI_FLAT_EVAL','net')} "
             f"rounds={_env('AZAI_FLAT_ROUNDS','39000')}"
             + (f" cand_mode={_cand_mode}" if _cand_mode in ("unified", "shape_pin", "unified_noupg") else ""))
    elif _search_kind == "unified":
        # **统一动作搜索**（2026-10-21）：不做类仲裁，直接枚举原子动作
        # （建塔/升级/降级/闪电/HOLD）逐个 rollout 打分，剩余预算复评 top-k。
        # 见 docs/idea_unified_bandit_search.md 与 code/my_ai/az_intent/unified_action_search.py
        from my_ai.az_intent.az_selfplay import load_three_models
        from my_ai.az_intent.unified_action_search import UnifiedActionSearch
        from my_ai.az_intent.value_override import describe as _vdesc, maybe_override as _vovr
        _, _, _value_model = load_three_models(ckpt)
        # 可选：叶子值网替换（`AZAI_VALUE_CKPT` 空 ⇒ 原样返回同一对象 ⇒ 行为逐位不变）
        _value_model = _vovr(_value_model)
        _value_model.eval()
        if _vdesc(_value_model):
            _log(_vdesc(_value_model))
        mcts = UnifiedActionSearch(
            net_fn,
            horizon=int(_env("AZAI_UNI_HORIZON", "32")),
            candidate_cap=int(_env("AZAI_UNI_CAP", "256")),
            reeval_topk=int(_env("AZAI_UNI_TOPK", "8")),
            total_rounds=int(_env("AZAI_UNI_ROUNDS", "39000")),
            min_samples_per_candidate=int(_env("AZAI_UNI_MINSAMP", "30")),
            max_lightning=int(_env("AZAI_UNI_MAXLIGHT", "32")),
            value_model=_value_model,
            feature_extractor=feat,
            value_batch=int(_env("AZAI_FLAT_BATCH", "32")),
            eval_mode=_env("AZAI_FLAT_EVAL", "net"),
            verbose=bool(int(_env("AZAI_UNI_VERBOSE", "0"))),
            **_kw,
        )
        _log(f"engine=unified horizon={_env('AZAI_UNI_HORIZON','32')} "
             f"cap={_env('AZAI_UNI_CAP','256')} topk={_env('AZAI_UNI_TOPK','8')} "
             f"rounds={_env('AZAI_UNI_ROUNDS','39000')} "
             f"maxlight={_env('AZAI_UNI_MAXLIGHT','32')}")
    else:
        mcts = BundleMCTS(net_fn, **_kw)
    _log(f"engine built: ckpt={ckpt} iters={_env('AZAI_ITERS','256')} "
         f"depth={_env('AZAI_DEPTH','4')} mode={_env('AZAI_MODE','joint')} "
         f"pospin={_env('AZAI_POSPIN','argmax')} t_class={_env('AZAI_TCLASS','0.5')} "
         f"t_pos={_env('AZAI_TPOS','1.0')} temp={_env('AZAI_TEMP','1e-6')}")
    return mcts, model, feat


def main() -> int:
    _install_reporters()
    io = ProtocolIO()
    trace = bool(int(_env("AZAI_TRACE", "1")))
    verify = bool(int(_env("AZAI_VERIFY", "1")))
    temperature = float(_env("AZAI_TEMP", "1e-6"))

    init = io.recv_line()
    if init is None:
        _log("!! 没有收到 init 行就 EOF 了")
        return 1
    player, seed = (int(x) for x in init.split())
    STATS["seed"] = int(seed)   # 2026-10-28：过滤闸门/日志要带 seed（原来只有局部变量）
    _log(f"init: player={player} seed={seed} pid={os.getpid()}")

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType

    mcts, model, feat = build_engine(seed, player)
    _ENGINE["mcts"] = mcts   # 供 _report 读取（原来是 main 的局部变量 ⇒ _report 里 NameError 被吞）

    facade = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    # 裸观测 dump（可选）：**边打边追加**（见文件头注释：结束时写会赶不上 bridge 的硬杀）
    _DUMP["dir"] = _env("AZAI_DUMP_RAW_DIR", "")
    _DUMP["seed"] = seed
    _DUMP["player"] = player
    _DUMP["fh"] = None
    _DUMP["n"] = 0
    _AZ["seed"] = seed
    _AZ["player"] = player
    if _DUMP["dir"]:
        import numpy as np
        out = Path(_DUMP["dir"])
        if not out.is_absolute():
            out = Path(MVS.REPO_ROOT) / out       # ⚠️ bridge 用 cwd=exe.parent 起我们
        out.mkdir(parents=True, exist_ok=True)
        _DUMP["path"] = str(out / f"obs_seed{seed:05d}_p{player}.bin")
        _log(f"裸观测 dump 已开启 -> {_DUMP['path']}")

    def _record_obs(res=None) -> None:
        """追加一条裸记录：board + stats + player + **实际选中的类(3)** + **合法类掩码(24)**。

        "选中的类"口径与 `code/my_ai/az_intent/report_posnet_diag.py:44-57` 一致
        （= 访问量最高候选里、该 head 采样次数最多的 intent 的类号）；没有就记 255。
        掩码由解码器自己推（`intent_decoding=True`）⇒ 与"能不能真解出操作"一致。
        """
        _record_az(res, facade, player)
        if not _DUMP["dir"]:
            return
        import numpy as np
        from my_ai.decoder import make_class_mask, make_position_masks
        obs = feat.encode_observation(facade, player, np.zeros(96))
        pm = make_position_masks(facade, player, intent_decoding=True)
        cm = make_class_mask(facade, player, position_mask=pm, intent_decoding=True)
        chosen = [255, 255, 255]
        ic = getattr(res, "intent_counts", None) if res is not None else None
        if ic:
            # 2026-10-22（审计）：BundleSearchResult 的真实字段是 **visit_policy**，
            # 原来写 "visit" ⇒ 恒为 None ⇒ dump 里"选中类"永远取候选 0（而这份 dump 是值网训练数据）
            vis = getattr(res, "visit_policy", None) or getattr(res, "visit", None)
            j = int(np.argmax(np.asarray(vis, dtype=float))) if vis is not None and len(vis) else 0
            for h in range(3):
                d = ic[j][h] if j < len(ic) and h < len(ic[j]) else {}
                if d:
                    chosen[h] = int(max(d.items(), key=lambda kv: kv[1])[0][0])
        if _DUMP["fh"] is None:
            _DUMP["fh"] = open(_DUMP["path"], "ab")
        _DUMP["fh"].write(np.asarray(obs["board"], dtype=np.float16).tobytes())
        _DUMP["fh"].write(np.asarray(obs["stats"], dtype=np.float16).tobytes())
        _DUMP["fh"].write(bytes([int(player)]))
        _DUMP["fh"].write(bytes(int(x) & 0xFF for x in chosen))
        _DUMP["fh"].write(bytes(int(x) & 0xFF for x in np.asarray(cm, dtype=np.uint8)))
        _DUMP["n"] += 1

    # ── 亚军候选注入（2026-10-26；开关式；`AZAI_ORACLE_INJECT` 为空 ⇒ 下面全是 no-op）──────
    #   设计/兜底/统计字段见 `docs/oracle_inject_design.md`；查询器本身在
    #   `code/test_match/runnerup_oracle.py`（协议已 30/30、90/90 验证，本文件**不改它**）。
    def _oracle_bundle(ops):
        """`list[Operation]` → bundle 元组（`(op_type, arg0, arg1)` 的元组）；**有一条非法 ⇒ None**。

        ⚠️ 合法性**必须**在这里逐条查：`facade.apply_operation_list` 走冷路径，
        **非法操作被静默跳过**（`code/cpp_engine/binding.cpp:46-59` 的返回值被
        `game_state_facade.py:167-170` 丢掉）⇒ 不查就会把一个"半条命"的候选塞进搜索。
        `pending=` 把前序 op 的金币/塔位影响算进去 —— 与亚军自己 `cpp_lure_v4.cpp:122-128` 一致。
        """
        accepted: list = []
        for op in ops:
            try:
                ok = bool(facade.can_apply_operation(int(player), op, pending=accepted))
            except Exception as exc:  # noqa: BLE001
                _log(f"[oracle] ⚠️ can_apply_operation 抛异常（按非法处理）: "
                     f"{type(exc).__name__}: {exc}")
                ok = False
            if not ok:
                _log(f"[oracle] ⚠️ 第 {facade.round_index} 回合亚军给的 op 非法 ⇒ **整条候选丢弃**："
                     f"{(int(op.op_type), int(op.arg0), int(op.arg1))}（它这一手="
                     f"{[(int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops]}）")
                return None
            accepted.append(op)
        # 2026-10-27：这条路径**本来就有**逐条 can_apply（上面的循环）⇒ 注入的 op 不会非法；
        # 这里只补计数，进 `_oracle_stats_text()` 的 `checked=`，让"注入通道查了 N 条"可见。
        _ORACLE["checked"] += len(ops)
        return tuple((int(o.op_type), int(o.arg0), int(o.arg1)) for o in accepted)

    def _oracle_init() -> None:
        """按环境变量决定要不要启用注入。**默认（空）= 不 import、不起子进程、零耗时。**"""
        exe = _env("AZAI_ORACLE_INJECT", "").strip()
        _ORACLE["seed"], _ORACLE["player"] = int(seed), int(player)
        # 2026-10-28：**默认 2.5s → 10s**。实测依据（docs/oracle_inject_ab_20261028.md §4.1.1、§4.3）：
        #   贵配置 16 局里单次查询 max = **6859 ms**（p95 ~1.4s），默认 2.5s 下 `--jobs 8` 时
        #   8 个注入臂有 **5 个**在 round 55/123/150/187/207 触发 `OracleTimeout`（每次都停在
        #   2515-2516ms，只是抖动尖峰），而超时后注入**永久停用**（读端错位，协议上不能重连）
        #   ⇒ 后半局被测臂退化成基准臂，主读数被稀释。10s ≈ 实测 max 的 1.5 倍、p95 的 7 倍，
        #   仍能抓真死；同时 10s 也是**游戏规则的每步上限**（规则内口径的硬边界）。
        _ORACLE["timeout"] = float(_env("AZAI_ORACLE_TIMEOUT", "10"))
        # 2026-10-29：重建/主动重启的开关（默认：重建 6 次、不主动重启 ⇒ **旧行为之外的唯一差别**就是
        #   "死了会重建"，而这一点正是 E4 的前置条件；`AZAI_ORACLE_REBUILD_MAX=0` 可退回旧行为）。
        try:
            _ORACLE["rebuild_max"] = max(0, int(_env("AZAI_ORACLE_REBUILD_MAX", "6") or 0))
        except ValueError:
            _ORACLE["rebuild_max"] = 6
            _log(f"[oracle] ⚠️ AZAI_ORACLE_REBUILD_MAX={_env('AZAI_ORACLE_REBUILD_MAX', '')!r} 不是整数 ⇒ 按 6")
        try:
            _ORACLE["restart_every"] = max(0, int(_env("AZAI_ORACLE_RESTART_EVERY", "0") or 0))
        except ValueError:
            _ORACLE["restart_every"] = 0
        # 2026-10-29：oracle 给 HOLD 时**不注入**（默认 0 = 照旧注入 HOLD）。
        #   用途：给"真外部动作"定价（HOLD 的功劳由 `AZAI_ALWAYS_HOLD` 单独控制住）。
        try:
            _ORACLE["skip_hold"] = 1 if int(_env("AZAI_ORACLE_SKIP_HOLD", "0") or 0) else 0
        except ValueError:
            _ORACLE["skip_hold"] = 0
        _ORACLE["off"] = True
        # 2026-10-28：**安慰剂臂**（`AZAI_ORACLE_PLACEBO=1`，见 docs/placebo_inject_ab_20261028.md）
        #   = 走**同一条**注入通道，但候选不由亚军给、本地随机抽一个合法 op（**不查 oracle、不起子进程**）。
        #   `AZAI_ORACLE_PLACEBO_HOLD_P`（默认 0）是该决策走 HOLD 的概率：P 臂 0（每次都给一个非 HOLD op，
        #   干预面最大 = "候选数效应"的上界检验），P2 臂 0.78（对齐亚军 78% 的 HOLD，比较"内容分布"）。
        #   ⚠️ 两个环境变量都不设 ⇒ 下面这几行只是两次 `os.environ.get`，行为逐位不变。
        try:
            _ph_on = bool(int(_env("AZAI_ORACLE_PLACEBO", "0") or 0))
        except ValueError:
            _ph_on = False
            _log(f"[placebo] ⚠️ AZAI_ORACLE_PLACEBO={_env('AZAI_ORACLE_PLACEBO', '')!r} 不是整数"
                 f" ⇒ 按不启用处理")
        try:
            _ph_hold = float(_env("AZAI_ORACLE_PLACEBO_HOLD_P", "0") or 0)
        except ValueError:
            _ph_hold = 0.0
        _ORACLE["placebo"] = _ph_on
        _ORACLE["placebo_hold_p"] = min(1.0, max(0.0, _ph_hold))
        if _ph_on:
            if not hasattr(mcts, "set_extra_bundles"):
                _ORACLE["placebo"] = False
                _log(f"[placebo] ⚠️ {type(mcts).__name__} 没有额外候选通道（只有 flat/unified 有）"
                     f" ⇒ 不启用安慰剂注入")
                return
            _ORACLE["off"] = True       # 明确：**不查 oracle**（obj 保持 None）
            _ORACLE["obj"] = None
            if exe:
                _log(f"[placebo] ⚠️ AZAI_ORACLE_INJECT 也非空 ⇒ 本局**按安慰剂臂跑**（不查 oracle）")
            _log(f"[placebo] 安慰剂注入已启用: seed={seed} player={player} "
                 f"FLAT_SEED={_env('AZAI_FLAT_SEED', '0')} hold_p={_ORACLE['placebo_hold_p']:g} "
                 f"（候选 = 本地随机**合法** op；不查询 oracle ⇒ 无 ~0.8s 等待）")
            return
        if not exe:
            _log("[oracle] AZAI_ORACLE_INJECT 为空 ⇒ 不启用候选注入（零开销）")
            return
        if not hasattr(mcts, "set_extra_bundles"):
            _log(f"[oracle] ⚠️ {type(mcts).__name__} 没有额外候选通道（只有 flat/unified 有）"
                 f" ⇒ 不启用注入")
            return
        try:
            _exe = Path(exe)
            if not _exe.is_absolute():
                _exe = (Path(MVS.REPO_ROOT) / _exe).resolve()
            # ⚠️ 只有启用时才 import 查询器（它内部会 import match_via_sdk、起进程）
            _spec = importlib.util.spec_from_file_location(
                "runnerup_oracle_for_azai", _HERE / "runnerup_oracle.py")
            _mod = importlib.util.module_from_spec(_spec)
            assert _spec.loader is not None
            _spec.loader.exec_module(_mod)
            _ORACLE["obj"] = _mod.RunnerUpOracle(
                seed=int(seed), player=int(player), exe=str(_exe),
                timeout=float(_ORACLE["timeout"]),
                verbose=bool(int(_env("AZAI_ORACLE_DEBUG", "0"))))
            _ORACLE["mod"] = _mod            # 2026-10-29：重建时要再 `RunnerUpOracle(...)`
            _ORACLE["exe"] = str(_exe)
            _ORACLE["off"] = False
            _log(f"[oracle] 注入已启用: exe={_exe} seed={seed} player={player} "
                 f"timeout={_ORACLE['timeout']:g}s（**每回合都要喂**，跳回合会永久错位）"
                 f" 重建上限={_ORACLE['rebuild_max']} 主动重启每={_ORACLE['restart_every'] or '关'}")
        except Exception as exc:  # noqa: BLE001
            _ORACLE["off"] = True
            _ORACLE["obj"] = None
            _ORACLE["err"] = f"{type(exc).__name__}: {exc}"
            _log(f"[oracle] ⚠️ 启用失败 ⇒ 本局不注入（决策照常，绝不影响对局）: {_ORACLE['err']}")

    def _oracle_set_opp(round_index: int, ops) -> None:
        """登记「对手在第 round_index 回合的操作」（**P0 用**；P1 走 `ask(opponent_ops=)`）。

        协议规定 P0 的读序是 `opp_ops(N-1) → state(N)`，所以 P0 登记的是**上一回合**的块；
        P1 是 `state(N) → opp_ops(N)`，即本回合的块（见 runnerup_oracle_protocol.md §2.2）。
        """
        if _ORACLE["off"] or _ORACLE["obj"] is None:
            return
        try:
            # 2026-10-29：主动重启后回合号有偏移 ⇒ 按重标后的回合号登记（默认 off=0，逐字不变）
            _ORACLE["obj"].set_opponent_ops(
                int(round_index) - int(_ORACLE.get("relabel_off", 0) or 0), list(ops))
        except Exception as exc:  # noqa: BLE001
            _log(f"[oracle] ⚠️ 登记对手操作块失败（忽略，按空过处理）: {type(exc).__name__}: {exc}")

    def _placebo_step() -> None:
        """安慰剂臂的注入步骤：**不查 oracle**，本地生成 1 个合法候选（统计/日志与 A 臂同构）。

        * 尾部那 7 行是 `_oracle_step()` 尾部的**逐字复制**（故意不抽公共函数）：
          A 臂（亚军注入）的代码路径**一行都不动**，历史 A/B（81.2%）读数才继续成立；
        * 生成失败一律吞掉：本回合不注入、决策照常（与 A 臂同一条纪律）。
        """
        st = _ORACLE
        st["asks"] += 1
        _t0 = time.monotonic()
        try:
            _bundle, _kind = _placebo_bundle(facade, int(player),
                                             int(STATS.get("dec_n", 0)),
                                             int(_env("AZAI_FLAT_SEED", "0") or 0),
                                             float(st["placebo_hold_p"]))
        except Exception as exc:  # noqa: BLE001
            st["dead"] += 1
            st["err"] = f"{type(exc).__name__}: {exc}"
            st["last_bundle"] = None
            if hasattr(mcts, "set_extra_bundles"):
                mcts.set_extra_bundles(None)
            _log(f"[placebo] ⚠️ 第 {facade.round_index} 回合生成失败 ⇒ 本回合不注入"
                 f"（决策照常）: {st['err']}")
            return
        st["placebo_ms"].append((time.monotonic() - _t0) * 1000.0)
        st["ok"] += 1
        st["placebo_kinds"][_kind] = int(st["placebo_kinds"].get(_kind, 0)) + 1
        st["checked"] += len(_bundle)     # 口径同 A 臂：注入通道**逐条 can_apply 复核过**的 op 数
        if int(_env("AZAI_PLACEBO_VERBOSE", "0")):
            _log(f"[placebo] r{facade.round_index} n={STATS.get('dec_n')} -> "
                 f"{_bundle} ({_kind})")
        st["inject"] += 1
        st["pending"] = True
        st["last_bundle"] = _bundle
        # 与 `_oracle_step` 尾部逐字相同（见上面 docstring 第 1 条：故意复制）
        st["round_log"] = {"round": int(facade.round_index), "action": tuple(_bundle)}
        st["round_new"] = None
        st["round_dup"] = None
        mcts.set_extra_bundles([_bundle])        # ← 同一条注入通道（HOLD = 空 bundle `()`）

    def _oracle_step() -> None:
        """每回合**必做一次** ask（保持会话同步），把它的 ops 变成一个额外候选交给搜索。

        失败一律吞掉：本回合不注入、决策照常。"宁可少一个候选，也绝不能因为 oracle 让对局失败"。
        """
        st = _ORACLE
        st["pending"] = False
        # 2026-10-28：安慰剂臂 —— 走同一条通道，但候选由**本地**生成（不查 oracle、不起子进程）。
        #   默认 placebo=False ⇒ 这一支直接跳过，下面的代码与改动前逐字相同。
        if st["placebo"]:
            _placebo_step()
            return
        if st["off"] or st["obj"] is None:
            return
        # 2026-10-29：主动重启（`AZAI_ORACLE_RESTART_EVERY=N`；默认 0 ⇒ 这段一行都不进，
        #   行为与改动前逐位相同）。用 `asks`（= 已问过的回合数）当计数，N 回合换一个新进程。
        _re = int(st.get("restart_every", 0) or 0)
        if _re > 0 and int(st["asks"]) > 0 and int(st["asks"]) % _re == 0:
            _oracle_restart("oracle", st, facade)
            if st["obj"] is None:
                return
        st["asks"] += 1
        _opp = st["pending_opp"]
        st["pending_opp"] = None       # 一回合一次，绝不复用上一回合的块
        rnd = int(facade.round_index)
        _tries = 0
        while True:
            t0 = time.monotonic()
            try:
                if int(player) == 1:
                    ops = st["obj"].ask(_oracle_state(st, facade), opponent_ops=_opp)
                else:
                    ops = st["obj"].ask(_oracle_state(st, facade))
                break
            except Exception as exc:  # noqa: BLE001
                _dt = (time.monotonic() - t0) * 1000.0
                st["ms"].append(_dt)
                name = type(exc).__name__
                if "Timeout" in name:
                    st["timeouts"] += 1
                else:
                    st["dead"] += 1
                st["err"] = f"{name}: {exc}"
                # 读端已错位/影子自认终局 ⇒ 旧行为是**本局永久停用**；2026-10-29 起改成
                #   "重建会话 + 回放追赶 + 重试本回合"（次数上限 `rebuild_max`）。
                try:
                    st["obj"].close()
                except Exception:  # noqa: BLE001
                    pass
                _log(f"[oracle] ⚠️ 第 {rnd} 回合查询失败（{_dt:.0f}ms）"
                     f"⇒ {'尝试重建会话' if int(st.get('rebuild_max', 0)) > 0 else '本局停用注入'}: "
                     f"{name}: {exc}")
                if not _oracle_rebuild_try("oracle", st, facade, why="（崩溃后追赶）"):
                    st["obj"] = None
                    st["last_bundle"] = None
                    if hasattr(mcts, "set_extra_bundles"):
                        mcts.set_extra_bundles(None)
                    if int(st.get("rebuild_max", 0)) <= 0:
                        _log(f"[oracle] ⇒ **本局停用注入**，决策照常（AZAI_ORACLE_REBUILD_MAX=0）")
                    return
                _tries += 1
                if _tries > 2:      # 同一回合最多重试 2 次（防止"崩-重建-崩"死循环）
                    _log(f"[oracle] ⚠️ 第 {rnd} 回合重试 {_tries - 1} 次仍失败 ⇒ 本回合不注入")
                    st["last_bundle"] = None
                    if hasattr(mcts, "set_extra_bundles"):
                        mcts.set_extra_bundles(None)
                    return
                _log(f"[oracle] ↻ 第 {rnd} 回合：会话已重建，重试本回合（第 {_tries} 次）")
        dt_ms = (time.monotonic() - t0) * 1000.0
        st["ms"].append(dt_ms)
        st["ok"] += 1
        if not ops:
            st["hold"] += 1
        # 2026-10-29（父任务）：`AZAI_ORACLE_SKIP_HOLD=1` ⇒ 它给 HOLD 时**不注入**，只计数。
        #   为什么需要：冠军 oracle 恒给 `()`、亚军注入里 84.5% 也是 `()` ⇒ "注入有效"里 HOLD 的
        #   功劳必须被单独控制住（两臂都开 `AZAI_ALWAYS_HOLD`），两臂之差才是"**非 HOLD 外部动作**"
        #   的净价值。默认 0 ⇒ 与改动前逐字相同。
        if (not ops) and int(st.get("skip_hold", 0)):
            st["skipped_hold"] = int(st.get("skipped_hold", 0)) + 1
            st["last_bundle"] = None
            st["round_log"] = None
            st["round_new"] = st["round_dup"] = None
            if hasattr(mcts, "set_extra_bundles"):
                mcts.set_extra_bundles(None)     # 让 `_ahold_step` 只留 HOLD 那一个候选
            return
        if int(_env("AZAI_ORACLE_VERBOSE", "0")):
            _log(f"[oracle] r{facade.round_index} {dt_ms:.0f}ms -> "
                 f"{[(int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops]}")
        bundle = _oracle_bundle(ops)
        if bundle is None:                       # 有非法 op ⇒ 整条丢弃
            st["illegal"] += 1
            st["last_bundle"] = None
            mcts.set_extra_bundles(None)
            return
        st["inject"] += 1
        st["pending"] = True
        st["last_bundle"] = bundle
        # 2026-10-28：登记本回合的"待打印明细"（action 在这里就有；new/dup 要等搜索后才知道，
        # 由 `_oracle_after_search` 补；最终 `chosen` 由 `_oracle_log_round` 在出口补并打印）。
        st["round_log"] = {"round": int(facade.round_index), "action": tuple(bundle)}
        st["round_new"] = None
        st["round_dup"] = None
        mcts.set_extra_bundles([bundle])         # ← 注入通道（HOLD = 空 bundle `()`）

    def _oracle_after_search() -> None:
        """搜索结束：统计「去重后真正加进去几个」+「是不是选中了它」。"""
        st = _ORACLE
        # 2026-10-28：守卫从 `st["off"] or not st["pending"]` 改成只看 `pending` —— **行为等价**：
        #   `off=True`（默认/安慰剂臂）时 `pending` 只在安慰剂分支里被置 True，A 臂（off=False）不受影响。
        #   原来那句 `st["off"]` 会让安慰剂臂永远跳过这里（new/dup/hit 全为 0）。
        if not st["pending"]:
            return
        st["pending"] = False
        added = int(getattr(mcts, "last_extra", 0) or 0)
        if added:
            st["new"] += 1
            st["round_new"], st["round_dup"] = 1, 0
        else:
            st["dup"] += 1                       # 与网络候选重复（最常见：双方都 HOLD）
            st["round_new"], st["round_dup"] = 0, 1
            # 2026-10-28：dup 里"亚军也只是 HOLD"的那部分单独计数 —— 它决定了注入的**增量信息**
            # 有多少（HOLD 与 HOLD 重复不带来任何新动作，只是"它也没招"）。
            if tuple(st["last_bundle"] or ()) == ():
                st["dup_hold"] += 1
            else:
                st["dup_nonhold"] += 1
        if added and bool(getattr(mcts, "last_chosen_is_extra", False)):
            st["hit_sel"] += 1                   # 老口径（= 改动前的 `hit`，跨版本可比）
            _ls = getattr(mcts, "last_scores", None)
            _n = int(len(_ls)) if _ls is not None else 0   # ⚠️ 不能写 `_ls or []`（ndarray 真值歧义）
            _log(f"[{_inj_tag()}] 🎯 注入命中: round={facade.round_index} "
                 f"{'安慰剂' if st['placebo'] else '亚军'}候选 {st['last_bundle']} "
                 f"被搜索选中（本决策共 {_n} 个候选，其中额外 1 个）")

    def _oracle_log_round(final_ops) -> None:
        """**每决策一条**注入明细（2026-10-28）：事后可以统计"一致率"，不必开 VERBOSE。

        格式（stdout 绝不写，一律 stderr）：
            [oracle] inject round=<r> action=<...> dup=<0/1> new=<0/1> chosen=<...> hit=<0/1>
        * `action` = 亚军这一手（`()` = HOLD）；`chosen` = 我们**最终真的发出去**的 op 序列
          （同一个格式；`()` = 我们这回合 HOLD）—— 注意它已经过执行层过滤/闸门，
          所以这条日志能直接看出"我们改了亚军那一手没有"；
        * `dup` / `new` 互斥：`dup=1` = 亚军这一手与**已有候选**重复（没有新增候选），
          `new=1` = 它是**新增候选**；搜索没跑（`AZAI_FORCE_UPG` 早退）时两者为 `NA`；
        * `hit = int(chosen == action)`（**动作一致率**的分子）。

        整行只在 `AZAI_ORACLE_INJECT` / `AZAI_ORACLE_PLACEBO` 非空、且本回合**真的注入了候选**时出现
        ⇒ 默认零日志。标签随 `_inj_tag()` 变（`[oracle]` / `[placebo]`），字段与格式**逐字相同**
        ⇒ 两个臂的注入统计可以用同一个解析器读。
        """
        st = _ORACLE
        info = st.get("round_log")
        if not info:
            return
        st["round_log"] = None
        act = info["action"]
        chosen = tuple((int(o.op_type), int(o.arg0), int(o.arg1)) for o in (final_ops or ()))
        same = int(chosen == act)
        is_new, is_dup = st.get("round_new"), st.get("round_dup")
        if same:
            st["hit"] += 1
            if is_new:
                st["hit_new"] += 1
            if is_dup:
                st["hit_dup"] += 1

        def _b(x) -> str:
            return "NA" if x is None else str(int(x))

        _log(f"[{_inj_tag()}] inject round={info['round']} action={_fmt_act(act)} "
             f"dup={_b(is_dup)} new={_b(is_new)} chosen={_fmt_act(chosen)} hit={same}")

    def _oracle_close() -> None:
        _oracle_close_module()

    _oracle_init()

    # ── E4：双 oracle（B 会话）+ only-extra 候选集（2026-10-28；**默认全关**）────────────
    #   设计见本文件头 `_ORACLE_B` / `_ONLY` 的定义处与 `docs/us_vs_champion_20261028.md` §5。
    #   默认（`AZAI_ORACLE_INJECT_B` 空、`AZAI_ONLY_EXTRA*` 不设）⇒ 下面这些函数一个都不会被调用，
    #   唯一多出来的开销是 `decide()` 里的一个 `if _ORACLE_B["off"]`（恒 True）。
    def _oracle_b_init() -> None:
        """按 `AZAI_ORACLE_INJECT_B` 决定要不要开第二个会话。**默认空 = 什么都不做。**"""
        exe = _env("AZAI_ORACLE_INJECT_B", "").strip()
        if not exe:
            return
        st = _ORACLE_B
        st["seed"], st["player"] = int(seed), int(player)
        try:
            st["timeout"] = float(_env("AZAI_ORACLE_TIMEOUT_B", _env("AZAI_ORACLE_TIMEOUT", "20")))
        except ValueError:
            st["timeout"] = 20.0
        # 2026-10-29：B 会话也吃同一套重建/重启开关（各自独立计数）
        try:
            st["rebuild_max"] = max(0, int(_env("AZAI_ORACLE_REBUILD_MAX_B",
                                                _env("AZAI_ORACLE_REBUILD_MAX", "6")) or 0))
        except ValueError:
            st["rebuild_max"] = 6
        try:
            st["restart_every"] = max(0, int(_env("AZAI_ORACLE_RESTART_EVERY_B",
                                                  _env("AZAI_ORACLE_RESTART_EVERY", "0")) or 0))
        except ValueError:
            st["restart_every"] = 0
        try:
            st["skip_hold"] = 1 if int(_env("AZAI_ORACLE_SKIP_HOLD_B",
                                            _env("AZAI_ORACLE_SKIP_HOLD", "0")) or 0) else 0
        except ValueError:
            st["skip_hold"] = 0
        st["off"] = True
        if not hasattr(mcts, "set_extra_bundles"):
            _log("[oracle-b] ⚠️ 搜索器没有额外候选通道（只有 flat/unified 有）⇒ 不启用 B 会话")
            return
        try:
            _exe = Path(exe)
            if not _exe.is_absolute():
                # ⚠️ 与 A 会话同一条纪律：`/d/...` 形态会让 `is_absolute()` 判 False
                #    ⇒ 拼成 `<repo>/d/...` ⇒ Popen FileNotFoundError ⇒ **静默不启用**。
                _exe = (Path(MVS.REPO_ROOT) / _exe).resolve()
            _spec = importlib.util.spec_from_file_location(
                "runnerup_oracle_for_azai_b", _HERE / "runnerup_oracle.py")
            _mod = importlib.util.module_from_spec(_spec)
            assert _spec.loader is not None
            _spec.loader.exec_module(_mod)
            # ⚠️ 显式给 stderr 路径：默认名只含 (seed, player, pid)，A/B 两会话会撞同一个文件
            _stderr = (Path(MVS.REPO_ROOT) / "match_results" / "oracle_logs"
                       / f"oracle_b_seed{int(seed)}_p{int(player)}_{os.getpid()}.stderr.log")
            st["obj"] = _mod.RunnerUpOracle(
                seed=int(seed), player=int(player), exe=str(_exe),
                timeout=float(st["timeout"]), stderr_path=_stderr,
                verbose=bool(int(_env("AZAI_ORACLE_DEBUG", "0"))))
            st["mod"] = _mod            # 2026-10-29：重建用
            st["exe"] = str(_exe)
            st["off"] = False
            _log(f"[oracle-b] B 会话已启用（E4 的第二条外部动作）: exe={_exe} seed={seed} "
                 f"player={player} timeout={st['timeout']:g}s stderr={_stderr}"
                 f" 重建上限={st['rebuild_max']} 主动重启每={st['restart_every'] or '关'}")
        except Exception as exc:  # noqa: BLE001
            st["off"] = True
            st["obj"] = None
            st["err"] = f"{type(exc).__name__}: {exc}"
            _log(f"[oracle-b] ⚠️ 启用失败 ⇒ 本局没有 B 动作（A 会话/决策照常，绝不影响对局）: {st['err']}")

    def _oracle_bundle_b(ops):
        """B 会话回的操作 → bundle；**有一条非法 ⇒ None**（口径与 A 的 `_oracle_bundle` 相同）。

        故意**复制**而不是给 `_oracle_bundle` 加参数：A 臂（亚军注入，81.2% 那批）走的代码
        路径必须逐字不变，历史读数才继续可比。
        """
        accepted: list = []
        for op in ops:
            try:
                ok = bool(facade.can_apply_operation(int(player), op, pending=accepted))
            except Exception as exc:  # noqa: BLE001
                _log(f"[oracle-b] ⚠️ can_apply_operation 抛异常（按非法处理）: "
                     f"{type(exc).__name__}: {exc}")
                ok = False
            if not ok:
                _log(f"[oracle-b] ⚠️ 第 {facade.round_index} 回合 B 给的 op 非法 ⇒ **整条候选丢弃**："
                     f"{(int(op.op_type), int(op.arg0), int(op.arg1))}")
                return None
            accepted.append(op)
        _ORACLE_B["checked"] += len(ops)
        return tuple((int(o.op_type), int(o.arg0), int(o.arg1)) for o in accepted)

    def _oracle_b_ask():
        """问 B 会话一次（每回合**必问一次**，跳回合会让它读端永久错位）。失败 ⇒ None。

        2026-10-29：失败后不再"本局永久停用"，而是走与 A 相同的**重建 + 回放**（次数上限同源）。
        """
        st = _ORACLE_B
        if st["off"] or st["obj"] is None:
            return None
        _re = int(st.get("restart_every", 0) or 0)
        if _re > 0 and int(st["asks"]) > 0 and int(st["asks"]) % _re == 0:
            _oracle_restart("oracle-b", st, facade)
            if st["obj"] is None:
                return None
        st["asks"] += 1
        _opp = st["pending_opp"]
        st["pending_opp"] = None        # 一回合一次，绝不复用上一回合的块
        rnd = int(facade.round_index)
        _tries = 0
        while True:
            t0 = time.monotonic()
            try:
                if int(player) == 1:
                    ops = st["obj"].ask(_oracle_state(st, facade), opponent_ops=_opp)
                else:
                    ops = st["obj"].ask(_oracle_state(st, facade))
                break
            except Exception as exc:  # noqa: BLE001
                _dt = (time.monotonic() - t0) * 1000.0
                st["ms"].append(_dt)
                name = type(exc).__name__
                if "Timeout" in name:
                    st["timeouts"] += 1
                else:
                    st["dead"] += 1
                st["err"] = f"{name}: {exc}"
                try:
                    st["obj"].close()
                except Exception:  # noqa: BLE001
                    pass
                _log(f"[oracle-b] ⚠️ 第 {rnd} 回合 B 查询失败（{_dt:.0f}ms）"
                     f"⇒ {'尝试重建 B 会话' if int(st.get('rebuild_max', 0)) > 0 else '本局停用 B'}: "
                     f"{name}: {exc}")
                if not _oracle_rebuild_try("oracle-b", st, facade, why="（B 崩溃后追赶）"):
                    st["obj"] = None
                    st["last_bundle"] = None
                    return None
                _tries += 1
                if _tries > 2:
                    _log(f"[oracle-b] ⚠️ 第 {rnd} 回合 B 重试 {_tries - 1} 次仍失败 ⇒ 本回合没有 B")
                    st["last_bundle"] = None
                    return None
        dt_ms = (time.monotonic() - t0) * 1000.0
        st["ms"].append(dt_ms)
        st["ok"] += 1
        if not ops:
            st["hold"] += 1
        # 2026-10-29：B 会话也吃同一个开关（`AZAI_ORACLE_SKIP_HOLD`；默认 0 ⇒ 行为不变）。
        #   E4 里 B = 亚军；本开关在 E4 上不要开（那里要的就是"两个外部候选"）。
        if (not ops) and int(st.get("skip_hold", 0)):
            st["skipped_hold"] = int(st.get("skipped_hold", 0)) + 1
            st["last_bundle"] = None
            st["round_log"] = None
            return None
        if int(_env("AZAI_ORACLE_VERBOSE", "0")):
            _log(f"[oracle-b] r{facade.round_index} {dt_ms:.0f}ms -> "
                 f"{[(int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops]}")
        bundle = _oracle_bundle_b(ops)
        if bundle is None:
            st["illegal"] += 1
            st["last_bundle"] = None
            return None
        st["last_bundle"] = bundle
        st["round_log"] = {"round": int(facade.round_index), "action": tuple(bundle)}
        return bundle

    def _oracle_b_step() -> None:
        """搜索**前**：A 走原路径、B 紧随其后，两条外部动作**一起**写进额外候选通道。

        与 `_oracle_step()` 的关系：A 的那次 `set_extra_bundles([bundle])` 已经写完，
        这里**覆盖**成 `[A 的动作, B 的动作]`（只多一次 set 调用，A 的查询/统计/日志都不动）。
        """
        _oracle_step()                     # ← 原路径（未启用 = no-op）
        b = _oracle_b_ask()
        if b is None:
            return                         # B 未启用 / 失败 ⇒ 保留 A 写进通道的那份
        base: list = []
        if (not _ORACLE["off"]) and _ORACLE.get("pending") and _ORACLE.get("last_bundle") is not None:
            base.append(tuple(_ORACLE["last_bundle"]))
        base.append(tuple(b))
        _ORACLE_B["inject"] += 1
        mcts.set_extra_bundles(base)

    def _oracle_b_set_opp(round_index: int, ops) -> None:
        """把「对手本回合的操作块」也登记给 B（与 A 的 `_oracle_set_opp` 同语义）。"""
        st = _ORACLE_B
        if st["off"] or st["obj"] is None:
            return
        try:
            # 2026-10-29：与 A 同一套回合号偏移（默认 0 ⇒ 逐字不变）
            st["obj"].set_opponent_ops(
                int(round_index) - int(st.get("relabel_off", 0) or 0), list(ops))
        except Exception as exc:  # noqa: BLE001
            _log(f"[oracle-b] ⚠️ 登记对手操作块失败（忽略，按空过处理）: {type(exc).__name__}: {exc}")

    def _oracle_b_log_round(final_ops) -> None:
        """每决策一条 B 的明细（格式与 A 的 `inject round=` 同构；默认零日志）。"""
        st = _ORACLE_B
        info = st.get("round_log")
        if not info:
            return
        st["round_log"] = None
        chosen = tuple((int(o.op_type), int(o.arg0), int(o.arg1)) for o in (final_ops or ()))
        same = int(chosen == info["action"])
        if same:
            st["hit"] += 1
        _log(f"[oracle-b] inject round={info['round']} action={_fmt_act(info['action'])} "
             f"chosen={_fmt_act(chosen)} hit={same}")

    def _xor_log_round(final_ops) -> None:
        """`[xor]` 双 oracle 逐回合记录（E4/E4R；E2 只有 A ⇒ B 侧全 `NA`）。**默认零日志**。

        ⚠️ **必须在 `_oracle_log_round()` / `_oracle_b_log_round()` 之前调用**：它只**读**
        两个会话的 `round_log`（那两行会把它们置 None），因此 A/B 的日志与行为一个字节都不变。
        `dup_c` / `dup_r` 的口径 = 该动作**是否已在网络候选集里**（`mcts.last_root.bundles`）：
        `1` = 我们本来就有（不带来新信息），`0` = 真·增量。
        """
        a_info = _ORACLE.get("round_log")
        b_info = _ORACLE_B.get("round_log")
        if a_info is None and b_info is None:
            return
        a = tuple(tuple(int(x) for x in op) for op in (a_info["action"] if a_info else ()))
        b = tuple(tuple(int(x) for x in op) for op in (b_info["action"] if b_info else ()))
        chosen = tuple((int(o.op_type), int(o.arg0), int(o.arg1)) for o in (final_ops or ()))
        rnd = int((a_info or b_info).get("round", 0))
        try:
            _root = getattr(mcts, "last_root", None)
            net = {tuple(tuple(int(x) for x in op) for op in k)
                   for k in (getattr(_root, "bundles", None) or ())}
        except Exception:  # noqa: BLE001
            net = set()
        has_a, has_b = a_info is not None, b_info is not None
        x = _XOR
        x["rounds"] += 1
        x["have_a"] += int(has_a)
        x["have_b"] += int(has_b)
        if has_a:
            x["champ_hold"] += int(a == ())
            x["champ_ops"] += len(a)
            x["champ_new"] += int(a not in net)
            x["chosen_eq_champ"] += int(chosen == a)
        if has_b:
            x["ru_hold"] += int(b == ())
            x["ru_ops"] += len(b)
            x["ru_new"] += int(b not in net)
            x["chosen_eq_ru"] += int(chosen == b)
        if has_a and has_b:
            _agree = int(a == b)
            x["agree"] += _agree
            x["both_hold"] += int(a == () and b == ())
            if not _agree:
                x["disagree"] += 1
                x["eq_champ_when_disagree"] += int(chosen == a)
        elif has_a:
            # E2（单会话）：把 agree 记成 NA，不污染分母
            pass
        if has_a and has_b:
            x["chosen_eq_neither"] += int(chosen != a and chosen != b)
        elif has_a:
            x["chosen_eq_neither"] += int(chosen != a)

        def _b3(v) -> str:
            # ⚠️ 只有"该会话本回合没有动作"才是 NA；`0` 是**有效读数**（= 不在网络候选里 = 新信息）
            return "NA" if v is None else str(int(v))

        _log(f"[xor] round={rnd} champ={_fmt_act(a) if has_a else 'NA'} "
             f"ru={_fmt_act(b) if has_b else 'NA'} "
             f"agree={_b3(int(a == b) if (has_a and has_b) else None)} "
             f"dup_c={_b3(int(a in net) if has_a else None)} "
             f"dup_r={_b3(int(b in net) if has_b else None)} "
             f"chosen={_fmt_act(chosen)} "
             f"hit_c={_b3(int(chosen == a) if has_a else None)} "
             f"hit_r={_b3(int(chosen == b) if has_b else None)}")

    def _dscore_log_round(final_ops) -> None:
        """`[dscore]` 逐决策记录**两个额外候选在本评估器下的分数差**（2026-10-29；用户要求）。

        ⚠️ 它**只读**（`mcts.last_scores` / `mcts.last_root.bundles` / `mcts.last_extra`），
        不改任何决策；默认（没有额外候选）第一行就 return ⇒ **零日志、零影响**。

        口径（与 `flat_rollout_greedy.search()` 的数组对齐关系一致）：
          * `last_scores[i]` 对**搜索结果的** `bundles[i]`：**网络候选在前**，额外候选追加在**末尾**；
            `only_extra` 时网络候选一个都不进 ⇒ `n_net=0`，`bundles` 全是额外候选。
          * ⚠️ **不能**用 `mcts.last_root.bundles` 当候选表：那是 MCTS 根的网络候选（`K` 条、
            不含 HOLD），与 `last_scores` **不对齐**（2026-10-29 第一版就踩了这个，读数全是
            `d=NA pick=na`）。额外候选的唯一可靠来源是喂给通道的 `mcts.extra_bundles`，
            去重规则与 `flat_rollout_greedy._collect_extras()` 逐字一致（按顺序去重、保留 HOLD）。
          * `d`   = 两个额外候选的 `|Δscore|`（`n_extra≥2` 才有；`HOLD` 也是一个候选）；
          * `dvn` = `max(额外候选分) − max(网络候选分)` —— E2（单会话）时这就是
            "注入候选 vs 我们自己候选"的分数差（父任务要求的口径）；
          * `pick` = 我们**最终发出去**的那一手落在 `hi`（额外候选里分最高的）/`lo`/`net`（网络候选）
            /`na`。E4 应≈全 `hi`（评估器按分数挑），E4-rand 应≈一半 `hi` ⇒ **臂是否真的不同**的正向对照。
        """
        if _ORACLE.get("off", True) and _ORACLE_B.get("off", True):
            return                       # 两个会话都关 ⇒ 不可能有额外候选（默认路径）
        try:
            n_extra = int(getattr(mcts, "last_extra", 0) or 0)
            _ls = getattr(mcts, "last_scores", None)
            if n_extra <= 0 or _ls is None or len(_ls) < n_extra:
                return
            # 额外候选 = 通道里那串（按顺序去重，与 `_collect_extras` 同规则）
            ext, _seen = [], set()
            for b in (getattr(mcts, "extra_bundles", None) or ()):
                k = tuple(tuple(int(x) for x in op) for op in (b or ()))
                if k in _seen:
                    continue
                _seen.add(k)
                ext.append(k)
            if len(ext) < n_extra:        # 通道被清过/不一致 ⇒ 宁可不记，也不记错的
                return
            ext = ext[:n_extra]
            # 网络候选集合（只用来判 `pick=net` 的归属；`last_root.bundles` 就是那份 K 条网络束）
            _root = getattr(mcts, "last_root", None)
            _seen_net = {tuple(tuple(int(x) for x in op) for op in (b or ()))
                         for b in (getattr(_root, "bundles", None) or ())}
            only_extra = bool(getattr(mcts, "only_extra", False))
            n_net = 0 if only_extra else max(0, len(_ls) - n_extra)
            ext_s = [float(_ls[i]) for i in range(n_net, n_net + n_extra)]
            net_s = [float(_ls[i]) for i in range(n_net)]
            chosen = tuple((int(o.op_type), int(o.arg0), int(o.arg1)) for o in (final_ops or ()))
            d = _DS
            d["rounds"] += 1
            dd = None
            pick = "only1"
            if n_extra >= 2:
                d["n2"] += 1
                dd = max(ext_s) - min(ext_s)
                d["ds"].append(dd)
                hi = max(range(len(ext_s)), key=lambda i: ext_s[i])
                if chosen == ext[hi]:
                    d["pick_hi"] += 1
                    pick = "hi"
                elif chosen in {ext[j] for j in range(len(ext)) if j != hi}:
                    d["pick_lo"] += 1
                    pick = "lo"
                elif chosen in _seen_net:
                    d["pick_net"] += 1
                    pick = "net"
                else:
                    d["pick_na"] += 1
                    pick = "na"
            else:
                d["n1"] += 1
                # 只有一个候选 ⇒ 没有"选哪个"的问题（两臂按构造相同）⇒ `pick` 记 `only1`，
                # **不**算进 pick_hi/pick_lo（否则 `pick_hi/n2` 这个正向对照会被 n1 撑爆）。
                pick = "only1"
            dvn = (max(ext_s) - max(net_s)) if net_s else None
            if dvn is not None:
                d["dvn"].append(dvn)
            _d_txt = "NA" if dd is None else "%.4f" % dd
            _dvn_txt = "NA" if dvn is None else "%+.4f" % dvn
            _s_txt = ", ".join("%+.4f" % v for v in ext_s)
            _net_txt = "NA" if not net_s else "%+.4f" % max(net_s)
            _log(f"[dscore] round={int(facade.round_index)} n_extra={n_extra} n_net={n_net} "
                 f"d={_d_txt} dvn={_dvn_txt} pick={pick} s_extra=[{_s_txt}] s_netmax={_net_txt}")
        except Exception as exc:  # noqa: BLE001
            _DS["err"] = f"{type(exc).__name__}: {exc}"

    def _only_extra_enable() -> None:
        """读 `AZAI_ONLY_EXTRA` / `AZAI_ONLY_EXTRA_RANDOM`（**不设 = 0 = 关**）。"""
        st = _ONLY
        raw_on = _env("AZAI_ONLY_EXTRA", "0").strip()
        raw_rd = _env("AZAI_ONLY_EXTRA_RANDOM", "0").strip()
        on = raw_on not in ("", "0")
        rd = raw_rd not in ("", "0")
        if not (on or rd):
            return
        if not on:
            # `AZAI_ONLY_EXTRA_RANDOM=1` **蕴含** only-extra（"同一候选集里随机挑一个"）
            on = True
            _log("[onlyextra] AZAI_ONLY_EXTRA_RANDOM=1 ⇒ 自动把 AZAI_ONLY_EXTRA 当 1（同一候选集）")
        if not hasattr(mcts, "set_only_extra"):
            _log(f"[onlyextra] ⚠️ {type(mcts).__name__} 不支持 only_extra（只有 flat 有）⇒ 不启用")
            return
        try:
            st["seed"] = int(_env("AZAI_ONLY_EXTRA_SEED", _env("AZAI_FLAT_SEED", "0")) or 0)
        except ValueError:
            st["seed"] = 0
        mcts.set_only_extra(True)
        st["on"], st["random"] = True, bool(rd)
        _log(f"[onlyextra] 候选集=**只保留**额外候选: seed={seed} player={player} "
             f"random={int(rd)} rng_seed={st['seed']} search={type(mcts).__name__} "
             f"K={getattr(mcts, 'k', '?')}（网络候选不进候选集；其余逻辑不变）")

    def _only_extra_rng(dec_n: int):
        """随机挑的随机流：`sha256("onlyextra|<seed>|<dec_n>")` ⇒ 与 rollout 种子**完全独立**、
        同一 (seed, 决策序号) 恒得同一个 `Random` ⇒ 可复现（不依赖时间/哈希随机化）。"""
        h = hashlib.sha256(f"onlyextra|{int(_ONLY['seed'])}|{int(dec_n)}".encode()).digest()
        return random.Random(int.from_bytes(h[:8], "big"))

    def _only_extra_pick(res):
        """`AZAI_ONLY_EXTRA_RANDOM=1` ⇒ **不看分数**，在候选集里随机挑一个（默认原样返回）。

        候选集 = 本决策写进通道的额外候选（= A/B 两条外部动作；HOLD 也是一个候选）。
        只改 `chosen_bundle` / `chosen_index`，**不动** `last_scores`（诊断读数保留评估器的意见）。
        """
        if not _ONLY["random"]:
            return res
        try:
            seen, cands = set(), []
            for b in (getattr(mcts, "extra_bundles", None) or ()):
                k = tuple(tuple(int(x) for x in op) for op in (b or ()))
                if k in seen:
                    continue
                seen.add(k)
                cands.append(k)
        except Exception as exc:  # noqa: BLE001
            _ONLY["err"] = f"{type(exc).__name__}: {exc}"
            return res
        if not cands:
            return res
        dec_n = int(STATS.get("dec_n", 0))
        pick = cands[_only_extra_rng(dec_n).randrange(len(cands))]
        _ONLY["decided"] += 1
        _ONLY["ncand"][len(cands)] = int(_ONLY["ncand"].get(len(cands), 0)) + 1
        _ONLY["pick"][repr(pick)] = int(_ONLY["pick"].get(repr(pick), 0)) + 1
        try:
            res.chosen_bundle = pick
            res.chosen_index = list(res.bundles).index(pick)
        except ValueError:
            res.chosen_index = -1
        _log(f"[onlyextra] 🎲 round={facade.round_index} n={dec_n} cands={len(cands)} "
             f"picked={_fmt_act(pick)}（随机挑，不看评估器分数）")
        return res


    # ── 升级候选进束（2026-10-28；`AZAI_UPG_BUNDLES=N`，**默认 0 = 完全关闭**）───────────
    #   背景（`docs/why_no_upgrade_20261028.md`）：现役配置下整束 48 个候选被钉在同一个类上
    #   （`bundle_mcts.py:870` 的 `pinned=[_ccls]*3`），那个类 100% ∈ {类 0, 类 16}；类 0
    #   数学上不可能产出升级（`decoder.py:452-456`）、类 16 走降级分支 ⇒ 我方升级 0/3175 op，
    #   尽管 42.3% 的回合**合法且买得起**（22368 个 op 全没发）。
    #   本开关**只补候选、不改搜索算法**：把最多 N 条单 op 升级候选 `((12, tower_id, target),)`
    #   走**已有的**额外候选通道 `set_extra_bundles`（`flat_rollout_greedy.py:417-453`，注入/
    #   安慰剂在用的那条）塞进去，让它们和另外 48 条**同台**被 flat rollout 打分。
    #   ⚠️ 与 `AZAI_MC_UPG` 的区别（别混）：那个是"MC 仲裁选中后**直接发这一手**、绕过 48 束
    #      评估器"（`az_bridge_ai.py:1381-1384`），三次自对弈 A/B 都更弱（23.4/18.8/31.2%）；
    #      这个是"**成为候选**、由同一个评估器裁决"。设计/读数见 `docs/upgrade_bundles_20261028.md`。
    #   N=0（默认）⇒ 三个钩子函数第一行就 return：不枚举、不 dry-run、不碰通道、不打日志。

    def _upg_enable() -> None:
        """读 `AZAI_UPG_BUNDLES`。**不设 = 0 = 关**；非法值/无通道 ⇒ 也按关处理（绝不让它改默认行为）。"""
        raw = _env("AZAI_UPG_BUNDLES", "0")
        try:
            n = int(str(raw).strip() or 0)
        except ValueError:
            _log(f"[upg] ⚠️ AZAI_UPG_BUNDLES={raw!r} 不是整数 ⇒ 按 0（关闭）处理")
            n = 0
        if n < 0:
            _log(f"[upg] ⚠️ AZAI_UPG_BUNDLES={n} < 0 ⇒ 按 0（关闭）处理")
            n = 0
        if n > 0 and not hasattr(mcts, "set_extra_bundles"):
            _log(f"[upg] ⚠️ {type(mcts).__name__} 没有额外候选通道（只有 flat/unified 有）⇒ 不启用升级候选")
            n = 0
        _UPG["n"] = int(n)
        if n > 0:
            _log(f"[upg] 升级候选已启用: 每回合最多 {n} 条（seed={seed} player={player} "
                 f"K={getattr(mcts, 'k', '?')} search={type(mcts).__name__}；"
                 f"通道=set_extra_bundles（与注入同一入口））")

    def _upg_bundles(n_max: int) -> list:
        """枚举"合法且买得起"的升级 bundle `((12, tower_id, target),)`，**确定性**取前 n_max 条。

        * 顺序 = `tower_id` 升序 × `TOWER_UPGRADE_TREE` 的目标顺序（**塔外层、目标内层**，与
          `bundle_mcts.py:784-820` 的 `for _tw: for _tgt:` 同构；**不用 RNG** ⇒ 逐局可复现）；
        * 合法性 = `facade.can_apply_operation(...)`（**只读** dry-run；与 `_oracle_bundle:1022`
          同一个检查器：它复刻冷路径的金币/塔位/营地上限 ⇒ "买得起"天然含在里面，
          `docs/why_no_upgrade_20261028.md` §4.3 的"方法 B"就是它，与"买得起"口径**同值**）；
        * 单 op ⇒ `pending` 恒空，不存在"半块非法"（那是 `AZAI_NO_DOWNGRADE` 多 op 块的问题）；
        * 异常一律 **fail-closed**（当非法丢掉）：宁可本回合少一条候选，也绝不发没验过的 op。
        """
        from SDK.utils.constants import TOWER_UPGRADE_TREE as _TREE
        pl = int(player)
        out: list = []
        try:
            # 便宜的预筛：升级价最低 60（二级）/ 200（三级）⇒ 金币 < 60 时**一条都不可能**，
            #   直接省掉 towers×targets 次 dry-run（`game_state_facade.py:304-310`）。
            if int(facade.coins[pl]) < 60:
                return out
            tws = sorted(facade.towers_of(pl), key=lambda z: int(z.tower_id))
        except Exception as exc:  # noqa: BLE001
            _UPG["err"] = f"towers_of: {type(exc).__name__}: {exc}"
            return out
        for tw in tws:
            for tgt in (_TREE.get(int(tw.tower_type), ()) or ()):
                op = (12, int(tw.tower_id), int(tgt))
                try:
                    ok = bool(facade.can_apply_operation(
                        pl, Operation(OperationType(op[0]), op[1], op[2])))
                except Exception as exc:  # noqa: BLE001
                    _UPG["err"] = f"can_apply: {type(exc).__name__}: {exc}"
                    ok = False
                if not ok:
                    continue
                out.append((op,))
                if len(out) >= int(n_max):
                    return out
        return out

    def _upg_step() -> None:
        """搜索**前**：把最多 N 条升级候选写进额外候选通道（N=0 ⇒ 第一行返回，零影响）。"""
        st = _UPG
        if st["n"] <= 0:
            return
        st["decided"] += 1
        rnd = int(facade.round_index)
        # ① 先摘掉**自己上一回合**写进通道的那份：`extra_bundles` 是**跨回合持久**的字段
        #    （`search()` 读 `self.extra_bundles`，`set_extra_bundles` 只在调用时改它）⇒ 不摘就会
        #    逐回合累加。⚠️ 只摘"我写的那些"，注入臂（A 臂/安慰剂臂）那份必须留住（合并而非覆盖）。
        try:
            prev = list(getattr(mcts, "extra_bundles", None) or ())
        except Exception:  # noqa: BLE001
            prev = []
        mine_prev = tuple(st.get("set_prev") or ())
        base = [b for b in prev
                if tuple(tuple(int(x) for x in op) for op in b) not in mine_prev]
        st["mine"], st["set_prev"] = None, ()
        bundles = _upg_bundles(int(st["n"]))
        if bundles:
            # ② 再用独立校验器复核一遍（`_block_legal_gate`：`MVS.check_ops`，与 `decide_final`
            #    自检同一个、fail-closed）。⚠️ **必须逐条（每条 bundle 单独）**调用，不能把 4 条
            #    当成一个块：它们对评估器是**互斥的备选**（同回合只会发一条），而闸门的 `pending=`
            #    语义是"同一回合先后执行" ⇒ 整块查会把"同一座塔升不同目标"误判成
            #    `tower_already_used_in_round`（2026-10-28 冒烟实测：4 条全被误拦）。
            _kept: list = []
            for b in bundles:
                bad = _block_legal_gate(facade, int(player),
                                        [Operation(OperationType(12), b[0][1], b[0][2])])
                if bad:
                    st["gate_drop"] += 1
                    _log(f"[upg] ⚠️ round={rnd} 闸门拦下 1 条升级候选 {b[0]} ⇒ 丢掉"
                         f"（bad={list(bad)[:1]}）")
                    continue
                _kept.append(b)
            bundles = _kept
        if not bundles:
            st["no_legal"] += 1
            mcts.set_extra_bundles(base if base else None)   # 注入臂的候选原样保留
            _log(f"[upg] round={rnd} 加入 0 条升级候选 targets=[]")
            return
        keys = tuple(tuple(tuple(int(x) for x in op) for op in b) for b in bundles)
        st["rounds"] += 1
        st["offers"] += len(bundles)
        st["kinds"][len(bundles)] = int(st["kinds"].get(len(bundles), 0)) + 1
        for b in bundles:
            k = int(b[0][2])
            st["targets"][k] = int(st["targets"].get(k, 0)) + 1
        _log(f"[upg] round={rnd} 加入 {len(bundles)} 条升级候选 "
             f"targets=[{', '.join(f'(12,{b[0][1]},{b[0][2]})' for b in bundles)}]")
        mcts.set_extra_bundles(list(base) + list(bundles))   # ← 与注入**同一条**通道
        st["mine"] = keys
        st["set_prev"] = keys

    def _upg_after_search(res) -> None:
        """搜索**后**：这些升级候选有没有被评估器选中 + 实测每回合的 `(n_cand, n_rollouts)`。"""
        st = _UPG
        mine = st.get("mine")
        if not mine:
            return
        st["mine"] = None
        try:
            chosen = tuple(tuple(int(x) for x in k) for k in (res.chosen_bundle or ()))
        except Exception:  # noqa: BLE001
            chosen = ()
        hit = int(chosen in mine)
        st["sel"] += hit
        added = int(getattr(mcts, "last_extra", 0) or 0)
        st["added"] += added
        if hit:
            for _op in chosen:
                st["sel_targets"][int(_op[2])] = int(st["sel_targets"].get(int(_op[2]), 0)) + 1
        _ls = getattr(mcts, "last_scores", None)
        n_cand = int(len(_ls)) if _ls is not None else 0
        n_roll = int(getattr(mcts, "last_rollouts", 0) or 0)   # flat 里**实际用的** n_rollouts
        _bk = f"{n_cand}/{n_roll}"
        st["budget"][_bk] = int(st["budget"].get(_bk, 0)) + 1
        _log(f"[upg] sel round={int(facade.round_index)} hit={hit} mine={len(mine)} extra={added} "
             f"n_cand={n_cand} n_rollouts={n_roll} chosen={_fmt_act(chosen)}")

    def _upg_count_sent(final_ops) -> None:
        """出口：最终**真的发出去**的 op 里数 `op=12`（执行层过滤/闸门之后；N=0 时零成本）。"""
        if _UPG["n"] <= 0:
            return
        try:
            _UPG["sent"] += sum(1 for o in (final_ops or ()) if int(o.op_type) == 12)
        except Exception:  # noqa: BLE001
            pass
        # 预算稀释的**实测**（每个决策都记，包括"本回合没塞升级"的）：`n_cand`/`n_rollouts`
        #   直接取 flat 自己算好/用掉的那两个量（`flat_rollout_greedy.py:521-523, 591`），不猜。
        try:
            _ls = getattr(mcts, "last_scores", None)
            _nc = int(len(_ls)) if _ls is not None else 0
            _nr = int(getattr(mcts, "last_rollouts", 0) or 0)
            _bk = f"{_nc}/{_nr}"
            _UPG["budget"][_bk] = int(_UPG["budget"].get(_bk, 0)) + 1
        except Exception:  # noqa: BLE001
            pass
        # 周期汇总（bridge 收尾时会直接 terminate 我们 ⇒ 主读数必须随对局落盘）
        if _UPG["decided"] and _UPG["decided"] % 50 == 0:
            _log(f"[upg] 统计: {_upg_stats_text()}")

    _upg_enable()
    # 2026-10-28：E4 的两个开关（默认关 ⇒ 各是一行 `os.environ.get`）
    _oracle_b_init()
    _only_extra_enable()

    # ── 2026-10-29：HOLD 恒为候选（`AZAI_ALWAYS_HOLD=1`，**默认关**；消融臂用）────────────
    #   背景（父任务 2026-10-29）：冠军 oracle 实际恒给 `()`（`force_pass_` 锁死），亚军注入臂
    #   8117 次注入里 84.5% 也是 HOLD ⇒ "注入有效"很可能只是"把 HOLD 补进候选集"这一个效应。
    #   本开关把这一个效应**单独**做出来：不需要任何 oracle，候选集里始终有一个空 bundle。
    #   默认不设 ⇒ `_AHOLD["on"]=False` ⇒ 两个钩子第一行就 return，**行为逐位不变**
    #   （证据：`_only_extra_zero_impact.sh` 那套 `RESULT` 逐字段比较仍适用）。
    def _ahold_enable() -> None:
        raw = _env("AZAI_ALWAYS_HOLD", "0").strip()
        if raw in ("", "0"):
            return
        if not hasattr(mcts, "set_extra_bundles"):
            _log(f"[ahold] ⚠️ {type(mcts).__name__} 没有额外候选通道 ⇒ 不启用 HOLD 恒为候选")
            return
        _AHOLD["on"] = True
        _log(f"[ahold] HOLD 恒为候选已启用（AZAI_ALWAYS_HOLD={raw!r}）: seed={seed} "
             f"player={player} search={type(mcts).__name__} K={getattr(mcts, 'k', '?')}"
             f"（空 bundle 走**已有**的额外候选通道；已存在时由去重跳过）")

    def _ahold_step() -> None:
        """搜索**前**：把 `()` 放进额外候选通道（与 `AZAI_UPG_BUNDLES`/注入同一条入口）。

        ⚠️ 通道**跨回合保留**（`_UPG` 那边专门有"下回合先摘掉"的处理）⇒ 这里先把上一回合自己放的
        `()` 摘掉再放，否则"这一回合的 HOLD 在不在候选里"就永远读到上一回合的答案。
        **判"在不在"不在这里做**（要看去重后的结果，见 `_ahold_after_search` 的 `last_extra`）。
        """
        if not _AHOLD["on"]:
            return
        try:
            base = [tuple(tuple(int(x) for x in op) for op in (b or ()))
                    for b in (getattr(mcts, "extra_bundles", None) or ())]
            base = [b for b in base if b != ()]        # 摘掉自己上一回合放的那份
            base.append(())
            mcts.set_extra_bundles(base)
        except Exception as exc:  # noqa: BLE001
            _AHOLD["err"] = f"{type(exc).__name__}: {exc}"

    def _ahold_after_search(res) -> None:
        """搜索**后**：统计"我们最终 HOLD 的比例"+"被架构剥夺的决策被拿回多少"+ 每 50 决策打一行。

        ⚠️ `added` 的口径必须用 **`mcts.last_extra`**（= 去重后**真的进候选集**的额外候选数）：
        通道里放没放 `()` 是**输入**，`last_extra` 才是"网络候选里本来有没有 HOLD"的**答案**。
        （第一版用"通道里原来有没有 `()`"判断，被跨回合保留的通道骗了，实测只有 2% —— 明显偏低。）
        """
        if not _AHOLD["on"]:
            return
        _AHOLD["dec"] = int(_AHOLD["dec"]) + 1
        try:
            added = int(getattr(mcts, "last_extra", 0) or 0) > 0
        except Exception:  # noqa: BLE001
            added = False
        if added:
            _AHOLD["added"] = int(_AHOLD["added"]) + 1
        try:
            is_hold = not (getattr(res, "chosen_bundle", None) or ())
        except Exception:  # noqa: BLE001
            is_hold = False
        if is_hold:
            _AHOLD["hold"] = int(_AHOLD["hold"]) + 1
            if added:
                _AHOLD["chose_after_add"] = int(_AHOLD["chose_after_add"]) + 1
            else:
                _AHOLD["present_kept"] = int(_AHOLD["present_kept"]) + 1
        elif not added:
            _AHOLD["present_moved"] = int(_AHOLD["present_moved"]) + 1
        if int(_AHOLD["dec"]) % 50 == 0:
            _log(f"[ahold] 累计: {_ahold_stats_text()}")

    _ahold_enable()          # 2026-10-29：读 `AZAI_ALWAYS_HOLD`（不设 = 关，零开销）

    def _probe_upgrade_once() -> None:
        """🔬 诊断（`AZAI_PROBE_UPG=1`）：**在真实对局的引擎环境里**问清"升级 op 到底能不能生效"。

        动机（2026-10-07 第 3 轮）：分侧统计显示我方 **从不升级**（196 局 0 次，对手 9.4/局），
        而我**离线**写的探针在重放状态上**连建塔都测不出效果** ⇒ 离线结论不可用（见 experiment_log 02:1x）。
        所以这里把探针搬进 AI：用**真实状态**、拿 **native 的返回值**、并同时做**建塔对照**。
        """
        if _env("AZAI_PROBE_UPG", "0") != "1" or STATS.get("_upg_probe_done"):
            return
        try:
            from SDK.backend.model import Operation as _Op
            from SDK.utils.constants import OperationType as _OT
            tws = facade.towers_of(int(player))
            if not tws or int(facade.coins[player]) < 60:
                return
            w = tws[0]
            _log(f"🔬 探针环境: round={STATS['rounds']} coins={list(facade.coins)} "
                 f"我方塔数={len(tws)} 首塔=({w.x},{w.y}) type={w.tower_type} level={w.level} hp={w.hp}")
            can = facade.can_apply_operation(int(player), _Op(_OT.UPGRADE_TOWER, int(w.x), int(w.y)))
            c = facade.clone()
            before = [(t.tower_id, int(t.level), int(t.hp)) for t in c.towers_of(int(player))]
            cb = list(c.coins)
            # ⚠️ 必须走 **facade** 方法（它会 `_cache.clear()`）；直接调 `_g.apply_operation_list` 会**绕过缓存失效**
            #    ⇒ 随后读 `towers_of/coins` 拿到的是**过期值**（2026-10-07 我就是这么被骗了一轮）。
            c.apply_operation_list(int(player), [_Op(_OT.UPGRADE_TOWER, int(w.x), int(w.y))])
            after = [(t.tower_id, int(t.level), int(t.hp)) for t in c.towers_of(int(player))]
            _log(f"🔬 升级 op(12,{w.x},{w.y}): can_apply={can} "
                 f"金币 {cb}->{list(c.coins)} 变化={'有' if before != after else '**无**'} {before} -> {after}")
            # 对照：同环境下建塔（应生效 ⇒ 证明探针方法本身有效）
            from my_ai.decoder import make_position_masks as _mpm
            pm = _mpm(facade, int(player), intent_decoding=True)
            ys, xs = (pm[0] > 0).nonzero()
            if len(ys):
                c2 = facade.clone()
                b2 = len(c2.towers_of(int(player))); cb2 = list(c2.coins)
                cost = facade.build_tower_cost(facade.tower_count(int(player)))
                can11 = facade.can_apply_operation(int(player), _Op(_OT.BUILD_TOWER, int(xs[0]), int(ys[0])))
                c2.apply_operation_list(int(player), [_Op(_OT.BUILD_TOWER, int(xs[0]), int(ys[0]))])
                _log(f"🔬 建塔对照 op(11,{int(xs[0])},{int(ys[0])}): cost={cost} can_apply={can11} "
                     f"塔数 {b2}->{len(c2.towers_of(int(player)))} 金币 {cb2}->{list(c2.coins)}")
            STATS["_upg_probe_done"] = True
        except Exception as exc:  # noqa: BLE001
            _log(f"🔬 探针异常: {type(exc).__name__}: {exc}")
            STATS["_upg_probe_done"] = True

    def decide() -> list:
        # 每回合**必做**：喂 oracle（保持会话同步）+ 把它的 ops 当额外候选注入。
        # 2026-10-28：`AZAI_ORACLE_INJECT_B` 非空（E4）⇒ 走双会话那条（A 的动作原样保留）；
        #   否则走**原来的** `_oracle_step()`（逐字不变，A 臂/基准臂的历史读数继续成立）。
        if _ORACLE_B["off"]:
            _oracle_step()
        else:
            _oracle_b_step()
        _probe_upgrade_once()
        # 🧪 强制升级实验（`AZAI_FORCE_UPG=1`）：只要有钱有塔，就**直接发 upgrade op**（不做搜索）
        #    目的：在**真实对局**里判定"我们发的升级 op 引擎到底认不认"（离线/克隆探针已证明不可靠）。
        if _env("AZAI_FORCE_UPG", "0") in ("1", "2"):
            try:
                _tws = facade.towers_of(int(player))
                if _tws and int(facade.coins[player]) >= 60:
                    _w = _tws[0]
                    if _env("AZAI_FORCE_UPG", "0") == "2":
                        # ✅ **正确形式**（2026-10-07 从 SDK 官方样例查明）：
                        #   `Operation(UPGRADE_TOWER, tower_id, int(target_type))`
                        #   —— 而 `(x, y)` 是**建塔**的参数；我第一版误用了 (x,y) ⇒ 引擎静默拒绝、塔等级永远 0。
                        from SDK.utils.constants import TOWER_UPGRADE_TREE as _TREE
                        _tgts = _TREE.get(_w.tower_type, ())
                        _tgt = int(_tgts[0]) if _tgts else 1
                        _ops = [Operation(OperationType(int(12)), int(_w.tower_id), int(_tgt))]
                    else:
                        _ops = [Operation(OperationType(int(12)), int(_w.x), int(_w.y))]
                    STATS["forced_upg"] = int(STATS.get("forced_upg", 0)) + 1
                    _log(f"🧪 强制升级(mode={_env('AZAI_FORCE_UPG', '0')}): round={STATS['rounds']} "
                         f"塔 id={_w.tower_id} ({_w.x},{_w.y}) type={_w.tower_type} level={getattr(_w,'level','?')} "
                         f"coins={facade.coins[player]} 发={[(int(o.op_type), o.arg0, o.arg1) for o in _ops]}")
                    return _ops
            except Exception as _exc:  # noqa: BLE001
                _log(f"🧪 强制升级异常: {type(_exc).__name__}: {_exc}")
        # ⬆️ 升级候选（`AZAI_UPG_BUNDLES=N`；**默认 0 ⇒ 这一行立刻返回、行为逐位不变**）：
        #    必须在 `mcts.search()` **之前**塞进通道（`search()` 才读 `self.extra_bundles`）。
        _upg_step()
        _ahold_step()          # 2026-10-29：`AZAI_ALWAYS_HOLD=1` ⇒ 候选集里恒有 HOLD（默认 no-op）
        from my_ai.az_intent.timing import T as _T
        with _T("search"):
            res = mcts.search(facade, player, temperature=temperature)
        # 2026-10-28（E4R）：`AZAI_ONLY_EXTRA_RANDOM=1` 时**不看分数**随机挑一个候选；
        #   默认（random=False）⇒ 原样返回，一个字节都不改。
        res = _only_extra_pick(res)
        _oracle_after_search()   # 注入统计（去重后是否真的加进去 / 是否被选中）
        _upg_after_search(res)   # 升级候选：有没有被**同一个**评估器选中 + 实测预算
        _ahold_after_search(res)  # 2026-10-29：HOLD 恒为候选臂的读数（默认 no-op）
        # 显式 op（升级）：class+position 通路表达不了 ⇒ 直接把 MC 选中的那个 op 发出去。
        _expl = getattr(mcts, 'mc_last_explicit', None)
        if _expl is not None:
            _log(f"⬆️ 显式 op（升级）: round={STATS['rounds']} op={_expl}")
            return [Operation(OperationType(int(_expl[0])), int(_expl[1]), int(_expl[2]))]
        _record_obs(res)
        chosen = res.chosen_bundle or ()
        ops = [Operation(OperationType(int(k[0])), int(k[1]), int(k[2])) for k in chosen]
        # 2026-10-28：下面四个"事后删 op"的诊断开关共用的闸门基准（过滤前 op 数 + 原块引用）。
        # 只有 `len(ops) != _n_raw`（真的删过东西）才复核整块合法性 ⇒ 开关全关时零成本、零行为变化。
        _n_raw = len(ops)
        _raw_ops = ops          # 只存引用（四个过滤块都是新建列表、不原地改）⇒ 不复制、不改行为
        # 🧪 诊断（`AZAI_NO_DOWNGRADE=1`，预注册 `prereg_20261007_no_downgrade.md`）：
        #   **在"实际执行"层去掉降级 op**（与"从候选菜单排除类 16"不同——那是选层，这是执行层消融）。
        #   依据：256 局败局签名里我方"升级+降级"在负局更高（Cliff's d = −0.302），而我们 10.8 次/局、从不升级
        #   ⇒ "建塔→降级→再建"的 churn 可能在自毁。去掉后若该回合没有别的 op，就等于 HOLD（返回 []）。
        if _env("AZAI_NO_DOWNGRADE", "0") == "1":
            _kept = [o for o in ops if int(o.op_type) != 13]
            if len(_kept) != len(ops):
                STATS["down_killed"] = int(STATS.get("down_killed", 0)) + (len(ops) - len(_kept))
                _log(f"🧪 抑制降级: round={STATS['rounds']} 去掉 {len(ops) - len(_kept)} 个"
                     f"（累计 {STATS['down_killed']}）；剩余={len(_kept)}")
            ops = _kept
            # 2026-10-27：**去掉降级后的剩余 op 必须重查**——降级是"退钱"操作，去掉它之后
            # 后面的建塔/升级可能就变成"钱不够"（非法）。
            # 2026-10-28：这里原来调的是 `_legal_check(..., "after_no_downgrade")`（只记录、不拦），
            # 现在**改由下面那道闸门统一处理**（同一个检查器，但会真的把整块丢掉）。
            # 为什么不保留这条独立自检：它会把"**没发出去**的 op"也计进 `ILLEGAL_STATS` ⇒
            # `illegal_total` 不再是"真的发出去了几个非法 op"（判据会失真）；
            # 闸门拦下的量由 `gate_dropped` 单独计，日志里有 `🛡️ [gate]` 原文。
        # 🧪 诊断（`AZAI_LIGHT_MIN_COINS=180`，预注册 `prereg_20261007_light_reserve.md`）：
        #   **闪电储备门槛**：金币 < 阈值时**不放闪电**（该回合等于 HOLD）⇒ 复现"手写储备规则"那条纪律
        #   （项目史上最大的一次准入增益 +13.3pp，机理 = 省下的钱变成塔）。只在**执行层**做，不动搜索/网络。
        _lmin = float(_env("AZAI_LIGHT_MIN_COINS", "0"))
        if _lmin > 0:
            _kept2 = [o for o in ops if not (int(o.op_type) == 21 and int(facade.coins[player]) < _lmin)]
            if len(_kept2) != len(ops):
                STATS["light_gated"] = int(STATS.get("light_gated", 0)) + (len(ops) - len(_kept2))
                _log(f"🧪 储备门槛: round={STATS['rounds']} 金币={int(facade.coins[player])} < {_lmin:g}"
                     f" ⇒ 去掉闪电（累计 {STATS['light_gated']}）；剩余={len(_kept2)}")
            ops = _kept2
        # 🧪 诊断（`AZAI_BUILD_MIN_COINS=τ`，预注册 `prereg_20261007_build_gate.md`）：
        #   **建塔花钱门槛**：金币 < τ 时不建塔（该回合等于 HOLD）⇒ 省下的钱留给闪电（闪电是承重的，见 L180）。
        #   依据：我方建塔 24.8 vs 对手 16.0；建塔数在**负局更多**（Cliff d = −0.148）；
        #   两端消融已知：建塔 0 → 0.5312、建塔 35.5 → 0.5156 ⇒ 现状 24.8 未必在峰上。
        _bmin = float(_env("AZAI_BUILD_MIN_COINS", "0"))
        if _bmin > 0:
            _kept3 = [o for o in ops if not (int(o.op_type) == 11 and int(facade.coins[player]) < _bmin)]
            if len(_kept3) != len(ops):
                STATS["build_gated"] = int(STATS.get("build_gated", 0)) + (len(ops) - len(_kept3))
                _log(f"🧪 建塔门槛: round={STATS['rounds']} 金币={int(facade.coins[player])} < {_bmin:g}"
                     f" ⇒ 去掉建塔（累计 {STATS['build_gated']}）；剩余={len(_kept3)}")
            ops = _kept3
        # 🧪 Phase 0 安全网（`AZAI_BAN_OPS`，逗号分隔 op 类型，如 "21" = 闪电）：
        #   掩码层禁类（`AZAI_BAN_CLASSES`）**本应**让被禁的类永远选不到；这里再兜一层，
        #   并且**每次触发都记数** —— 若触发数 > 0，说明掩码禁类不彻底（Phase 0 的验证读数）。
        _bano = _env("AZAI_BAN_OPS", "").strip()
        if _bano:
            _banned = {int(x) for x in _bano.split(",") if x.strip().isdigit()}
            _kept4 = [o for o in ops if int(o.op_type) not in _banned]
            if len(_kept4) != len(ops):
                STATS["banned_op_killed"] = int(STATS.get("banned_op_killed", 0)) + (len(ops) - len(_kept4))
                _log(f"🧪 安全网: round={STATS['rounds']} 去掉被禁 op {len(ops) - len(_kept4)} 个"
                     f"（累计 {STATS['banned_op_killed']}）")
            ops = _kept4
        # 🛡️ **整块合法性闸门**（2026-10-28）：上面四个开关都是"事后从已合法的块里删 op"，
        #   删掉"降级退钱"会让剩下的建塔变成"钱不够"（§5.3 的 2/2 局非法就是这么来的）。
        #   所以只要**真的删过东西**，就逐条复核剩余整块；有一条非法 ⇒ **整块丢弃 ⇒ 本回合 HOLD**。
        #   开关全关（默认）时 `len(ops) == _n_raw` ⇒ 这段**一次都不进**，行为与改动前逐字相同。
        if len(ops) != _n_raw:
            _bad = _block_legal_gate(facade, int(player), ops)
            STATS["filter_gate_checks"] = int(STATS.get("filter_gate_checks", 0)) + 1
            if _bad:
                STATS["filter_gate_dropped"] = int(STATS.get("filter_gate_dropped", 0)) + 1
                _log(f"🛡️ [gate] 过滤后整块非法 ⇒ **整块丢弃（本回合 HOLD）**: "
                     f"round={STATS['rounds']} 过滤前="
                     f"{[(int(o.op_type), int(o.arg0), int(o.arg1)) for o in _raw_ops]} "
                     f"过滤后={[(int(o.op_type), int(o.arg0), int(o.arg1)) for o in ops]} "
                     f"非法={[(b.get('idx'), b.get('op'), b.get('reason')) for b in _bad]} "
                     f"（累计丢弃 {STATS['filter_gate_dropped']}/{STATS['filter_gate_checks']} 次）")
                ops = []
        return ops

    def _legal_check(pl: int, ops: list, tag: str) -> list:
        """**只读**逐条合法性自检：非法就逐条打 stderr（带 round/side/op/reason）+ 计数。

        绝不改变决策：改状态的只有 `facade.apply_operation_list`，这里只问 `can_apply`
        （dry + clone 版冷路径交叉验证，见 `match_via_sdk.check_ops`）。检查器自身出错也只记日志。
        `tag` 标出这一手来自哪条改写路径（decide_final / after_no_downgrade / shadow_apply）。
        """
        try:
            bad = MVS.check_ops(facade, int(pl), ops, ILLEGAL_STATS, seed=int(seed),
                                round_index=int(STATS.get("rounds", 0)), tag=str(tag),
                                cold=False, log=_log)
        except Exception as exc:  # noqa: BLE001
            _log(f"⚠️ 合法性自检异常（忽略，不影响对局）: {type(exc).__name__}: {exc}")
            return []
        STATS["illegal_checked"] = ILLEGAL_STATS.checked
        STATS["illegal"] = ILLEGAL_STATS.total          # 让既有的 illegal= 日志行变成**真的**
        STATS["hold"] = ILLEGAL_STATS.holds[0] + ILLEGAL_STATS.holds[1]
        return bad

    def _apply(pl: int, ops: list) -> None:
        """把操作打进影子状态，并统计非法数（影子与裁判的分歧会在这里先露头）。"""
        from my_ai.az_intent.timing import T as _T
        # ⚠️ 旧写法 `bad = facade.apply_operation_list(...)` 的返回值**恒为 None**
        #    （facade:167-170 丢掉了 binding 的计数）⇒ STATS["illegal"] 从来没数到过东西。
        #    现在先做**只读**校验（在计时器外，保证 apply 的耗时读数可比），再照原样应用。
        _legal_check(pl, ops, "shadow_apply")
        with _T("apply"):
            facade.apply_operation_list(pl, ops)

    try:
        def _decide_timed():
            """计时包装（p0/p1 两条决策路径共用）。⚠️ 计时必须覆盖**两处**决策点：
            我第一版只写在 `player == 0` 分支里，导致**镜像局（我方执后手）完全没有耗时数据**（2026-10-06）。

            2026-10-20 起增加**分阶段分解**（`AZAI_TIMING=1`）：用 `az_intent.timing.T` 的
            嵌套计时器，inclusive/exclusive 自动区分，并在每条决策上做一致性自检
            （root exclusive 之和 ≈ total，偏差 >2% 记为 check_ok=false）。
            """
            from my_ai.az_intent.timing import T
            T.start(round=int(STATS.get("rounds", 0)), dec_n=int(STATS.get("dec_n", 0)))
            with T("decision"):
                _t = time.monotonic()
                _ops = decide()
                _dt = time.monotonic() - _t
            T.finish()
            # 2026-10-31（`AZAI_CAND_VERBOSE=1`，**纯日志、默认关**）：把**真的发出去的**那串 op
            #   逐回合落盘。动机：统一候选 A/B 要报"行为差异"（HOLD 率 / 降级 churn / 升级次数 /
            #   同 op 类型多 op），而这些**只能**从实际 op 算；本批的 `_tmp_ladder` 不透传
            #   `--trace-ops`（对局日志里没有 op 轨迹）。不设该变量 ⇒ 一个字节都不多打。
            if _CAND_VERBOSE[0]:
                _log(f"[act] round={int(STATS.get('rounds', 0))} n={len(_ops)} "
                     f"ops={[(int(o.op_type), int(o.arg0), int(o.arg1)) for o in _ops]}")
            # 2026-10-28：**每决策一条**注入明细（`[oracle] inject round=… action=… dup=… new=…
            # chosen=… hit=…`）。放在这里而不是 `decide()` 里面：这里的 `_ops` 是**最终真的发出去的**
            # 那串 op（已过执行层过滤/闸门），所以 `hit` 才是"我们改没改亚军那一手"。
            # ⚠️ 它**只读**、只在 `AZAI_ORACLE_INJECT` 非空且本回合真的注入时打 ⇒ 默认零日志零影响。
            # 2026-10-28：**双 oracle 逐回合记录**（E4/E4R；E2 只有 A）。⚠️ 必须在下面两个
            # `_oracle_*_log_round` **之前**：它只读 `round_log`，那两行会把它置 None。
            # 默认（两个会话都没开）⇒ 第一行就 return，零日志零成本。
            _xor_log_round(_ops)
            # 2026-10-29：候选分数差（E2/E4/E4R；默认零日志）。放在 `_xor_log_round` 之后、
            #   `_oracle_*_log_round` 之前/之后都行（它不读 `round_log`），这里紧跟 xor 便于对照。
            _dscore_log_round(_ops)
            _oracle_log_round(_ops)
            # 2026-10-28：B 会话（E4）的每决策明细；默认 `_ORACLE_B["round_log"]` 是 None ⇒ 零日志
            _oracle_b_log_round(_ops)
            # 2026-10-28：出口数一下"最终真的发出去几个升级 op"（`AZAI_UPG_BUNDLES` 关时零成本）
            _upg_count_sent(_ops)
            # 2026-10-27：出口自检 —— 一处覆盖 decide() 的**所有** return 分支
            # （AZAI_FORCE_UPG 的升级 op / mc 显式升级 op(12) / HOLD 空列表 / 各改写后缀）。
            # 放在计时器外 ⇒ `decision` 的耗时读数与改动前口径一致。
            _legal_check(int(player), _ops, "decide_final")
            STATS["dec_n"] = int(STATS.get("dec_n", 0)) + 1
            STATS["dec_times"].append(round(_dt, 3))   # 2026-10-26：p50/p95 用（只进日志）
            if _dt > float(STATS.get("dec_max_s", 0.0)):
                STATS["dec_max_s"] = round(_dt, 2)
            _thr = float(_env("AZAI_SLOW_DECISION_SEC", "20"))
            if _dt >= _thr:
                STATS["dec_slow"] = int(STATS.get("dec_slow", 0)) + 1
                _log(f"⚠️ 慢决策 round={STATS['rounds']} 用时={_dt:.2f}s 阈值={_thr:g}s"
                     f" （超过 bridge 读超时会记 verdict=timeout）")
            # 固定阈值计数（10/20/30s；`dec_slow` 那条线的阈值可配 ⇒ 口径会漂，这条不会）
            for _th in _DEC_THRESHOLDS:
                if _dt >= _th:
                    STATS[f"dec_ge{_th}"] = int(STATS.get(f"dec_ge{_th}", 0)) + 1
            # ⚠️ 必须**对局中**周期打印：bridge 结束时会直接 terminate()，atexit 里的「退出报告」
            #    在正常对局里根本不会执行 ⇒ dec_max/dec_slow 只在异常局可见，等于没用。
            _every = int(_env("AZAI_TIMING_EVERY", "25"))
            if _every > 0 and STATS["dec_n"] % _every == 0:
                _bi = ""
                try:
                    from my_ai.az_intent.mc_quiet import BIDIR_STATS as _BS
                    if int(_BS.get("n", 0)) > 0:
                        _bi = (f" mc_bidir_n={_BS['n']} 打到终局={_BS['terminal']} "
                               f"平均推进={_BS['steps'] / max(1, _BS['n']):.1f}回合 "
                               f"空过={_BS['noop']}")
                    from my_ai.az_intent import mc_quiet as _mq
                    if int(_mq.SPEND_STATS.get("calls", 0)) > 0:
                        _ss = _mq.SPEND_STATS
                        _bi += (f" rollout_calls={_ss['calls']} noop={_ss['noop']}"
                                f"({_ss['noop'] / max(1, _ss['calls']):.0%})"
                                f" classes={_mq.ROLLOUT_CLASSES}"
                                f" rollout_light={_ss.get('light', 0)}"
                                f" light_masked={_ss.get('light_blocked', 0)}")
                except Exception:
                    pass
                _log(f"⏱️ 耗时累计: dec_n={STATS['dec_n']} dec_max={STATS['dec_max_s']}s "
                     f"dec_slow={STATS['dec_slow']} {_dec_hist_text()}{_bi}")
                # 2026-10-28（规则内口径）：p50/p95 **两臂都打**（原来只 Injection 臂打 ⇒
                # 基准臂的长尾只能靠收尾报告，而收尾常被 bridge 硬杀）。只多一行/25 决策。
                _dt_all = STATS["dec_times"]
                _log(f"⏱️ 决策耗时 p50={_pct(_dt_all, 0.50):.2f}s "
                     f"p95={_pct(_dt_all, 0.95):.2f}s max={max(_dt_all):.2f}s "
                     f"n={len(_dt_all)}")
                # 2026-10-27：非法操作自检也**对局中**周期打印（同样因为收尾常被硬杀）
                _log("🔎 非法操作自检: " + ILLEGAL_STATS.summary(int(seed), False))
                # 2026-10-28：过滤闸门读数（拦下的半块**没有发出去** ⇒ 不能混进上面的 illegal_total）
                if int(STATS.get("filter_gate_checks", 0)):
                    _log(f"🛡️ 过滤闸门累计: checks={STATS['filter_gate_checks']} "
                         f"dropped={STATS.get('filter_gate_dropped', 0)}"
                         f"（dropped = 整块丢弃 ⇒ 该回合 HOLD）")
                # 2026-10-26：注入统计也要**对局中**周期打印（收尾报告常被 bridge 硬杀打断）
                if _ORACLE["asks"] and (_ORACLE["obj"] is not None or _ORACLE["placebo"]):
                    # ↑ 2026-10-28：安慰剂臂没有 oracle 对象（`obj is None`）但**有**注入读数
                    #   ⇒ 条件里补 `or placebo`。默认/A 臂两者都为 False/True 时行为与改动前逐字相同。
                    _log(f"[{_inj_tag()}] 注入累计: {_oracle_stats_text()}")
                # 2026-10-28：B 会话（E4）的周期读数；默认 asks=0 ⇒ 不打
                if _ORACLE_B["asks"]:
                    _log(f"[oracle-b] 注入累计: {_oracle_b_stats_text()}")
                if _ONLY["on"]:
                    _log(f"[onlyextra] 累计: decided={_ONLY['decided']} ncand={_ONLY['ncand']} "
                         f"pick={_ONLY['pick']}")
                # 2026-10-28：双 oracle 的周期汇总（默认 rounds=0 ⇒ 不打；收尾那行常被 bridge 硬杀）
                if _XOR["rounds"]:
                    _log(f"[xor-sum] {_xor_summary_text()}")
                # 2026-10-29：候选分差的周期汇总（同上；`[dscore-sum]` 的收尾行同样会被硬杀吃掉）
                if _DS.get("rounds"):
                    _log(f"[dscore-sum] {_dscore_summary_text()}")
            return _ops

        while not facade.terminal:
            if player == 0:
                self_ops = _decide_timed()
                io.send_ops(self_ops)
                _apply(0, self_ops)
                opp = io.recv_ops()
                if opp is None:
                    break
                _apply(1, opp)
                # P0：对手操作块的语义是「上一回合」（它先出招、后收对手行）⇒ 现在登记 round_index
                _oracle_set_opp(facade.round_index, opp)
                _oracle_b_set_opp(facade.round_index, opp)   # 2026-10-28：E4 的 B 会话（默认 no-op）
            else:
                opp = io.recv_ops()
                if opp is None:
                    break
                _apply(0, opp)
                # P1：本回合的对手操作块，`_oracle_step` 里 ask(opponent_ops=) 会用掉它
                if not _ORACLE["off"]:
                    _ORACLE["pending_opp"] = ops_to_text(opp)
                if not _ORACLE_B["off"]:      # 2026-10-28：E4 的 B 会话（默认 off ⇒ 不做任何事）
                    _ORACLE_B["pending_opp"] = ops_to_text(opp)
                self_ops = _decide_timed()
                io.send_ops(self_ops)
                _apply(1, self_ops)

            # ---- 本回合结束：收"推进后"的局面文本，并和影子状态对拍 ----
            facade.advance_round()
            expected = state_to_text(facade)
            n_expect = len(expected.split("\n")) - 1   # 末尾的 "" 不算一行
            got: list[str] = []
            eof = False
            for _ in range(n_expect):
                ln = io.recv_line()
                if ln is None:
                    eof = True
                    break
                got.append(ln)
            if eof:
                break
            got_text = "\n".join(got) + "\n"
            if verify and got_text != expected:
                STATS["mismatch"] += 1
                if trace or STATS["mismatch"] <= 3:
                    _log(f"!! 第 {STATS['rounds']} 回合局面文本对拍不一致"
                         f"（mismatch #{STATS['mismatch']}）")
                    gl, el = got_text.split("\n"), expected.split("\n")
                    for i in range(max(len(gl), len(el))):
                        a = gl[i] if i < len(gl) else "<缺>"
                        b = el[i] if i < len(el) else "<缺>"
                        if a != b:
                            _log(f"   line{i}: 收到={a!r}  影子={b!r}")
            STATS["rounds"] += 1
            if trace:
                # 每回合一行轨迹（bridge 结束时会硬杀我们，收尾报告打不出来，
                # 所以累计计数必须随每回合落盘；最后一行就是终局状态）。
                _log(f"round={STATS['rounds']} hp={[b.hp for b in facade.bases]} "
                     f"coins={list(facade.coins)} towers={len(facade.towers)} "
                     f"lv={[(t.x, t.y, int(getattr(t, 'level', -1)), int(t.tower_type)) for t in facade.towers_of(int(player))]} "
                     f"lv_opp={[(t.x, t.y, int(getattr(t, 'level', -1))) for t in facade.towers_of(1 - int(player))]} "
                     f"mismatch={STATS['mismatch']} illegal={STATS['illegal']}")
    except BrokenPipeError:
        _log("bridge 关掉了管道（正常收尾）")
    except Exception as exc:  # noqa: BLE001
        import traceback
        _log(f"!! EXC {type(exc).__name__}: {exc}")
        sys.stderr.write(traceback.format_exc())
        return 1

    # ⚠️ 统计要在**"结束"之前**打：bridge 一旦拿到终局就 TerminateProcess 我们，
    #    "结束"之后的行有概率被砍掉（实测：注入统计那行丢过一次）。
    if _ORACLE["asks"]:
        _dt_all = STATS["dec_times"]
        _log(f"[{_inj_tag()}] 注入统计: {_oracle_stats_text()}")
        _log(f"决策耗时: n={len(_dt_all)} p50={_pct(_dt_all, 0.50):.2f}s "
             f"p95={_pct(_dt_all, 0.95):.2f}s max={max(_dt_all):.2f}s")
    # 2026-10-28：B 会话 / only-extra 的收尾读数（默认各自为 0/False ⇒ 不打）
    if _ORACLE_B["asks"]:
        _log(f"[oracle-b] 注入统计: {_oracle_b_stats_text()}")
    if _ONLY["on"]:
        _log(f"[onlyextra] 统计: random={int(_ONLY['random'])} decided={_ONLY['decided']} "
             f"ncand={_ONLY['ncand']} pick={_ONLY['pick']}"
             + (f" err={_ONLY['err']}" if _ONLY.get("err") else ""))
    if _XOR.get("rounds"):
        # 2026-10-29：收尾的 xor 汇总（debug 报告里常被 bridge 硬杀打断 ⇒ 命中率低但留底）
        _log(f"[xor-sum] {_xor_summary_text()}")
    if _DS.get("rounds"):
        _log(f"[dscore-sum] {_dscore_summary_text()}")
    _oracle_close()
    # 2026-10-27：非法操作自检汇总（**本进程自己发出去的那一手**；裁判侧另有一份，见 bridge 的
    # `illegal-summary`）。这里用 `illegal_total=` 措辞、且不加前缀大写 ILLEGAL。
    _log("非法操作自检汇总: " + ILLEGAL_STATS.summary(int(seed), False))
    _log(f"结束: rounds={STATS['rounds']} terminal={facade.terminal} winner={facade.winner} "
         f"hp={[b.hp for b in facade.bases]} mismatch={STATS['mismatch']} "
         f"illegal={STATS['illegal']} illegal_checked={STATS.get('illegal_checked', 0)} "
         f"hold={STATS.get('hold', 0)}")
    # ⚠️ 显式在这里 dump：bridge 收尾时会**硬杀**子进程（Windows TerminateProcess ⇒ atexit/SIGTERM
    # 都不一定跑得到）⇒ 不能只靠 atexit（实测：只挂 atexit 时一个文件都没写出来）。
    _dump_value_data()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

