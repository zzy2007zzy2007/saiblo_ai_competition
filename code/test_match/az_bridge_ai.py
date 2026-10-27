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
"""
from __future__ import annotations

import importlib.util
import os
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

# ── 非法操作自检（2026-10-27）──────────────────────────────────────────────────────
# 为什么需要它：`GameStateFacade.apply_operation_list` 返回 **None**
# （`game_state_facade.py:167-170` 把 binding 的 accepted 计数丢掉了），而冷路径本身
# **静默跳过**非法 op ⇒ 本文件原来的 `bad = facade.apply_operation_list(...)` +
# `len(bad) if bad else 0` **恒为 0**（`STATS["illegal"]` 从来没真的数过任何东西，
# 和 bridge 侧 `if il0 or il1:` 是同一个坑）。
# 现在改用 `match_via_sdk.check_ops`（逐条 can_apply_dry + can_apply_cold 交叉验证、**只读**），
# 只记日志/计数，**不改变任何决策**。覆盖的高风险改写路径：
#   decide_final（出口，含强制升级 / mc 显式升级 op(12) / HOLD）
#   after_no_downgrade（AZAI_NO_DOWNGRADE 去掉降级后的剩余 op —— 去掉降级会**少一笔收入**，
#                       后面的建塔/升级可能因此变成"钱不够"⇒ 非法）
#   shadow_apply（影子状态应用前，两侧都查）
#   oracle_bundle（AZAI_ORACLE_INJECT 注入的候选，见 `_oracle_bundle` 里原有的逐条检查）
ILLEGAL_STATS = MVS.IllegalStats()

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
    "hit": 0,             # 注入的候选被搜索选中的次数
    "illegal": 0,         # 因非法操作丢掉整条候选的次数
    "timeouts": 0, "dead": 0,
    "err": "",
    "ms": [],             # 每次 ask 的墙钟（毫秒）
    "pending_opp": None,  # P1 用：本回合的对手操作块原文
    "last_bundle": None,  # 上一回合注入的 bundle（命中日志用）
    "pending": False,     # 上一回合是否交过候选（用于统计 new/dup）
}


def _pct(vals: list, q: float) -> float:
    """最近的百分位（样本少时退化到端点，够看）。"""
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    k = int(round(q * (len(s) - 1)))
    return float(s[max(0, min(len(s) - 1, k))])


def _oracle_stats_text() -> str:
    """注入统计的一行文本（本功能的**主读数**）。字段含义见 docs/oracle_inject_design.md §3.5。"""
    st = _ORACLE
    txt = (f"asks={st['asks']} ok={st['ok']} checked={st['checked']} inject={st['inject']} "
           f"new={st['new']} "
           f"dup={st['dup']} hit={st['hit']} illegal={st['illegal']} "
           f"timeout={st['timeouts']} dead={st['dead']}")
    if st["ms"]:
        txt += (f" avg_ms={sum(st['ms']) / len(st['ms']):.0f} p50_ms={_pct(st['ms'], 0.50):.0f} "
                f"p95_ms={_pct(st['ms'], 0.95):.0f} max_ms={max(st['ms']):.0f}")
    txt += f" hit_rate={(100.0 * st['hit'] / max(1, st['new'])):.1f}%"
    if st["err"]:
        txt += f" err={st['err']}"
    return txt


def _oracle_close_module() -> None:
    """释放注入会话（模块级：供 atexit / SIGTERM 兜底；`main()` 正常收尾也会显式调）。"""
    _ob = _ORACLE.get("obj")
    if _ob is not None:
        try:
            _ob.close()
        except Exception:  # noqa: BLE001
            pass
        _ORACLE["obj"] = None


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
    _log(f"退出报告: rounds={STATS['rounds']} mismatch={STATS['mismatch']} "
         f"illegal={STATS['illegal']}{_mc} dec_n={STATS.get('dec_n')} "
         f"dec_max={STATS.get('dec_max_s')}s dec_slow={STATS.get('dec_slow')}")
    _dt = STATS.get("dec_times") or []
    if _dt:
        _log(f"决策耗时: n={len(_dt)} p50={_pct(_dt, 0.50):.2f}s p95={_pct(_dt, 0.95):.2f}s "
             f"max={max(_dt):.2f}s")
    if _ORACLE.get("asks"):
        _log(f"[oracle] 注入统计: {_oracle_stats_text()}")
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
        mcts = FlatRolloutGreedy(
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
            **_kw,
        )
        _log(f"engine=flat rollouts={_env('AZAI_FLAT_ROLLOUTS','16')} "
             f"horizon={_env('AZAI_FLAT_HORIZON','64')} "
             f"batch={_env('AZAI_FLAT_BATCH','32')} eval={_env('AZAI_FLAT_EVAL','net')} "
             f"rounds={_env('AZAI_FLAT_ROUNDS','39000')}")
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
        _ORACLE["timeout"] = float(_env("AZAI_ORACLE_TIMEOUT", "2.5"))
        _ORACLE["off"] = True
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
            _ORACLE["exe"] = str(_exe)
            _ORACLE["off"] = False
            _log(f"[oracle] 注入已启用: exe={_exe} seed={seed} player={player} "
                 f"timeout={_ORACLE['timeout']:g}s（**每回合都要喂**，跳回合会永久错位）")
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
            _ORACLE["obj"].set_opponent_ops(int(round_index), list(ops))
        except Exception as exc:  # noqa: BLE001
            _log(f"[oracle] ⚠️ 登记对手操作块失败（忽略，按空过处理）: {type(exc).__name__}: {exc}")

    def _oracle_step() -> None:
        """每回合**必做一次** ask（保持会话同步），把它的 ops 变成一个额外候选交给搜索。

        失败一律吞掉：本回合不注入、决策照常。"宁可少一个候选，也绝不能因为 oracle 让对局失败"。
        """
        st = _ORACLE
        st["pending"] = False
        if st["off"] or st["obj"] is None:
            return
        st["asks"] += 1
        _opp = st["pending_opp"]
        st["pending_opp"] = None       # 一回合一次，绝不复用上一回合的块
        t0 = time.monotonic()
        try:
            if int(player) == 1:
                ops = st["obj"].ask(facade, opponent_ops=_opp)
            else:
                ops = st["obj"].ask(facade)
        except Exception as exc:  # noqa: BLE001
            _dt = (time.monotonic() - t0) * 1000.0
            st["ms"].append(_dt)
            name = type(exc).__name__
            if "Timeout" in name:
                st["timeouts"] += 1
            else:
                st["dead"] += 1
            st["err"] = f"{name}: {exc}"
            # 读端已错位 ⇒ 本局**永久停用**（重建必须从 r0 重放，见设计文档 §3.4）
            try:
                st["obj"].close()
            except Exception:  # noqa: BLE001
                pass
            st["obj"] = None
            st["last_bundle"] = None
            if hasattr(mcts, "set_extra_bundles"):
                mcts.set_extra_bundles(None)
            _log(f"[oracle] ⚠️ 第 {facade.round_index} 回合查询失败（{_dt:.0f}ms）"
                 f"⇒ **本局停用注入**，决策照常: {name}: {exc}")
            return
        dt_ms = (time.monotonic() - t0) * 1000.0
        st["ms"].append(dt_ms)
        st["ok"] += 1
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
        mcts.set_extra_bundles([bundle])         # ← 注入通道（HOLD = 空 bundle `()`）

    def _oracle_after_search() -> None:
        """搜索结束：统计「去重后真正加进去几个」+「是不是选中了它」。"""
        st = _ORACLE
        if st["off"] or not st["pending"]:
            return
        st["pending"] = False
        added = int(getattr(mcts, "last_extra", 0) or 0)
        if added:
            st["new"] += 1
        else:
            st["dup"] += 1                       # 与网络候选重复（最常见：双方都 HOLD）
        if added and bool(getattr(mcts, "last_chosen_is_extra", False)):
            st["hit"] += 1
            _ls = getattr(mcts, "last_scores", None)
            _n = int(len(_ls)) if _ls is not None else 0   # ⚠️ 不能写 `_ls or []`（ndarray 真值歧义）
            _log(f"[oracle] 🎯 注入命中: round={facade.round_index} 亚军候选 {st['last_bundle']} "
                 f"被搜索选中（本决策共 {_n} 个候选，其中额外 1 个）")

    def _oracle_close() -> None:
        _oracle_close_module()

    _oracle_init()

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
        _oracle_step()   # 每回合**必做**：喂 oracle（保持会话同步）+ 把它的 ops 当额外候选注入
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
        from my_ai.az_intent.timing import T as _T
        with _T("search"):
            res = mcts.search(facade, player, temperature=temperature)
        _oracle_after_search()   # 注入统计（去重后是否真的加进去 / 是否被选中）
        # 显式 op（升级）：class+position 通路表达不了 ⇒ 直接把 MC 选中的那个 op 发出去。
        _expl = getattr(mcts, 'mc_last_explicit', None)
        if _expl is not None:
            _log(f"⬆️ 显式 op（升级）: round={STATS['rounds']} op={_expl}")
            return [Operation(OperationType(int(_expl[0])), int(_expl[1]), int(_expl[2]))]
        _record_obs(res)
        chosen = res.chosen_bundle or ()
        ops = [Operation(OperationType(int(k[0])), int(k[1]), int(k[2])) for k in chosen]
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
            # 后面的建塔/升级可能就变成"钱不够"（非法）。只记日志，不改决策。
            _legal_check(int(player), ops, "after_no_downgrade")
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
                     f"dec_slow={STATS['dec_slow']}{_bi}")
                # 2026-10-27：非法操作自检也**对局中**周期打印（同样因为收尾常被硬杀）
                _log("🔎 非法操作自检: " + ILLEGAL_STATS.summary(int(seed), False))
                # 2026-10-26：注入统计也要**对局中**周期打印（收尾报告常被 bridge 硬杀打断）
                if _ORACLE["asks"] and _ORACLE["obj"] is not None:
                    _dt_all = STATS["dec_times"]
                    _log(f"⏱️ 决策耗时 p50={_pct(_dt_all, 0.50):.2f}s "
                         f"p95={_pct(_dt_all, 0.95):.2f}s max={max(_dt_all):.2f}s")
                    _log(f"[oracle] 注入累计: {_oracle_stats_text()}")
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
            else:
                opp = io.recv_ops()
                if opp is None:
                    break
                _apply(0, opp)
                # P1：本回合的对手操作块，`_oracle_step` 里 ask(opponent_ops=) 会用掉它
                if not _ORACLE["off"]:
                    _ORACLE["pending_opp"] = ops_to_text(opp)
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
        _log(f"[oracle] 注入统计: {_oracle_stats_text()}")
        _log(f"决策耗时: n={len(_dt_all)} p50={_pct(_dt_all, 0.50):.2f}s "
             f"p95={_pct(_dt_all, 0.95):.2f}s max={max(_dt_all):.2f}s")
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

