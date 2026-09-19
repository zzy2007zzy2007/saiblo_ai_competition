#!/usr/bin/env python3
"""
从 SelfPlay 对战回合级 JSON 日志生成结构化对战摘要报告。

用法:
  format_battle_detail.py <json_path>

输出 6 个板块: 摘要卡 / 战略阶段演变 / 关键事件 / 经济指标 / 双方对比 / 科技发展。
数据来源: round_details[] 和 actions[] (日志 JSON)。
"""
import json
import os
import sys
from collections import defaultdict
from typing import List, Dict, Tuple


# ──────────────────────────────────────────────
# 模块一：动作推导引擎
# ──────────────────────────────────────────────

TOWER_COST_BASE = 15
TOWER_UPGRADE_L1 = 60
TOWER_UPGRADE_L2 = 200
TECH_COST = 200


def _calc_tower_cost(tower_count: int) -> int:
    n = tower_count
    return TOWER_COST_BASE * (3 ** (n // 2)) * (2 if n % 2 else 1)


def derive_actions(rounds: List[dict]) -> Tuple[List[dict], dict, dict]:
    """
    从相邻回合状态反推动作。
    返回:
      actions: 每步的[动作1, 动作2, ...]列表
      tower_history: 每座塔的生命周期 {id: {first, last, type, levels, destroyed}}
      summary: 动作汇总统计
    """
    actions_per_step = []
    tower_registry = {}
    tower_history = {}
    action_counts = defaultdict(int)

    for i, r in enumerate(rounds):
        step = r["step"]
        curr_towers = {t["id"]: t for t in r.get("our_towers", [])}
        curr_tech = r.get("our_tech", {})
        step_actions = []

        if i == 0:
            # 第一轮没有前一步，记录初始塔
            for tid, t in curr_towers.items():
                if tid not in tower_registry:
                    tower_registry[tid] = t
                    tower_history[tid] = {
                        "first": step, "last": step, "type": t["type"],
                        "levels": [(step, t["level"])], "destroyed": False
                    }
            actions_per_step.append([])
            continue

        prev = rounds[i - 1]
        prev_towers = {t["id"]: t for t in prev.get("our_towers", [])}
        prev_tech = prev.get("our_tech", {})

        # 检测 BUILD: 新出现的塔 id
        for tid, t in curr_towers.items():
            if tid not in prev_towers:
                cost_estimate = _calc_tower_cost(len(prev_towers))
                step_actions.append({
                    "type": "BUILD", "tower_id": tid,
                    "tower_type": t["type"], "position": t["position"],
                    "level": t["level"],
                    "coins_before": prev.get("our_coins", 0),
                    "coins_after": r.get("our_coins", 0),
                    "cost_estimate": cost_estimate,
                })
                action_counts["BUILD"] += 1
                tower_registry[tid] = t
                tower_history[tid] = {
                    "first": step, "last": step, "type": t["type"],
                    "levels": [(step, t["level"])], "destroyed": False
                }

        # 检测 DESTROY: 消失的塔 id
        for tid, t in prev_towers.items():
            if tid not in curr_towers:
                if tid in tower_history:
                    tower_history[tid]["last"] = step
                    tower_history[tid]["destroyed"] = True
                # 不向 step_actions 添加 DESTROY（这是被动事件）

        # 检测 UPGRADE: 同一塔 level 增加
        for tid, t in curr_towers.items():
            if tid in prev_towers and t["level"] > prev_towers[tid]["level"]:
                step_actions.append({
                    "type": "UPGRADE", "tower_id": tid,
                    "tower_type": t["type"],
                    "level_from": prev_towers[tid]["level"],
                    "level_to": t["level"],
                })
                action_counts["UPGRADE"] += 1
                if tid in tower_history:
                    tower_history[tid]["levels"].append((step, t["level"]))
                    tower_history[tid]["last"] = step

        # 检测 TECH_UPGRADE
        for key in ("gen_speed", "ant_strength"):
            if curr_tech.get(key, 0) > prev_tech.get(key, 0):
                step_actions.append({
                    "type": "TECH_UPGRADE", "key": key,
                    "from": prev_tech.get(key, 0),
                    "to": curr_tech.get(key, 0),
                })
                action_counts["TECH_UPGRADE"] += 1

        # 更新 tower 最后可见步
        for tid in curr_towers:
            if tid in tower_history:
                tower_history[tid]["last"] = step

        if not step_actions:
            action_counts["NO_OP"] += 1

        actions_per_step.append(step_actions)

    # 整理摘要
    summary = dict(action_counts)
    return actions_per_step, tower_history, summary


# ──────────────────────────────────────────────
# 模块二：阶段划分算法
# ──────────────────────────────────────────────

def segment_phases(rounds: List[dict], actions_per_step: List[dict]) -> List[dict]:
    """
    将整局对战划分为若干战略阶段。
    划分依据：塔数变化≥2 或 金币趋势反转 或 HP 斜率突变。
    """
    MIN_SEGMENT = 15

    # 计算每步的显著变化得分
    change_points = []
    for i in range(1, len(rounds)):
        r = rounds[i]
        p = rounds[i - 1]
        score = 0
        curr_tower_count = len(r.get("our_towers", []))
        prev_tower_count = len(p.get("our_towers", []))
        if abs(curr_tower_count - prev_tower_count) >= 2:
            score += 2
        hp_drop = p.get("our_hp", 50) - r.get("our_hp", 50)
        if hp_drop >= 3:
            score += 1
        coins_diff = abs(r.get("our_coins", 0) - p.get("our_coins", 0))
        if coins_diff >= 50:
            score += 1
        # 有非 NO-OP 动作
        if i - 1 < len(actions_per_step) and actions_per_step[i - 1]:
            score += 1
        change_points.append((r["step"], score))

    # 基于变化点做粗分割
    segments = []
    seg_start = 0
    seg_start_step = rounds[0]["step"]
    acc_score = 0

    for idx, (step, score) in enumerate(change_points):
        acc_score += score
        step_idx = idx + 1
        if acc_score >= 4 and (step_idx - seg_start) >= MIN_SEGMENT:
            segments.append((seg_start, step_idx, seg_start_step, step))
            seg_start = step_idx
            seg_start_step = step
            acc_score = 0

    # 最后一段
    if seg_start < len(rounds):
        segments.append((seg_start, len(rounds) - 1, seg_start_step, rounds[-1]["step"]))

    # 如果段数太少（<3），用等分法
    if len(segments) < 3:
        total = len(rounds)
        part = max(total // 5, MIN_SEGMENT)
        segments = []
        for i in range(0, total, part):
            end = min(i + part, total - 1)
            if end - i < 5:
                continue
            segments.append((i, end, rounds[i]["step"], rounds[end]["step"]))
            if end >= total - 1:
                break
        if len(segments) < 3:
            segments = [(0, len(rounds) - 1, rounds[0]["step"], rounds[-1]["step"])]

    # 生成阶段摘要
    phases = []
    for start_idx, end_idx, start_step, end_step in segments:
        seg_rounds = rounds[start_idx: end_idx + 1]
        seg_actions = actions_per_step[start_idx: end_idx + 1] if start_idx < len(actions_per_step) else []

        # 塔数范围
        tower_counts = [len(r.get("our_towers", [])) for r in seg_rounds]
        tower_min = min(tower_counts) if tower_counts else 0
        tower_max = max(tower_counts) if tower_counts else 0

        # 金币范围
        coins_list = [r.get("our_coins", 0) for r in seg_rounds]
        coins_min = min(coins_list) if coins_list else 0
        coins_max = max(coins_list) if coins_list else 0

        # HP 范围
        hp_start = seg_rounds[0].get("our_hp", 50)
        hp_end = seg_rounds[-1].get("our_hp", 50)

        # 动作计数
        action_counts = defaultdict(int)
        for sa in seg_actions:
            if not sa:
                action_counts["NO_OP"] += 1
            else:
                for a in sa:
                    action_counts[a["type"]] += 1

        phases.append({
            "start_step": start_step, "end_step": end_step,
            "tower_range": f"{tower_min}~{tower_max}",
            "coins_range": f"${coins_min}~${coins_max}",
            "hp": f"{hp_start}→{hp_end}",
            "actions": dict(action_counts),
            "total_steps": end_step - start_step + 1,
        })

    return phases


# ──────────────────────────────────────────────
# 模块三：关键事件筛选
# ──────────────────────────────────────────────

def find_key_events(rounds: List[dict], actions_per_step: List[dict]) -> List[dict]:
    """
    从 242 步中筛选 <=8 条事实性关键事件。
    规则纯机械，不做价值判断。
    """
    events = []

    # 1. 首次建塔
    for i, sa in enumerate(actions_per_step):
        for a in sa:
            if a["type"] == "BUILD":
                r = rounds[i]
                events.append({
                    "step": r["step"],
                    "desc": f"首次建塔 (BASIC @[{a['position'][0]},{a['position'][1]}])，建塔前金币 ${a['coins_before']:.0f}，建塔后 ${a['coins_after']:.0f}"
                })
                break
        if events:
            break

    # 2. 建塔后金币接近耗尽 (< 10)
    for i, sa in enumerate(actions_per_step):
        for a in sa:
            if a["type"] == "BUILD" and a["coins_after"] < 10:
                if i < len(rounds):
                    r = rounds[i]
                    events.append({
                        "step": r["step"],
                        "desc": f"建塔，建塔前金币 ${a['coins_before']:.0f}，建塔后 ${a['coins_after']:.0f}"
                    })
                    break

    # 3. 金币积累但无操作（连续 >15 步金币 >100, 无动作）
    noop_streak = 0
    for i, sa in enumerate(actions_per_step):
        if i < len(rounds):
            r = rounds[i]
            coins = r.get("our_coins", 0)
            if not sa and coins > 100:
                noop_streak += 1
                if noop_streak == 20:
                    events.append({
                        "step": r["step"],
                        "desc": f"金币 ${coins:.0f}，连续 NO-OP ≥20 步"
                    })
            else:
                noop_streak = 0

    # 4. 连续 NO-OP > 30 步
    noop_streak2 = 0
    for i, sa in enumerate(actions_per_step):
        if not sa:
            noop_streak2 += 1
        else:
            if noop_streak2 >= 30 and i < len(rounds):
                events.append({
                    "step": rounds[i - 1]["step"],
                    "desc": f"连续 NO-OP {noop_streak2} 步"
                })
            noop_streak2 = 0
    # 末尾检查
    if noop_streak2 >= 30 and len(rounds) > 0:
        events.append({
            "step": rounds[-1]["step"],
            "desc": f"连续 NO-OP {noop_streak2} 步"
        })

    # 5. 单步 HP 大幅下降 (>3)
    for i in range(1, len(rounds)):
        drop = rounds[i - 1].get("our_hp", 50) - rounds[i].get("our_hp", 50)
        if drop >= 4:
            events.append({
                "step": rounds[i]["step"],
                "desc": f"单步 HP 下降 {drop} 点 (HP {rounds[i-1].get('our_hp', 50)}→{rounds[i].get('our_hp', 50)})"
            })
            break
        enemy_drop = rounds[i - 1].get("enemy_hp", 50) - rounds[i].get("enemy_hp", 50)
        if enemy_drop >= 4:
            events.append({
                "step": rounds[i]["step"],
                "desc": f"敌方单步 HP 下降 {enemy_drop} 点 (HP {rounds[i-1].get('enemy_hp', 50)}→{rounds[i].get('enemy_hp', 50)})"
            })
            break

    # 6. 塔数归零（防线全破）
    for i, r in enumerate(rounds):
        if not r.get("our_towers") and i > 0 and rounds[i - 1].get("our_towers"):
            events.append({
                "step": r["step"],
                "desc": f"我方塔数降至 0 (此前 {len(rounds[i-1].get('our_towers', []))} 塔)"
            })
            break

    # 7. 升级/科技升级
    for i, sa in enumerate(actions_per_step):
        for a in sa:
            if a["type"] == "UPGRADE":
                if i < len(rounds):
                    r = rounds[i]
                    events.append({
                        "step": r["step"],
                        "desc": f"升级 (id={a['tower_id']} {a['tower_type']} L{a['level_from']}→L{a['level_to']})"
                    })
                    break
        if events and len([e for e in events if "升级" in e.get("desc", "")]) > 0:
            # 只记录第一个升级
            pass
    # 实际只允许第一个升级
    upgrade_event_exists = False
    for a in events:
        if "升级" in a.get("desc", ""):
            upgrade_event_exists = True
            break

    if not upgrade_event_exists:
        for i, sa in enumerate(actions_per_step):
            found = False
            for a in sa:
                if a["type"] == "UPGRADE":
                    if i < len(rounds):
                        r = rounds[i]
                        events.append({
                            "step": r["step"],
                            "desc": f"升级 (id={a['tower_id']} {a['tower_type']} L{a['level_from']}→L{a['level_to']})"
                        })
                        found = True
                        break
                elif a["type"] == "TECH_UPGRADE":
                    if i < len(rounds):
                        r = rounds[i]
                        events.append({
                            "step": r["step"],
                            "desc": f"科技升级 ({a['key']} L{a['from']}→L{a['to']})"
                        })
                        found = True
                        break
            if found:
                break

    # 去重：相同 desc 的只保留一条
    seen = set()
    unique_events = []
    for e in events:
        key = e["desc"][:30]
        if key not in seen:
            seen.add(key)
            unique_events.append(e)

    # 去重后按 step 排序，最多 8 条
    unique_events.sort(key=lambda x: x["step"])
    return unique_events[:8]


# ──────────────────────────────────────────────
# 模块四：统计指标
# ──────────────────────────────────────────────

def compute_metrics(rounds: List[dict], actions_per_step: List[dict],
                    tower_history: dict, action_summary: dict) -> dict:
    """计算补充指标面板所需的所有数据。"""
    total_builds = action_summary.get("BUILD", 0)
    total_upgrades = action_summary.get("UPGRADE", 0)
    total_tech = action_summary.get("TECH_UPGRADE", 0)
    total_noop = action_summary.get("NO_OP", 0)
    total_steps = len(rounds) - 1 if rounds else 0

    # 总收入 = 终局金币 + 累计(建塔费用+升级费用+科技费用) - 初始金币
    initial_coins = rounds[0].get("our_coins", 50) if rounds else 50
    final_coins = rounds[-1].get("our_coins", 0) if rounds else 0

    total_spent_on_build = 0
    tower_count_at_build = 0
    for i, sa in enumerate(actions_per_step):
        for a in sa:
            if a["type"] == "BUILD":
                total_spent_on_build += a.get("cost_estimate", 0)
            elif a["type"] == "UPGRADE":
                if a["level_from"] == 0:
                    total_spent_on_build += TOWER_UPGRADE_L1
                else:
                    total_spent_on_build += TOWER_UPGRADE_L2
            elif a["type"] == "TECH_UPGRADE":
                total_spent_on_build += TECH_COST

    total_income = final_coins + total_spent_on_build - initial_coins
    if total_income < 0:
        total_income = 0
    utilization = (total_spent_on_build / total_income * 100) if total_income > 0 else 0

    # 塔生命周期统计
    tower_lifespans = []
    for tid, th in tower_history.items():
        lifespan = th["last"] - th["first"]
        if lifespan > 0:
            tower_lifespans.append(lifespan)
    avg_lifespan = sum(tower_lifespans) / len(tower_lifespans) if tower_lifespans else 0

    # 敌方统计
    enemy_builds = 0
    for i in range(1, len(rounds)):
        prev_enemy = {t["id"] for t in rounds[i - 1].get("enemy_towers", [])}
        curr_enemy = {t["id"] for t in rounds[i].get("enemy_towers", [])}
        new_towers = curr_enemy - prev_enemy
        if new_towers:
            enemy_builds += len(new_towers)

    # HP 总损失
    our_hp_loss = rounds[0].get("our_hp", 50) - rounds[-1].get("our_hp", 50) if rounds else 0
    enemy_hp_loss = rounds[0].get("enemy_hp", 50) - rounds[-1].get("enemy_hp", 50) if rounds else 0

    # 科技状态
    final_tech = rounds[-1].get("our_tech", {}) if rounds else {}
    enemy_final_tech = rounds[-1].get("enemy_tech", {}) if rounds else {}

    return {
        "total_steps": total_steps,
        "total_income": total_income,
        "total_spent": total_spent_on_build,
        "final_coins": final_coins,
        "utilization": utilization,
        "builds": total_builds,
        "upgrades": total_upgrades,
        "tech_upgrades": total_tech,
        "noop": total_noop,
        "avg_tower_lifespan": avg_lifespan,
        "steps_per_build": total_steps / total_builds if total_builds else 0,
        "enemy_builds": enemy_builds,
        "our_hp_loss": our_hp_loss,
        "enemy_hp_loss": enemy_hp_loss,
        "final_tech": final_tech,
        "enemy_final_tech": enemy_final_tech,
    }


# ──────────────────────────────────────────────
# 模块五：格式化输出
# ──────────────────────────────────────────────

def format_report(data: dict) -> str:
    """生成完整的 Markdown 摘要报告。"""
    info = data["info"]
    phases = data["phases"]
    events = data["events"]
    metrics = data["metrics"]
    action_summary = data["action_summary"]
    tower_history = data["tower_history"]
    rounds = data["rounds"]

    lines = []

    # ── 顶部摘要卡 ──
    pos = "先手" if info["player_position"] == 0 else "后手"
    result_str = "胜" if info["result"] == "win" else "败"
    sep = "═" * 63
    lines.append(sep)
    total_actions = sum(action_summary.values())
    build_pct = action_summary.get("BUILD", 0) / total_actions * 100 if total_actions else 0
    upg_pct = action_summary.get("UPGRADE", 0) / total_actions * 100 if total_actions else 0
    noop_pct = action_summary.get("NO_OP", 0) / total_actions * 100 if total_actions else 0
    tech_pct = action_summary.get("TECH_UPGRADE", 0) / total_actions * 100 if total_actions else 0

    lines.append(
        f" Ep{info['episode']:>4}  {pos}  {result_str}  "
        f"{info['total_rounds']}r  {info['duration']:.1f}s  "
        f"奖励{info['total_reward']:+.0f}  "
        f"终局 HP {info['final_our_hp']}:{info['final_enemy_hp']}"
    )
    # 动作汇总
    action_parts = []
    if action_summary.get("BUILD", 0):
        action_parts.append(f"BUILD×{action_summary['BUILD']}({build_pct:.1f}%)")
    if action_summary.get("UPGRADE", 0):
        action_parts.append(f"UPG×{action_summary['UPGRADE']}({upg_pct:.1f}%)")
    if action_summary.get("TECH_UPGRADE", 0):
        action_parts.append(f"TECH×{action_summary['TECH_UPGRADE']}({tech_pct:.1f}%)")
    if action_summary.get("NO_OP", 0):
        action_parts.append(f"NO-OP×{action_summary['NO_OP']}({noop_pct:.1f}%)")
    lines.append(" 动作: " + "  ".join(action_parts))
    lines.append(sep)
    lines.append("")

    # ── 战略阶段演变 ──
    lines.append("阶段    步数        塔数        金币              HP            动作")
    lines.append("─────── ───────── ────────── ──────────────── ────────────── ────────────────────")
    for i, ph in enumerate(phases, 1):
        act_str = "  ".join(f"{k}×{v}" for k, v in sorted(ph["actions"].items()))
        lines.append(
            f"  {i}    {ph['start_step']:>3}~{ph['end_step']:<3}  "
            f"{ph['tower_range']:<10}  "
            f"{ph['coins_range']:<16}  "
            f"{ph['hp']:<14}  "
            f"{act_str}"
        )

    # 全局合计
    lines.append(" " + "─" * 98)
    total_action_str = "  ".join(f"{k}×{v}" for k, v in sorted(action_summary.items()))
    coins_range = f"${metrics['total_income']:.0f}收入 ${metrics['total_spent']:.0f}支出"
    hp_start = info.get("final_our_hp", 0) + metrics["our_hp_loss"] if "final_our_hp" in info else 50
    lines.append(
        f" 合计   {rounds[0]['step'] if data['rounds'] else 0}~{rounds[-1]['step'] if data['rounds'] else 0}  "
        f"—          {coins_range:<16}  "
        f"HP 50→{info.get('final_our_hp', '?')}         "
        f"{total_action_str}"
    )
    lines.append("")

    # ── 关键事件 ──
    lines.append("─ 关键事件 " + "─" * 42)
    if not events:
        lines.append("  (无显著事件)")
    else:
        for e in events:
            lines.append(f"  Step {e['step']:>4}: {e['desc']}")
    lines.append("")

    # ── 补充指标 ──
    lines.append("─ 经济指标 " + "─" * 43)
    lines.append(f"  总收入 ${metrics['total_income']:.0f}  "
                 f"总支出 ${metrics['total_spent']:.0f}  "
                 f"剩余 ${metrics['final_coins']:.0f}  "
                 f"利用率 {metrics['utilization']:.0f}%")
    lines.append(f"  建塔 {metrics['builds']} 次  "
                 f"{metrics['steps_per_build']:.1f} 步/塔  "
                 f"塔均寿 {metrics['avg_tower_lifespan']:.0f} 步")
    lines.append("")

    lines.append("─ 双方对比 " + "─" * 43)
    lines.append(f"                 我方     敌方")
    lines.append(f"  建塔次数        {metrics['builds']:<6}  {metrics['enemy_builds']:<6}")
    lines.append(f"  升级次数        {metrics['upgrades']:<6}  0")
    lines.append(f"  科技升级        {metrics['tech_upgrades']:<6}  0")
    lines.append(f"  终局金币        ${metrics['final_coins']:<5.0f}  "
                 f"${rounds[-1].get('enemy_coins', 0) if data['rounds'] else 0:<5.0f}")
    final_our_towers = len(rounds[-1].get("our_towers", [])) if data["rounds"] else 0
    final_enemy_towers = len(rounds[-1].get("enemy_towers", [])) if data["rounds"] else 0
    lines.append(f"  终局塔数        {final_our_towers:<6}  {final_enemy_towers:<6}")
    lines.append(f"  终局 HP         {info.get('final_our_hp', '?')}        {info.get('final_enemy_hp', '?')}")
    lines.append(f"  HP 总损失       {metrics['our_hp_loss']:<6}  {metrics['enemy_hp_loss']:<6}")
    lines.append("")

    tech = metrics["final_tech"]
    lines.append("─ 科技发展 " + "─" * 43)
    lines.append(f"  gen_speed:     L{tech.get('gen_speed', 0)}")
    lines.append(f"  ant_strength:  L{tech.get('ant_strength', 0)}")
    # 科技升级可能性
    if metrics["total_income"] >= 200:
        lines.append(f"  收入 ${metrics['total_income']:.0f} ≥200g, 科技升级在经济上可行")
    else:
        lines.append(f"  收入 ${metrics['total_income']:.0f} <200g, 未达到科技升级门槛")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def generate_report(log_path: str) -> str:
    """从 JSON 日志路径生成完整报告，返回 Markdown 字符串。"""
    if not os.path.exists(log_path):
        return f"错误：文件不存在 — {log_path}"

    with open(log_path) as f:
        raw = json.load(f)

    rounds = raw.get("round_details", [])
    if not rounds:
        return "错误：日志中没有回合数据"

    info = {
        "episode": raw.get("episode", 0),
        "opponent_id": raw.get("opponent_id", "?"),
        "player_position": raw.get("player_position", 0),
        "result": raw.get("result", "?"),
        "total_reward": raw.get("total_reward", 0),
        "total_rounds": len(rounds) - 1,
        "start_time": raw.get("start_time", ""),
        "end_time": raw.get("end_time", ""),
        "final_our_hp": rounds[-1].get("our_hp", 0) if rounds else 0,
        "final_enemy_hp": rounds[-1].get("enemy_hp", 0) if rounds else 0,
    }
    if info["start_time"] and info["end_time"]:
        from datetime import datetime
        try:
            start_dt = datetime.strptime(info["start_time"], "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(info["end_time"], "%Y-%m-%d %H:%M:%S")
            info["duration"] = (end_dt - start_dt).total_seconds()
        except ValueError:
            info["duration"] = 0
    else:
        info["duration"] = 0

    actions_per_step, tower_history, action_summary = derive_actions(rounds)
    phases = segment_phases(rounds, actions_per_step)
    events = find_key_events(rounds, actions_per_step)
    metrics = compute_metrics(rounds, actions_per_step, tower_history, action_summary)

    data = {
        "info": info,
        "rounds": rounds,
        "phases": phases,
        "events": events,
        "metrics": metrics,
        "action_summary": action_summary,
        "tower_history": tower_history,
    }
    return format_report(data)


def format_battle_detail(log_path: str) -> None:
    """从 JSON 日志生成完整报告并打印。"""
    report = generate_report(log_path)
    print(report)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成 SelfPlay 对战摘要报告")
    parser.add_argument("log_path", help="JSON 对战日志路径")
    args = parser.parse_args()
    format_battle_detail(args.log_path)


if __name__ == "__main__":
    main()
