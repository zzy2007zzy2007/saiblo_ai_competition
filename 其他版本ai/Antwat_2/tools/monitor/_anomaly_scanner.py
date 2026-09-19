#!/usr/bin/env python3
"""
训练异常扫描器 (ppo_v2 适配版)
快速扫描训练日志和指标，发现异常但不做深入诊断。
输出结构化的异常清单。

数据源:
  - batch_metrics.jsonl        PPO update 级训练指标
  - selfplay_battle_log.jsonl  每局 SelfPlay 对战数据
  - training_{time}.log        loguru INFO 级别日志
  - training_error_{time}.log  loguru ERROR 级别日志

配置阈值:
  - 从 ppo_antwar.yaml 读取 max_grad_norm / max_grad_norm_vf
  - 通过 --max_grad_norm / --max_grad_norm_vf CLI 参数传入
"""
import argparse
import glob
import json
import os
import re
import sys
import math


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def scan_metrics(
    batch_metrics_path: str,
    battle_log_path: str,
    max_grad_norm: float = 2.5,
    max_grad_norm_vf: float = 1000.0,
) -> list:
    """扫描 batch_metrics.jsonl + selfplay_battle_log.jsonl

    Args:
        max_grad_norm: 策略梯度裁剪阈值（YAML ppo.max_grad_norm）
        max_grad_norm_vf: 价值梯度裁剪阈值（YAML ppo.max_grad_norm_vf）
    """
    records = load_jsonl(batch_metrics_path)
    battle_records = load_jsonl(battle_log_path)
    findings = []

    if not records:
        findings.append({"severity": "WARN", "type": "no_data",
                         "msg": "batch_metrics.jsonl 为空或不存在"})
        return findings

    # 为每条记录添加 step 序号
    total = len(records)
    for i, r in enumerate(records, start=1):
        r['_step'] = i

    # 基础统计
    findings.append({"severity": "INFO", "type": "summary",
                     "msg": f"共 {total} 条 PPO update 记录"})

    # 1. NaN 检测
    for r in records:
        for key in ["grad_norm", "grad_norm_policy", "grad_norm_value"]:
            v = r.get(key)
            if isinstance(v, str) and v.upper() in ("NAN", "INF"):
                findings.append({"severity": "CRITICAL", "type": "nan_in_metric",
                                 "msg": f"Step {r['_step']} 的 {key} 值为 {v}"})
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                findings.append({"severity": "CRITICAL", "type": "nan_in_metric",
                                 "msg": f"Step {r['_step']} 的 {key} 值为 {v}"})

    # 2. entropy 过低 — 熵坍塌
    low_entropy = [r for r in records if r.get("entropy", 1) < 0.01]
    if low_entropy:
        first_low = low_entropy[0]
        findings.append({
            "severity": "CRITICAL", "type": "entropy_collapse",
            "msg": f"entropy < 0.01 自 Step {first_low['_step']} 起, "
                   f"共 {len(low_entropy)}/{total} 次, "
                   f"最低 entropy={min(r['entropy'] for r in low_entropy):.6f}"
        })

    # 3. clip_fraction 过高
    high_clip = [r for r in records if r.get("clip_fraction", 0) > 0.5]
    if high_clip:
        max_c = max(high_clip, key=lambda x: x["clip_fraction"])
        findings.append({
            "severity": "WARN", "type": "high_clip",
            "msg": f"clip_fraction > 0.5 共 {len(high_clip)}/{total} 次, "
                   f"最高 clip_fraction={max_c['clip_fraction']:.4f} (Step {max_c['_step']})"
        })

    # 4. 策略梯度爆炸 — 检查 grad_norm_policy 是否超过裁剪阈值的 10×
    policy_threshold = max_grad_norm * 10.0
    high_grad = [r for r in records
                 if isinstance(r.get("grad_norm_policy"), (int, float))
                 and r["grad_norm_policy"] > policy_threshold]
    if high_grad:
        max_g = max(high_grad, key=lambda x: x["grad_norm_policy"])
        findings.append({
            "severity": "WARN", "type": "grad_explosion",
            "msg": f"grad_norm_policy > {policy_threshold:.1f} (max_grad_norm×10) "
                   f"共 {len(high_grad)}/{total} 次, "
                   f"最大={max_g['grad_norm_policy']:.1f} (Step {max_g['_step']})"
        })

    # 5. nan_skip_count > 0 — 批次跳过
    skipped = [r for r in records if r.get("nan_skip_count", 0) > 0]
    if skipped:
        total_skipped = sum(r["nan_skip_count"] for r in skipped)
        findings.append({
            "severity": "WARN", "type": "nan_skip_count",
            "msg": f"nan_skip_count > 0 共 {len(skipped)}/{len(records)} 次, "
                   f"累计跳过 {total_skipped} 个 minibatch"
        })

    # 6. NO-OP 占比过高（策略坍缩）
    high_noop = [r for r in records
                 if isinstance(r.get("type_ratios"), dict)
                 and r["type_ratios"].get("noop_ratio", 0) > 0.95]
    if not high_noop:
        high_noop = [r for r in records if r.get("type_noop_ratio", 0) > 0.95]
    if high_noop:
        first_hn = high_noop[0]
        findings.append({
            "severity": "CRITICAL", "type": "noop_collapse",
            "msg": f"NO-OP 占比 > 95% 自 Step {first_hn['_step']} 起, "
                   f"共 {len(high_noop)}/{total} 次"
        })

    # 7. value_grad 偏高 — 阈值 = max_grad_norm_vf × 0.8（对齐裁剪阈值）
    vf_warn_threshold = max_grad_norm_vf * 0.8
    vf_records = [r for r in records
                  if isinstance(r.get("grad_norm_value"), (int, float))
                  and r["grad_norm_value"] > vf_warn_threshold]
    if vf_records:
        max_vf = max(vf_records, key=lambda x: x["grad_norm_value"])
        findings.append({
            "severity": "WARN", "type": "value_grad_high",
            "msg": f"grad_norm_value > {vf_warn_threshold:.1f} (max_grad_norm_vf×0.8) "
                   f"共 {len(vf_records)}/{total} 次, "
                   f"最大 {max_vf['grad_norm_value']:.1f} (Step {max_vf['_step']})"
        })

    # 8. 损失发散（policy_loss 持续上升）
    policy_losses = [(r['_step'], r.get('policy_loss', 0))
                     for r in records if 'policy_loss' in r]
    if len(policy_losses) >= 10:
        recent = policy_losses[-10:]
        rising = all(recent[i][1] < recent[i+1][1] for i in range(len(recent)-1))
        if rising and recent[-1][1] > recent[0][1] * 1.5:
            findings.append({
                "severity": "WARN", "type": "loss_divergence",
                "msg": f"policy_loss 持续上升 10 轮: {recent[0][1]:.4f} → {recent[-1][1]:.4f} "
                       f"(Step {recent[0][0]}-{recent[-1][0]})"
            })

    # 9. 奖励异常（reward_mean 显著偏离）
    reward_means = [r.get('reward_mean') for r in records
                    if isinstance(r.get('reward_mean'), (int, float))]
    if reward_means:
        mean_r = sum(reward_means) / len(reward_means)
        std_r = (sum((m - mean_r) ** 2 for m in reward_means) / len(reward_means)) ** 0.5
        recent_means = reward_means[-5:]
        outliers = [m for m in recent_means if abs(m - mean_r) > 3 * max(std_r, 1e-6)]
        if outliers:
            findings.append({
                "severity": "WARN", "type": "reward_anomaly",
                "msg": f"reward_mean 偏离 3σ: 全局 mean={mean_r:.2f}, std={std_r:.2f}, "
                       f"最近值异常: {[f'{m:.2f}' for m in outliers]}"
            })

    # 10. value_loss 尖峰 — 直接检查 value_loss 的异常离群值
    value_losses = [(r['_step'], r.get('value_loss', 0))
                    for r in records if 'value_loss' in r]
    if len(value_losses) >= 4:
        losses = [v for _, v in value_losses]
        median_vl = sorted(losses)[len(losses) // 2]
        spikes = [(s, v) for s, v in value_losses
                  if v > median_vl * 5 and v > 1.0]  # 超过中位数 5 倍 + 非零
        if spikes:
            max_spike = max(spikes, key=lambda x: x[1])
            findings.append({
                "severity": "WARN", "type": "value_loss_spike",
                "msg": f"value_loss 存在尖峰: {len(spikes)}/{total} 次超过中位数的 5× "
                       f"(median={median_vl:.1f}), "
                       f"最大={max_spike[1]:.1f} (Step {max_spike[0]})"
            })

    # 11. 经济崩溃检测
    if battle_records:
        eco_episodes = []
        for b in battle_records:
            own_coins = b.get('end_own_coins')
            enemy_coins = b.get('end_enemy_coins')
            if (isinstance(own_coins, (int, float))
                    and isinstance(enemy_coins, (int, float))
                    and enemy_coins > 0
                    and own_coins / enemy_coins < 0.5):
                eco_episodes.append(b)
        if eco_episodes:
            worst = min(eco_episodes, key=lambda x: x['end_own_coins'] / max(x['end_enemy_coins'], 1))
            ratio = worst['end_own_coins'] / max(worst['end_enemy_coins'], 1)
            findings.append({
                "severity": "WARN", "type": "economic_collapse",
                "msg": f"经济崩溃: end_own_coins/end_enemy_coins < 0.5 "
                       f"共 {len(eco_episodes)}/{len(battle_records)} 次, "
                       f"最低 ratio={ratio:.2f} (Ep {worst['episode']})"
            })

    return findings


def scan_log(training_dir: str) -> list:
    """扫描所有日志文件中的 WARNING 和 ERROR 记录。

    覆盖:
      - training/training_*.log
      - training/training_error_*.log
      - evaluation/eval_battle_round_log_*.log
      - selfplay/sp_all.log
    """
    findings = []

    run_dir = os.path.dirname(training_dir)
    train_dir = training_dir
    eval_dir = os.path.join(run_dir, "evaluation")
    sp_dir = os.path.join(run_dir, "selfplay")

    all_log_files = []
    for d, label in [(train_dir, "training"), (eval_dir, "evaluation"), (sp_dir, "selfplay")]:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".log"):
                all_log_files.append((os.path.join(d, f), label))

    if not all_log_files:
        return [{"severity": "WARN", "type": "no_log",
                 "msg": "未找到任何 .log 文件"}]

    findings.append({"severity": "INFO", "type": "summary",
                     "msg": f"扫描 {len(all_log_files)} 个日志文件"})

    # ── 逐文件提取 WARNING / ERROR 行 ─────────────────────────────
    warn_error_re = re.compile(r"\| (WARNING|ERROR)\s+\|.*?\|\s+(.*)")

    raw_errors = []
    raw_warnings = []
    for filepath, label in all_log_files:
        try:
            with open(filepath) as f:
                content = f.read()
        except (IOError, OSError):
            continue
        for m in warn_error_re.finditer(content):
            level = m.group(1)
            msg = m.group(2).strip()
            entry = (label, os.path.basename(filepath), msg)
            if level == "ERROR":
                raw_errors.append(entry)
            else:
                raw_warnings.append(entry)

    # ── 汇总统计 ─────────────────────────────────────────────────
    findings.append({"severity": "INFO", "type": "log_stats",
                     "msg": f"日志中共发现 ERROR {len(raw_errors)} 条, WARNING {len(raw_warnings)} 条"})

    if not raw_errors and not raw_warnings:
        findings.append({"severity": "INFO", "type": "log_clean",
                         "msg": "所有日志中未发现 ERROR / WARNING"})
        return findings

    # ── 分组相似消息 ─────────────────────────────────────────────
    def extract_pattern(msg: str) -> str:
        p = msg
        p = re.sub(r'\b\d+\b', '<N>', p)                    # 纯数字
        p = re.sub(r'0x[0-9a-fA-F]+', '<HEX>', p)           # 十六进制
        p = re.sub(r'[0-9a-f]{8,}', '<HASH>', p)             # 长 hash
        p = re.sub(r'\d+\.\d+', '<F>', p)                    # 浮点数
        return p

    def group_messages(entries, entry_type):
        """将相似消息分组。
        entry_type: "error" → CRITICAL, "warning" → WARN（不按数量升级）
        """
        groups = {}
        for entry in entries:
            label, filename, msg = entry
            pattern = extract_pattern(msg)
            if pattern not in groups:
                groups[pattern] = []
            groups[pattern].append(entry)

        # ERROR → CRITICAL, WARNING → WARN（保持不变，不因数量升级）
        base_severity = "CRITICAL" if entry_type == "error" else "WARN"

        results = []
        for pattern, items in sorted(groups.items(), key=lambda x: -len(x[1])):
            count = len(items)
            _, _, sample_msg = items[0]
            if len(sample_msg) > 150:
                sample_msg = sample_msg[:147] + "..."

            sources = set(f"{l}/{f}" for l, f, _ in items)
            source_str = ", ".join(sorted(sources)[:3])
            if len(sources) > 3:
                source_str += f" (+{len(sources)-3})"

            results.append({
                "severity": base_severity,
                "type": "log_warning" if entry_type == "warning" else "log_error",
                "msg": f"[{source_str}] (×{count}) {sample_msg}"
            })

        return results

    error_findings = group_messages(raw_errors, "error")
    warn_findings = group_messages(raw_warnings, "warning")

    findings.extend(error_findings)
    findings.extend(warn_findings)

    # ── 特定异常模式检测 ──────────────────────────────────────────
    all_content = ""
    for fp, _ in all_log_files:
        try:
            with open(fp) as f:
                all_content += f.read() + "\n"
        except (IOError, OSError):
            pass

    if not all_content:
        return findings

    grad_nan_count = all_content.count("[权重监控] 梯度 NaN")
    if grad_nan_count > 0:
        findings.append({
            "severity": "CRITICAL", "type": "grad_nan",
            "msg": f"梯度 NaN 检测到 {grad_nan_count} 次"
        })

    weight_nan_count = all_content.count("[权重监控] 权重 NaN")
    if weight_nan_count > 0:
        findings.append({
            "severity": "CRITICAL", "type": "weight_nan",
            "msg": f"权重 NaN 写入检测到 {weight_nan_count} 次"
        })

    loss_skip = re.findall(r"loss 异常.*?跳过此批次", all_content)
    if loss_skip:
        findings.append({
            "severity": "WARN", "type": "loss_anomaly",
            "msg": f"loss 异常跳过 {len(loss_skip)} 次"
        })

    time_matches = re.findall(r"训练总耗时.*?(\d+\.?\d*)秒", all_content)
    if time_matches:
        total_sec = float(time_matches[-1])
        findings.append({
            "severity": "INFO", "type": "total_time",
            "msg": f"训练总耗时: {total_sec:.1f}s ({total_sec/60:.1f}min)"
        })

    return findings


def main():
    parser = argparse.ArgumentParser(description="训练异常扫描器")
    parser.add_argument("training_dir", help="训练目录路径 (outputs/<run_id>/training)")
    parser.add_argument("--max_grad_norm", type=float, default=2.5,
                        help="策略梯度裁剪阈值 (来自 YAML ppo.max_grad_norm)")
    parser.add_argument("--max_grad_norm_vf", type=float, default=1000.0,
                        help="价值梯度裁剪阈值 (来自 YAML ppo.max_grad_norm_vf)")
    args = parser.parse_args()

    training_dir = args.training_dir

    if not os.path.isdir(training_dir):
        print(json.dumps([{"severity": "ERROR", "type": "dir_not_found",
                           "msg": f"目录不存在: {training_dir}"}]))
        sys.exit(1)

    findings = []

    # 扫描指标（传递配置阈值）
    batch_metrics_path = os.path.join(training_dir, "batch_metrics.jsonl")
    battle_log_path = os.path.join(training_dir, "selfplay_battle_log.jsonl")
    findings.extend(scan_metrics(
        batch_metrics_path,
        battle_log_path,
        max_grad_norm=args.max_grad_norm,
        max_grad_norm_vf=args.max_grad_norm_vf,
    ))

    # 扫描日志
    findings.extend(scan_log(training_dir))

    # 统计
    criticals = [f for f in findings if f["severity"] == "CRITICAL"]
    warnings = [f for f in findings if f["severity"] == "WARN"]
    infos = [f for f in findings if f["severity"] == "INFO"]

    if not criticals and not warnings:
        findings.append({"severity": "INFO", "type": "clean", "msg": "未发现异常"})

    findings.append({
        "severity": "INFO", "type": "scan_summary",
        "msg": f"扫描完成: CRITICAL={len(criticals)}, WARN={len(warnings)}, INFO={len(infos)}"
    })

    print(json.dumps(findings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
