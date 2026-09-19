#!/usr/bin/env python3
"""
训练异常扫描器 (ppo_v2 适配版)
快速扫描训练日志和指标，发现异常但不做深入诊断。
输出结构化的异常清单。

数据源变化:
  - training_metrics_history.jsonl → batch_metrics.jsonl + selfplay_battle_log.jsonl
  - training.log → training_{time}.log (loguru 轮转)
  - 移除: hard_entropy_penalty、value_pred_mean、valid_actions_min 检测
"""
import glob
import json
import os
import re
import sys
import math
from collections import defaultdict


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def scan_metrics(batch_metrics_path: str, battle_log_path: str) -> list:
    """扫描 batch_metrics.jsonl + selfplay_battle_log.jsonl"""
    records = load_jsonl(batch_metrics_path)
    battle_records = load_jsonl(battle_log_path)
    findings = []

    if not records:
        findings.append({"severity": "WARN", "type": "no_data",
                         "msg": "batch_metrics.jsonl 为空或不存在"})
        return findings

    # 基础统计
    eps = [r["episode"] for r in records]
    last_ep = eps[-1]
    findings.append({"severity": "INFO", "type": "summary",
                     "msg": f"共 {len(records)} 条 PPO update 记录, 最新 Ep {last_ep}"})

    # 1. NaN 检测 — grad_norm / grad_norm_policy / grad_norm_value
    nan_found = False
    for r in records:
        for key in ["grad_norm", "grad_norm_policy", "grad_norm_value"]:
            v = r.get(key)
            if isinstance(v, str) and v.upper() in ("NAN", "INF"):
                findings.append({"severity": "CRITICAL", "type": "nan_in_metric",
                                 "msg": f"Ep {r['episode']} 的 {key} 值为 {v}"})
                nan_found = True
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                findings.append({"severity": "CRITICAL", "type": "nan_in_metric",
                                 "msg": f"Ep {r['episode']} 的 {key} 值为 {v}"})
                nan_found = True

    # 2. entropy 过低 — 熵坍塌
    low_entropy = [r for r in records if r.get("entropy", 1) < 0.01]
    if low_entropy:
        first_low = low_entropy[0]
        findings.append({
            "severity": "CRITICAL", "type": "entropy_collapse",
            "msg": f"entropy < 0.01 自 Ep {first_low['episode']} 起, "
                   f"共 {len(low_entropy)}/{len(records)} 次, "
                   f"最低 entropy={min(r['entropy'] for r in low_entropy):.6f}"
        })

    # 3. clip_fraction 过高
    high_clip = [r for r in records if r.get("clip_fraction", 0) > 0.5]
    if high_clip:
        max_c = max(high_clip, key=lambda x: x["clip_fraction"])
        findings.append({
            "severity": "WARN", "type": "high_clip",
            "msg": f"clip_fraction > 0.5 共 {len(high_clip)}/{len(records)} 次, "
                   f"最高 clip_fraction={max_c['clip_fraction']:.4f} (Ep {max_c['episode']})"
        })

    # 4. grad_norm 异常 — 梯度爆炸
    high_grad = [r for r in records
                 if isinstance(r.get("grad_norm"), (int, float))
                 and r["grad_norm"] > 500]
    if high_grad:
        max_g = max(high_grad, key=lambda x: x["grad_norm"])
        findings.append({
            "severity": "WARN", "type": "grad_explosion",
            "msg": f"grad_norm > 500 共 {len(high_grad)}/{len(records)} 次, "
                   f"最大 grad_norm={max_g['grad_norm']:.1f} (Ep {max_g['episode']})"
        })

    # 5. skipped_minibatches > 0 — 批次跳过
    skipped = [r for r in records if r.get("skipped_minibatches", 0) > 0]
    if skipped:
        total_skipped = sum(r["skipped_minibatches"] for r in skipped)
        findings.append({
            "severity": "WARN", "type": "skipped_minibatches",
            "msg": f"skipped_minibatches > 0 共 {len(skipped)}/{len(records)} 次, "
                   f"累计跳过 {total_skipped} 个 minibatch"
        })

    # 6. NO-OP 占比过高（策略坍缩）
    high_noop = [r for r in records
                 if isinstance(r.get("type_ratios"), dict)
                 and r["type_ratios"].get("noop_ratio", 0) > 0.95]
    if not high_noop:
        # 兼容旧字段名
        high_noop = [r for r in records if r.get("type_noop_ratio", 0) > 0.95]
    if high_noop:
        first_hn = high_noop[0]
        findings.append({
            "severity": "CRITICAL", "type": "noop_collapse",
            "msg": f"NO-OP 占比 > 95% 自 Ep {first_hn['episode']} 起, "
                   f"共 {len(high_noop)}/{len(records)} 次"
        })

    # 7. value_grad 过高
    vf_records = [r for r in records if r.get("grad_norm_value", 0) > 200]
    if vf_records:
        max_vf = max(vf_records, key=lambda x: x.get("grad_norm_value", 0))
        findings.append({
            "severity": "WARN", "type": "value_grad_high",
            "msg": f"grad_norm_value > 200 共 {len(vf_records)}/{len(records)} 次, "
                   f"最大 {max_vf['grad_norm_value']:.1f} (Ep {max_vf['episode']})"
        })

    # 8. 损失发散（policy_loss 持续上升）
    policy_losses = [(r['episode'], r.get('policy_loss', 0))
                     for r in records if 'policy_loss' in r]
    if len(policy_losses) >= 10:
        recent = policy_losses[-10:]
        rising = all(recent[i][1] < recent[i+1][1] for i in range(len(recent)-1))
        if rising and recent[-1][1] > recent[0][1] * 1.5:
            findings.append({
                "severity": "WARN", "type": "loss_divergence",
                "msg": f"policy_loss 持续上升 10 轮: {recent[0][1]:.4f} → {recent[-1][1]:.4f} "
                       f"(Ep {recent[0][0]}-{recent[-1][0]})"
            })

    # 9. 奖励异常（reward_mean 或 reward_std 异常）
    reward_means = [r.get('reward_mean') for r in records
                    if isinstance(r.get('reward_mean'), (int, float))]
    reward_stds = [r.get('reward_std') for r in records
                   if isinstance(r.get('reward_std'), (int, float))]
    if reward_means:
        mean_r = sum(reward_means) / len(reward_means)
        std_r = (sum((m - mean_r) ** 2 for m in reward_means) / len(reward_means)) ** 0.5
        # 检测最近 5 条是否偏离 3 sigma
        recent_means = reward_means[-5:]
        outliers = [m for m in recent_means if abs(m - mean_r) > 3 * max(std_r, 1e-6)]
        if outliers:
            findings.append({
                "severity": "WARN", "type": "reward_anomaly",
                "msg": f"reward_mean 偏离 3σ: 全局 mean={mean_r:.2f}, std={std_r:.2f}, "
                       f"最近值异常: {[f'{m:.2f}' for m in outliers]}"
            })

    # 10. value_loss 尖峰
    if reward_stds:
        mean_std = sum(reward_stds) / len(reward_stds)
        if mean_std > 10:
            findings.append({
                "severity": "WARN", "type": "value_loss_spike",
                "msg": f"reward_std 平均值偏高: {mean_std:.2f}, "
                       f"可能表示 value_loss 存在尖峰"
            })

    # 11. 经济崩溃检测（从 selfplay_battle_log.jsonl 读取）
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
                "msg": f"经济崩溃: end_own_coins/end_enemy_coins < 0.5 共 {len(eco_episodes)}/{len(battle_records)} 次, "
                       f"最低 ratio={ratio:.2f} (Ep {worst['episode']})"
            })

    return findings


def scan_log(log_dir: str) -> list:
    """扫描 training_{time}.log 和 training_error_{time}.log 中的异常"""
    findings = []

    # 查找日志文件
    log_pattern = os.path.join(log_dir, "training_*.log")
    log_files = sorted(glob.glob(log_pattern))

    # 优先找 error 日志
    error_logs = [f for f in log_files if "error" in os.path.basename(f)]
    main_logs = [f for f in log_files if "error" not in os.path.basename(f)]

    content = ""
    for log_file in error_logs + main_logs:
        try:
            with open(log_file) as f:
                content += f.read() + "\n"
        except (IOError, OSError):
            pass

    if not content:
        return [{"severity": "WARN", "type": "no_log", "msg": "training_*.log 不存在"}]

    # 1. 梯度 NaN (ERROR 级别)
    grad_nan_count = content.count("[权重监控] 梯度 NaN")
    nan_error_count = len(re.findall(r"ERROR.*[Nn]an|ERROR.*[Ii]nf", content))
    if grad_nan_count > 0:
        findings.append({
            "severity": "CRITICAL", "type": "grad_nan",
            "msg": f"梯度 NaN 检测到 {grad_nan_count} 次 — 查看 [权重监控] 梯度 NaN 行"
        })
    if nan_error_count > grad_nan_count:
        findings.append({
            "severity": "WARN", "type": "nan_in_log",
            "msg": f"日志中发现 {nan_error_count} 处 NaN/Inf 相关 ERROR 记录"
        })

    # 2. 权重 NaN
    weight_nan_count = content.count("[权重监控] 权重 NaN")
    if weight_nan_count > 0:
        findings.append({
            "severity": "CRITICAL", "type": "weight_nan",
            "msg": f"权重 NaN 写入检测到 {weight_nan_count} 次"
        })

    # 3. loss 异常
    loss_skip = re.findall(r"loss 异常.*?跳过此批次", content)
    if loss_skip:
        findings.append({
            "severity": "WARN", "type": "loss_anomaly",
            "msg": f"loss 异常跳过 {len(loss_skip)} 次"
        })

    # 4. 梯度范数过大
    grad_warnings = re.findall(r"梯度范数.*?远超过阈值", content)
    if grad_warnings:
        findings.append({
            "severity": "WARN", "type": "grad_warning",
            "msg": f"梯度范数远超过阈值 {len(grad_warnings)} 次"
        })

    # 5. 价值网络梯度范数过大
    value_grad_warnings = re.findall(r"价值网络参数.*?梯度范数过大", content)
    if value_grad_warnings:
        findings.append({
            "severity": "WARN", "type": "value_grad_warning",
            "msg": f"价值网络参数梯度范数过大 {len(value_grad_warnings)} 次"
        })

    # 6. 对手胜率异常
    win_rates = re.findall(r"对手胜率.*?(\d+\.?\d*)", content)
    bad_rates = [float(x) for x in win_rates if float(x) < 0.2]
    if bad_rates:
        findings.append({
            "severity": "WARN", "type": "low_win_rate",
            "msg": f"对手胜率 < 20% 共 {len(bad_rates)} 次"
        })

    # 7. 训练总时长
    time_matches = re.findall(r"训练总耗时.*?(\d+\.?\d*)秒", content)
    if time_matches:
        total_sec = float(time_matches[-1])
        findings.append({
            "severity": "INFO", "type": "total_time",
            "msg": f"训练总耗时: {total_sec:.1f}s ({total_sec/60:.1f}min)"
        })

    return findings


def main():
    if len(sys.argv) < 2:
        print(json.dumps([{"severity": "ERROR", "type": "usage",
                           "msg": "用法: python3 _anomaly_scanner.py <training_dir>"}]))
        sys.exit(1)

    training_dir = sys.argv[1]

    if not os.path.isdir(training_dir):
        print(json.dumps([{"severity": "ERROR", "type": "dir_not_found",
                           "msg": f"目录不存在: {training_dir}"}]))
        sys.exit(1)

    findings = []

    # 扫描指标
    batch_metrics_path = os.path.join(training_dir, "batch_metrics.jsonl")
    battle_log_path = os.path.join(training_dir, "selfplay_battle_log.jsonl")
    findings.extend(scan_metrics(batch_metrics_path, battle_log_path))

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
