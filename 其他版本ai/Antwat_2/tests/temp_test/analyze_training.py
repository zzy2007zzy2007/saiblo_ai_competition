#!/usr/bin/env python3
import json
from datetime import datetime
import yaml
from collections import defaultdict


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    # Load all data files
    base_dir = '/Users/raymondzeng/Documents/trae_projects/AntWar'
    training_metrics = load_json(f'{base_dir}/temp_server5_logs/training_metrics.json')
    system_metrics = load_json(f'{base_dir}/temp_server5_logs/system_metrics.json')
    battle_stats = load_json(f'{base_dir}/temp_server5_logs/battle_stats.json')
    config = load_yaml(f'{base_dir}/temp_server5_logs/config.yaml')
    
    print("=" * 120)
    print(" " * 40 + "SERVER5 - PPO_V1 训练状态全面分析报告")
    print("=" * 120)
    print()
    
    # ============================================================
    # 1. 训练基本信息
    # ============================================================
    print("\033[1;34m" + "=" * 120 + "\033[0m")
    print("\033[1;34m1. 训练基本信息\033[0m")
    print("\033[1;34m" + "=" * 120 + "\033[0m")
    print()
    
    print("  \033[1m1.1 完整参数配置:\033[0m")
    print("  -" * 60)
    
    training_cfg = config['training']
    ppo_cfg = config['ppo']
    network_cfg = config['network']
    selfplay_cfg = config['selfplay']
    
    print(f"    \033[1m训练配置:\033[0m")
    print(f"      - 总训练轮数 (total_episodes): {training_cfg['total_episodes']}")
    print(f"      - 批量大小 (batch_size): {ppo_cfg['batch_size']}")
    print(f"      - 并行环境数 (num_envs): {training_cfg['num_envs']}")
    print(f"      - 对战次数 (n_battles): {training_cfg['n_battles']}")
    print(f"      - 对战间隔 (battle_interval): {training_cfg['battle_interval']}")
    print(f"      - 模型保存间隔 (save_interval): {training_cfg['save_interval']}")
    print(f"      - 评估间隔 (eval_interval): {training_cfg['eval_interval']}")
    print(f"      - 对手更新间隔 (opponent_update_interval): {training_cfg['opponent_update_interval']}")
    print()
    
    print(f"    \033[1mPPO算法配置:\033[0m")
    print(f"      - 学习率 (lr): {ppo_cfg['lr']}")
    print(f"      - Clip参数 (clip_eps): {ppo_cfg['clip_eps']}")
    print(f"      - 折扣因子 (gamma): {ppo_cfg['gamma']}")
    print(f"      - GAE lambda (gae_lambda): {ppo_cfg['gae_lambda']}")
    print(f"      - 熵系数 (ent_coef): {ppo_cfg['ent_coef']}")
    print(f"      - 最终熵系数 (ent_coef_final): {ppo_cfg['ent_coef_final']}")
    print(f"      - 梯度裁剪阈值 (max_grad_norm): {ppo_cfg['max_grad_norm']}")
    print(f"      - PPO epoch数 (ppo_epochs): {ppo_cfg['ppo_epochs']}")
    print(f"      - 价值函数系数 (vf_coef): {ppo_cfg['vf_coef']}")
    print(f"      - 学习率预热轮数 (lr_warmup_episodes): {ppo_cfg['lr_warmup_episodes']}")
    print(f"      - 初始学习率 (lr_warmup_init): {ppo_cfg['lr_warmup_init']}")
    print()
    
    print(f"    \033[1m神经网络配置:\033[0m")
    print(f"      - 动作维度 (action_dim): {network_cfg['action_dim']}")
    print(f"      - 棋盘形状 (board_shape): {network_cfg['board_shape']}")
    print(f"      - 全局维度 (global_dim): {network_cfg['global_dim']}")
    print(f"      - 隐藏层维度 (hidden_dim): {network_cfg['hidden_dim']}")
    print()
    
    print(f"    \033[1m自对战配置:\033[0m")
    print(f"      - 利用概率 (exploit_prob): {selfplay_cfg['exploit_prob']}")
    print(f"      - 探索概率 (explore_prob): {selfplay_cfg['explore_prob']}")
    print(f"      - 最小对手对局数 (min_opponent_games): {selfplay_cfg['min_opponent_games']}")
    print(f"      - 对手池大小 (opponent_pool_size): {selfplay_cfg['opponent_pool_size']}")
    print()
    
    print(f"    \033[1m系统配置:\033[0m")
    print(f"      - 设备 (device): {config['system']['device']}")
    print(f"      - 随机种子 (seed): {config['system']['seed']}")
    print()
    
    print("  \033[1m1.2 训练启动命令完整参数:\033[0m")
    print("  -" * 60)
    print("    python train.py --config config.yaml --model ppo_v1")
    print()
    
    print("  \033[1m1.3 训练目录路径和日志文件位置:\033[0m")
    print("  -" * 60)
    print("    远程目录: /root/autodl-tmp/AntWar/ppo_v1/logs/[timestamp]/")
    print("    日志文件: /root/autodl-tmp/AntWar/ppo_v1/logs/[timestamp]/training/training.log")
    print("    指标文件: /root/autodl-tmp/AntWar/ppo_v1/logs/[timestamp]/training/training_metrics.json")
    print("    对战统计: /root/autodl-tmp/AntWar/ppo_v1/logs/[timestamp]/battle_stats.json")
    print("    系统资源: /root/autodl-tmp/AntWar/ppo_v1/logs/[timestamp]/system_metrics.json")
    print()
    
    # ============================================================
    # 2. 训练进度详细信息
    # ============================================================
    print("\033[1;34m" + "=" * 120 + "\033[0m")
    print("\033[1;34m2. 训练进度详细信息\033[0m")
    print("\033[1;34m" + "=" * 120 + "\033[0m")
    print()
    
    latest = training_metrics['latest']
    history = training_metrics['history']
    stats = training_metrics['stats']
    
    # Deduplicate history (some entries have duplicates)
    unique_history = []
    seen_episodes = set()
    for entry in history:
        ep = entry['episode']
        if ep not in seen_episodes:
            seen_episodes.add(ep)
            unique_history.append(entry)
    
    history = unique_history
    history.sort(key=lambda x: x['episode'])
    
    print(f"  \033[1m2.1 当前训练轮数和总轮数:\033[0m")
    print(f"    - 当前轮数: {latest['episode']} / {training_cfg['total_episodes']}")
    progress_pct = (latest['episode'] / training_cfg['total_episodes']) * 100
    print(f"    - 完成进度: {progress_pct:.1f}%")
    print()
    
    # Calculate timestamps
    timestamps = []
    for entry in history:
        if 'timestamp' in entry:
            try:
                ts = datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S')
                timestamps.append(ts)
            except:
                pass
    
    if len(timestamps) >= 2:
        start_time = timestamps[0]
        current_time = timestamps[-1]
        total_duration = current_time - start_time
        total_minutes = total_duration.total_seconds() / 60
        avg_episode_time = total_duration.total_seconds() / len(history) if len(history) > 0 else 0
        
        print(f"  \033[1m2.2 训练开始时间和当前时间:\033[0m")
        print(f"    - 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    - 当前时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print(f"  \033[1m2.3 训练总时长:\033[0m")
        print(f"    - 总时长: {total_minutes:.2f} 分钟")
        print(f"    - 即: {total_duration}")
        print()
        
        print(f"  \033[1m2.4 已完成的训练步数:\033[0m")
        print(f"    - 总步数: {latest['steps']:,}")
        print()
        
        print(f"  \033[1m2.5 平均每轮训练耗时:\033[0m")
        print(f"    - 平均每轮: {avg_episode_time:.2f} 秒")
        print()
        
        # Recent episodes timing
        print(f"  \033[1m2.6 最近20轮的训练耗时明细:\033[0m")
        print(f"    {'轮次':>6} | {'时间戳':>20} | {'奖励':>10} | {'损失':>14} | {'步数':>10}")
        print(f"    " + "-" * 80)
        
        recent_history = history[-20:] if len(history) >= 20 else history
        for entry in recent_history:
            ts_str = entry.get('timestamp', 'N/A')[:20]
            ep = entry['episode']
            reward = entry['avg_reward']
            loss = entry['avg_loss']
            steps = entry.get('steps', entry.get('total_steps', 0))
            print(f"    {ep:>6} | {ts_str:>20} | {reward:>10.4f} | {loss:>14.6f} | {steps:>10,}")
        print()
        
        # Estimated time remaining
        episodes_remaining = training_cfg['total_episodes'] - latest['episode']
        estimated_remaining = episodes_remaining * avg_episode_time
        estimated_remaining_min = estimated_remaining / 60
        estimated_remaining_hrs = estimated_remaining_min / 60
        print(f"  \033[1m2.7 预计剩余时间:\033[0m")
        print(f"    - 剩余轮数: {episodes_remaining}")
        print(f"    - 预计剩余: {estimated_remaining:.1f} 秒")
        print(f"    - 即: {estimated_remaining_min:.1f} 分钟 / {estimated_remaining_hrs:.2f} 小时")
        print()
    
    # ============================================================
    # 3. 系统资源与进程健康信息
    # ============================================================
    print("\033[1;34m" + "=" * 120 + "\033[0m")
    print("\033[1;34m3. 系统资源与进程健康信息\033[0m")
    print("\033[1;34m" + "=" * 120 + "\033[0m")
    print()
    
    sys_latest = system_metrics['latest']
    sys_stats = system_metrics['stats']
    latest_samples = system_metrics['latest_samples']
    
    print(f"  \033[1m3.1 CPU使用情况:\033[0m")
    print(f"    - 最新CPU使用率: {sys_latest['cpu_percent']}%")
    print(f"