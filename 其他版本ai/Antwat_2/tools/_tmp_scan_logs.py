import json, glob

log_dir = "/root/autodl-tmp/AntWar/outputs/20260611_181319"

# batch_metrics
with open(f"{log_dir}/training/batch_metrics.jsonl") as f:
    lines = f.readlines()
print(f"Total PPO batches: {len(lines)}")
for line in lines:
    d = json.loads(line)
    print(f"Ep {d['episode']:>4d} | PLoss={d.get('policy_loss',0):.4f} VLoss={d.get('value_loss',0):.1f} Entropy={d.get('entropy',0):.3f} ClipFrac={d.get('clip_fraction',0):.4f} KL={d.get('approx_kl',0):.5f} | GradTotal={d.get('total_grad_norm',0):.1f} GradV={d.get('grad_norm_value',0):.1f} GradP={d.get('grad_norm_policy',0):.2f} | RetMean={d.get('return_mean',0):.1f} RetStd={d.get('return_std',0):.1f} | Lr={d.get('lr',0):.6f}")

# aux losses
print()
for line in lines:
    d = json.loads(line)
    aux_items = []
    for k in ['aux_tower_loss','aux_gold_loss','aux_enemy_tower_loss','aux_enemy_gold_loss','aux_base_loss']:
        if k in d:
            aux_items.append(f"{k}={d[k]:.2f}")
    if aux_items:
        print(f"Ep {d['episode']:>4d} | {' | '.join(aux_items)}")

# episode_batch stats
print()
with open(f"{log_dir}/training/episode_batch_train_stats.jsonl") as f:
    lines2 = f.readlines()
print(f"Total episode batches: {len(lines2)}")
for line in lines2:
    d = json.loads(line)
    print(f"Ep {d['episode']:>4d} | WinRate={d.get('win_rate',0):.1f} AvgReward={d.get('avg_reward',0):.1f} AvgRounds={d.get('avg_rounds',0):.1f} MaxBal={d.get('avg_max_own_coins',0):.1f} TotInc={d.get('avg_total_own_coin_income',0):.1f}")

# error logs
for ef in sorted(glob.glob(f"{log_dir}/training/training_error_*.log")):
    with open(ef) as f:
        content = f.read().strip()
        if content:
            print(f"\n--- Error log ({ef}) ---\n{content[:2000]}")
        else:
            print(f"\n--- Error log ({ef}) (empty) ---")

# check for key warnings
print()
with open(f"{log_dir}/training/training_*.log") as f:
    pass  # can't glob in open
# use glob
for lf in sorted(glob.glob(f"{log_dir}/training/training_*.log")):
    if 'error' in lf:
        continue
    with open(lf) as f:
        content = f.read()
    # count warnings
    wc = content.count('WARNING')
    ec = content.count('ERROR')
    nc = content.count('NaN') + content.count('nan')
    sc = content.count('skip_update')
    # unique warning types
    grad_warnings = content.count('[大梯度]')
    total_grad_warnings = content.count('[总梯度异常]')
    print(f"\nLog: {lf}")
    print(f"  WARNING count: {wc}, ERROR count: {ec}, NaN count: {nc}, skip_update: {sc}")
    print(f"  [大梯度] warnings: {grad_warnings}, [总梯度异常] warnings: {total_grad_warnings}")
    
    # show a few sample grad values
    for line in content.split('\n'):
        if '[大梯度]' in line and 'value_head' in line:
            print(f"  Sample: {line.strip()}")
            break
