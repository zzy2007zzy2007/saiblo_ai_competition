#!/usr/bin/env python3
"""
NaN 蔓延路径根因分析
"""
import json, math, re

# ========== 1. 权重快照深度分析 ==========
print("=" * 70)
print("1. 权重快照 NaN 蔓延路径")
print("=" * 70)

with open("ppo/outputs/20260525_172045/training/weight_stats.jsonl") as f:
    weights = [json.loads(l) for l in f]

# 提取所有参数名
all_params = list(weights[-1]["weights"].keys())
print(f"总参数数: {len(all_params)}")

# 按网络层分组
layers = {}
for name in all_params:
    layer = ".".join(name.split(".")[:2])
    layers.setdefault(layer, []).append(name)

for wr in weights:
    ep = wr["episode"]
    print(f"\n--- Ep {ep} ---")
    for layer, params in sorted(layers.items()):
        nan_in_layer = sum(1 for p in params if any(
            math.isnan(wr["weights"][p].get(k, 0))
            for k in ["mean", "std", "min", "max"]
        ))
        total = len(params)
        if nan_in_layer > 0:
            print(f"  {layer}: {nan_in_layer}/{total} NaN")
            for p in params:
                ws = wr["weights"][p]
                if any(math.isnan(ws.get(k, 0)) for k in ["mean","std","min","max"]):
                    print(f"    {p.split('.')[-1]}: mean={ws['mean']}, std={ws['std']}, "
                          f"grad_mean={ws.get('grad_mean','N/A')}, grad_norm={ws.get('grad_norm','N/A')}")

# ========== 2. NaN 首次出现的层关系 ==========
print("\n" + "=" * 70)
print("2. 架构中 NaN 首次出现的层")
print("=" * 70)

# 查看 Ep 40 的 NaN 参数
ep40 = weights[0]
nan_params_40 = {n: ws for n, ws in ep40["weights"].items()
                 if any(math.isnan(ws.get(k, 0)) for k in ["mean","std","min","max"])}
print(f"Ep 40 首次 NaN 共 {len(nan_params_40)} 个参数:")
for name, ws in sorted(nan_params_40.items()):
    print(f"  {name}")
    for k in ["mean", "std", "min", "max", "grad_mean", "grad_std", "grad_norm"]:
        if k in ws:
            v = ws[k]
            if isinstance(v, float) and math.isnan(v):
                print(f"    {k}: NaN")
            else:
                print(f"    {k}: {v}")

# 分析 NaN 是否源于 0 梯度
print("\n  grad_norm 值:")
for name, ws in sorted(nan_params_40.items()):
    gn = ws.get("grad_norm", "N/A")
    print(f"    {name}: grad_norm={gn}")

# ========== 3. 从 training.log 提取 NaN 相关条目 ==========
print("\n" + "=" * 70)
print("3. training.log 中 NaN/ratio/warning 条目")
print("=" * 70)

with open("ppo/outputs/20260525_172045/training/training.log") as f:
    lines = f.readlines()

# 提取所有含 NaN/warning/异常 的日志
keywords = ["NaN", "nan", "warning", "警告", "异常", "梯度", "grad_norm", "ratio"]
found = []
for i, line in enumerate(lines):
    if any(k.lower() in line.lower() for k in keywords):
        found.append((i, line.rstrip()))

print(f"共 {len(found)} 条相关日志 (总行数: {len(lines)})")
print()

# 展示所有 NaN 相关行
print("--- NaN/警告 日志详情 ---")
prev_ep = -1
for line_no, line in found:
    # 提取 episode 编号
    ep_match = re.search(r'[Ee]p(?:isode)?[:\s]*(\d+)', line)
    ep = int(ep_match.group(1)) if ep_match else -1
    if ep != prev_ep:
        print(f"\n  --- Ep {ep} ---" if ep >= 0 else "")
        prev_ep = ep
    # 截断长行
    display = line[:200] + "..." if len(line) > 200 else line
    print(f"  L{line_no}: {display}")

# ========== 4. 分析熵系数更新前后的比值变化 ==========
print("\n" + "=" * 70)
print("4. 训练指标时间线（按熵排序的关键事件）")
print("=" * 70)

with open("ppo/outputs/20260525_172045/training/training_metrics_history.jsonl") as f:
    metrics = [json.loads(l) for l in f]

def safe(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0.0
    return v

# 计算每个 update 的指标变化
print(f"{'Ep':>4s}  {'entropy':>9s}  {'ent_coef':>8s}  {'ratio_m':>7s}  {'ratio_s':>7s}  "
      f"{'clip':>7s}  {'grad_norm':>9s}  {'value_m':>7s}  {'value_s':>7s}  "
      f"{'hard_pen':>7s}  {'valid':>5s}  {'noop%':>6s}")
print("-" * 100)

for r in metrics:
    ep = r["episode"]
    ent = r["entropy"]
    ent_coef = r.get("entropy_coef", 0)
    rm = safe(r.get("ratio_mean", 0))
    rs = safe(r.get("ratio_std", 0))
    cf = safe(r.get("clip_fraction", 0))
    gn = r.get("gradient_norm", "?")
    if isinstance(gn, str):
        gn_disp = gn
    else:
        gn_disp = f"{gn:>7.1f}" if not math.isnan(gn) else "    NaN"
    vm = safe(r.get("value_input_mean", 0))
    vs = safe(r.get("value_input_std", 0))
    hp = safe(r.get("hard_entropy_penalty", 0))
    va = r.get("valid_actions_mean", 0)
    noop = safe(r.get("type_noop", 0)) * 100
    if ep in [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192]:
        print(f"{ep:>4d}  {ent:>9.6f}  {ent_coef:>8.4f}  {rm:>7.2f}  {rs:>7.2f}  "
              f"{cf:>7.4f}  {gn_disp:>9}  {vm:>7.2f}  {vs:>7.2f}  "
              f"{hp:>7.2f}  {va:>5.1f}  {noop:>5.1f}%")

# ========== 5. 关键时序分析 ==========
print("\n" + "=" * 70)
print("5. NaN 蔓延关键时序")
print("=" * 70)

# 从 metrics 中找到：
# 1. target_heads.0 首次损坏的时刻 - 从日志推断
# 2. value_head 首次损坏的时刻
# 3. NaN 从 2 个参数扩散到 46 个的过程

print("""
已知事实:
  Ep 40:  policy_head.target_heads.0.bias + value_head.bias 为 NaN（共 2 个）
  Ep 80:  同上 2 个 NaN（无扩散）
  Ep 120: 同上 2 个 NaN（无扩散）
  Ep 160: 全部 46 个参数 NaN（完全污染）

关键问题:
  Q1: 为什么 target_heads.0 (build_tower) 和 value_head 先损坏?
  Q2: 为什么 NaN 从 Ep 40 到 Ep 120 维持了 3 个快照（80 episodes）不扩散?
  Q3: 为什么在 Ep 120 到 Ep 160 之间突然扩散到全部参数?

假设分析:
""")

# Q1 分析
print("Q1: target_heads.0 和 value_head 为何先损坏")
print("""
  target_heads.0 → build_tower 目标头
    该头有 16 个输出（max_targets=16），但有效动作只有 1-10（flat_end=11）
    当 NO-OP 占比 85%+ 时，build_tower 几乎不被采样
    → 该头的梯度几乎为零或极高方差
    → 反向传播时少数样本产生极端梯度 → NaN

  value_head → 价值网络
    终局奖励 ±100 vs 单步奖励 ~0.6 的量级差异
    GAE 将终局信号回传到整个轨迹
    → 价值估计方差极大 → 梯度爆炸 → NaN
""")

# Q2 分析
print("Q2: NaN 为何维持 80 episodes 不扩散")
print("""
  NO-OP 占比 85%+ 意味着大部分 forward pass 不经过 target_heads.0
  只有 ~7% 的 build_tower 动作经过该头
  当 build_tower 头损坏后输出 NaN:
    flat_logits[1:11] = type_logit[1] + target_logits[0] = finite + NaN = NaN
    但 NO-OP 占比 85%+，模型很少选这些 NaN 动作
    → NaN 被"冻结"在少数不被使用的参数中，不影响主要训练过程

  value_head 损坏后输出 NaN:
    values = NaN → _clean_tensor(NaN) → torch.tensor(0.0, detached)
    → value_loss 被替换为 0.0，不影响训练
    → 但 optimizer.step() 仍然可能用 NaN 梯度更新 value_head
    → 然而 value_head.bias 已经是 NaN，更新后仍然是 NaN
    → NaN 被"锁死"在 value_head 中不扩散

  关键保护机制:
    _clean_tensor 在 value_loss 上成功隔离了 NaN
    只要 NO-OP 占比足够高，其他头不依赖损坏的参数
""")

# Q3 分析
print("Q3: Ep 120→160 之间为何突然扩散")
print("""
  可能的触发条件组合:
    1. Ep 136 时 entropy=0.124（已经连续多次下降）
    2. Ep 136-144 间某些 batch 中 build_tower 合法概率突然变高
       → 更多样本经过 target_heads.0 → NaN 影响更多 loss
    3. Ep 144 ratio 失控 (ratio_mean=160.8, ratio_std=2551.7):
       policy_loss → -INF/NaN → _clean_tensor 返回 0.0(detached)
       但计算图中其他路径的 NaN 通过 loss.backward() 传播
    4. clip_grad_norm_(NaN) → 所有梯度 NaN
    5. optimizer.step() → 全部 46 个参数写入 NaN

  关键转折点: Ep 144 的 ratio 失控不是"首次"NaN，而是"压死骆驼的最后一根稻草"
  它让原本被隔离的 NaN 通过梯度裁剪这个"单点故障"扩散到全参数
""")

# ========== 6. 已有防御机制分析 ==========
print("=" * 70)
print("6. 现有防御机制在 NaN 面前的表现")
print("=" * 70)

print("""
  防御 1: _clean_tensor (NaN loss → 0.0)
    保护了 value_head NaN 不污染 value_loss ✓
    但 detached=True 割断了梯度，价值网络从此不学习 ✗
    
  防御 2: torch.clamp (NaN → NaN)
    对 NaN 完全无效 ✗
    (已经通过 log_ratio clamp 修复 ✓)
    
  防御 3: clip_grad_norm_ (NaN → 全 NaN)
    单点故障：一个参数的 NaN 梯度导致所有参数 NaN ✗
    
  防御 4: NaN skip check (loss > LOSS_MAX)
    只能检查 loss 本身，不能检查权重中的 NaN ✗

  综合分析:
    Ep 40-120 期间 NaN 被隔离是因为"幸运"（NO-OP 占主导，不依赖损坏参数）
    而非系统设计上的容错
    任何一个使得损坏参数被调用的条件变化都可能导致 NaN 扩散
""")

# ========== 7. 结论和建议 ==========
print("=" * 70)
print("7. 结论与修复建议")
print("=" * 70)

print("""
结论:
  1. NaN 并非 Ep 144 首次出现，而是从 Ep 40 就开始存在
  2. Ep 144 的 ratio 失控是"扩散事件"(propagation event)，不是"首次事件"(first event)
  3. NaN 在 Ep 40-120 被隔离是因为 NO-OP 占 85%+，"碰巧"不依赖损坏参数
  4. clip_grad_norm_ 是单点故障——一个 NaN 梯度污染所有参数

修复优先级:
  P0: 在 optimizer.step() 之后增加权重级 NaN 检测（文档中修复三）
      检测到 NaN 后回滚 checkpoint + skip batch
      这是防止 NaN 扩散到全参数的最后防线
  
  P1: clip_grad_norm_ 之前过滤 NaN 梯度
      如果某个 param.grad 包含 NaN，将其替换为 0.0 而不是让 total_norm=NaN
  
  P2: 理解 target_heads.0 和 value_head 为何易损
      target_heads.0: 输出维度(16)远大于有效动作数(10)，梯度方差大
      value_head: 终局奖励信号量级不匹配
      可能需要调整初始化或学习率
""")
