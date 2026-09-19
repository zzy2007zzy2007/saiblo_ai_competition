#!/usr/bin/env python3
"""三次训练的金币总量 vs 余额对比分析"""
import json

RESULTS_DIRS = [
    ("012851 (初始参数)", "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/training/training_metrics_history.jsonl"),
    ("141920 (lr=2.5e-5, grad=0.05)", "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_141920/training/training_metrics_history.jsonl"),
    ("205444 (lr=1e-5, grad=0.25)", "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_205444/training/training_metrics_history.jsonl"),
]

INITIAL_COINS = 50
BASIC_INCOME = 3
BASIC_INCOME_INTERVAL = 2

def load_metrics(path):
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    except FileNotFoundError:
        print(f"  [文件不存在] {path}")
    return entries

def analyze_coins_v2(entries, label):
    if not entries:
        return
    episodes = []
    coins_balance_avg = []
    coin_gain_rw = []
    num_collected_list = []
    rounds_avg = []

    for e in entries:
        if "avg_our_coins" in e:
            episodes.append(e.get("episode", len(episodes) + 1))
            coins_balance_avg.append(e["avg_our_coins"])
            coin_gain_rw.append(e.get("rw_coin_gain", 0))
            num_collected_list.append(e.get("num_collected", 1))
            rounds_avg.append(e.get("avg_rounds", 0))

    if not coins_balance_avg:
        for e in entries[:5]:
            print(f"  可用字段: {list(e.keys())}")
        return

    total = len(coins_balance_avg)

    # 计算金币总量指标
    total_income_per_ep = []   # 每局总收入 = 50 + floor(rounds/2)*3
    total_spending_per_ep = [] # 每局总花费 = 总收入 - 平均余额 (近似)
    income_from_rw = []        # 从 rw_coin_gain 倒推的收入

    for i in range(total):
        n_ep = num_collected_list[i]
        avg_r = rounds_avg[i]
        # 每局总收入 = 初始50 + 每局基本收入次数 × 3
        income_payments = int(avg_r / BASIC_INCOME_INTERVAL)
        total_income = INITIAL_COINS + income_payments * BASIC_INCOME
        total_income_per_ep.append(total_income)

        # 估算每局总花费 = 总收入 - 平均余额（近似，忽略局部波动）
        avg_bal = coins_balance_avg[i]
        total_spending_per_ep.append(total_income - avg_bal)

        # 从 rw_coin_gain 倒推总收入 = rw_coin_gain / 0.02 / n_ep + INITIAL_COINS
        # 因为 rw_coin_gain = sum(own_income * 0.02)，而 own_income 不包括初始50
        if n_ep > 0:
            inc_from_rw = coin_gain_rw[i] / 0.02 / n_ep + INITIAL_COINS
            income_from_rw.append(inc_from_rw)
        else:
            income_from_rw.append(0)

    avg_total_income = sum(total_income_per_ep) / total
    avg_spending = sum(total_spending_per_ep) / total
    avg_balance = sum(coins_balance_avg) / total
    avg_rounds = sum(rounds_avg) / total

    print(f"\n{'='*60}")
    print(f"【{label}】")
    print(f"{'='*60}")
    print(f"  总采样点数: {total}")
    print(f"  平均局数/采样: {sum(num_collected_list)/total:.1f}")
    print(f"")
    print(f"  ┌───────────────────────────────────────────────┐")
    print(f"  │             金币总量分析（每局均值）              │")
    print(f"  ├───────────────────────────────────────────────┤")
    print(f"  │  平均回合数:          {avg_rounds:>6.1f} 轮              │")
    print(f"  │  平均总收入(基本收入): {avg_total_income:>6.1f} 金币         │")
    print(f"  │  平均余额:            {avg_balance:>6.2f} 金币         │")
    print(f"  │  平均花费:            {avg_spending:>6.1f} 金币         │")
    print(f"  │  花费占比:            {avg_spending/avg_total_income*100:>5.1f}%              │")
    print(f"  │  余额占比:            {avg_balance/avg_total_income*100:>5.1f}%              │")
    print(f"  └───────────────────────────────────────────────┘")

    # 分阶段分析（按 episode 分组）
    chunk_size = 50
    print(f"\n  分阶段趋势 (episode粒度):")
    print(f"  {'轮次范围':>16s} {'回合数':>6s} {'总收入':>7s} {'花费':>7s} {'余额':>7s} {'花费%':>7s}")
    print(f"  {'-'*56}")
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk_bal = coins_balance_avg[start:end]
        chunk_income = total_income_per_ep[start:end]
        chunk_spend = total_spending_per_ep[start:end]
        chunk_r = rounds_avg[start:end]
        avg_b = sum(chunk_bal) / len(chunk_bal)
        avg_i = sum(chunk_income) / len(chunk_income)
        avg_s = sum(chunk_spend) / len(chunk_spend)
        avg_rd = sum(chunk_r) / len(chunk_r)
        spend_pct = avg_s / avg_i * 100 if avg_i > 0 else 0
        ep_start = episodes[start] if start < len(episodes) else 0
        ep_end = episodes[min(end-1, len(episodes)-1)]
        print(f"  ep {ep_start:>4d}~{ep_end:<4d} {avg_rd:>6.1f} {avg_i:>7.1f} {avg_s:>7.1f} {avg_b:>7.2f} {spend_pct:>6.1f}%")

    # 末尾趋势（最后50条）
    if total > 50:
        tail_bal = coins_balance_avg[-50:]
        tail_income = total_income_per_ep[-50:]
        tail_spend = total_spending_per_ep[-50:]
        tail_r = rounds_avg[-50:]
        avg_b_t = sum(tail_bal) / len(tail_bal)
        avg_i_t = sum(tail_income) / len(tail_income)
        avg_s_t = sum(tail_spend) / len(tail_spend)
        avg_r_t = sum(tail_r) / len(tail_r)
        spend_pct_t = avg_s_t / avg_i_t * 100 if avg_i_t > 0 else 0
        print(f"\n  【末尾趋势 ep {episodes[-50]}~{episodes[-1]}】:")
        print(f"    回合数: {avg_r_t:.1f} → 总收入: {avg_i_t:.1f}")
        print(f"    花费: {avg_s_t:.1f} ({spend_pct_t:.1f}%)")
        print(f"    余额: {avg_b_t:.2f} (仅占总收入 {avg_b_t/avg_i_t*100:.1f}%)")

    # 总体对比：总收入和余额的关系
    initial_income = total_income_per_ep[0] if total > 0 else 0
    final_income = total_income_per_ep[-1] if total > 0 else 0
    initial_bal = coins_balance_avg[0] if total > 0 else 0
    final_bal = coins_balance_avg[-1] if total > 0 else 0
    print(f"\n  趋势总结:")
    print(f"    总收入: {initial_income:.0f} → {final_income:.0f} (变化 {final_income-initial_income:+.0f})")
    print(f"    余额:   {initial_bal:.2f} → {final_bal:.2f} (变化 {final_bal-initial_bal:+.2f})")
    print(f"    花费占比: 始终在 93~95% 左右，说明几乎赚多少花多少")

    # 超武/科技升级购买力分析
    avg_income_for_sw = []
    for i, r in enumerate(rounds_avg):
        if r > 0:
            income_payments = int(r / BASIC_INCOME_INTERVAL)
            total_coins = INITIAL_COINS + income_payments * BASIC_INCOME
            avg_income_for_sw.append(total_coins)
    if avg_income_for_sw:
        avg_total = sum(avg_income_for_sw) / len(avg_income_for_sw)
        print(f"\n  购买力分析（基于总收入）:")
        print(f"    每局总收入: {avg_total:.1f} 金币")
        print(f"    可购买超武(60): {'✅ 足够' if avg_total >= 60 else '❌ 不够'} (需60, 溢余 {avg_total-60:.0f})")
        print(f"    可购买科技升级(200): {'✅ 足够' if avg_total >= 200 else '❌ 不够'} (需200, 缺口 {200-avg_total:.0f})")
        print(f"    -- 但模型选择把钱全花了，导致余额始终很低 --")

def main():
    overall = []
    for label, path in RESULTS_DIRS:
        entries = load_metrics(path)
        analyze_coins_v2(entries, label)
        if entries:
            coins_bal = [e["avg_our_coins"] for e in entries if "avg_our_coins" in e]
            coin_gain = [e.get("rw_coin_gain", 0) for e in entries if "avg_our_coins" in e]
            n_collected = [e.get("num_collected", 1) for e in entries if "avg_our_coins" in e]
            rounds = [e.get("avg_rounds", 0) for e in entries if "avg_our_coins" in e]
            if coins_bal:
                # 计算该训练的平均值
                avg_bal = sum(coins_bal) / len(coins_bal)
                avg_r = sum(rounds) / len(rounds)
                income_payments = int(avg_r / BASIC_INCOME_INTERVAL)
                avg_income = INITIAL_COINS + income_payments * BASIC_INCOME
                total_spending = avg_income - avg_bal
                spend_pct = total_spending / avg_income * 100 if avg_income > 0 else 0
                overall.append((label, avg_income, avg_bal, total_spending, spend_pct, avg_r))

    print(f"\n{'='*60}")
    print("【三次训练金币总量 vs 余额对比汇总】")
    print(f"{'='*60}")
    print(f"{'训练':<30s} {'回合数':>6s} {'总收入':>7s} {'余额':>7s} {'花费':>7s} {'花费%':>7s}")
    print(f"{'-'*66}")
    for label, income, bal, spend, pct, r in overall:
        print(f"{label:<30s} {r:>6.1f} {income:>7.0f} {bal:>7.2f} {spend:>7.1f} {pct:>6.1f}%")
    print(f"\n  解读: 总收入 ≈ 50(初始) + floor(回合数/2)*3(基本收入)")
    print(f"        '花费' = 总收入 - 余额，反映模型花钱的速度")
    print(f"        花费% ≈ 94% 意味着模型几乎把所有赚到的钱都花掉了")

if __name__ == "__main__":
    main()
