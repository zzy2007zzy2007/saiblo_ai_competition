"""跨代 Top3 循环赛评估脚本

从 server 的训练输出中提取每代 Top3 种子，构建 30 人循环赛。
不依赖训练进程，独立运行。

用法:
    python3 tools/cross_gen_round_robin.py \\
        --output-dir outputs/20260614_225347 \\
        --device cuda \\
        --n-battles 2 \\
        --max-workers 12
"""

import argparse
import json
import os
import sys
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

# 确保项目路径在 sys.path 中
_PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_DIR / "src"))
sys.path.insert(0, str(_PROJECT_DIR.parent / "Ant-Game"))

from ppo_ga.network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ppo_ga.battle.ga_war_agent import GAWarAgent
from ppo_ga.battle.agent_loader import AgentLoader
from ppo_ga.battle.battle_simulator import _execute_battle


# ========================
# Worker 函数（spawn 子进程）
# ========================

_WORKER_DIAG_LOG = "/tmp/ppo_v10_cross_gen_worker.log"


def _reconfigure_loguru():
    from loguru import logger as _wlogger
    _wlogger.remove()
    _wlogger.add(
        _WORKER_DIAG_LOG, level="WARNING", rotation="10 MB", retention=3,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def _battle_worker(args: Tuple[bytes, bytes, int, int]) -> list:
    """在子进程中执行两个 agent 间的所有对战。

    args: (agent_a_bytes, agent_b_bytes, n_battles, max_rounds)
    返回: list of result dicts
    """
    _reconfigure_loguru()
    a_bytes, b_bytes, n_battles, max_rounds = args
    agent_a = AgentLoader.deserialize(a_bytes)
    agent_b = AgentLoader.deserialize(b_bytes)
    # _ensure_compatible 在 spawn 模式下可能不需要，但安全起见
    from ppo_ga.battle.battle_simulator import _ensure_compatible
    agent_a = _ensure_compatible(agent_a)
    agent_b = _ensure_compatible(agent_b)

    results = []
    total = n_battles * 2  # 先后手各一局
    for i in range(total):
        first = i % 2
        r = _execute_battle(agent_a, agent_b, first, max_rounds)
        r["_agent_a"] = getattr(agent_a, "name", "?")
        r["_agent_b"] = getattr(agent_b, "name", "?")
        results.append(r)
    return results


# ========================
# TrueSkill 计算（最小实现）
# ========================

def _compute_trueskill(payoff: Dict[str, Dict[str, Dict[str, int]]],
                       population_ids: List[str]) -> Dict[str, dict]:
    """从 payoff 矩阵计算 TrueSkill ratings（简化版，无外部依赖）。

    使用迭代最大似然近似。
    """
    from math import sqrt, exp, log
    from collections import defaultdict

    n = len(population_ids)
    idx_map = {pid: i for i, pid in enumerate(population_ids)}

    # 收集所有对战记录
    matchups = []
    for pid_a in population_ids:
        for pid_b in population_ids:
            rec = payoff.get(pid_a, {}).get(pid_b, {})
            w = rec.get("wins", 0)
            l = rec.get("losses", 0)
            if w + l > 0:
                matchups.append((idx_map[pid_a], idx_map[pid_b], w, l))

    if not matchups:
        return {pid: {"mu": 25.0, "sigma": 8.333} for pid in population_ids}

    # 简化 TrueSkill: 用 Elo + Bayesian 近似
    mu = np.full(n, 25.0)
    sigma = np.full(n, 8.333)

    # 迭代更新（Bayesian 近似）
    for _ in range(20):
        mu_new = mu.copy()
        for ia, ib, w, l in matchups:
            total = w + l
            # 对方 skill 估计
            opp_mu = mu[ib]
            opp_var = sigma[ib] ** 2

            for _ in range(1):  # 简化：每对只更新一次
                # A 的更新
                expected = total * _sigmoid((mu[ia] - opp_mu) / sqrt(1 + opp_var))
                mu_new[ia] += 0.1 * (w - expected)
                # B 的更新
                expected_b = total * _sigmoid((opp_mu - mu[ia]) / sqrt(1 + sigma[ia] ** 2))
                mu_new[ib] += 0.1 * (l - expected_b)

        mu = mu_new
        # sigma 收缩
        sigma = sigma * 0.95 + 0.05 * 8.333

    return {
        population_ids[i]: {"mu": float(mu[i]), "sigma": float(sigma[i])}
        for i in range(n)
    }


def _sigmoid(x):
    from math import exp
    return 1.0 / (1.0 + exp(-x))


# ========================
# 主逻辑
# ========================

def load_top3_models(output_dir: str, device: torch.device) -> List[dict]:
    """加载每个 generation 的 Top3 种子模型。"""
    generations_dir = os.path.join(output_dir, "generations")
    gen_dirs = sorted(
        [d for d in os.listdir(generations_dir) if d.startswith("gen_")],
        key=lambda x: int(x.split("_")[-1]),
    )

    entries = []
    for gdir in gen_dirs:
        seed_dir = os.path.join(generations_dir, gdir, "seed_models")
        br_path = os.path.join(generations_dir, gdir, "battle_results.json")

        # 从 battle_results 确认排名
        top3_ranks = [1, 2, 3]
        seed_ids = {}
        if os.path.exists(br_path):
            with open(br_path) as f:
                br = json.load(f)
            for s in br.get("seeds", []):
                if s.get("seed_rank") in top3_ranks:
                    seed_ids[s["seed_rank"]] = s["id"]

        for rank in top3_ranks:
            pt_path = os.path.join(seed_dir, f"seed_{rank}.pt")
            if not os.path.exists(pt_path):
                continue

            checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
            network = AntWarPolicyValueNetwork(hidden_dim=256)
            if device.type == "cuda":
                network = network.to(device)
            network.load_state_dict(checkpoint["policy_state_dict"])
            network.eval()

            ind_id = checkpoint.get("individual_id", f"{gdir}_seed_{rank}")
            agent = GAWarAgent(player_id=0, network=network, device=device)
            agent.name = ind_id  # 用于识别

            entries.append({
                "gen": gdir,
                "rank": rank,
                "id": ind_id,
                "mu": checkpoint.get("trueskill_mu", 0),
                "agent": agent,
                "agent_bytes": AgentLoader.serialize(agent),
            })

    return entries


def compute_payoff(entries: List[dict], n_battles: int, max_workers: int,
                   max_rounds: int = 512):
    """运行全对全循环赛。"""
    n = len(entries)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"Total individuals: {n}")
    print(f"Total pairs: {len(pairs)}")
    print(f"Total games: {len(pairs) * n_battles * 2}")
    print()

    # 初始化 payoff
    payoff: Dict[str, Dict[str, Dict[str, int]]] = {}
    for e in entries:
        pid = e["id"]
        payoff[pid] = {}
    for i, j in pairs:
        pid_a = entries[i]["id"]
        pid_b = entries[j]["id"]
        payoff[pid_a][pid_b] = {"wins": 0, "losses": 0, "draws": 0}
        payoff[pid_b][pid_a] = {"wins": 0, "losses": 0, "draws": 0}

    # 构建任务
    tasks = []
    for i, j in pairs:
        a = entries[i]
        b = entries[j]
        tasks.append({
            "i": i, "j": j,
            "id_a": a["id"], "id_b": b["id"],
            "args": (a["agent_bytes"], b["agent_bytes"], n_battles, max_rounds),
        })

    # 并行执行
    start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
        futures = {pool.submit(_battle_worker, t["args"]): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                results = fut.result(timeout=600)
            except Exception as e:
                print(f"  ERROR {t['id_a']} vs {t['id_b']}: {e}")
                continue

            w = sum(1 for r in results if r.get("result") == "agent1_win")
            l = sum(1 for r in results if r.get("result") == "agent2_win")
            d = sum(1 for r in results if r.get("result") == "draw")

            payoff[t["id_a"]][t["id_b"]] = {"wins": w, "losses": l, "draws": d}
            payoff[t["id_b"]][t["id_a"]] = {"wins": l, "losses": w, "draws": d}

            completed += 1
            elapsed = time.time() - start
            rate = completed / max(elapsed, 1) * 60
            eta = (len(tasks) - completed) / max(rate, 0.01)
            print(f"  [{completed}/{len(tasks)}] {t['id_a']} vs {t['id_b']}: "
                  f"{w}-{l}-{d}  ({rate:.1f} pairs/min, ETA {eta:.0f}s)")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    return payoff


def main():
    parser = argparse.ArgumentParser(description="Cross-Generation Top3 Round Robin")
    parser.add_argument("--output-dir", required=True, help="Training output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-battles", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--max-rounds", type=int, default=512)
    parser.add_argument("--result-dir", default=None,
                        help="Directory to save results (default: output-dir/cross_gen_rr)")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    result_dir = args.result_dir or os.path.join(output_dir, "cross_gen_rr")
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")
    print(f"Result dir: {result_dir}")

    # 1. 加载模型
    print("\n=== Loading Top3 models from each generation ===")
    entries = load_top3_models(output_dir, device)

    # 去重（同 ID 可能出现在不同代 - 本场景中不会，但安全起见）
    seen = set()
    unique_entries = []
    for e in entries:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique_entries.append(e)
    entries = unique_entries

    print(f"Loaded {len(entries)} unique individuals from {len(set(e['gen'] for e in entries))} generations:")
    for e in entries:
        print(f"  {e['gen']} rank={e['rank']}  {e['id']} (mu={e['mu']:.2f})")

    # 2. 运行循环赛
    print(f"\n=== Round Robin: {len(entries)} individuals, "
          f"{len(entries)*(len(entries)-1)//2} pairs, n_battles={args.n_battles} ===")
    payoff = compute_payoff(entries, args.n_battles, args.max_workers, args.max_rounds)

    # 3. 计算 TrueSkill
    print("\n=== Computing TrueSkill ratings ===")
    pop_ids = [e["id"] for e in entries]
    trueskill = _compute_trueskill(payoff, pop_ids)

    # 4. 排名
    ranked = sorted(trueskill.items(), key=lambda x: x[1]["mu"], reverse=True)

    # 5. 总胜率
    total_winrates = {}
    for pid in pop_ids:
        total_w = 0
        total_g = 0
        for opp, rec in payoff.get(pid, {}).items():
            total_w += rec.get("wins", 0)
            total_g += rec.get("wins", 0) + rec.get("losses", 0) + rec.get("draws", 0)
        total_winrates[pid] = total_w / max(total_g, 1) * 100

    # 6. 输出
    print("\n=== Final Rankings ===\n")
    print(f"{'Rank':<5} {'ID':<25} {'mu':>8} {'Win%':>8} {'Generation':>10}")
    print("-" * 60)
    for rank, (pid, ts) in enumerate(ranked, 1):
        gen = pid.split("_")[0] + "_" + pid.split("_")[1] if "_" in pid else "?"
        wr = total_winrates.get(pid, 0)
        print(f"{rank:<5} {pid:<25} {ts['mu']:8.2f} {wr:7.1f}% {gen:>10}")

    # 7. 保存结果
    result = {
        "description": f"Cross-generation Top3 Round Robin ({len(entries)} individuals)",
        "n_battles": args.n_battles,
        "individuals": [{"id": pid, "gen": e["gen"], "rank": e["rank"],
                          "original_mu": e["mu"]}
                         for e in entries for pid in [e["id"]]],
        "rankings": [{"id": pid, "mu": ts["mu"], "sigma": ts["sigma"],
                       "win_rate_total": total_winrates[pid]}
                      for pid, ts in ranked],
        "payoff": payoff,
    }

    result_path = os.path.join(result_dir, "cross_gen_rr_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to: {result_path}")

    summary_path = os.path.join(result_dir, "cross_gen_rr_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Cross-Generation Top3 Round Robin Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Rank':<5} {'ID':<25} {'mu':>8} {'Win%':>8} {'Gen':>8}\n")
        f.write("-" * 60 + "\n")
        for rank, (pid, ts) in enumerate(ranked, 1):
            gen = pid.split("_")[0] + "_" + pid.split("_")[1]
            wr = total_winrates.get(pid, 0)
            f.write(f"{rank:<5} {pid:<25} {ts['mu']:8.2f} {wr:7.1f}% {gen:>8}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
