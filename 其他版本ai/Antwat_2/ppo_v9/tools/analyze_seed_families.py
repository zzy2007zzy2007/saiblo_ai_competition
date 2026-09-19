#!/usr/bin/env python3
"""分析种子家族关系：计算余弦相似度矩阵并聚类。

用法:
    python tools/analyze_seed_families.py <gen_dir>
"""

import sys
import os
import json
import numpy as np
import torch
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def extract_weights(pt_path: str) -> np.ndarray:
    """从 .pt 文件加载 state_dict 并展平为一维权重向量。"""
    data = torch.load(pt_path, map_location="cpu", weights_only=True)

    # 种子文件格式: {"policy_state_dict": {...}, "elo_rating": ..., ...}
    if isinstance(data, dict) and "policy_state_dict" in data:
        state_dict = data["policy_state_dict"]
    elif isinstance(data, dict) and "state_dict" in data:
        state_dict = data["state_dict"]
    elif isinstance(data, dict):
        state_dict = data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    tensors = []
    for name, param in sorted(state_dict.items()):
        if isinstance(param, torch.Tensor):
            tensors.append(param.cpu().numpy().flatten())
    return np.concatenate(tensors).astype(np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def find_families(cos_matrix: np.ndarray, names: list, threshold: float = 0.85):
    """用并查集找出家族（连通分量），返回家族分组。"""
    n = len(names)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if cos_matrix[i, j] > threshold:
                union(i, j)

    families = defaultdict(list)
    for i in range(n):
        families[find(i)].append(i)

    return list(families.values())


def main():
    if len(sys.argv) < 2:
        gen_dir = "/root/autodl-tmp/AntWar/ppo_v9/outputs/20260612_164342/generations/gen_0005"
    else:
        gen_dir = sys.argv[1]

    seed_dir = os.path.join(gen_dir, "seed_models")
    if not os.path.isdir(seed_dir):
        print(f"ERROR: {seed_dir} not found")
        sys.exit(1)

    # 收集种子文件
    seed_files = sorted([
        f for f in os.listdir(seed_dir) if f.endswith(".pt")
    ], key=lambda x: int(x.replace("seed_", "").replace(".pt", "")))

    n_seeds = len(seed_files)
    print(f"Found {n_seeds} seed models in {seed_dir}")
    print()

    # 加载权重
    print("Loading weights...")
    weights = {}
    for f in seed_files:
        path = os.path.join(seed_dir, f)
        rank = int(f.replace("seed_", "").replace(".pt", ""))
        weights[rank] = extract_weights(path)
        print(f"  {f}: {weights[rank].shape[0]:,} params")

    param_count = list(weights.values())[0].shape[0]
    print(f"\nParam count per model: {param_count:,}")
    print()

    # 计算余弦相似度矩阵
    print(f"Computing cosine similarity matrix ({n_seeds}x{n_seeds})...")
    ranks = sorted(weights.keys())
    cos_matrix = np.zeros((n_seeds, n_seeds))
    for i, ri in enumerate(ranks):
        for j, rj in enumerate(ranks):
            if i <= j:
                cos = cosine_similarity(weights[ri], weights[rj])
                cos_matrix[i, j] = cos
                cos_matrix[j, i] = cos

    # 输出矩阵
    # 先获取 ELO 信息
    eval_file = os.path.join(gen_dir, "evaluation.json")
    pop_file = os.path.join(gen_dir, "population_stats.json")

    # 尝试从日志获取 ELO 排名
    seed_ids = {}
    # 从 evaluation.json 获取 individual_id 映射
    if os.path.exists(eval_file):
        with open(eval_file) as f:
            eval_data = json.load(f)
        # 按 seed_rank 映射
        for ind_id, results in eval_data.items():
            for ref_name, stats in results.items():
                if "seed_rank" not in locals():
                    break

    print("\n" + "=" * 85)
    print("余弦相似度矩阵 (Gen 5 Seeds)")
    print("=" * 85)

    # 表头
    header = f"{'':>18}"
    for r in ranks:
        header += f"  seed_{r:1d}"
    print(header)
    print("-" * 85)

    for i, ri in enumerate(ranks):
        row = f"  seed_{ri:1d}"
        for j, rj in enumerate(ranks):
            v = cos_matrix[i, j]
            if i == j:
                row += f"   1.000"
            else:
                row += f"  {v:6.3f}"
        print(row)

    # 家族分析
    print()
    print("=" * 85)
    print("家族分析 (cos > 0.85)")
    print("=" * 85)
    names = [f"seed_{r}" for r in ranks]
    families_085 = find_families(cos_matrix, names, 0.85)
    print(f"\n阈值 0.85: {len(families_085)} 个家族")
    for idx, fam in enumerate(families_085):
        members = [names[i] for i in fam]
        # 计算家族内部 cos 范围
        if len(members) >= 2:
            intra = [cos_matrix[i, j] for i_idx, i in enumerate(fam) for j in fam[i_idx+1:]]
            print(f"  家族 {idx+1}: {', '.join(members)} "
                  f"(内部 cos: [{min(intra):.3f}, {max(intra):.3f}])")
        else:
            print(f"  家族 {idx+1}: {', '.join(members)} (独立)")

    print(f"\n阈值 0.90: {len(find_families(cos_matrix, names, 0.90))} 个家族")
    families_09 = find_families(cos_matrix, names, 0.90)
    for idx, fam in enumerate(families_09):
        members = [names[i] for i in fam]
        if len(members) >= 2:
            intra = [cos_matrix[i, j] for i_idx, i in enumerate(fam) for j in fam[i_idx+1:]]
            print(f"  家族 {idx+1}: {', '.join(members)} "
                  f"(内部 cos: [{min(intra):.3f}, {max(intra):.3f}])")
        else:
            print(f"  家族 {idx+1}: {', '.join(members)} (独立)")

    print(f"\n阈值 0.95: {len(find_families(cos_matrix, names, 0.95))} 个家族")
    families_095 = find_families(cos_matrix, names, 0.95)
    for idx, fam in enumerate(families_095):
        members = [names[i] for i in fam]
        if len(members) >= 2:
            intra = [cos_matrix[i, j] for i_idx, i in enumerate(fam) for j in fam[i_idx+1:]]
            print(f"  家族 {idx+1}: {', '.join(members)} "
                  f"(内部 cos: [{min(intra):.3f}, {max(intra):.3f}])")
        else:
            print(f"  家族 {idx+1}: {', '.join(members)} (独立)")

    # 总结
    print()
    print("=" * 85)
    print("总结")
    print("=" * 85)
    max_cos = max(cos_matrix[i, j] for i in range(n_seeds) for j in range(i + 1, n_seeds))
    mean_cos = np.mean([cos_matrix[i, j] for i in range(n_seeds) for j in range(i + 1, n_seeds)])
    print(f"最大种子间 cos: {max_cos:.4f}")
    print(f"平均种子间 cos: {mean_cos:.4f}")

    # 检查是否有逐位拷贝 (cos > 0.95)
    hard_dups = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            if cos_matrix[i, j] > 0.95:
                hard_dups.append((names[i], names[j], cos_matrix[i, j]))
    if hard_dups:
        print(f"\n⚠️  发现 {len(hard_dups)} 对近乎拷贝的种子 (cos > 0.95):")
        for a, b, c in hard_dups:
            print(f"  {a} ↔ {b}: cos={c:.4f}")
    else:
        print("\n✅ 未发现逐位拷贝 (cos > 0.95)")

    # 家族垄断检查
    largest_family = max(families_085, key=len)
    if len(largest_family) >= 5:
        print(f"\n⚠️  家族垄断风险：最大家族有 {len(largest_family)} 个种子")
        print(f"  成员: {', '.join(names[i] for i in largest_family)}")
    else:
        print(f"\n✅ 无家族垄断风险：最大家族只有 {len(largest_family)} 个种子")


if __name__ == "__main__":
    main()
