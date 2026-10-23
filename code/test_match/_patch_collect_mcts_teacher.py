"""给采集器加 `--teacher mcts`：用 BundleMCTS(iterations=256) 的 **visit_policy** 当标签（2026-10-23 夜）。

为什么换教师（今晚的结论）：
  · 值网 rollout 排序当教师：Spearman 从 -0.11 练到 +0.428，k=24 棋力始终 44-53%（噪声带）
    ⇒ **值网排序与"赢"弱相关，蒸馏它换不来棋力**；
  · 而 `k=160 vs k=1 = 100%（32-0）` ⇒ 搜索本身很强；
  · 搜索的强很可能来自**多点平均降噪**，`visit_policy` 正是把这个平均压进标签，
    单次 rollout 的值没有这个信息（rollout 是确定性的，N 次 = 同一个数）。

实现：
  · `--teacher {rollout,mcts}`，默认 rollout（旧行为）；
  · mcts：`BundleMCTS(iterations=256, k=24, ...)`，标签 = `visit_policy` 归一化后的概率；
  · 同时把 `PRIOR_MAX_ROUNDS` 环境变量接进来（便于冒烟）。
"""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "test_match/collect_prior_labels.py"
t = P.read_text(encoding="utf-8")

# ① 顶部：教师开关 + MAX_ROUNDS 可配
old = "MAX_ROUNDS = 400"
new = ('MAX_ROUNDS = int(__import__("os").environ.get("PRIOR_MAX_ROUNDS", "400"))\n'
       'TEACHER = __import__("os").environ.get("PRIOR_TEACHER", "rollout")   # rollout | mcts\n'
       'MCTS_ITERS = int(__import__("os").environ.get("PRIOR_MCTS_ITERS", "256"))\n'
       'MCTS_K = int(__import__("os").environ.get("PRIOR_MCTS_K", "24"))')
assert old in t, "① 锚点未命中"
t = t.replace(old, new, 1)

# ② 教师搜索构造：mcts 时换 BundleMCTS
old2 = """    def make_search(k):
        return FlatRolloutGreedy(net_fn, k=k, rollouts_per_candidate=1, horizon=H,
                                 max_total_rounds=10 ** 9, value_model=vm,
                                 feature_extractor=feat, rollout_mode="idle")"""
new2 = '''    def make_search(k):
        if TEACHER == "mcts":
            from my_ai.az_intent.bundle_mcts import BundleMCTS
            return BundleMCTS(net_fn, iterations=MCTS_ITERS, k=MCTS_K, sample_mult=15,
                              search_mode="pos-only", pos_pin="mc_quiet", mc_every=1,
                              mc_horizon=16, mc_root_only=1, mc_cells=4, mc_score="net",
                              mc_net_samples=4)
        return FlatRolloutGreedy(net_fn, k=k, rollouts_per_candidate=1, horizon=H,
                                 max_total_rounds=10 ** 9, value_model=vm,
                                 feature_extractor=feat, rollout_mode="idle")'''
assert old2 in t, "② 锚点未命中"
t = t.replace(old2, new2, 1)

# ③ 取标签：mcts 用 visit_policy，rollout 用 last_scores
old3 = """                    vals = np.asarray(search.last_scores, dtype=np.float32)"""
new3 = """                    if TEACHER == "mcts":
                        _res = getattr(search, "last_result", None)
                        vp = getattr(_res, "visit_policy", None) if _res is not None else None
                        if vp is None:
                            vp = getattr(search, "visit_policy", None)
                        vals = np.asarray(vp if vp is not None else search.last_scores,
                                          dtype=np.float32)
                    else:
                        vals = np.asarray(search.last_scores, dtype=np.float32)"""
assert old3 in t, "③ 锚点未命中"
t = t.replace(old3, new3, 1)

# ④ 记录输出文件里带上教师类型
t = t.replace("    print(f\"  [done] g{seed}:", "    print(f\"  [done] teacher={TEACHER} g{seed}:")

P.write_text(t, encoding="utf-8")
print("已加 --teacher mcts（visit_policy 当标签）")
