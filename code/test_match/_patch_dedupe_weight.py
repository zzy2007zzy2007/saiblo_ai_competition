"""去重加权（dedupe-weighted）平均（2026-10-23）。

问题（N=128 实验暴露）：
  C=.3/k=1 下约 **62% 的样本与"静止"轨迹完全相同**（实测 20/32 同叶子）。
  N 越大，均值越收敛到那条"静止"轨迹的估值 ⇒ 候选之间的差异被冲淡 ⇒
  采样曲线非单调：N=8 50.0% → 32 59.4% → **64 68.8%** → 128 46.9%。

修法：按**轨迹 token 序列**分组，**每组只贡献 1 份权重**（组内取该组的代表值），
      这样"静止"那份重复 80 次也只算 1 次；同时输出"有效样本数"便于观察。

实现：给 FlatRolloutGreedy 加 `rollout_dedupe_weight: bool = False`，
      在树版 rollout（已有 token 机制）里顺手收集每条样本的 token 序列，
      最后按序列分组求均值。
"""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "my_ai/az_intent/flat_rollout_greedy.py"
t = P.read_text(encoding="utf-8")

# ① 参数
old = "                 rollout_share_prefix: bool = False,"
new = ("                 rollout_share_prefix: bool = False,\n"
       "                 # 2026-10-23：按轨迹去重加权（每条不同轨迹只算一次），避免大 N 被"
       "重复的\"静止\"轨迹稀释\n"
       "                 rollout_dedupe_weight: bool = False,")
assert old in t, "① 锚点未命中"
t = t.replace(old, new, 1)
t = t.replace("        self.rollout_share_prefix = bool(rollout_share_prefix)",
              "        self.rollout_share_prefix = bool(rollout_share_prefix)\n"
              "        self.rollout_dedupe_weight = bool(rollout_dedupe_weight)\n"
              "        self.last_effective_samples = None", 1)

# ② 树版：收集 token 序列 + 去重加权
old2 = """        for rng in rngs:
            cur = 0
            for _r in range(self.horizon):
                st = states[cur]
                if st.terminal:
                    break
                ops = self._decide_actions(st, rng, ctx, player, _r)
                token = tuple(sorted(ops)) if ops else ()
                key = (cur, token)"""
new2 = """        leaf_tokens = []
        for rng in rngs:
            cur = 0
            _seq = []
            for _r in range(self.horizon):
                st = states[cur]
                if st.terminal:
                    break
                ops = self._decide_actions(st, rng, ctx, player, _r)
                token = tuple(sorted(ops)) if ops else ()
                _seq.append(token)
                key = (cur, token)"""
assert old2 in t, "② 锚点未命中"
t = t.replace(old2, new2, 1)

old3 = """            leaf_states.append(states[cur])
        self.last_tree_nodes = getattr(self, \"last_tree_nodes\", 0) + next_id
        if not leaf_states:
            return 0.0
        _lv = self._leaf_values(leaf_states, player)     # ← 批量评估（与朴素路径一致）
        return float(np.mean(_lv))"""
new3 = """            leaf_states.append(states[cur])
            leaf_tokens.append(tuple(_seq))
        self.last_tree_nodes = getattr(self, \"last_tree_nodes\", 0) + next_id
        if not leaf_states:
            return 0.0
        _lv = self._leaf_values(leaf_states, player)     # ← 批量评估（与朴素路径一致）
        if not self.rollout_dedupe_weight:
            self.last_effective_samples = getattr(self, \"last_effective_samples\", 0) + len(_lv)
            return float(np.mean(_lv))
        # 去重加权：相同 token 序列的样本只算一次
        groups: dict = {}
        for _tok, _v in zip(leaf_tokens, _lv):
            groups.setdefault(_tok, []).append(float(_v))
        reps = [float(np.mean(v)) for v in groups.values()]
        self.last_effective_samples = (getattr(self, \"last_effective_samples\", 0)
                                       + len(reps))
        return float(np.mean(reps))"""
assert old3 in t, "③ 锚点未命中"
t = t.replace(old3, new3, 1)

P.write_text(t, encoding="utf-8")
print("已加入 rollout_dedupe_weight（按轨迹 token 去重加权）")
