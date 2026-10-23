"""rollout 随机性的"衰减"模式（2026-10-23 用户提议）。

问题（已实测）：`rollout_mode=random` 恒定 p=0.5 ⇒ 双方每回合反复建塔/拆塔（churn），
叶子局面被搅成值网没见过的样子，HP 几乎不变但塔数/金币乱掉 ⇒ 34.4%（显著更差）。

用户提议：
  · 只在**第一次决策**随机（之后静止）；或
  · 以 `p(r) = C · e^(−r/k)` 衰减（开局有随机、随后迅速回到安静）。

这样"对手会有反应"这件事被建模了，但叶子状态仍接近值网训练分布。
新增 `rollout_prob_mode = const | decay | first`，参数 `rollout_decay_c` / `rollout_decay_k`。
"""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "my_ai/az_intent/flat_rollout_greedy.py"
t = P.read_text(encoding="utf-8")

# ① 参数
old = "                 rollout_random_side: str = \"opponent\",  # opponent=只随机对手(推荐) / both=双方"
new = ("                 rollout_random_side: str = \"opponent\",  # opponent=只随机对手(推荐) / both=双方\n"
       "                 # 2026-10-23：随机概率随回合衰减（避免整段 churn 污染叶子状态）\n"
       "                 rollout_prob_mode: str = \"const\",   # const | decay | first\n"
       "                 rollout_decay_c: float = 0.6,        # 衰减模式的首回合概率 C\n"
       "                 rollout_decay_k: float = 3.0,        # 衰减常数 k：p(r)=C*exp(-r/k)")
assert old in t, "① 锚点未命中"
t = t.replace(old, new, 1)

# ② 字段
old2 = "        self.rollout_random_side = rollout_random_side"
new2 = ("        self.rollout_random_side = rollout_random_side\n"
        "        assert rollout_prob_mode in (\"const\", \"decay\", \"first\"), rollout_prob_mode\n"
        "        self.rollout_prob_mode = rollout_prob_mode\n"
        "        self.rollout_decay_c = float(rollout_decay_c)\n"
        "        self.rollout_decay_k = max(1e-6, float(rollout_decay_k))")
assert old2 in t, "② 锚点未命中"
t = t.replace(old2, new2, 1)

# ③ _random_actions 增加 round_idx 参数，并按模式算概率
old3 = "    def _random_actions(self, sim, rng, ctx, root_player=None) -> None:"
new3 = ("    def _random_actions(self, sim, rng, ctx, root_player=None, round_idx: int = 0) -> None:\n"
        "        # 按模式算本回合的出招概率（2026-10-23 用户提议：衰减/仅首回合）\n"
        "        if self.rollout_prob_mode == \"first\":\n"
        "            _p = self.rollout_decay_c if round_idx == 0 else 0.0\n"
        "        elif self.rollout_prob_mode == \"decay\":\n"
        "            import math as _math\n"
        "            _p = self.rollout_decay_c * _math.exp(-round_idx / self.rollout_decay_k)\n"
        "        else:\n"
        "            _p = self.rollout_act_prob\n"
        "        if _p <= 0.0:\n"
        "            return")
assert old3 in t, "③ 锚点未命中"
t = t.replace(old3, new3, 1)

t = t.replace("            if rng.random() >= self.rollout_act_prob:", "            if rng.random() >= _p:", 1)

# ④ 调用处传回合序号
old5 = "                        self._random_actions(sim, _rng, _rctx, player)"
new5 = "                        self._random_actions(sim, _rng, _rctx, player, _round_idx)"
assert old5 in t, "⑤ 锚点未命中"
t = t.replace(old5, new5, 1)

# ⑤ 循环里维护 _round_idx
old6 = """                for _ in range(self.horizon):
                    if sim.terminal:
                        break
                    if _rng is not None:"""
new6 = """                _round_idx = 0
                for _ in range(self.horizon):
                    if sim.terminal:
                        break
                    if _rng is not None:"""
assert old6 in t, "⑥ 锚点未命中"
t = t.replace(old6, new6, 1)
t = t.replace("                    sim.advance_round()\n                    total_rounds += 1",
              "                    sim.advance_round()\n                    _round_idx += 1\n                    total_rounds += 1", 1)

P.write_text(t, encoding="utf-8")
print("已加入 rollout_prob_mode = const / decay / first")
