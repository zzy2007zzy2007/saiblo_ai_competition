"""加「只随机对手」选项（2026-10-23）。

依据（读亚军源码 `simulate_round(rng)`）：
  · 亚军随机的是**环境/敌方**：蚂蚁移动 `move_phase(rng)`、传送 `teleport_phase(rng)`、
    敌方生成 `spawn_enemy_ant(rng)`（lure_v4 里 `ignore_enemy_spawns=true` 关掉了生成）；
  · **我方始终按候选计划走**（`apply_operations(simulator, plan.ops)`），之后是
    `reactive_turn_operations`（按规则的反应式操作，**不是随机**）。
  · 我们上一版把"自己也随机"了 ⇒ 破坏候选动作的价值 ⇒ 实测 34.4%（CI[-12.3,-2.6]）显著更差。
  ⇒ 新增 `rollout_random_side`：opponent（只随机对手，推荐）/ both（上一版，留作对照）。
"""
from pathlib import Path

P = Path(__file__).resolve().parents[1] / "my_ai/az_intent/flat_rollout_greedy.py"
t = P.read_text(encoding="utf-8")

# ① 构造参数
old = "                 rollout_mode: str = \"idle\",        # idle=现状(双方不出招) / random=随机出招"
new = ("                 rollout_mode: str = \"idle\",        # idle=现状(双方不出招) / random=随机出招\n"
       "                 rollout_random_side: str = \"opponent\",  # opponent=只随机对手(推荐) / both=双方")
assert old in t, "① 锚点未命中"
t = t.replace(old, new, 1)

# ② 字段
old2 = "        self.rollout_seed = int(rollout_seed)"
new2 = ("        self.rollout_seed = int(rollout_seed)\n"
        "        assert rollout_random_side in (\"both\", \"opponent\"), rollout_random_side\n"
        "        self.rollout_random_side = rollout_random_side")
assert old2 in t, "② 锚点未命中"
t = t.replace(old2, new2, 1)

# ③ _random_actions 只看对手
old3 = "    def _random_actions(self, sim, rng, ctx) -> None:"
new3 = "    def _random_actions(self, sim, rng, ctx, root_player=None) -> None:"
assert old3 in t, "③ 锚点未命中"
t = t.replace(old3, new3, 1)

old4 = "        for _pl in (0, 1):"
new4 = ("        _sides = (0, 1)\n"
        "        if self.rollout_random_side == \"opponent\" and root_player is not None:\n"
        "            _sides = (1 - int(root_player),)   # 只让对手出招；我方打完候选 op 后不再动\n"
        "        for _pl in _sides:")
assert old4 in t, "④ 锚点未命中"
t = t.replace(old4, new4, 1)

# ④ 调用处传入 player
old5 = "                        self._random_actions(sim, _rng, _rctx)"
new5 = "                        self._random_actions(sim, _rng, _rctx, player)"
assert old5 in t, "⑤ 锚点未命中"
t = t.replace(old5, new5, 1)

P.write_text(t, encoding="utf-8")
print("已支持 rollout_random_side=opponent（只随机对手）")
