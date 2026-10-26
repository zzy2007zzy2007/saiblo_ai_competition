"""给 FlatRolloutGreedy 加 last_samples（每动作每分支的分数），由环境变量 AZAI_ROLLOUT_SAMPLES 控制。"""
from pathlib import Path

p = Path("code/my_ai/az_intent/flat_rollout_greedy.py")
t = p.read_text(encoding="utf-8")

if "last_samples" not in t:
    # 1) 初始化
    t = t.replace("        self.last_scores = None",
                  "        self.last_scores = None\n        self.last_samples = None   # list[list[float]]：每候选每分支的分数（开关打开时记录）", 1)
    # 2) 记录点
    t = t.replace(
        "            scores[index] = float(self._leaf_values(leaves, player).mean())",
        "            _lv = self._leaf_values(leaves, player)\n"
        "            if _REC_SAMPLES:\n"
        "                _samples_buf[index] = [float(v) for v in _lv]\n"
        "            scores[index] = float(_lv.mean())",
        1,
    )
    # 3) 在 rollout 循环前建缓冲
    t = t.replace("        scores = np.zeros(len(bundles), dtype=np.float64)",
                  "        scores = np.zeros(len(bundles), dtype=np.float64)\n"
                  "        _samples_buf = [None] * len(bundles) if _REC_SAMPLES else None", 1)
    # 4) 收尾
    t = t.replace("        self.last_scores = scores",
                  "        self.last_scores = scores\n        if _REC_SAMPLES:\n            self.last_samples = _samples_buf", 1)
    # 5) 模块级开关
    t = t.replace("import numpy as np",
                  "import os as _os\nimport numpy as np\n\n_REC_SAMPLES = _os.environ.get(\"AZAI_ROLLOUT_SAMPLES\", \"0\") == \"1\"", 1)
    p.write_text(t, encoding="utf-8")
    print("  已加 last_samples 记录开关")
else:
    print("  已有 last_samples")
