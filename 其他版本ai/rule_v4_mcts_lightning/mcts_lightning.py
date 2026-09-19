"""rule_v4 的**闪电位置**改由我们的网络决定；其余（含"何时放闪电"）逐字照原规则。

**不改 rule_v4 本体**：本文件是它的子类，只覆盖 `choose_bundle` —— 拿到原规则选出的
bundle 后，把 `OperationType.USE_LIGHTNING_STORM` 那个 op 的 (arg0, arg1) 换掉，其余原样返回。
`choose_bundle` 的四条策略（冷却/金币判断、拆塔、升级、建塔、hold）全部沿用，所以
"什么时候放闪电"完全没变；闪电的花费与位置无关（固定 90），所以也不影响攒钱逻辑。

顺带把 `log()` 关掉：原实现每回合写 `ai_decisions.log`，跑评测会把它撑到 GB 级
（2026-09-19 实测已 909MB）。

`pos_source`：
  * ``"net"``   —— 用我们训好的位置网（`posnet_r1.pt`）的 `action_map[17]` 在合法格上取 argmax
                   （1 次前向，这是**可部署**的形态）
  * ``"value"`` —— 用价值网对该类每个合法格做 1-ply 评估取最优（约 270 次前向，oracle 形态）
  * ``None``    —— 不覆盖（**零方差对照臂**：未改的 rule_v4 打未改的 rule_v4 应为 1.0000）

注意：我们的网络是在**我们自己智能体的局面分布**上训的，rule_v4 的局面不同
⇒ 这个实验本身就是一次**迁移性检验**（见 docs/experiment_log.md）。
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_REPO = _HERE.parents[1]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(1 if str(_p) != str(_REPO / "Ant-Game") else 0, str(_p))

from SDK.backend.model import Operation, OperationType       # noqa: E402
from SDK.utils.actions import ActionBundle                    # noqa: E402
from ai import AI as _RuleV4AI                                # noqa: E402

LIGHTNING_CLASS = 17          # 意图空间里的动作类 17 = 闪电（与 decoder/pos 管线一致）


class AI_LightningSearch(_RuleV4AI):
    def __init__(self, seed: int | None = None, pos_source: str | None = "net",
                 ckpt: str = "training_history/vprior/posnet_r1.pt",
                 max_actions: int = 96) -> None:
        super().__init__(seed=seed, max_actions=max_actions)
        self.pos_source = pos_source
        self.ckpt = ckpt
        self._feat = None
        self._net = None
        self._vmodel = None
        self.n_override = 0        # 统计：实际改过几次位置
        self.n_lightning = 0       # 统计：见过几次闪电
        if pos_source is not None:
            self._load()

    def log(self, message) -> None:      # 关掉 GB 级日志
        pass

    def _load(self) -> None:
        import numpy as np                                    # noqa: F401
        import torch
        from SDK.utils.features import FeatureExtractor
        from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
        torch.set_num_threads(1)
        self._feat = FeatureExtractor(max_actions=96)
        self._net, self._raw_net = make_net_fn_from_ckpt(self.ckpt, self._feat)
        if self.pos_source == "value":
            _, _, self._vmodel = load_three_models(self.ckpt)
        if self.pos_source == "mcts":
            from my_ai.az_intent.bundle_mcts import BundleMCTS
            # 把 head 0 的 argmax 强制成闪电 ⇒ `pos_pin="argmax"` 就会把类钉在 17
            # （BundleMCTS 的 pos-only 是"钉住每个头自己的 argmax"，没有"指定类"的入口，
            #  所以这里在 net_fn 外面套一层，不改搜索代码）。
            def _pinned_net(st, pl):
                out = self._raw_net(st, pl)
                hl = list(out["head_logits"])
                hl0 = np.asarray(hl[0], dtype=np.float32).copy()
                hl0[LIGHTNING_CLASS] = float(hl0.max()) + 1.0
                hl[0] = hl0
                out = dict(out)
                out["head_logits"] = hl
                return out
            self._mcts = BundleMCTS(_pinned_net, iterations=256, max_depth_rounds=4, k=24,
                                    t_class=0.5, t_pos=1.0, seed=0,
                                    search_mode="pos-only", skip_single_candidate=True)

    # ── 位置来源 ──────────────────────────────────────────────────────────
    def _legal_lightning_cells(self, state, player):
        """闪电在该局面下的全部可执行落点。用与 pos 管线同一套掩码。"""
        import numpy as np
        from my_ai.decoder import make_position_masks
        pm = make_position_masks(state, player, intent_decoding=True)
        if LIGHTNING_CLASS >= len(pm):
            return None, pm
        return np.argwhere(pm[LIGHTNING_CLASS]), pm

    def _pick_cell(self, state, player):
        """返回 (x, y) 或 None（None = 保留原规则的选择）。"""
        import numpy as np
        import torch
        cells, pm = self._legal_lightning_cells(state, player)
        if cells is None or len(cells) == 0:
            return None
        if self.pos_source == "mcts":
            res = self._mcts.search(state, player, temperature=0.0)
            for op in res.chosen_bundle:
                if int(op[0]) == int(OperationType.USE_LIGHTNING_STORM):
                    return (int(op[1]), int(op[2]))
            return None
        obs = self._feat.encode_observation(state, player, np.zeros(96))
        b = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        s = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
        with torch.no_grad():
            out = self._net(b, s)

        if self.pos_source == "net":
            am = out["action_map"][0].numpy()
            vals = am[LIGHTNING_CLASS][cells[:, 0], cells[:, 1]]
            return tuple(int(v) for v in cells[int(np.argmax(vals))])

        # "value"：逐格建 post-state 过价值网（对手视角取负 = 我方优势）
        from my_ai.decoder import decode_head, make_class_mask
        am = out["action_map"][0].numpy()
        cm = make_class_mask(state, player, position_mask=pm, intent_decoding=True)
        hl0 = out["head1_logits"][0].numpy()
        posts, keep = [], []
        for (x, y) in cells:
            am2 = am.copy(); am2[LIGHTNING_CLASS] = -1e9
            am2[LIGHTNING_CLASS, x, y] = 1e9
            op = decode_head(hl0, am2, cm.copy(), pm.copy(), state, player,
                             class_id=LIGHTNING_CLASS, temperature=0.0,
                             pos_temperature=0.0, intent_decoding=True)
            if op is None or not state.can_apply_operation(player, op, ()):
                continue
            post = state.clone(); post.apply_operation_list(player, [op])
            ob = self._feat.encode_observation(post, 1 - player, np.zeros(96))
            posts.append((ob["board"], ob["stats"])); keep.append((int(x), int(y)))
        if len(keep) < 2:
            return None
        adv = []
        for i in range(0, len(posts), 128):
            bb = torch.from_numpy(np.stack([t[0] for t in posts[i:i + 128]])).float()
            ss = torch.from_numpy(np.stack([t[1] for t in posts[i:i + 128]])).float()
            with torch.no_grad():
                v = self._vmodel(bb, ss)["value"].squeeze(-1).numpy()
            adv.extend((-v).tolist())
        return keep[int(np.argmax(adv))]

    # ── 唯一的改动点 ──────────────────────────────────────────────────────
    def choose_bundle(self, state, player, bundles=None) -> ActionBundle:
        b = super().choose_bundle(state, player, bundles)
        if self.pos_source is None:
            return b
        idx = [i for i, op in enumerate(b.operations)
               if op.op_type == OperationType.USE_LIGHTNING_STORM]
        if not idx:
            return b
        self.n_lightning += 1
        cell = self._pick_cell(state, player)
        if cell is None:
            return b
        i = idx[0]
        old = b.operations[i]
        if (int(cell[0]), int(cell[1])) == (int(old.arg0), int(old.arg1)):
            return b                      # 与原规则同格，不用改
        self.n_override += 1
        ops = list(b.operations)
        ops[i] = Operation(old.op_type, int(cell[0]), int(cell[1]))
        return ActionBundle(name=b.name + "+posnet", operations=tuple(ops), score=b.score)
