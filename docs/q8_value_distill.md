## 8 步期望值蒸馏（2026-10-20）

目标从"叶子评估"换成亚军决策实际用的量 `mean_terminal_score`（8 回合 rollout 的均值）。

**数据**：`training_history/distill_top2/q8/`
| 文件 | 内容 |
|---|---|
| `q8_dataset_big.jsonl` | 286,434 行 / 1,750 状态（40 局 × 22 回合 × 双方），每行 = (状态, 候选计划) → 64 次 rollout 的均值 + 63 个状态特征 + 41 个计划特征 |
| `q8_ent_big.npz` | 同上的实体张量版（塔/蚁 token + plan token 的格子位置），供 `train_q8_value.py` 用 |

**模型**：`code/my_ai/entity_transformer/q8_value_model.py`（带 plan token 的实体 Transformer，0.52M 参数）

**结果**（按**整局**切分：30 局训练 / 10 局测试，438 个测试状态）：

| 模型 | 状态内 Spearman | top-1 | **「永远 hold」基线** | 该动时动了 | 前 12 名成对排序 |
|---|---|---|---|---|---|
| GBDT（手工特征） | +0.656 | 42.7% | **76.9%** | 29.7% | 43.7% |
| **实体 Transformer（286k 行）** | **+0.822** | 77.2% | **76.9%** | 5.9% | 49.5%（≈随机） |

**结论：都打不过「永远 hold」。** 实体网络的状态内整体排序相关能到 0.822（明显强于手工特征的 0.656），
但**前 12 名之间的细排序等于随机**、`该动时动了` 只有 5.9% ⇒ **学出来的 8 步期望值不能替代搜索**。

原因清楚：决策要的是**分差 ~71 分的分辨率**，而模型在这批数据上的残差量级远大于它；
先验的 +80 又刚好压在这个量级上，所以模型输出几乎永远被 hold 压住。

**checkpoint**
| 文件 | 说明 |
|---|---|
| `checkpoints/q8_ent_big.pt` | 实体网络，286k 行 / 按局切分，状态内 ρ 0.822、top-1 77.2% |
| `checkpoints/q8_value_resid_z.pt` | 早期小数据（50k 行 / 按状态切分）版本，ρ 0.71 |
| `checkpoints/q8_value.pt` | 首版（abs 目标）冒烟产物，无参考价值 |

## ⚠️ 坑：两套 ants schema 并存

| 数据 | ants 字段 | 布局 |
|---|---|---|
| `F:\...\lowq_e2\*.ents.jsonl`（5000 局老数据） | **9 项** | `[id, player, x, y, hp, max_hp, kind, age, behavior]` |
| `distill_top2/abl_linear2_raw/ents_*.jsonl`（`cpp_probe_match.dump_state` 产的） | **12 项** | `[..., kind, age, **level**, behavior, status, frozen]` |

`build_lowq_entity_dataset.py` 用 `ant[:9]` + index 8 当 behavior —— **对 9 项那版是对的**；
若拿它去读 12 项那版，就会把 **level 当 behavior**、behavior/status 全丢。
`build_q8_entity_dataset.py` 是按 **12 项** 写的，只用于新 dump。

