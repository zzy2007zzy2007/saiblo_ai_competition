# 池化 × tau 标签组合实验（2026-09-12）

## 动机

池化消融（`docs/az_value_head_spatial_pooling_plan.md`）**是在 terminal 标签下做的**：
6 种池化的价值头输入构造不同，但标签都是"每局一个常数"的 terminal 血量差。

而 terminal 标签的一个硬伤刚被量化（2026-09-12 核查 `warm_polonly`）：

| 每局 \|HP 差\| | 局数 | 占比 |
|---|---|---|
| = 1（标签 0.05，基本噪声）| 54 | 16% |
| ≤ 2 | 94 | 28% |
| ≥ 5 | 171 | 51% |
| ≥ 10 | 36 | 11% |

**每局只有一个标量标签**，且 28% 的局连这个标量都接近 0 → 逐帧的 board/stats 不同，
但监督信号几乎不变 → 价值头没法学"这个位置/走法好不好"，只能学"这局大概谁赢"。

**tau 加权血量差**（`az_train.add_weighted_labels`，gamma=exp(-1/tau)）是**逐帧变化**的
标签：每帧的标签 = 从该帧起、指数加权的未来血量差均值。tau=50 → gamma≈0.980，
视野约 50 帧（≈25 回合），比 terminal 平滑但**有逐帧结构**。

**假设**：池化（空间信息）× tau 标签（逐帧信号）**可能有交互**——如果标签本身
逐帧没信息，那么价值头给再细的空间输入也学不到位置依赖；只有逐帧有信号时，
"塔在哪个位置"才有可能体现到 value 上。这正是池化在 terminal 下全灭的一个可能解释。

## 关键便利：标签可以离线重算，不用重采数据

`warm_polonly` 的 npz 存了**逐帧** `stats`（42 维）和 `player`，而 `stats[1] = hp_delta`
（`Ant-Game/SDK/utils/features.py:136` 的 `named` 字典，`round_ratio` 在 0、`hp_delta` 在 1）。
`az_train.add_weighted_labels` 也正是只读 `stats[1]` + `player`。

→ **同一批 336 局**就能算出 tau 标签，**唯一变量 = 标签**，采集成本为 0。

## 改动清单

`code/my_ai/az_intent/value_warmup.py`：

1. 新增 `_exp_weighted_labels(stats, player, *, tau, label_scale, label_mode, mix_alpha)`：
   **逐局**重算标签，公式与 `az_train.add_weighted_labels` **逐步一致**（同一 gamma、
   同一 P0 统一视角、同一 clip/scale、同一按 player 翻符号）
2. `load_all_data(npz_paths, label_cfg=None)`：`label_cfg` 为 None 或 mode=="terminal"
   → 用 npz 里存的 `value_target`（**向后兼容，默认不变**）；否则逐局重算
3. 新增参数：`--label-mode {terminal,abs,rel,mix}`（默认 `terminal`）、`--tau`、`--label-scale`、
   `--label-mix-alpha`；标签配置写进 checkpoint 元数据
4. 打印标签统计（mean/std/range）+ **每局标签幅度分布**（就是上面那张表），便于判读
5. 同时加载 npz 的 `player`（现在没加载）

## 正确性验证（必须先做）

用**同一局**交叉验证：`data_polonly` 的 pkl 走 `az_train.add_weighted_labels`，
`warm_polonly` 的对应 npz 走新的 `_exp_weighted_labels`，**逐帧标签必须一致**
（float16 stats 的量化误差内）。不一致就不许开跑。

## 实验设计

**6 臂**：`gap`（基线池化）+ 5 种新池化 `gapmask/gapmax/region/grid/attn`，
全部用 **tau=50 / scale=2 / mode=abs** 标签，其余照抄基线的价值头配方：

- `--hotstart training_history/az_fixed/gen0120_bn_init.pt`
- `--data-dir training_history/az_fixed/warm_polonly --skip-collect`
- `--epochs 10 --freeze-backbone --device auto`
- 只有 `--value-pool` 和 `--label-mode/--tau/--label-scale` 变

流程：训练 → `convert_no_bn_to_bn` → `make_mix_checkpoint`（策略固定为 `az_r10`）。

### 评价（两阶段，控制时间）

- **阶段 1（粗筛）**：各臂 32 局 vs rule_v4。只作淘汰用，**不作结论**
  （已两次验证 vs rule_v4 小局数是不可靠判据）
- **阶段 2（判据）**：粗筛最好的 2 臂各 64 局 **search-vs-search**，对手 =
  terminal 基线 `mix_r10p_vw_pol_frozen.pt`。这是灵敏判据

## 判读

- 某臂 svs > **60%** → **池化 × tau 有交互**，继续沿该池化加深
- 全部 svs ≈ 50% → 池化的全灭不是"标签太粗"导致的；tau 标签本身在 6 臂上都没救
- 还要看标签幅度分布：tau 标签若仍挤在 0 附近（一局只赢 1 血 → 未来均值≈1 →
  标签≈0.05×2=0.1），说明**病根在"对局本身不决定性"**，而不是标签形式 → 回到对手池

## 风险

- **tau 标签可能不改变幅度问题**：加权未来均值在"贴身局"里同样接近 0，scale=2 只是
  乘个常数。所以本实验更可能验证"标签形式不是瓶颈"，而不是救活池化
- rel 模式与搜索读 value 的方式（绝对评估）不匹配（历史结论），故主用 abs
- 向后兼容：默认 `--label-mode terminal` 时逐位复现现有 336 局结果

## ⚠️ 关键发现（2026-09-12，实现后、跑之前发现）

抽查 336 局，把 tau 标签与"当前血量差"（`stats[1]`，**玩家视角**，即网络可以直接从
输入里抄到的量）做相关：

| 标签形式 | 与当前血量差的相关 | 备注 |
|---------|------------------|------|
| tau50 **abs** s2 | **0.937** | R²≈0.88 → 88% 方差可被"抄输入"解释 |
| tau50 mix s2 | **0.985** | 更严重 |
| tau50 rel s2 | −0.241 | 模式设计上就减掉了 d_t |

**含义**：血量差是慢变量、自相关极强 → "未来血量差的指数加权平均"必然≈当前血量差。
于是 tau-abs 标签**大部分是当前输入的直接函数**，价值头只要学"把 stats[1] 乘 2 输出"
就能把 loss 压得很低。实测印证：tau-abs 的 value loss ≈ **0.0126**，而 terminal 基线
≈ **0.05**——标签幅度更大（std 0.425 vs 0.295）却更容易拟合，正是捷径的signature。

这给历史上 tau 家族全灭（s1/s2/s4/s6 = 9.4/21.9/12.5/9.4%，均 < terminal 29.7%）
提供了**机制解释**：这类标签让价值头学的是"复述当前血量差"，而不是"评估棋盘"。

**推论**：本实验（6 臂 tau-abs × 池化）预期仍会失败。它真正的价值是**确认**这一点，
并把结论从"又一个杠杆无效"升级为"标签工程这条线为什么无效"。要真正避开捷径，
需要 **rel 模式**（或任何相对当前状态的量），但 rel 幅度被压缩（std 0.15，27% 样本
贴近 0）。

### 局内逐帧信号对比（支持"terminal 无逐帧信号"的原假设）

| 标签 | 全局 std | 局内(P0)逐帧 std |
|------|---------|-----------------|
| terminal（每局常数）| 0.2945 | **0.0000** |
| tau50 abs s2 | 0.4245 | **0.2344** |
| tau50 rel s2 | 0.1525 | 0.1468 |
| tau50 mix s2 | 0.4228 | 0.2485 |

terminal 的局内逐帧方差**恒为 0** → 同一局所有帧监督信号完全相同 → 池化再细也学不出
位置依赖。这仍然是池化全灭的一个合理解释；但 tau-abs 又引入了捷径问题。
