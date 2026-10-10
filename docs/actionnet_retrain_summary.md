# 重训动作网（禁闪电子博弈）能拿到什么：实测总结

> 目标（2026-10-10 用户下达，预注册 `docs/prereg_20261010_actionnet_retrain.md`）：
> 在**双方都禁闪电**的子博弈里，用**只重训动作网**（`class_state` / `pos_state`，`value_state` 与部署配置全程冻结）
> 的办法，让对**冻结基线**的配对胜率出现 **+10pp** 的可观测提升（固定段 seed 7..70：`p̂ ≥ 0.78`，
> incumbent `MC1` = 0.6797）。次级门槛：冻结随机段 `p̂ ≥ 0.7512`（= MC1 0.7812 − 3pp）。
> 本文是**无论成败都要交**的那份总结；所有读数都来自 `run_logged` 归档（见 `docs/experiment_log.md` 第 79–84 轮）。

---

## 1. 一句话结论

**在这个子博弈里，"只重训动作网"目前拿不到 +10pp；两条路线有实测判据（B 掉 11.7pp、A 连"教师可学性"闸门都过不去），
第三条（C 类头）已证明"类头确实控制行为"并正在跑判据。** 真正的 +18~28pp 杠杆是**部署侧的 MC 仲裁覆盖率**（`MC_EVERY` 5→1），
不是动作网参数本身——这与"`S1b` 蒸馏 +41.9pp"那次的历史经验**不矛盾但方向不同**：那次换的是动作网的**来源**（SDK 蒸馏），
不是把一个已收敛的动作网**再训得更收敛**。

---

## 2. 实测结论（每条都有判据尺度读数）

| 路线 | 做法（唯一变量） | 固定段 64 对 `p̂` | 对照 incumbent `MC1` | 判定 |
|---|---|---|---|---|
| **B** | 把动作网来源的 SDK 蒸馏 30 → **90 epoch**（训练侧 cls 0.6715→0.6288、map 0.1491→0.1418），`value` 逐位不变 | **0.5625**（CI [0.4766,0.6484]，0% 无效，判词全 engine；执先 33/64、执后 39/64） | 0.6797 | ❌ **−11.7pp**（配对 McNemar 更差 22 / 更好 12 ⇒ **p=0.1214**）|
| **A** | 用**安静 rollout**给位置候选打分当教师，蒸馏进 `pos_state` | **未投判据** | — | ❌ **被自己的闸门判死**（见 §3.2）|
| **C** | 用 AWR 加权 BC 把"**MC/搜索实际执行的类**"蒸进 `class_state`（只改类头） | 进行中 | — | ⏳ 机制审计已过（§3.3）|
| **D** | 换更好 critic ⇒ 重新打标 ⇒ 重训位置头 | 未做 | — | 备用（本项目对价值网轴五次否证）|

**B 的机制列几乎不变**（30.3 建 / 19.3 降 / 6.3 升 / 45.9 有操作回合 / 240 回合，incumbent 29.0/19.9/7.8/46.2/243，均 5.46 min/局）
⇒ **行为微变、棋力掉 11.7pp**。训练损失更好而棋力更差 ⇒ **"训练指标 ≠ 棋力"的第 N 次实证**
（同型：`phold10 −7.3pp`、价值网"离线更准、棋力更差"、"训练 loss 更低但 rule_v4 胜率不涨"）。

---

## 3. 三条关键机制发现

### 3.1 "搜索 visit 教师"几乎没有信息（路线 A 的第一版死因）
在 incumbent 配置的 dump 上测 29,303 个决策：
- 真有位置选择（≥2 个落点）的只有 **838 = 2.9%**；
- 落点数只有 **2 个（696）/ 3 个（142）**；
- **教师分布归一化熵中位 1.000（100% 的决策 >0.9）**，**"决定性"（top1 ≥ 2×top2）= 0.0%**，**"近平局"（top1 ≤ 1.2×top2）= 100%**。

⇒ 搜索的 visit 分布在位置维上**基本是硬币**（树太薄 ⇒ visit 计数 1:1），**拿它当教师只能教噪声** ✓
（这正是历史把 `train_pos_distill` 关闭的原因）。

### 3.2 "2 格二选一"是结构性限制 ⇒ 位置头学不动（路线 A 修订版的死因）
- 改用**引擎安静 rollout**（确定性、有区分度）当教师：30 局试点得 46,080 行，
  `|top1−top2|` 中位 **0.100**、**≥0.05(=1 HP) 占 100%**、完全平局只 16.7% ⇒ **信号确实存在** ✓；
- 但即便**尝试全部合法格**，每个决策**只有 ~2 格可解码**（实测 3,072 位置决策/局、可成功强制落格的 ≈2 个）；
- 2 格 + 逐行 z-score ⇒ z 恒为 ±1 ⇒ **CE 只取决于"两格谁大"这一个离散量**；
- 全量 46k 行训练 6 epoch：**参数变化范数 0.15→0.22（参数确实在动）**，而 **train CE 1.0523 / top1 45.10%、
  val CE 1.0423 / top1 55.56% 逐位不变**（≈随机）⇒ **位置头学不会安静 rollout 的偏好** ⇒
  按预注册 §8.1（要求 top1 ≥0.65 且 val CE 降 ≥0.10）**不投判据**，并取消原定 240 局宽格采集（省 ~2 h）。

### 3.3 类头在现配置下**有活通道**（路线 C 的依据）
用干净的**只换类头**对照（`grafted_clsonly_audit.pt`：`class_state` 与 incumbent 仅 **13/94** 张量相同，
`pos_state`/`value_state` **94/94 逐位相同**），在**同 seed（3、3r、4、4r）**上与 incumbent 对比：

| 臂（4 局）| p̂ | 建塔 | 降级 | 升级 | **有操作回合** | 回合 | min/局 |
|---|---|---|---|---|---|---|---|
| incumbent | 1.0000 | **22.0** | 17.5 | 8.5 | **37.5** | 230 | 3.45 |
| 只换类头 | 0.7500 | **33.0** | 22.2 | 7.2 | **50.2** | 248 | 4.92 |

⇒ **建塔 +50%、有操作回合 +34%**（逐局动作行 78→95、60→97，日志哈希全不同）
⇒ 类头在 `POSPIN=mc_quiet + MC_EVERY=1` 下**确实控制行为**（通过 ②`decode_head` 合法性闸门 / ③类内塔型）。
**这不是重复历史**：`C156`"0/32 对翻转"发生在 `pos_pin=argmax` 时代（类头只被 argmax 读 ⇒ 与决策无关）。

---

## 4. 工程与口径教训（都踩过、都留了注释）

| 坑 | 现象 | 正确做法 |
|---|---|---|
| dump 语义 | 拿 `intent`（采样计数）当教师 ⇒ 自检 0/2000 | `cands`=候选束、**`visits`=与 cands 同序的访问分布**、`chosen`=**cands 下标**、`labels`=每 op 的 (类,x,y)、`intent`=**采样计数不是偏好**（`az_bridge_ai.py:183-213`）|
| 配对 | jsonl 前缀 `az_`、obs 前缀 `obs_` ⇒ 找不到 obs | 按 `seedNNNNN_pK` **后缀**配对；被硬杀的局会留**半条 JSONL** ⇒ 逐行容错 + `None` 占位保持对齐 |
| 坐标口径 | npz 存 `(x,y)`，训练器按 `am[..., xy[0], xy[1]]` 取 ⇒ **转置**（目标变噪声） | 存 **(row,col)=(y,x)**（历史 `vp_*.npz` 就是这个口径）|
| npz 键名 | 写了 `boards`/`cells`，训练器读 `board`/`cell`/`cnt` | `cnt`=**每行有效格数**（配 `valid = arange(MAXC) < cnt`）；`weight` 是**每行**标量 |
| 子进程导入 | 多进程子进程 `No module named 'SDK'` | 路径必须在**模块级**设置（`parents[3]` + `Ant-Game`/`code` 入 `sys.path`）|
| 控制台编码 | 打印 `⇒`(U+21D2) 在 GBK 下抛 `UnicodeEncodeError` | 一律 `PYTHONIOENCODING=utf-8` |
| 只试前 M 格 | "宽格采样"反而 **0 训练行**（可解码格常不在前几名） | 按值降序**尝试全部合法格**先收集，再**只记录分散的 M 个** |
| 裸调 ladder | 60 局全部 `verdict=aborted, rounds=0`（dump 每局 1 条） | 必须用**驱动**（注入 `AZAI_CKPT/MODE/...`）；`AZAI_AZ_DUMP`/`AZAI_DUMP_RAW_DIR` 在父进程设不会被覆盖 |
| 训练器"冻结" | train/val CE 6+ epoch 逐位不变，误以为训练器坏了 | 先打**参数变化范数**：参数在动 ⇒ 是**损失对参数不敏感**（C=2 + 逐行 z-score ⇒ CE 只取决于"谁大"）|
| 采集成本 | 打开 AZ dump 后对局**慢约 8 倍**（~5.5 min → ~24 min/局）| 采集预算按此估；60 局 ≈ 2.5–5 h |

---

## 5. 对"动作网到底能不能拿到 +10pp"的判断

- **B、A 两条已用判据尺度量过：拿不到**（B 甚至反向 11.7pp；A 的教师**不可学**）。
- **C 是唯一还有活通道的**（类头确实控制行为），但先验只有 ~15%：
  它改的是"类内塔型/合法性闸门/兜底 pin"，而**决策的主体（选不选、选哪个类）已被 MC 仲裁接管**（`MC_EVERY=1` 时接管率接近 100%）。
- **真正的杠杆在部署侧**：本项目目前唯一稳健的强度来源是 **MC 仲裁覆盖率 20%→100%**（固定段 +18pp、随机段 +28pp）。
  动作网的**来源**（`S1b` SDK 蒸馏 +41.9pp，单变量 0.3828→0.8016）也真实有效——但它有效的原因是**当时那张网还很弱**；
  把**已经收敛**的动作网再训得更收敛（B）不会有正收益 ⇒ **"重训"与"换来源"是两件事**。
- **若还要继续这条线，需要的不是更多 epoch，而是更好的教师**：例如搜索根的**子节点值/Q 值**（历史文档里"证据最强、从未实现"的那条），
  或把位置教师从"2 格二选一"换成**跨类比较**（把类与格一起当候选 ⇒ 目标不再是二值）——但这已触及"动作网"的边界。

---

## 6. 复现清单

```powershell
# 路线 B：拉长蒸馏 → 嫁接（保 value）→ 判据
python -u code/distill/train.py distill_data/ --out checkpoints/distill_sdk_A_e90.pt --epochs 90 --lambda-soft 0
python -u code/tools/graft_policy.py --base checkpoints/posnet_A_k5_m32.pt --policy checkpoints/distill_sdk_A_e90.pt --out checkpoints/grafted_sdk_A_e90.pt
powershell -File _tmp_arm.ps1 -Name E90_vs_baseline -Ckpt checkpoints/grafted_sdk_A_e90.pt -Mode pos-only -Pin mc_quiet `
  -Tclass 0.5 -McEvery 1 -Opp baseline -SeedSpec 7-70 -Jobs 8 -Repo <repo>   # ⇒ p̂=0.5625

# 路线 A：安静 rollout 教师（宽格采样）→ 自写训练器（含参数变化范数）
python -u code/my_ai/az_intent/collect_pos_rollout_teacher.py --checkpoint checkpoints/grafted_sdk_A.pt `
  --games 30 --workers 8 --max-cells 8 --rollout-rounds 256 --seed 20000 --out training_history/posroll/pilot30.npz
python -u code/my_ai/az_intent/train_pos_rollout.py --ckpt checkpoints/grafted_sdk_A.pt `
  --data training_history/posroll/pilot30_yx.npz --out checkpoints/grafted_sdk_A_posroll_s1.pt --epochs 6

# 路线 C：incumbent 配置下采 dump（训练专用 seed）→ ingest → 只训类头 → 判据
$env:AZAI_AZ_DUMP='data/posroll_c_policy/POSROLL_C'; $env:AZAI_DUMP_RAW_DIR='data/posroll_c_obs/POSROLL_C'
powershell -File _tmp_arm.ps1 -Name POSROLL_C_collect2 -Ckpt checkpoints/grafted_sdk_A.pt -Mode pos-only -Pin mc_quiet `
  -Tclass 0.5 -McEvery 1 -Opp baseline -SeedSpec 22000-22029 -Jobs 8 -Repo <repo>      # ⚠️ 冒烟写 seed 7 ⇒ ingest 前移出
python -u code/my_ai/az_intent/ingest_bridge_dump.py --raw-dir data/posroll_c_obs/POSROLL_C `
  --ladder-tag AB_POSROLL_C_collect2_vs_baseline --out-dir data/posroll_c_labeled
python -u code/my_ai/az_intent/train_class_outcome.py --ckpt checkpoints/grafted_sdk_A.pt `
  --data data/posroll_c_labeled --out checkpoints/grafted_sdk_A_clsC.pt --epochs 10 --beta 0.5 --class-weight auto
```

（本文随路线 C 的判据读数落地而更新；所有 ckpt/数据的 sha256 见 `docs/checkpoints.md`。）
