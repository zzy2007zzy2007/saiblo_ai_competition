# 预注册：**候选 C —— 类头从 1-ply 价值先验学习**（`posnet_C_class.pt`）

> 2026-10-05 07:1x 写（**在跑这批数据之前**）。计划见 `docs/class_head_value_prior_plan.md`，
> 前提检验见 `docs/next_step_survey_20261005.md` §3c。
>
> ⚠️ **性质**：这是**方法层**实验（把**我们自己的** 1-ply 价值排序蒸馏进类头）。
> 它**不模仿任何对手**的动作（目标 = 我们自己的价值网对"每个类的后手局面"的估值）。
> ⚠️ 判据、seed 集合、镜像方式、引擎**一律不动**；动的只是"我方"的 ckpt 里**一个 state**。

## 1. 臂（单变量：只换 `class_state`）

| | ckpt | 变化 |
|---|---|---|
| **对照 C0** | `training_history/vprior/posnet_A_k5_m32.pt` | —（读数直接取 `C1_a1_hist32`：**A1 配置**、seed `7..22`、**`p̂ = 0.5000`**）|
| **臂 C1** | `training_history/vprior/posnet_C_class.pt` | 只重训 `class_state`；**`pos_state`/`value_state` 逐位不动**（`save_three_net` 写回）|

- 训练命令（照抄）：
  ```
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/az_intent/train_class_prior.py \
      --ckpt training_history/vprior/posnet_A_k5_m32.pt \
      --data training_history/vprior/vp_A_k5_m32.npz \
      --epochs 30 --batch-size 256 --lr 1e-3 --target-tau 1.0 --val-frac 0.1 --seed 0 \
      --out training_history/vprior/posnet_C_class.pt
  ```
- **对手/评测配置 = A1**（`AZAI_K=24 AZAI_SAMPLE_MULT=15 AZAI_MODE=pos-only AZAI_SKIP1=1` +
  章程 §2 的其余钉死值）；判据与统计同 `prereg_20261005_search_config_ladder.md` §4/§5。

## 2. 样本量与判据（**先写死**）

- **快读数** = seed `7..22`（16 对 = **32 局**），`--jobs=8`，预期 ~8 min；
- **验收** = seed `7..70`（64 对 = 128 局），**只在快读数达标时跑**；
- **主读数** = `analyze_paired.py` 的配对分 `p̂`；
- **晋级规则**：`p̂ ≥ 0.55`（= 对照 **+5pp**；比配置阶梯的 +10pp **更严**，因为 §3c 已量出
  "有明确价值偏好的决策只占 ~12%" ⇒ **期望是小效应**）；或 `p̂ ≥ 0.70`；
- **有效性门槛**：同前（`verdict=engine`、`illegal=0`、无 INVALID、`terminal=True`；无效+缺失 >10% ⇒ 作废）；
- **禁止**：看数据后改样本量/seed/只报子集；**没有**延长规则。

## 3. 预注册的预测（把它变成真检验）

- **P-C1（数值）**：`p̂ ∈ [0.45, 0.60]`。理由：只有 ~12% 的 (决策, head) 对带有明确价值偏好，
  而这 12% 是否足以改变**胜负**完全未知。
- **P-C2（机制，最关键的一条）**：如果蒸馏真的改变了类轴行为，那么 A1 配置下我方的
  **建塔数应当 > 0**（现状是**恒为 0**：`pos-only` 钉类 + 继承类头几乎只选闪电）。
  ⇒ 若 `p̂` 没升但**建塔从 0 变成 >0**，那说明"类轴能被打开"、只是这一步的"质量"还不够 ——
  这**也是**有价值的信息（会把问题从"类轴打不开"改成"类轴打开了但方向不对"）。
- **P-C3（机制旁证，不作判据）**：训练器打印的 "policy vs value 一致率" 在训练后应当**上升**
  （训练前 ~40%）。若它不上升 ⇒ 蒸馏根本没进网络 ⇒ 数值读数无效、要先查代码。

## 4. 判定与后续（先写死）

1. `p̂ ≥ 0.55` ⇒ 跑 128 局验收；若验收 ≥0.70 ⇒ **走机械门**
   （`docs/gate_independent_verification_protocol.md`：无上下文独立验证 + 记档 + 用户确认）。
2. `0.45 < p̂ < 0.55` ⇒ 记"**类头蒸馏在本判据下测不到增益**"（进寄存器，标强度），
   下一步转向：候选 V（价值头阶梯，已有预注册）或"价值头 + 位置头 + 类头同批重训"（需新预注册）。
3. `p̂ ≤ 0.45` ⇒ 记"**类头蒸馏有害**"，并把"价值头自己的类级排序是噪声"列为首要嫌疑
   （这时应先修价值头，再谈蒸馏）。
4. 无论哪种结果，都**必须**同时报 §3 的机制读数（建塔数、一致率），否则这条读数**不可解释**。

## 5. 成本

训练（30 epoch、~26k 决策、GPU）**≈ 25–40 min**；快读数 **≈ 8 min**；验收 **≈ 30 min**。

## 6. 这批**不做**的事

- 不动 `pos_state`/`value_state`；不改判据/seed/镜像/引擎；不模仿对手动作；
- 不把 val 条件 CE 当判据（它只能证明"有没有学进去"）。
