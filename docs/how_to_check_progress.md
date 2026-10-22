# 进度速查（不用问 AI 自己也能看）

> 用途：想知道某条实验跑到哪了、现在什么战绩、我们的出招长什么样，照下面查即可。
> 所有命令在仓库根目录 `D:\2026智能体大赛_新2_distill` 下执行，Python 用
> `D:\anaconda3\envs\pytorch-gpu\python.exe`（或先 `conda activate pytorch-gpu`）。

## 一、信息都存在哪

| 想看什么 | 文件/目录 | 说明 |
|---|---|---|
| **实验总日志** | `docs/experiment_log.md` | 每次 `run_logged.sh` 都会追加：commit、命令、输出；末尾有 `result`（我事后补）|
| **每条实验的实时输出** | `training_history/runs/<时间戳>_<名字>/output.log` | **最直接**：里面有每局完成时的 `[seed N] flat=P0 score=1.0 rc=0 rounds=482 hp=(0,15) (60.5 min)` |
| **对亚军的对局明细** | `match_results/ladder_logs/<TAG>/<seed>.log` | 逐回合：双方原文 + 解析出的 op + RESULT |
| **A/B 内部对局明细** | `training_history/<AB目录>/seed<N>.log` | 同上；头部有两侧完整 env（可复现）|
| **我方决策过程（stderr）** | `match_results/ai{0,1}_az_bridge_ai_seed<N>.stderr.log` | 每回合 `coins/towers/hp/mismatch/illegal`；升级会打 `⬆️ 显式 op`；收尾有 `mc_taken/mc_fallback/mc_err` |
| **检查点说明** | `docs/checkpoints.md` | 每个 ckpt 的来源与读数 |
| **当前有哪些实验在跑** | `Get-ChildItem training_history/runs -Directory \| Sort-Object LastWriteTime -Descending \| Select -First 5` | 按时间排 |

## 二、现成脚本（直接给数）

```powershell
$PY = "D:\anaconda3\envs\pytorch-gpu\python.exe"

# ① 对亚军的战绩（中途也能看，会自己数已完成局数）
& $PY code\test_match\partial_arm.py <TAG>          # 例：VS_RUNNERUP_upg1fix4

# ② 内部 A/B 的胜负
& $PY code\test_match\reparse_ab.py <AB目录>         # 例：training_history/ab_upg1_vs_none

# ③ 内部 A/B 的血量差 + 95%CI（判断显著性看这个）
& $PY code\test_match\hp_margin_ab.py <AB目录>

# ④ 出招构成（建塔/降级/升级/闪电 的占比，A 侧 vs 对手）
& $PY code\test_match\op_mix_match.py match_results\ladder_logs\<TAG>

# ⑤ 实时状态总览（最近 N 分钟有更新的对局）
& $PY code\test_match\live_status.py 40 22
```

## 三、一句话看进度（最快）

```powershell
# 正在跑的实验 + 已完成的局数
Get-ChildItem training_history/runs -Directory | Sort-Object LastWriteTime -Desc |
  Select -First 3 Name, @{n='已完成局数';e={
    (Select-String -Path (Join-Path $_.FullName 'output.log') -Pattern 'rc=' -ErrorAction SilentlyContinue | Measure-Object).Count }}

# 或者直接刷 output.log 的最后几行
Get-Content (Get-ChildItem training_history/runs -Directory | Sort-Object LastWriteTime -Desc |
  Select -First 1 | ForEach-Object { Join-Path $_.FullName 'output.log' }) -Tail 5
```

## 四、当前（2026-10-22 10:30）在跑的实验

| 实验 | 目录 / TAG | 看进度 |
|---|---|---|
| **有升级(A) vs 无升级(B)**，配对 64 局 | `training_history/ab_upg1_vs_none` | `reparse_ab.py training_history/ab_upg1_vs_none` |
| （已停）打亚军 `cells4+MC_UPG=1` | `match_results/ladder_logs/VS_RUNNERUP_upg1fix4` | `partial_arm.py VS_RUNNERUP_upg1fix4` |

## 五、几个容易看错的地方

1. **`hp=(a, b)` 的顺序是"P0 的血, P1 的血"**，不是"我们/对手"。要知道自己是哪边：
   `ab_search_modes.py` 偶数种子 A 在 P0、奇数在 P1；打亚军的 ladder 里看 `p0_label`。
2. **中途的 `hp` 不能当战绩**：只有出现 `RESULT` 行才算一局结束（`partial_arm.py` 会自动过滤）。
3. **A/B 里的 `score`**：`score=1.0` 表示 **A 侧（flat-env 那一侧）赢**。
4. **亚军的对局 ≈1 小时/局**，内部 A/B ≈17 分钟/局；看到 `(60.5 min)` 是正常的，不是卡住。
5. **`training_history/runs/` 里可能有"孤儿"目录**（进程被杀，日志只有表头）——以 `output.log` 里有没有 `rc=` 行判断它是否真跑过。
