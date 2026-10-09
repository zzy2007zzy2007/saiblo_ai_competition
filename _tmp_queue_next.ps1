# _tmp_queue_next.ps1 —— 当前判据臂完成后的**自动接续队列**（按"机制上最该补的两条"排序）
#
#   守卫：先等 `AB_VSEARCH_dec_vs_baseline` 的分析文件出现（当前占满 8 线程的臂），再开始跑队列。
#   步骤 1：**K48 的判据复核**（延后项）——`MC1 + K=48` vs **冻结基线**、全固定段 128 局。
#           理由（写死在档）：K48 改的是**候选集本身**（不是菜单的"类个数"）⇒ 与 TOPK5 的机制**不同**，
#           不能仅凭 TOPK5 的"筛查增益不迁移"前例就把它否掉。
#   步骤 2：**搜索值网的"MC 评分器"闸门**——`MC1 + MC_SCORE=value_only` + 搜索值网 vs example、64 局（32 对）。
#           目的：补齐四方对照（rollout 0.9375 / 旧网 0.3438 / 终局网 0.4531 / **搜索值网 ?**）。
#   注：**价值迭代第二轮**是**有条件**的（预注册进入条件：第一轮判据 ≥0.72）⇒ 本脚本**不**自动执行它。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$GuardTag = 'AB_VSEARCH_dec_vs_baseline',
    [int]$GuardTimeoutMin = 240
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo

$guard = "match_results/ladder_logs/$GuardTag/_analysis.txt"
$t0 = Get-Date
Write-Host ("[queue] 等待守卫 $GuardTag 完成（最多 {0} min）… {1}" -f $GuardTimeoutMin, (Get-Date -Format 'HH:mm:ss'))
while (-not (Test-Path $guard) -and ((Get-Date) - $t0).TotalMinutes -lt $GuardTimeoutMin) { Start-Sleep -Seconds 45 }
if (-not (Test-Path $guard)) { Write-Host "[queue] 守卫超时 ⇒ 仍继续（可能该臂失败，交由后续人工判断）" }
Write-Host ("[queue] 守卫就绪 {0} ⇒ 开始步骤 1" -f (Get-Date -Format 'HH:mm:ss'))

# ── 步骤 1：K48 判据复核 ──
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name 'MC1_K48_dec' `
    -Ckpt 'checkpoints/grafted_sdk_A.pt' -Mode 'pos-only' -Pin 'mc_quiet' -Tclass '0.5' `
    -McEvery 1 -K 48 -Opp 'baseline' -SeedSpec '7-70' -Jobs 8 -Repo $Repo
Write-Host ("[queue] 步骤 1 完成 {0}" -f (Get-Date -Format 'HH:mm:ss'))

# ── 步骤 2：搜索值网的评分器闸门 ──
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name 'VSEARCH_gate' `
    -Ckpt 'checkpoints/grafted_sdk_A_vsearch.pt' -Mode 'pos-only' -Pin 'mc_quiet' -Tclass '0.5' `
    -McEvery 1 -McScore 'value_only' -Opp 'example' -SeedSpec '7-38' -Jobs 4 -Repo $Repo
Write-Host ("[queue] 步骤 2 完成 {0}" -f (Get-Date -Format 'HH:mm:ss'))
Write-Host "[queue] 队列结束。下一步（**有条件**）：若 $GuardTag 的判据 ≥0.72 ⇒ 按 docs/prereg_20261009_goal2_value_iter2.md 做价值迭代第二轮。"
