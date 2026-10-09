# _tmp_iter2.ps1 —— **价值迭代第二轮**全流程自动化（预注册：docs/prereg_20261009_goal2_value_iter2.md）
#
#   进入条件（已满足）：第一轮判据 `AB_VSEARCH_dec_vs_baseline` = **0.7188**，落在预注册的 0.68–0.72 带
#   ⇒ 允许再做**一轮**，且结论一律标"**不显著**"。
#
#   流程（每步都打印，便于事后核对）：
#     0. 守卫：等当前"接续队列"的两步（K48 判据复核、搜索值网闸门）都出分析文件；
#     1. 采集 60 局：**用第一轮的网**（`grafted_sdk_A_vsearch.pt`）+ MC1 配置，双方禁闪电，
#        **新的训练专用 seed 段 `10031..10060`**（与判据段/随机段/未见段全不重叠）+ 双 dump；
#     2. 转换：`azjsonl_to_value_npz`（目标 = 搜索回传的 root_value）；
#     3. 训练：`train_value_antwar`（同架构、按局切分、只替换 value_state）⇒ 产出 `grafted_sdk_A_vsearch2.pt`
#        **发散护栏（预注册）**：若按局 val_mae **劣于第一轮的 0.0718** ⇒ 判"自举发散"，**中止**，不跑判据臂；
#     4. 判据臂：`VSEARCH2_dec`（新网 + MC1 配置 vs 冻结基线、128 局）⇒ 与第一轮 0.7188、MC1 0.6797 比。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$Lo = 10031,
    [int]$Hi = 10060,
    [string]$Tag = 'ITER2_COLLECT',
    [double]$PrevMae = 0.0718,
    [int]$GuardTimeoutMin = 360
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo

# ── 0. 守卫：等当前队列两步完成 ──
$g1 = 'match_results/ladder_logs/AB_MC1_K48_dec_vs_baseline/_analysis.txt'
$g2 = 'match_results/ladder_logs/AB_VSEARCH_gate_vs_example/_analysis.txt'
$t0 = Get-Date
Write-Host ("[iter2] 等当前队列完成… {0}" -f (Get-Date -Format 'HH:mm:ss'))
while (((-not (Test-Path $g1)) -or (-not (Test-Path $g2))) -and ((Get-Date) - $t0).TotalMinutes -lt $GuardTimeoutMin) {
    Start-Sleep -Seconds 45
}
Write-Host ("[iter2] 守卫完成/超时 {0} ⇒ 开始采集" -f (Get-Date -Format 'HH:mm:ss'))

# ── 1. 采集（用第一轮的网 + MC1 配置）──
#     注意：`_tmp_collect_az.ps1` 内部把 AZAI_CKPT 钉成**冻结基线**（原始部署网）。
#     本轮要的是"**用第一轮的网**采搜索值" ⇒ 采完后另跑一次转换前，先把该网写进 env 覆盖：
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_collect_az.ps1" `
    -Lo $Lo -Hi $Hi -Jobs 4 -Tag $Tag -ObsDir 'data/i2_obs' -AkDir 'data/i2_policy' `
    -Ckpt 'checkpoints/grafted_sdk_A_vsearch.pt'
Write-Host ("[iter2] 采集完成 {0}" -f (Get-Date -Format 'HH:mm:ss'))

# ── 2. 转换 ──
& $Py -u code/my_ai/az_intent/azjsonl_to_value_npz.py --obs-dir "data/i2_obs/$Tag" --az-dir "data/i2_policy/$Tag" --out 'data/i2_npz'
Write-Host ("[iter2] 转换完成 {0}" -f (Get-Date -Format 'HH:mm:ss'))

# ── 3. 训练（含发散护栏）──
$log = "training_history/value_search_iter2.log"
& $Py -u code/my_ai/train_value_antwar.py --data-dirs data/i2_npz --base checkpoints/grafted_sdk_A.pt `
    --out checkpoints/grafted_sdk_A_vsearch2.pt --epochs 30 --batch-size 256 --lr 1e-3 --val-games 12 --save-net checkpoints/value_search_net2.pt 2>&1 |
    Tee-Object -FilePath $log | Select-Object -Last 3
$line = (Select-String -Path $log -Pattern 'best val_mae=([0-9.]+)' | Select-Object -Last 1)
$newMae = if ($line) { [double]$line.Matches[0].Groups[1].Value } else { 9.9 }
Write-Host ("[iter2] 第二轮按局 val_mae = {0}（第一轮 {1}）" -f $newMae, $PrevMae)
if ($newMae -gt $PrevMae) {
    Write-Host ("[iter2] ⛔ 触发**发散护栏**（预注册：劣于第一轮 0.0718）⇒ 中止，不跑判据臂。请人工判断是否为噪声。")
    exit 0
}

# ── 4. 判据臂 ──
& powershell -NoProfile -ExecutionPolicy Bypass -File "$Repo/_tmp_arm.ps1" -Name 'VSEARCH2_dec' `
    -Ckpt 'checkpoints/grafted_sdk_A_vsearch2.pt' -Mode 'pos-only' -Pin 'mc_quiet' -Tclass '0.5' `
    -McEvery 1 -Opp 'baseline' -SeedSpec '7-70' -Jobs 8 -Repo $Repo
Write-Host ("[iter2] 判据臂完成 {0}（对照：第一轮 0.7188 / MC1 0.6797）" -f (Get-Date -Format 'HH:mm:ss'))
