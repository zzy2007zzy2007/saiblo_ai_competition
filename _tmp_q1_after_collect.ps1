# _tmp_q1_after_collect.ps1 —— 等"随机钉类"采集跑完，然后按预注册 Q1 自动走完：
#   建缓存 → 只训价值头 → （若与对照逐张量相同则**不跑对局**、如实记录） → A1 配置 32 局快读数 → 判据统计
#
# 依据：`docs/prereg_20261005_valuehead_pinned_data.md`
#   * 对照 Q0 = A1：seed 7..22 前缀 = 0.5000；
#   * 晋级线 p̂ ≥ 0.55；
#   * 数据侧硬读数 P-Q1：新数据 `己方有塔` ≥ 30%（旧数据 0.0%）。
# 只执行预注册里写死的事；不做选择或结论。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$DataDir = 'training_history/vprior/data_pin002_100',
    [string]$Cache = 'training_history/vprior/vcache_pin002',
    [string]$OutCkpt = 'training_history/vprior/posnet_Q1_valpin.pt',
    [string]$Tag = 'Q1_valpin32',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

# ---------- 0) 等采集结束（连续两次看不到 az_selfplay 进程就开工） ----------
Write-Host "[q1] 等采集结束（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*az_selfplay.py*' })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[q1] 开工 $(Get-Date -Format 'HH:mm:ss')"

# ---------- 1) 数据侧硬读数（P-Q1）----------
$cov = & $Py -u code/test_match/check_tower_coverage.py $DataDir 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/training_history/vprior/_q1_coverage.txt", $cov,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $cov

# ---------- 2) 建缓存 + 只训价值头 ----------
& $runner -Name Q1_cache_build -CommandLine "$Py -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --data $DataDir --cache $Cache --build-cache-only"
& $runner -Name Q1_valtrain -CommandLine "$Py -u code/my_ai/az_intent/train_value_net.py --ckpt training_history/vprior/posnet_A_k5_m32.pt --cache $Cache --epochs 4 --lr 3e-4 --label-mode terminal --freeze-bn --out $OutCkpt"

# ---------- 3) 早停检查：产物是否与对照逐张量相同 ----------
$same = & $Py -u code/test_match/same_ckpt.py --a training_history/vprior/posnet_A_k5_m32.pt --b $OutCkpt 2>&1 | Out-String
Write-Host "[q1] same_ckpt: $($same.Trim())"
if ($same -match 'IDENTICAL') {
    Write-Host "[q1] ⛔ 产物与对照逐张量相同（早停回 epoch -1）⇒ 按构造为 null，**不跑对局**（预注册已写明）。"
    exit 0
}

# ---------- 4) 快读数：A1 配置 + 新 ckpt，seed 7..22 ----------
$env:AZAI_CKPT = $OutCkpt
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'
$env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$tokens = ((7..22) | ForEach-Object { "$_", "${_}r" }) -join ' '
& $runner -Name $Tag -CommandLine "$Py -u _tmp_ladder.py --tag=$Tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"

$out = & $Py -u code/test_match/analyze_paired.py --tag=$Tag --seeds=7-22 `
    --json="match_results/ladder_logs/$Tag/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$Tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[q1] 完成 $(Get-Date -Format 'HH:mm:ss')"
