# _tmp_fresh_ab.ps1 —— 全新 seed 段（103..166 双向 = 128 局）上的**同段对照复制**
#   依据：docs/prereg_20261005_reserve_ladder.md §6.3（2026-10-05 17:2x 修订 B）
#   两臂：A1（pos_pin=argmax，对照）与 reserve 180（当前操作配置）
#   目的：回答"E2 的 +13.3pp 是不是选择段（7..22）的运气"
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 180
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[fresh] 等机器空闲（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder|train_value_net|az_selfplay') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[fresh] 开工 $(Get-Date -Format 'HH:mm:ss')"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'; $env:AZAI_POSPRIOR = 'off'
$env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$tokens = ((103..166) | ForEach-Object { "$_", "${_}r" }) -join ' '

foreach ($arm in @(@('FRESH_A1_128', 'argmax'), @('FRESH_R180_128', 'reserve'))) {
    $tag = $arm[0]; $pin = $arm[1]
    $env:AZAI_POSPIN = $pin; $env:AZAI_RESERVE = '180'
    Write-Host "[fresh] == $tag（pospin=$pin）开始 $(Get-Date -Format 'HH:mm:ss')"
    & $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=103-166 `
        --json="match_results/ladder_logs/$tag/_analysis.json" 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
}
Write-Host "[fresh] 完成 $(Get-Date -Format 'HH:mm:ss')"
