# _tmp_s45_pipeline.ps1 —— 预注册 `docs/prereg_20261005_search_deepen_widen.md` 的 S4/S5 两臂
#   S4 = AZAI_DEPTH=16 ；S5 = AZAI_K=48（sample_mult 仍 15）
#   等机器空闲 → 每臂 32 局（seed 7..22）快读数 → 判据统计
# 对照：A1 同 seed 前缀 = 0.5000；S3（depth=8）同 seed = 0.5938。
# 晋级线（预注册 §2）：p̂ ≥ 0.60 或 ≥ 0.70。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 300
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[s45] 等机器空闲（最多 $WaitTimeoutMin min）…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -like '*az_selfplay.py*' -or $_.CommandLine -like '*_tmp_ladder.py*' -or $_.CommandLine -like '*train_value_net.py*' -or $_.CommandLine -like '*collect_value_prior.py*') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[s45] 开工 $(Get-Date -Format 'HH:mm:ss')"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'
$env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'
$env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$tokens = ((7..22) | ForEach-Object { "$_", "${_}r" }) -join ' '

function Run-Arm([string]$tag, [string]$depth, [string]$k) {
    $env:AZAI_DEPTH = $depth; $env:AZAI_K = $k
    Write-Host "[s45] $tag 开始 $(Get-Date -Format 'HH:mm:ss')  depth=$depth k=$k"
    & $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-22 `
        --json="match_results/ladder_logs/$tag/_analysis.json" 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
}

Run-Arm 'S4_depth16_32' '16' '24'
Run-Arm 'S5_k48_32' '4' '48'
Write-Host "[s45] 全部完成 $(Get-Date -Format 'HH:mm:ss')"
