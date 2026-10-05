# _tmp_a1_control.ps1 —— 补**同代码对照**（独立验证者限制②）：
#   在**当前 HEAD** 上重跑 A1 的 128 局（`POSPIN=argmax`，所有 MC/M5 新代码路径惰性），
#   与 10-05 的 `C1_a1_128` 对比 ⇒ 量化"代码漂移"对读数的影响。
#   等 M6bND_128 结束再跑（机器纪律：一次只跑一个多进程批）。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 180
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[a1ctl] 等 M6bND_128 结束…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[a1ctl] 开工 $(Get-Date -Format 'HH:mm:ss')"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_MENU3 = '0'; $env:AZAI_MC_EVERY = '0'; $env:AZAI_REL2ABS = '0'
Remove-Item Env:AZAI_MC_NODOWN -ErrorAction SilentlyContinue

$tokens = ((7..70) | ForEach-Object { "$_", "${_}r" }) -join ' '
$tag = 'A1_head_128'
& $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-70 `
    --json="match_results/ladder_logs/$tag/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[a1ctl] 完成 $(Get-Date -Format 'HH:mm:ss')"
