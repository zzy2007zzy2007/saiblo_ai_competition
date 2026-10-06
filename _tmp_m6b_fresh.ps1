# _tmp_m6b_fresh.ps1 —— 样本外补 **M6b 对照**（seed 103..166），把 "ROOT vs M6b" 也放到样本外比较
#   背景：独立验证指出 ROOT−M6b 的 +7.0pp 在 seed 7..70 上不显著（64 对，p=0.256）。
#   样本外（另一段 64 对）能给这条比较加一份独立证据。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 180
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[m6bfresh] 等机器空闲…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[m6bfresh] 开工 $(Get-Date -Format 'HH:mm:ss')"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'
Remove-Item Env:AZAI_MC_ROOTONLY -ErrorAction SilentlyContinue

$tokens = ((103..166) | ForEach-Object { "$_", "${_}r" }) -join ' '
$tag = 'M6b_fresh128'
& $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=103-166 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out

# 三段并排小结（样本外）
Write-Host "=== 样本外三段小结（seed 103..166）==="
foreach ($t in @('M6cROOT_fresh128', 'M6b_fresh128', 'A1_fresh128')) {
    $f = "$Repo/match_results/ladder_logs/$t/_analysis.txt"
    if (Test-Path $f) {
        $line = (Select-String -Path $f -Pattern '配对胜率').Line
        Write-Host ("{0}: {1}" -f $t, $line.Trim())
    }
}
Write-Host "[m6bfresh] 完成 $(Get-Date -Format 'HH:mm:ss')"
