# _tmp_m6c_root.ps1 —— M6c-ROOT：MC 只在根节点仲裁（预注册 mc_menu.md §4）
#   动机：`_expand` 在 256 次迭代里对每个叶节点都调用 ⇒ mc_quiet 现状每局 ~780 次 MC（且仲裁对手节点）。
#   root-only = 每次搜索 1 次 ⇒ 更便宜、语义更干净（只仲裁"部署时真正要做的那个决策"）。
#   对照 = M6b 的 0.6641（同代码同配置已逐局复现）。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 180
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[m6croot] 等机器空闲…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[m6croot] 开工 $(Get-Date -Format 'HH:mm:ss')"

$env:AZAI_CKPT = 'training_history/vprior/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
Remove-Item Env:AZAI_MC_M4 -ErrorAction SilentlyContinue
Remove-Item Env:AZAI_MC_NODOWN -ErrorAction SilentlyContinue

$tokens = ((7..70) | ForEach-Object { "$_", "${_}r" }) -join ' '
$tag = 'M6cROOT_128'
& $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
$out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-70 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[m6croot] 完成 $(Get-Date -Format 'HH:mm:ss')"
