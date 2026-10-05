# _tmp_vt_screen.ps1 —— 价值标签臂的 A1 配置快筛（预注册 §7.2：16 局 seed 7..14，各臂一份）
#   等 M5_rel5_4 跑完 → 依次跑 V-R50 / V-R100 / V-K50 / V-K100（A1 配置）→ 每臂出配对分
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 120
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[vt] 等 M5_rel5_4 结束…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[vt] 开工 $(Get-Date -Format 'HH:mm:ss')"

$tokens = ((7..14) | ForEach-Object { "$_", "${_}r" }) -join ' '
$arms = @(
    @('VR50_16',  'training_history/vprior/posnet_Vrel50.pt'),
    @('VR100_16', 'training_history/vprior/posnet_Vrel100.pt'),
    @('VK50_16',  'training_history/vprior/posnet_Vk50.pt'),
    @('VK100_16', 'training_history/vprior/posnet_Vk100.pt')
)
foreach ($a in $arms) {
    $tag = $a[0]; $ck = $a[1]
    if (-not (Test-Path $ck)) { Write-Host "[vt] 跳过 $tag（缺 $ck）"; continue }
    $env:AZAI_CKPT = $ck
    $env:AZAI_MENU3 = '0'
    $env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
    $env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
    $env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
    $env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
    Write-Host "[vt] == $tag 开始 $(Get-Date -Format 'HH:mm:ss')"
    & $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-14 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
}
Write-Host "[vt] 完成 $(Get-Date -Format 'HH:mm:ss')"
