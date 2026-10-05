# _tmp_vt_screen2.ps1 —— 价值标签臂的 A1 配置快筛（**修正部署 env**）
#   ⚠️ rel 标签的头输出的是"未来优势变化量"；搜索要跨不同局面比较 ⇒ 必须还原成绝对价值：
#      AZAI_VALUE_TANH=0 + AZAI_REL2ABS=1.0（= value_raw + stats[1]/HP_SCALE）
#   依据：code/my_ai/az_intent/az_selfplay.py 的 make_three_net_fn：
#      if value_rel_to_abs > 0: value = value / value_rel_to_abs + stats[1]/HP_SCALE
#   旧那批（VR50_16 等）忘了设 ⇒ 读数作废（VR50=0.375 不可信）。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 60
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[vt2] 等机器空闲…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[vt2] 开工 $(Get-Date -Format 'HH:mm:ss')"

$tokens = ((7..14) | ForEach-Object { "$_", "${_}r" }) -join ' '
$arms = @(
    @('VR50b_16',  'training_history/vprior/posnet_Vrel50.pt'),
    @('VR100b_16', 'training_history/vprior/posnet_Vrel100.pt'),
    @('VK50b_16',  'training_history/vprior/posnet_Vk50.pt'),
    @('VK100b_16', 'training_history/vprior/posnet_Vk100.pt')
)
foreach ($a in $arms) {
    $tag = $a[0]; $ck = $a[1]
    if (-not (Test-Path $ck)) { Write-Host "[vt2] 跳过 $tag（缺 $ck）"; continue }
    $env:AZAI_CKPT = $ck
    $env:AZAI_VALUE_TANH = '0'
    $env:AZAI_REL2ABS = '1.0'
    $env:AZAI_MENU3 = '0'
    $env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
    $env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
    $env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
    $env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
    Write-Host "[vt2] == $tag 开始 $(Get-Date -Format 'HH:mm:ss')"
    & $runner -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-14 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host $out
}
Write-Host "[vt2] 完成 $(Get-Date -Format 'HH:mm:ss')"
