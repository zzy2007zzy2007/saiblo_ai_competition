# _tmp_vk50_next.ps1 —— VK50b（rel+kgeo tau=50）晋级后的两步（预注册 §3：≥0.60 ⇒ 扩大样本）
#   ① A1 配置 **32 局（seed 7..22）** —— 把 16 局的 +12.5pp 钉得更实（对照 A1 = 0.5000）
#   ② **M5 4 局（seed 7..8 双向）** —— 预注册的**关键机制读数**（建塔数）；对照：M5+terminal 头 = 68/局
#   ⚠️ rel 头必须还原绝对价值：AZAI_VALUE_TANH=0 + AZAI_REL2ABS=1.0
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 90
)

Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

Write-Host "[vk50] 等机器空闲…"
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $empty = 0
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and ($_.CommandLine -match '_tmp_ladder') })
    if ($r.Count -eq 0) { $empty++; if ($empty -ge 2) { break } } else { $empty = 0 }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10
Write-Host "[vk50] 开工 $(Get-Date -Format 'HH:mm:ss')"

# 公共：rel 头的正确部署
$env:AZAI_VALUE_TANH = '0'
$env:AZAI_REL2ABS = '1.0'

# ① A1 配置 32 局
$env:AZAI_CKPT = 'training_history/vprior/posnet_Vk50.pt'
$env:AZAI_MENU3 = '0'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = 'argmax'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$t32 = ((7..22) | ForEach-Object { "$_", "${_}r" }) -join ' '
& $runner -Name VK50b_32 -CommandLine "$Py -u _tmp_ladder.py --tag=VK50b_32 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $t32"
$out = & $Py -u code/test_match/analyze_paired.py --tag=VK50b_32 --seeds=7-22 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/VK50b_32/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out

# ② M5 4 局（关键机制读数）
$env:AZAI_MENU3 = '1'
$env:AZAI_MODE = 'joint'; $env:AZAI_SKIP1 = '0'
& $runner -Name M5_k50_4 -CommandLine "$Py -u _tmp_ladder.py --tag=M5_k50_4 --jobs=4 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py 7 7r 8 8r"
$out2 = & $Py -u code/test_match/analyze_paired.py --tag=M5_k50_4 --seeds=7-8 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/M5_k50_4/_analysis.txt", $out2,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out2
Write-Host "[vk50] 完成 $(Get-Date -Format 'HH:mm:ss')"
