# _tmp_c2_class_read.ps1 —— 候选 C 的快读数（预注册 `docs/prereg_20261005_class_prior.md` §2）
#
# 做什么：等当前 ladder 退出 → 用 **A1 配置** + 新 ckpt（`posnet_C_class.pt`）跑
#         seed 7..22（16 对 = 32 局）→ 出配对分并存盘。
# 只执行预注册里写死的事；不做任何选择或结论。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Ckpt = 'training_history/vprior/posnet_C_class.pt',
    [string]$Tag = 'C2_class32',
    [string]$Seeds = '7-22',
    [string]$PosPin = 'argmax',
    [int]$Jobs = 8,
    [int]$WaitTimeoutMin = 90
)

Set-Location $Repo

$env:AZAI_CKPT = $Ckpt
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_POSPIN = $PosPin; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'

$parts = $Seeds -split '-'
$seedList = [int]$parts[0]..[int]$parts[1]
$tokens = ($seedList | ForEach-Object { "$_", "${_}r" }) -join ' '

# 等上一个 ladder 退出（机器纪律：一次只跑一个多进程程序）
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin); $saw = $false
while ((Get-Date) -lt $deadline) {
    $r = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like '*_tmp_ladder.py*' })
    if ($r.Count -gt 0) { $saw = $true }
    elseif ($saw) { Write-Host "[c2] 上一个 ladder 已退出 $(Get-Date -Format 'HH:mm:ss')"; break }
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 10

Write-Host "[c2] 快读数开始 $(Get-Date -Format 'HH:mm:ss')  tag=$Tag  ckpt=$Ckpt"
& "$Repo/code/run_logged.ps1" -Name $Tag -CommandLine "$Py -u _tmp_ladder.py --tag=$Tag --jobs=$Jobs --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $tokens"

$out = & $Py -u code/test_match/analyze_paired.py --tag=$Tag --seeds=$Seeds `
    --json="match_results/ladder_logs/$Tag/_analysis.json" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$Tag/_analysis.txt", $out,
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host $out
Write-Host "[c2] 完成 $(Get-Date -Format 'HH:mm:ss')"
