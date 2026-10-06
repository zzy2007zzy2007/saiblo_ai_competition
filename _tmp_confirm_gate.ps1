# _tmp_confirm_gate.ps1 —— 预注册 `docs/prereg_20261009_buildgate_confirm.md`：
#   「建塔门槛 τ=90」在**未见样本**上的确认实验
#   S1 = 固定段后半 seed 39..70（64 局；第 7 轮只跑过前缀 7..38）
#   S2 = 冻结随机段（128 局）
#   ⚠️ 对照都已存在：S1 用 M6cROOT_128 的 39..70；S2 用 M6cROOT_rand128（现状在随机段的 128 局）
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"

$env:AZAI_CKPT = 'checkpoints/posnet_A_k5_m32.pt'
$env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'
$env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'; $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'
$env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
$env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
$env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
$env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_TOPK = '0'; $env:AZAI_MC_MARGIN = '0'; $env:AZAI_MC_UPG = '0'
$env:AZAI_MC_SCORE = ''; $env:AZAI_MC_SELFSPEND = '0'
$env:AZAI_LIGHT_MIN_COINS = '0'
$env:AZAI_BUILD_MIN_COINS = '90'      # ← 本实验的干预
$env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
$env:AZAI_SLOW_DECISION_SEC = '30'

# ---- S1：固定段后半 39..70 ----
$s1 = (39..70 | ForEach-Object { "$_", "${_}r" }) -join ' '
Write-Host "[confirm] S1（39..70，64 局）开始 $(Get-Date -Format 'HH:mm:ss')"
& $runner -Name CF_gate_S1 -CommandLine "$Py -u _tmp_ladder.py --tag=CF_gate_S1 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $s1"
$o1 = & $Py -u code/test_match/analyze_paired.py --tag=CF_gate_S1 --seeds=39-70 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/CF_gate_S1/_analysis.txt", $o1, (New-Object System.Text.UTF8Encoding($false)))
Write-Host $o1

# ---- S2：冻结随机段 ----
$spec = (Get-Content -Raw -Encoding UTF8 "$Repo/training_history/vprior/_rand_seeds_spec.txt").Trim()
$s2 = (($spec -split ',') | ForEach-Object { $_, "${_}r" }) -join ' '
Write-Host "[confirm] S2（随机段，128 局）开始 $(Get-Date -Format 'HH:mm:ss')"
& $runner -Name CF_gate_S2 -CommandLine "$Py -u _tmp_ladder.py --tag=CF_gate_S2 --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py $s2"
$o2 = & $Py -u code/test_match/analyze_paired.py --tag=CF_gate_S2 --seeds="$spec" 2>&1 | Out-String
[System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/CF_gate_S2/_analysis.txt", $o2, (New-Object System.Text.UTF8Encoding($false)))
Write-Host $o2
Write-Host "[confirm] 全部完成 $(Get-Date -Format 'HH:mm:ss')"
