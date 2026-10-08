# _tmp_phase0.ps1 —— 预注册 `docs/prereg_20261017_phase0_nolightning.md` 的四个臂
#   A1 镜像(我方禁闪电 vs 我方禁闪电) / A2(vs 官方 example 带闪电) /
#   A3(vs 官方 example 禁闪电) / A4(vs rule_v4 禁闪电副本)
#   每臂 64 局 = 固定段前缀 7..38（32 对），jobs=8
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Only = ''
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"
$tokens = ((7..38 | ForEach-Object { "$_", "${_}r" }) -join ' ')

function Set-OurSide {
    $env:AZAI_CKPT = 'checkpoints/posnet_A_k5_m32.pt'
    $env:AZAI_ITERS = '256'; $env:AZAI_DEPTH = '4'; $env:AZAI_K = '24'; $env:AZAI_SAMPLE_MULT = '15'
    $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
    $env:AZAI_POSPRIOR = 'off'; $env:AZAI_TEMP = '1e-6'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
    $env:AZAI_POSPIN = 'mc_quiet'; $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_EVERY = '5'; $env:AZAI_MC_ROOTONLY = '1'
    $env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_TOPK = '0'; $env:AZAI_MC_MARGIN = '0'; $env:AZAI_MC_UPG = '0'
    $env:AZAI_MC_SCORE = ''; $env:AZAI_MC_SELFSPEND = '0'
    $env:AZAI_LIGHT_MIN_COINS = '0'; $env:AZAI_BUILD_MIN_COINS = '0'
    $env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
    $env:AZAI_SLOW_DECISION_SEC = '30'
    # ★ 本 Phase 的核心：掩码层禁闪电（+ 执行层安全网，触发会记数）
    $env:AZAI_BAN_CLASSES = '17'
    $env:AZAI_BAN_OPS = '21'
}

function Run-Arm([string]$Name, [string]$Ai1, [string]$BanOpp) {
    Set-OurSide
    if ($BanOpp -eq '1') { $env:AI_BAN_LIGHTNING = '1' } else { $env:AI_BAN_LIGHTNING = '0' }
    Write-Host "[$Name] ai1=$Ai1 AI_BAN_LIGHTNING=$($env:AI_BAN_LIGHTNING) 开始 $(Get-Date -Format 'HH:mm:ss')"
    & $runner -Name $Name -CommandLine "$Py -u _tmp_ladder.py --tag=$Name --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=$Ai1 $tokens"
    if (Test-Path "$Repo/match_results/ladder_logs/$Name") {
        $out = & $Py -u code/test_match/analyze_paired.py --tag=$Name --seeds=7-38 2>&1 | Out-String
        [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$Name/_analysis.txt", $out,
            (New-Object System.Text.UTF8Encoding($false)))
        Write-Host $out
    }
}

if ($Only -eq '' -or $Only -eq 'A1') { Run-Arm 'P0_A1_mirror'    'code/test_match/az_bridge_ai.py' '1' }
if ($Only -eq '' -or $Only -eq 'A2') { Run-Arm 'P0_A2_ex_light'  'code/test_match/ex_pkg/main.py'  '0' }
if ($Only -eq '' -or $Only -eq 'A3') { Run-Arm 'P0_A3_ex_nolight' 'code/test_match/ex_pkg/main.py' '1' }
if ($Only -eq '' -or $Only -eq 'A4') { Run-Arm 'P0_A4_rv4_nolight' 'code/test_match/nl_pkg/main.py' '0' }
Write-Host "[phase0] done $(Get-Date -Format 'HH:mm:ss')"
