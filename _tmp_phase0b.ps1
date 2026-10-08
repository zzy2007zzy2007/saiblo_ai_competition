# _tmp_phase0b.ps1 —— 预注册附录 `docs/prereg_20261017_phase0_addendum_economy.md`
#   E1(MC_EVERY=1) / E2(MC_UPG=1+MC_TOPK=3) / E3(POSPIN=playable) —— 都 vs 官方 example(禁闪电)
#   A5：随机化镜像（AZAI_TEMP=1.0，双方都是我方禁闪电）—— 判别 A1 的 p̂=1.0
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$Only = ''
)
Set-Location $Repo
$runner = "$Repo/code/run_logged.ps1"
$tokens = ((7..38 | ForEach-Object { "$_", "${_}r" }) -join ' ')

function Set-Base {
    $env:AZAI_CKPT = 'checkpoints/posnet_A_k5_m32.pt'
    $env:AZAI_DEPTH = '4'; $env:AZAI_SAMPLE_MULT = '15'
    $env:AZAI_MODE = 'pos-only'; $env:AZAI_SKIP1 = '1'; $env:AZAI_TCLASS = '0.5'; $env:AZAI_TPOS = '1.0'
    $env:AZAI_POSPRIOR = 'off'; $env:AZAI_VERIFY = '1'; $env:AZAI_TRACE = '1'
    $env:AZAI_MC_HORIZON = '256'; $env:AZAI_MC_ROOTONLY = '1'
    $env:AZAI_MC_BIDIR = '0'; $env:AZAI_MC_MARGIN = '0'
    $env:AZAI_MC_SCORE = ''; $env:AZAI_MC_SELFSPEND = '0'
    $env:AZAI_LIGHT_MIN_COINS = '0'; $env:AZAI_BUILD_MIN_COINS = '0'
    $env:AZAI_FORCE_UPG = '0'; $env:AZAI_NO_DOWNGRADE = '0'; $env:AZAI_PROBE_UPG = '0'
    $env:AZAI_SLOW_DECISION_SEC = '30'
    $env:AZAI_BAN_CLASSES = '17'; $env:AZAI_BAN_OPS = '21'
}

function Run([string]$Name, [string]$Ai1, [int]$BanOpp) {
    Write-Host "[$Name] start $(Get-Date -Format 'HH:mm:ss')"
    $env:AI_BAN_LIGHTNING = "$BanOpp"
    & $runner -Name $Name -CommandLine "$Py -u _tmp_ladder.py --tag=$Name --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=$Ai1 $tokens"
    if (Test-Path "$Repo/match_results/ladder_logs/$Name") {
        $out = & $Py -u code/test_match/analyze_paired.py --tag=$Name --seeds=7-38 2>&1 | Out-String
        [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$Name/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false)))
        Write-Host $out
    }
}

if ($Only -eq '' -or $Only -eq 'E1') {
    Set-Base; $env:AZAI_ITERS='256'; $env:AZAI_K='24'; $env:AZAI_POSPIN='mc_quiet'; $env:AZAI_MC_EVERY='1'; $env:AZAI_MC_UPG='0'; $env:AZAI_MC_TOPK='0'; $env:AZAI_TEMP='1e-6'
    Run 'P0_E1_mc1' 'code/test_match/ex_pkg/main.py' 1
}
if ($Only -eq '' -or $Only -eq 'E2') {
    Set-Base; $env:AZAI_ITERS='256'; $env:AZAI_K='24'; $env:AZAI_POSPIN='mc_quiet'; $env:AZAI_MC_EVERY='5'; $env:AZAI_MC_UPG='1'; $env:AZAI_MC_TOPK='3'; $env:AZAI_TEMP='1e-6'
    Run 'P0_E2_wide' 'code/test_match/ex_pkg/main.py' 1
}
if ($Only -eq '' -or $Only -eq 'E3') {
    Set-Base; $env:AZAI_ITERS='256'; $env:AZAI_K='24'; $env:AZAI_POSPIN='playable'; $env:AZAI_MC_EVERY='5'; $env:AZAI_MC_UPG='0'; $env:AZAI_MC_TOPK='0'; $env:AZAI_TEMP='1e-6'
    Run 'P0_E3_playable' 'code/test_match/ex_pkg/main.py' 1
}
if ($Only -eq '' -or $Only -eq 'A5') {
    Set-Base; $env:AZAI_ITERS='256'; $env:AZAI_K='24'; $env:AZAI_POSPIN='mc_quiet'; $env:AZAI_MC_EVERY='5'; $env:AZAI_MC_UPG='0'; $env:AZAI_MC_TOPK='0'; $env:AZAI_TEMP='1.0'
    Run 'P0_A5_stoch_mirror' 'code/test_match/az_bridge_ai.py' 1
}
Write-Host "[phase0b] done $(Get-Date -Format 'HH:mm:ss')"
