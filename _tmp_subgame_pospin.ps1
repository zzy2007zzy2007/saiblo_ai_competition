# _tmp_subgame_pospin.ps1 —— 预注册 docs/prereg_20261018_subgame_pospin.md
#   禁闪电子博弈里改"谁决定类"：C0 窄菜单 mc_quiet / P1 masked / P2 argmax(真原版) / P3 masked+TCLASS=1.0
param([string]$Repo = $PSScriptRoot, [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe')
Set-Location $Repo
function Set-Base {
    $env:AZAI_CKPT='checkpoints/posnet_A_k5_m32.pt'
    $env:AZAI_ITERS='256'; $env:AZAI_DEPTH='4'; $env:AZAI_K='24'; $env:AZAI_SAMPLE_MULT='15'
    $env:AZAI_MODE='pos-only'; $env:AZAI_SKIP1='1'; $env:AZAI_TPOS='1.0'; $env:AZAI_TEMP='1e-6'
    $env:AZAI_POSPRIOR='off'; $env:AZAI_VERIFY='1'; $env:AZAI_TRACE='1'; $env:AZAI_TIMING_EVERY='25'
    $env:AZAI_MC_HORIZON='256'; $env:AZAI_MC_ROOTONLY='1'; $env:AZAI_MC_MARGIN='0'; $env:AZAI_MC_SCORE=''
    $env:AZAI_LIGHT_MIN_COINS='0'; $env:AZAI_BUILD_MIN_COINS='0'
    $env:AZAI_FORCE_UPG='0'; $env:AZAI_NO_DOWNGRADE='0'; $env:AZAI_PROBE_UPG='0'
    $env:AZAI_MC_CELLS='0'; $env:AZAI_MC_TOP2='0'; $env:AZAI_MC_M4='0'; $env:AZAI_MC_NODOWN='0'
    $env:AZAI_MC_NO_LIGHT='0'; $env:AZAI_MC_SELFSPEND='0'; $env:AZAI_MC_SELFSPEND_MIX='1.0'
    $env:AZAI_MC_BIDIR='0'; $env:AZAI_MC_ROLLOUT_CLASSES='4'; $env:AZAI_SLOW_DECISION_SEC='60'
    $env:AZAI_BAN_CLASSES='17'; $env:AZAI_BAN_OPS='21'; $env:AI_BAN_LIGHTNING='1'
}
function Report-Arm([string]$tag) {
    foreach ($seeds in @('7-70','7-38','39-70')) {
        $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=$seeds 2>&1 | Out-String
        if ($seeds -eq '7-70') { [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false))) }
        $p = ($out -split "`n" | Select-String -Pattern '配对胜率' | Select-Object -First 1)
        Write-Host ("  [report] {0} seeds={1} | {2}" -f $tag, $seeds, ($(if ($p) { $p.Line.Trim() } else { 'p_hat=?' })))
    }
    $mech = & $Py -u code/test_match/phase0_report.py $tag 2>&1 | Select-Object -Skip 1 -First 1
    Write-Host ("  [mech] " + $mech)
}
function Run-Arm([string]$name, [string]$pin, [string]$tclass, [string]$menu) {
    Set-Base
    $env:AZAI_POSPIN = $pin
    $env:AZAI_TCLASS = $tclass
    if ($menu -eq 'wide') { $env:AZAI_MC_UPG='1'; $env:AZAI_MC_TOPK='3'; $env:AZAI_MC_EVERY='5' }
    else { $env:AZAI_MC_UPG='0'; $env:AZAI_MC_TOPK='0'; $env:AZAI_MC_EVERY='0' }
    $tag = "PS_$name"
    Write-Host ("=== arm {0} (pin={1} tclass={2} menu={3}) start {4}" -f $name,$pin,$tclass,$menu,(Get-Date -Format 'HH:mm:ss'))
    $tokens = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
    & "$Repo/code/run_logged.ps1" -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/ex_pkg/main.py $tokens"
    Report-Arm $tag
    Write-Host ("=== arm {0} done {1}" -f $name,(Get-Date -Format 'HH:mm:ss'))
}
Run-Arm 'C0_narrow'  'mc_quiet' '0.5' 'narrow'
Run-Arm 'P1_masked'  'masked'   '0.5' 'narrow'
Run-Arm 'P2_argmax'  'argmax'   '0.5' 'narrow'
Run-Arm 'P3_masksamp' 'masked'  '1.0' 'narrow'
Write-Host ("[ALL DONE] " + (Get-Date -Format 'HH:mm:ss'))
