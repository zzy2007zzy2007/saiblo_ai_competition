# _tmp_ablation_eval.ps1 —— 头消融 + 类维度交给学习的评估（预注册 docs/prereg_20261019_goal2_head_ablation.md）
#   两个测量语境：
#     vsExample  : ai0=候选 bridge, ai1=ex_pkg（官方 example）⇒ 与锚点 S0=0.3828 / 完整蒸馏=0.8016 比
#     vsBaseline : ai0=候选 bridge, ai1=base_pkg（冻结基线）⇒ **本 goal 的判据**（两段各 ≥80% 才算）
#   每臂先 2 局冒烟（我方建塔 ≤0.5/局 ⇒ 判退化、跳过 128 局）。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo

function Set-Base {
    param([string]$Ckpt, [string]$Mode, [string]$Pin, [string]$Tclass)
    $env:AZAI_CKPT = $Ckpt
    $env:AZAI_ITERS='256'; $env:AZAI_DEPTH='4'; $env:AZAI_K='24'; $env:AZAI_SAMPLE_MULT='15'
    $env:AZAI_MODE=$Mode; $env:AZAI_SKIP1='1'; $env:AZAI_TPOS='1.0'; $env:AZAI_TEMP='1e-6'
    $env:AZAI_POSPRIOR='off'; $env:AZAI_VERIFY='1'; $env:AZAI_TRACE='1'; $env:AZAI_TIMING_EVERY='25'
    $env:AZAI_POSPIN=$Pin; $env:AZAI_TCLASS=$Tclass
    $env:AZAI_MC_HORIZON='256'; $env:AZAI_MC_EVERY='5'; $env:AZAI_MC_ROOTONLY='1'; $env:AZAI_MC_MARGIN='0'
    $env:AZAI_MC_SCORE=''; $env:AZAI_MC_SELFSPEND='0'; $env:AZAI_MC_SELFSPEND_MIX='1.0'
    $env:AZAI_MC_BIDIR='0'; $env:AZAI_MC_NO_LIGHT='0'; $env:AZAI_MC_CELLS='0'; $env:AZAI_MC_TOP2='0'
    $env:AZAI_MC_M4='0'; $env:AZAI_MC_NODOWN='0'; $env:AZAI_MC_UPG='1'; $env:AZAI_MC_TOPK='3'
    $env:AZAI_LIGHT_MIN_COINS='0'; $env:AZAI_BUILD_MIN_COINS='0'
    $env:AZAI_FORCE_UPG='0'; $env:AZAI_NO_DOWNGRADE='0'; $env:AZAI_PROBE_UPG='0'
    $env:AZAI_SLOW_DECISION_SEC='60'
    $env:AZAI_BAN_CLASSES='17'; $env:AZAI_BAN_OPS='21'; $env:AI_BAN_LIGHTNING='1'
}

function Our-Builds([string]$tag) {
    $line = & $Py -u code/test_match/phase0_report.py $tag 2>&1 | Select-Object -Skip 1 -First 1
    if (-not $line) { return -1.0 }
    $m = [regex]::Match($line, '^\S+\s+\d+\s+\S+\s+([\d.]+)/')
    if ($m.Success) { return [double]$m.Groups[1].Value }
    return -1.0
}

function Run-Arm([string]$name, [string]$ckpt, [string]$mode, [string]$pin, [string]$tclass, [string]$opp) {
    Set-Base -Ckpt $ckpt -Mode $mode -Pin $pin -Tclass $tclass
    $tag = "AB_${name}_vs_$opp"
    $oppPath = if ($opp -eq 'example') { 'code/test_match/ex_pkg/main.py' } else { 'code/test_match/base_pkg/main.py' }
    Write-Host ("=== {0} (ckpt={1} mode={2} pin={3} tclass={4} opp={5}) smoke {6}" -f $name,$ckpt,$mode,$pin,$tclass,$opp,(Get-Date -Format 'HH:mm:ss'))
    & $Py -u _tmp_ladder.py --tag="SMK_$tag" --jobs=2 --ai0=code/test_match/az_bridge_ai.py --ai1=$oppPath 7 7r 2>&1 | Out-Null
    $b = Our-Builds "SMK_$tag"
    Write-Host ("  [smoke] 我方建塔/局 = {0}" -f $b)
    if ($b -le 0.5) { Write-Host ("  [SKIP] {0} 冒烟即退化 ⇒ 不跑 128 局" -f $name); return }
    $tokens = ((7..70 | ForEach-Object { "$_", "${_}r" }) -join ' ')
    & "$Repo/code/run_logged.ps1" -Name $tag -CommandLine "$Py -u _tmp_ladder.py --tag=$tag --jobs=8 --ai0=code/test_match/az_bridge_ai.py --ai1=$oppPath $tokens"
    $out = & $Py -u code/test_match/analyze_paired.py --tag=$tag --seeds=7-70 2>&1 | Out-String
    [System.IO.File]::WriteAllText("$Repo/match_results/ladder_logs/$tag/_analysis.txt", $out, (New-Object System.Text.UTF8Encoding($false)))
    $out -split "`n" | Select-String -Pattern '配对胜率|完整配对数|有效性门槛' | ForEach-Object { "  " + $_.Line.Trim() }
    Write-Host ("  [mech] " + (& $Py -u code/test_match/phase0_report.py $tag 2>&1 | Select-Object -Skip 1 -First 1))
    Write-Host ("=== {0} done {1}" -f $name,(Get-Date -Format 'HH:mm:ss'))
}

# ── 消融（理解"哪个头"）：vs example，与锚点 0.3828 / 0.8016 比 ──
Run-Arm 'A_maponly'  'checkpoints/grafted_maponly.pt' 'pos-only' 'mc_quiet' '0.5' 'example'
Run-Arm 'B_clsonly'  'checkpoints/grafted_clsonly.pt' 'pos-only' 'mc_quiet' '0.5' 'example'
Run-Arm 'C_phold10'  'checkpoints/grafted_phold10.pt' 'pos-only' 'mc_quiet' '0.5' 'example'
# ── 类维度交给学习（goal 判据：vs 基线）──
Run-Arm 'D_joint_full'   'checkpoints/grafted_sdk_A.pt' 'joint' 'mc_quiet' '0.5' 'baseline'
Run-Arm 'D2_joint_full_t1' 'checkpoints/grafted_sdk_A.pt' 'joint' 'mc_quiet' '1.0' 'baseline'
Run-Arm 'E_joint_phold10' 'checkpoints/grafted_phold10.pt' 'joint' 'mc_quiet' '1.0' 'baseline'
Run-Arm 'F_masked_phold10' 'checkpoints/grafted_phold10.pt' 'pos-only' 'masked' '0.5' 'baseline'
Write-Host ("[ALL DONE] " + (Get-Date -Format 'HH:mm:ss'))
