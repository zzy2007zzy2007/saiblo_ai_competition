# _tmp_rand_seg.ps1 —— 跑**随机段**（64 个冻结 seed = 128 局），**原样复用**某个候选 run 的配置
#
#   为什么这么写：判据要求"两段唯一差异 = seed"。所以这里从候选自己的 run 归档里
#   读回 `env.txt`（全部 AZAI_*）与 `cmd.txt`（--ai0/--ai1/--jobs），只换 tag 与 token 列表。
#
#   用法：
#     powershell -File _tmp_rand_seg.ps1 -SrcTag AB_VSEARCH_dec_vs_baseline -OutTag VSEARCH_rand_diag
#
#   ⚠️ 口径提醒：**本 goal 的规则是"固定段不达阈值就不跑随机段"**。本脚本用于**补充诊断**
#      （明确标注：该候选固定段 0.7188 < 0.80 ⇒ 随机段读数**不构成成功声明**，只用于报告完整性）。
param(
    [Parameter(Mandatory=$true)][string]$SrcTag,
    [Parameter(Mandatory=$true)][string]$OutTag,
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$SeedsFile = 'docs/random_segment_seeds.txt',
    [int]$Jobs = 8
)
if (-not $Repo) { $Repo = Split-Path -Parent $PSCommandPath }
Set-Location $Repo

$run = Get-ChildItem training_history/runs -Directory | Where-Object Name -like "*_$SrcTag" | Sort-Object Name -Descending | Select-Object -First 1
if (-not $run) { Write-Host "[rand] 找不到 $SrcTag 的 run 归档 ⇒ 停"; exit 2 }
$cmd = Get-Content -Raw -Encoding UTF8 (Join-Path $run.FullName 'cmd.txt')
$ai0 = ([regex]::Match($cmd, '--ai0=(\S+)')).Groups[1].Value
$ai1 = ([regex]::Match($cmd, '--ai1=(\S+)')).Groups[1].Value
Write-Host "[rand] 复用 $($run.Name) 的配置：ai0=$ai0  ai1=$ai1"
foreach ($line in Get-Content -Encoding UTF8 (Join-Path $run.FullName 'env.txt')) {
    if ($line -match '^([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { Set-Item -Path ("Env:" + $Matches[1]) -Value $Matches[2] }
}
$seeds = ((Get-Content -Encoding UTF8 $SeedsFile | Where-Object { $_ -and -not $_.StartsWith('#') } | Select-Object -Last 1) -split ',') |
    ForEach-Object { $_.Trim() } | Where-Object { $_ }
$tokens = ($seeds | ForEach-Object { $_, "${_}r" }) -join ' '
Write-Host ("[rand] {0} 个 seed ⇒ {1} 局；开始 {2}" -f $seeds.Count, (2 * $seeds.Count), (Get-Date -Format 'HH:mm:ss'))
& "$Repo/code/run_logged.ps1" -Name $OutTag -CommandLine "$Py -u _tmp_ladder.py --tag=$OutTag --jobs=$Jobs --ai0=$ai0 --ai1=$ai1 $tokens"
Write-Host ("[rand] 完成 {0}；分析：{1} -u code/test_match/analyze_paired.py --tag $OutTag --seeds=<随机段列表>" -f (Get-Date -Format 'HH:mm:ss'), $Py)
