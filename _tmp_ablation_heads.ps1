# _tmp_ablation_heads.ps1 —— 头消融（本 goal 第 1 轮）
#   A_maponly : --lambda-class 0   （只训动作图）⇒ 检验"位置/动作图是否就是 +42pp 的来源"
#   B_clsonly : --lambda-map 0     （只训类头）
#   C_phold10 : --p-hold 0.1       （HOLD 降采样重蒸：让类头分布可用，为 joint 铺路）
#   每个训完即 graft（class/pos = 蒸馏网; value 保留原网）并跑 2 局冒烟（vs ex_pkg）。
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe'
)
Set-Location $Repo

function Train-And-Graft([string]$name, [string[]]$extra) {
    $out = "checkpoints/distill_$name.pt"
    $grafted = "checkpoints/grafted_$name.pt"
    $log = "training_history/distill_$name.log"
    Write-Host ("=== train {0} start {1}  extra={2}" -f $name, (Get-Date -Format 'HH:mm:ss'), ($extra -join ' '))
    & $Py -u code/distill/train.py distill_data/ --out $out --epochs 30 --lambda-soft 0 --log $log @extra 2>&1 |
        Select-String -Pattern 'epoch=30|epoch=1/|Saved|cls=' | Select-Object -Last 3 | ForEach-Object { "   " + $_.Line.Trim() }
    & $Py -u code/tools/graft_policy.py --base checkpoints/posnet_A_k5_m32.pt --policy $out --out $grafted 2>&1 |
        Select-String -Pattern '覆盖|已写出' | ForEach-Object { "   " + $_.Line.Trim() }
    Write-Host ("=== train {0} done {1}" -f $name, (Get-Date -Format 'HH:mm:ss'))
}

Train-And-Graft 'maponly' @('--lambda-class', '0')
Train-And-Graft 'clsonly' @('--lambda-map', '0')
Train-And-Graft 'phold10' @('--p-hold', '0.1')
Write-Host ("[ALL DONE] " + (Get-Date -Format 'HH:mm:ss'))
