# _tmp_distill_pos.ps1 —— 预注册 `docs/prereg_20261013_distill_k48_cells.md` 第 2 步：**蒸馏**
#   学生 = 位置头（只改 pos_state）；教师 = K48 搜索的"格子访问分布"（pkl 里的 intent_counts）
#   做法：az_train.py --policy-only --freeze-backbone（策略损失 = 按 visit 边缘化到 per-head intent）
#        训练后**只把 pos_state 嫁接回原 ckpt**（显式保证单变量）
param(
    [string]$Repo = $PSScriptRoot,
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [string]$DataDir = 'training_history/vprior/data_k48_80',
    [string]$Out = 'training_history/vprior/posnet_dk48.pt',
    [int]$Epochs = 6,
    [string]$Lr = '3e-4',
    [switch]$PosSingle
)
Set-Location $Repo
$n = (Get-ChildItem $DataDir -Filter '*.pkl' -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "[distill] 数据 $DataDir : $n 个 pkl；epochs=$Epochs lr=$Lr pos-single=$PosSingle"
if ($n -lt 10) { Write-Host '[distill] 数据太少，先别训'; exit 1 }

$psFlag = @(); if ($PosSingle) { $psFlag = @('--pos-single') }
& "$Repo/code/run_logged.ps1" -Name distill_pos_k48 -CommandLine "$Py -u code/my_ai/az_intent/az_train.py --init checkpoints/posnet_A_k5_m32.pt --policy-dir $DataDir --checkpoint training_history/vprior/_dk48_tmp.pt --epochs $Epochs --lr $Lr --policy-only --freeze-backbone --batch-size 64 --t-class 0.5 --t-pos 1.0 $($psFlag -join ' ')"

# --- 只把 pos_state 嫁接回原 ckpt（显式单变量）---
@'
import sys, torch
src, dst, out = sys.argv[1], sys.argv[2], sys.argv[3]
a = torch.load(src, map_location="cpu", weights_only=False)
b = torch.load(dst, map_location="cpu", weights_only=False)
if "pos_state" not in b:
    raise SystemExit("训练产物里没有 pos_state")
grafted = 0
for k in list(b["pos_state"].keys()):
    if k in a["pos_state"] and a["pos_state"][k].shape == b["pos_state"][k].shape:
        a["pos_state"][k] = b["pos_state"][k].clone(); grafted += 1
torch.save(a, out)
print(f"[graft] pos_state 张量嫁接 {grafted} 个 -> {out}")
'@ | Set-Content -Encoding UTF8 _tmp_graft.py
& $Py -u _tmp_graft.py checkpoints/posnet_A_k5_m32.pt training_history/vprior/_dk48_tmp.pt $Out
Remove-Item _tmp_graft.py
Write-Host "[distill] 产物 $Out ；下一步用 verify 脚本核单变量"
