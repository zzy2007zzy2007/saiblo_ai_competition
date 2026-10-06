# _tmp_r4_clshead_pipeline.ps1 —— 闭环重训（预注册 `docs/prereg_20261007_classhead_distill.md`）的**后续三步**：
#   ① 等采集完成（`loop_r4_collect300`，300 局）
#   ② ingest（pinned 标签）+ **前置检查 P0**（单类 >95% 就停）
#   ③ 训练类头（30 epoch）→ **核子网哈希**（只有 class_state 变）→ 打印结果
#   ⚠️ 训练只是"造 ckpt"；**评估（64 局筛选 / 两段验收）另跑**，且按机械门需要独立验证。
param(
    [string]$Repo = 'D:/2026智能体大赛_新2',
    [string]$Py = 'D:/anaconda3/envs/pytorch-gpu/python.exe',
    [int]$WaitTimeoutMin = 180
)
Set-Location $Repo

# ① 等采集
$deadline = (Get-Date).AddMinutes($WaitTimeoutMin)
while ((Get-Date) -lt $deadline) {
    $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -match 'az_selfplay' })
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Seconds 60
}
Write-Host "[r4] 采集结束（或无进程），开始 ingest $(Get-Date -Format 'HH:mm:ss')"

# ② ingest
& $Py -u code/my_ai/az_intent/ingest_selfplay_classes.py --src training_history/vprior/data_loop_r4_300 `
    --dst training_history/vprior/data_loop_r4_300_exec --label pinned 2>&1 | Tee-Object -Variable ingestOut | Out-Null
$ingestOut | Select-String -Pattern '被选中的类分布|类\s+\d+:|未产出' | ForEach-Object { $_.Line }
$outTxt = $ingestOut -join "`n"
# P0 前置检查：解析最大占比
$m = [regex]::Matches($outTxt, '类\s+(\d+):\s+(\d+)\s+\(([\d.]+)%\)')
if ($m.Count -gt 0) {
    $top = $m | Sort-Object { [double]($_.Groups[3].Value) } -Descending | Select-Object -First 1
    $pct = [double]$top.Groups[3].Value
    Write-Host "[r4] P0 前置检查：最大类占比 = $pct%（类 $($top.Groups[1].Value)）"
    if ($pct -gt 95.0) { Write-Host "[r4] !! P0 不合格（>95%）⇒ 按预注册**停止**，不训练"; exit 2 }
}

# ③ 训练 + 子网哈希核对
& $Py -u code/my_ai/az_intent/train_class_outcome.py --ckpt checkpoints/posnet_A_k5_m32.pt `
    --data training_history/vprior/data_loop_r4_300_exec `
    --out training_history/vprior/posnet_R4_clshead.pt --epochs 30 --class-weight-power 0.5 2>&1 |
    Select-Object -Last 12
@'
import hashlib
from pathlib import Path
import torch

def flat(sd, prefix=""):
    out = {}
    for k, v in sd.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict): out.update(flat(v, kk + "."))
        else: out[kk] = v
    return out

def sub_hashes(p):
    sd = torch.load(p, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = flat(sd)
    res = {}
    for tag in ("class_state", "pos_state", "value_state"):
        ks = sorted(k for k in sd if k.startswith(tag))
        h = hashlib.md5()
        for k in ks:
            h.update(k.encode())
            v = sd[k]
            h.update(v.detach().cpu().numpy().tobytes() if hasattr(v, "detach") else repr(v).encode())
        res[tag] = h.hexdigest()[:12] if ks else "-"
    return res

a = sub_hashes("checkpoints/posnet_A_k5_m32.pt")
b = sub_hashes("training_history/vprior/posnet_R4_clshead.pt")
print("A(现用)      :", a)
print("R4(重训后)   :", b)
print("⇒ 只有 class_state 变？", ("是 ✓" if (a["pos_state"] == b["pos_state"] and a["value_state"] == b["value_state"]
      and a["class_state"] != b["class_state"]) else "**否（本预注册作废，需重写）**"))
'@ | Set-Content -Encoding UTF8 _tmp_hash_r4.py
& $Py -u _tmp_hash_r4.py
Remove-Item _tmp_hash_r4.py
Write-Host "[r4] 流水线完成 $(Get-Date -Format 'HH:mm:ss')（评估另跑）"
