"""自检：progress.py 的输出是否与用户给的参考格式逐字符一致。"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code/test_match"))
import progress  # noqa: E402

t = progress.GameTrace("SELFTEST", "820000000")
t.round(0, [], [], hp=[50, 50], coins=[50, 50])
t.round(2, [(11, 4, 16)], [], hp=[50, 50], coins=[38, 53])
t.round(3, [(13, 0, -1)], [], hp=[50, 50], coins=[54, 56])
t.round(9, [(12, 3, 2), (21, 7, 7)], [(23, 4, 4)], hp=[12, 49], coins=[60, 200])
t.terminal(1, [0, 2], 441)

mine = (REPO / "training_history/progress/SELFTEST_seed820000000.txt").read_text(
    encoding="utf-8").splitlines()
ref = Path(r"D:\2026智能体大赛_新2\training_history\az_bn_kgeo\data\batch2"
           r"\az_progress_seed820000000.txt").read_text(
    encoding="utf-8", errors="replace").splitlines()

print("=== 我们生成的 ===")
for ln in mine:
    print("  " + ln)
print("\n=== 与参考前 3 行逐字符对照 ===")
for a, b in zip(mine[:3], ref[:3]):
    wa = a.find("P1=") - a.find(" P0=") - 1
    wb = b.find("P1=") - b.find(" P0=") - 1
    print(f"  mine={a[:64]!r}")
    print(f"  ref ={b[:64]!r}")
    print(f"  P0段宽度 mine={wa} ref={wb}   {'✅一致' if wa == wb else '❌不一致'}")
print("\n参考的 terminal 行:", ref[-1])
print("我们的 terminal 行  :", mine[-1])
