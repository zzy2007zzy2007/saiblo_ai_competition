"""三路径对照：rule_v4 vs rule_v4，同 seed 7-14，看「引擎 / 结算 API」各贡献了什么。

  P1  rule_v4_lightning_match.py --null                    C++ facade + apply_operation_list
  P2  同上 + --python-engine                               Python SDK + apply_operation_list
  P3  _tmp_rv4_self_resolve_turn.py                        Python SDK + resolve_turn
"""
import glob
import pathlib
import re


def _latest(tag: str) -> pathlib.Path:
    d = sorted(glob.glob(f"training_history/runs/*{tag}/output.log"))
    assert d, f"没找到 {tag} 的日志"
    return pathlib.Path(d[-1])


def parse_lightning(path: pathlib.Path) -> dict:
    """rule_v4_lightning_match.py 的输出：取 our_player=0 的局（(us,opp) 就是 (hp0,hp1)）。"""
    out = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\[game\] seed=(\d+) our_player=0 (\w+) us=(\d+) opp=(\d+) rounds=(\d+)", ln)
        if m:
            out[int(m.group(1))] = (int(m.group(5)), int(m.group(3)), int(m.group(4)), m.group(2))
    return out


def parse_p3(path: pathlib.Path) -> dict:
    out = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\[game\] seed=(\d+) rounds=(\d+) winner=(\S+) hp=(\d+),(\d+)", ln)
        if m:
            out[int(m.group(1))] = (int(m.group(2)), int(m.group(4)),
                                    int(m.group(5)), "w=%s" % m.group(3))
    return out


p1 = parse_lightning(_latest("ctrl_rv4_self_inprocess"))
p2 = parse_lightning(_latest("P2_rv4_self_python_engine"))
p3 = parse_p3(_latest("P3_rv4_self_resolve_turn"))

print("P1 = C++ facade + apply_operation_list   (rule_v4_lightning_match --null)")
print("P2 = Python SDK  + apply_operation_list   (同上 + --python-engine)")
print("P3 = Python SDK  + resolve_turn           (_tmp_rv4_self_resolve_turn.py)")
print()
print("%-5s | %-24s | %-24s | %-24s | %s" % ("seed", "P1 rounds,hp0,hp1", "P2", "P3", "谁和谁一致"))
for s in sorted(set(p1) | set(p2) | set(p3)):
    a, b, c = p1.get(s), p2.get(s), p3.get(s)
    ab = "P1=P2" if a and b and a[:3] == b[:3] else ""
    ac = "P1=P3" if a and c and a[:3] == c[:3] else ""
    bc = "P2=P3" if b and c and b[:3] == c[:3] else ""
    f = lambda x: "%4s %2s,%2s" % (x[0], x[1], x[2]) if x else "  --"
    print("%-5d | %-24s | %-24s | %-24s | %s" % (
        s, f(a), f(b), f(c), " ".join(x for x in (ab, ac, bc) if x) or "**三个都不同**"))
print()
# 胜负是否一致（用 hp 判侧，官方级联第①级）
print("胜负（按残血判侧，P1/P2 的 WIN/LOSS 文字与 hp 一致）：")
for s in sorted(set(p1) | set(p2) | set(p3)):
    def side(x):
        if not x:
            return "?"
        return "p0" if x[1] > x[2] else ("p1" if x[2] > x[1] else "draw")
    print("  seed %-3d  P1=%s  P2=%s  P3=%s" % (s, side(p1.get(s)), side(p2.get(s)), side(p3.get(s))))
