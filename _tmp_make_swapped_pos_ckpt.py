"""造一个"换过位置网"的三网 ckpt：pos_state 换成蒸馏版网络的 state_dict。

动机（用户 2026-09-18）：现在的动作网是遗传/BC 训练出来的，而遗传训练里"动作类选得好"
提升就大、位置选烂了影响也不大 ⇒ 位置图可能被训坏了。而 `distill_data*/model.pt` 是
**直接拟合 SDK 启发式评分函数**（ActionReport/score 蒸馏，见 docs/distill_example_ai.md）
的产物，每个合法操作都有监督 ⇒ 它的位置图可能反而相对好。

用法: python _tmp_make_swapped_pos_ckpt.py <蒸馏ckpt> <输出ckpt>
只改 pos_state 一个键，其余顶层元数据逐字沿用源三网 ckpt。
"""
import sys
import torch

SRC = "training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt"


def main() -> None:
    dist_path, out_path = sys.argv[1], sys.argv[2]
    src = torch.load(SRC, map_location="cpu", weights_only=False)
    dist = torch.load(dist_path, map_location="cpu", weights_only=False)
    st = dist.get("model_state", dist)

    ref = src["pos_state"]
    assert set(st) == set(ref), "键名不一致，先核对架构"
    bad = [k for k in ref if tuple(ref[k].shape) != tuple(st[k].shape)]
    assert not bad, f"形状不一致: {bad[:5]}"

    out = dict(src)
    out["pos_state"] = {k: v.clone() for k, v in st.items()}
    out["pos_state_source"] = dist_path
    torch.save(out, out_path)

    n_diff = sum(1 for k in ref if not torch.equal(ref[k], st[k]))
    print("源三网: %s" % SRC)
    print("位置网换成: %s" % dist_path)
    print("写出: %s" % out_path)
    print("  张量 %d 个全部对齐；与原来的 pos_state 相比 %d/%d 个张量不同"
          % (len(st), n_diff, len(ref)))
    print("  顶层键: %s" % list(out.keys()))


if __name__ == "__main__":
    main()
