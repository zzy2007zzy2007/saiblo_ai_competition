"""Convert az_fixed self-play pkl data into value_warmup-format npz anchor data.

Experiment (isolate A/D "data source" vs B/C "training method"): az_train's
pkl self-play data stores board/stats per decision, plus a terminal value_target.
value_warmup's anchor targets (action_map + head_logits) must come from the SAME
model used as init.  So here we re-forward every stored state through the original
gen_0120 init model to produce matching anchor targets.  This lets us train a
value head with value_warmup's exact method + init, but on az_train's search
self-play states — isolating whether the *data source* is the problem.

Output npz per game (same schema as value_warmup collect): board/stats/action_map/
head_logits/player/value_target.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def convert_one(model, feat, pkl_path: Path, out_path: Path) -> int:
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)["samples"]
    boards, stats, a_maps, h_logits, players, v_targets = [], [], [], [], [], []
    with torch.no_grad():
        for s in samples:
            b = torch.from_numpy(s["board"]).unsqueeze(0).float()
            st = torch.from_numpy(s["stats"]).unsqueeze(0).float()
            out = model(b, st)
            boards.append(s["board"])
            stats.append(s["stats"])
            a_maps.append(out["action_map"].squeeze(0).numpy().astype(np.float16))
            h_logits.append(np.stack(
                [out[f"head{i + 1}_logits"].squeeze(0).numpy()
                 for i in range(model.num_heads)], axis=0).astype(np.float16))
            players.append(s["player"])
            v_targets.append(s.get("value_target", 0.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        board=np.stack(boards), stats=np.stack(stats),
        action_map=np.stack(a_maps), head_logits=np.stack(h_logits),
        player=np.asarray(players, dtype=np.int8),
        value_target=np.asarray(v_targets, dtype=np.float32),
    )
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="az_fixed pkl data dir (batch dirs)")
    parser.add_argument("--out-dir", required=True, help="output npz dir")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from my_ai.az_intent.train import build_model
    from SDK.utils.features import FeatureExtractor
    model = build_model("training_history/ga_ss_20260730_093908/gen_0120.pt")
    model.eval()
    feat = FeatureExtractor(max_actions=96)
    torch.set_num_threads(1)

    pkls = sorted(Path(args.data_dir).rglob("az_selfplay_seed*.pkl"))
    print(f"[convert] {len(pkls)} pkl files -> {args.out_dir}", flush=True)
    total = 0
    import concurrent.futures
    def work(p):
        out = Path(args.out_dir) / (p.stem + ".npz")
        return convert_one(model, feat, p, out)
    # single-process forward (model is shared); parallel would need per-worker model
    for p in pkls:
        n = convert_one(model, feat, p, Path(args.out_dir) / (p.stem + ".npz"))
        total += n
        if len(pkls) > 8:
            pass
    print(f"[convert] done: {total} samples -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
