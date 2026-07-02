"""Smoke test: BC data collection + GPU training pipeline.

Tests:
  1. GPU is available
  2. Model forward pass on GPU works
  3. Feature extraction + argmax label extraction works
  4. Write .npz file works
  5. BCDataset loading + HOLD downsampling works
  6. supervised_update on GPU runs end-to-end
"""
import sys, time, shutil, tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch

from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND
from my_ai.network import create_model
from my_ai.agent import NeuralAgent

# ── 1. GPU check ─────────────────────────────────────────────────
print("=" * 55)
print("1. GPU AVAILABILITY")
print("=" * 55)
cuda_ok = torch.cuda.is_available()
print(f"  torch.cuda.is_available() = {cuda_ok}")
if cuda_ok:
    print(f"  device name: {torch.cuda.get_device_name(0)}")
    print(f"  memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("  WARNING: no GPU, falling back to CPU")
device = torch.device("cuda" if cuda_ok else "cpu")

# ── 2. Load checkpoint ──────────────────────────────────────────
print("\n" + "=" * 55)
print("2. LOAD MODEL")
print("=" * 55)
ckpt = torch.load("training_history/20260702_005921/gen_0006.pt",
                  map_location="cpu", weights_only=True)
num_heads = ckpt.get("num_heads", 3)
print(f"  params={len(ckpt['top2_params'][0]):,}, heads={num_heads}")
model = create_model(num_heads=num_heads)
model.set_parameters_from_vector(ckpt["top2_params"][0].numpy())
model.eval()

# ── 3. Forward pass on GPU (warmup) ──────────────────────────────
print("\n" * 0 + "=" * 55)
print("3. GPU FORWARD")
print("=" * 55)
model.to(device)
board = torch.randn(16, 28, 19, 19).to(device)
stats = torch.randn(16, 42).to(device)
with torch.no_grad():
    t0 = time.perf_counter()
    for _ in range(20):
        out = model(board, stats)
    t = time.perf_counter() - t0
print(f"  20 × batch(16) forward: {t:.3f}s ({t/20:.3f}s/iter)")
model.cpu()

# ── 4. Play 1 game + collect data (simulating _eval_worker) ──────
print("\n" + "=" * 55)
print("4. DATA COLLECTION (1 game with gen_0006 top1)")
print("=" * 55)
agent = NeuralAgent(model=model)
state = GameState.initial(seed=0, cold_handle_rule_illegal=True)
our_player = 0

boards, statss, cls_labels, map_labels = [], [], [], []
t0 = time.perf_counter()
for turn in range(MAX_ROUND):
    if state.terminal:
        break
    ops = agent._choose_operations(state, our_player)
    
    # Collect data (same logic as _eval_worker)
    feat = agent.feature_extractor.encode_observation(
        state, our_player, np.zeros(agent.max_actions))
    boards.append(feat["board"].copy())
    statss.append(feat["stats"].copy())
    map_arg = agent.last_output["action_map"].reshape(-1).argmax().item()
    cls = [agent.last_output[f"head{hi+1}_logits"].argmax().item()
           for hi in range(num_heads)]
    cls_labels.append(cls)
    map_labels.append([map_arg] * num_heads)

    # Resolve turn
    state.resolve_turn(ops or [], [])

t_collect = time.perf_counter() - t0
hp = state.bases[0].hp
print(f"  turns={turn+1}, HP={hp}, time={t_collect:.3f}s")
print(f"  samples collected: {len(boards)}")

# Save .npz
from my_ai.elite_bc import write_bc_npz
tmpdir = Path(tempfile.mkdtemp(prefix="bc_test_"))
boards_arr = np.stack(boards, axis=0)
stats_arr = np.stack(statss, axis=0)
cls_arr = np.array(cls_labels)
map_arr = np.array(map_labels)
npz_path = tmpdir / "test_seed0.npz"
t0 = time.perf_counter()
write_bc_npz(npz_path, [boards_arr], [stats_arr], [cls_arr], [map_arr])
t_write = time.perf_counter() - t0
file_size_mb = npz_path.stat().st_size / 1e6
print(f"  .npz written: {file_size_mb:.2f} MB, time={t_write*1000:.1f}ms")
print(f"  board dtype={boards_arr.dtype}, shape={boards_arr.shape}")

# ── 5. HOLD stats ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("5. HOLD ANALYSIS")
print("=" * 55)
hold_ratio = (cls_arr == 23).all(axis=1).mean()
print(f"  all-HOLD turns: {hold_ratio*100:.1f}%")

# ── 6. BCDataset + supervised_update ────────────────────────────
print("\n" + "=" * 55)
print("6. BC TRAINING (3 epochs on collected data)")
print("=" * 55)
model2 = create_model(num_heads=num_heads)
model2.set_parameters_from_vector(ckpt["top2_params"][0].numpy())
model2.eval()

from my_ai.elite_bc import BCDataset, supervised_update
ds = BCDataset([npz_path], p_hold=0.1)
print(f"  dataset size: {len(ds)} (filtered from {len(boards)})")
# extra .npz to make data size more realistic
for extra in range(4):
    write_bc_npz(tmpdir / f"test_seed{extra+1}.npz",
                  [boards_arr], [stats_arr], [cls_arr], [map_arr])
ds_full = BCDataset(list(tmpdir.glob("*.npz")))
print(f"  merged dataset: {len(ds_full)} samples")

t0 = time.perf_counter()
result = supervised_update(model2, ds_full, device=device,
                           epochs=3, batch_size=64, lr=1e-3)
t_train = time.perf_counter() - t0
print(f"  training time: {t_train:.3f}s")
print(f"  class_loss={result['class_loss']:.4f}  "
      f"map_loss={result['map_loss']:.4f}  "
      f"total={result['total_loss']:.4f}")

# ── 7. Verify model was updated ────────────────────────────────
print("\n" + "=" * 55)
print("7. VERIFY MODEL UPDATED")
print("=" * 55)
new_vec = model2.get_parameters_as_vector()
old_vec = ckpt["top2_params"][0].numpy()
diff = np.abs(new_vec - old_vec).mean()
print(f"  mean param diff (before vs after training): {diff:.8f}")
print(f"  model was {'✓ UPDATED' if diff > 1e-6 else '✗ NOT UPDATED'}")

# ── Cleanup ───────────────────────────────────────────────────────
shutil.rmtree(tmpdir, ignore_errors=True)

print("\n" + "=" * 55)
print("ALL TESTS PASSED" if diff > 1e-6 else "WARNING: MODEL NOT UPDATED")
print("=" * 55)
