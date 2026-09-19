import torch, os

seed_dir = "/root/autodl-tmp/AntWar/ppo_v9/outputs/20260612_164342/generations/gen_0004/seed_models"

weights = {}
for fname in sorted(os.listdir(seed_dir)):
    if not fname.endswith(".pt"):
        continue
    ckpt = torch.load(os.path.join(seed_dir, fname), map_location="cpu", weights_only=True)
    ind_id = ckpt["individual_id"]
    sd = ckpt["policy_state_dict"]
    flat = []
    for k, v in sorted(sd.items()):
        flat.append(v.float().flatten())
    w = torch.cat(flat)
    weights[fname] = (ind_id, w)

print("Gen4 seeds weight pairwise cosine similarity:")
header = "           "
for f1 in sorted(weights):
    header += f"  {f1}"
print(header)
for f1 in sorted(weights):
    id1, w1 = weights[f1]
    line = f"{f1:>11s}"
    for f2 in sorted(weights):
        _, w2 = weights[f2]
        cos = (w1 @ w2) / (w1.norm() * w2.norm())
        line += f"  {cos.item():.6f}"
    print(line)

# Gen4 vs Gen3 parents
gen3_dir = "/root/autodl-tmp/AntWar/ppo_v9/outputs/20260612_164342/generations/gen_0003/seed_models"
print()
print("Gen4 vs Gen3 parent weight distance:")
for gf in ["seed_1.pt", "seed_2.pt", "seed_3.pt"]:
    ckpt4 = torch.load(os.path.join(seed_dir, gf), map_location="cpu", weights_only=True)
    parents = ckpt4.get("parent_ids", [])
    id4 = ckpt4["individual_id"]
    sd4 = ckpt4["policy_state_dict"]
    w4 = torch.cat([v.float().flatten() for k, v in sorted(sd4.items())])
    
    for p_id in parents:
        for gf3 in sorted(os.listdir(gen3_dir)):
            if not gf3.endswith(".pt"):
                continue
            ckpt3 = torch.load(os.path.join(gen3_dir, gf3), map_location="cpu", weights_only=True)
            if ckpt3["individual_id"] == p_id:
                sd3 = ckpt3["policy_state_dict"]
                w3 = torch.cat([v.float().flatten() for k, v in sorted(sd3.items())])
                cos = (w4 @ w3) / (w4.norm() * w3.norm())
                l2 = (w4 - w3).norm().item()
                print(f"  {id4} ({gf}) vs parent {p_id}: cos={cos.item():.6f}, L2={l2:.2f}")

# Entropy check: distribution of absolute weight values
print()
print("Weight magnitude stats (first 3 seeds):")
for gf in ["seed_1.pt", "seed_2.pt", "seed_3.pt"]:
    _, w = weights[gf]
    print(f"  {gf}: mean={w.abs().mean().item():.6f}, std={w.std().item():.6f}, max={w.abs().max().item():.4f}")
