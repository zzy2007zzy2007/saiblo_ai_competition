import torch, os
seed_dir = '/root/autodl-tmp/AntWar/ppo_v9/outputs/20260612_164342/generations/gen_0005/seed_models'
for f in sorted(os.listdir(seed_dir), key=lambda x: int(x.replace('seed_','').replace('.pt',''))):
    d = torch.load(os.path.join(seed_dir, f), map_location='cpu', weights_only=True)
    print(f"{f}: id={d['individual_id']} elo={d['elo_rating']:.1f} parents={d['parent_ids']}")
