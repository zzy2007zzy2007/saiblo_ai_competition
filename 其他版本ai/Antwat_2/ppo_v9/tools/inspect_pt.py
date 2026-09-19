import torch, sys
data = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(f"type: {type(data).__name__}")
if isinstance(data, dict):
    print(f"keys: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())[:10]}")
            for k2, v2 in list(v.items())[:3]:
                print(f"    {k2}: {type(v2).__name__} {str(v2)[:80]}")
        else:
            print(f"  {k}: {type(v).__name__} = {str(v)[:120]}")
