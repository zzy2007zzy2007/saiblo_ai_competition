"""Quick GPU test from pytorch-gpu environment."""
import torch, sys
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = (x @ y).sum()
    print(f"GPU compute OK: {z.item():.4f}")
print("DONE")
