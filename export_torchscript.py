import torch
from model import CNN

model = torch.load(r"D:\TA Protonx\Tùy biến và deploy mô hình riêng\triton-inference-server\model.pth").eval()

traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
torch.jit.save(traced_model, "traced_model.pt")