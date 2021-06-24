import torch

print("cuda:0" if torch.cuda.is_available() else "cpu")