import torch
import habana_frameworks.torch.core as htcore
import pdb
# Initialize W and dead tensors
device = 'hpu'  # Change to 'cuda' if testing on GPU
x = torch.randn(4096,4096,device=device)
H = x.matmul(x.t())
W = torch.randn(4096, 4096, device=device)  # 2D tensor
W = W.float()
dead = torch.diag(H) == 0
# Attempt to zero out columns in W based on 'dead'
H[dead,dead] = 1
W[:,dead]=0

print(H,W)