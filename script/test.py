import torch

t = torch.Tensor([1.0, 2.0, 3.0])
z = torch.Tensor([2.0, 3.0, 4.0])

q = torch.div(t, z)
print(q)