import torch

x= torch.tensor([1., 2., 3., 4., 5., 6.]).reshape(6, 1)
x1 = torch.ones(32, 6)
print(x.size())
print(x1.size())
y = torch.mm(x1, x)
print(x)
print(x1)
print(y)


