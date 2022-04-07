import torch 

a = torch.randn(16, 512)
print(a.shape)
b = torch.randn(16, 32, 2048)
a = a.unsqueeze(2)
print(a.shape)
a = a.expand(-1,-1,2048)

print(a.shape)