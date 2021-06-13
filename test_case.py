import torch

matrix = torch.eye(5) * 10
vector = torch.ones(5)

for i in range(100):
    vector = matrix @ vector

output = torch.log10(vector)
print(output)
# >>> tensor([nan, nan, nan, nan, nan])


import torch
from stensor import stensor # STensor wrapper function

matrix = torch.eye(5) * 10
vector = torch.ones(5)
vector = stensor(vector)    # Conversion to STensor

for i in range(100):
    vector = matrix @ vector

output = torch.log10(vector)
print(output)
# >>> stensor([100., 100., 100., 100., 100.])


x = stensor(torch.arange(5))
print(x)
# >>> stensor([0., 1., 2., 3., 4.])
print(x.torch())
# >>> tensor([0., 1., 2., 3., 4.])


print(vector)
# >>> stensor([inf, inf, inf, inf, inf])
# inf and/or zero entries may be artifact of conversion
# to Tensor, use repr to view underlying data

print(repr(vector))
# >>> STensor(data=
# tensor([0.5715, 0.5715, 0.5715, 0.5715, 0.5715]), scale=
# tensor([333.]))


y = torch.randn(2, 3, 4)
y = stensor(y, stable_dims=[0, 2])
print(y.shape)
# >>> torch.Size([2, 3, 4])

print(y.scale.shape)
# >>> torch.Size([2, 1, 4])