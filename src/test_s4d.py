import torch
import sys
sys.path.append('.')

from s4d import S4Model

model = S4Model(d_input=1, d_output=8, d_model=256, n_layers=4)
x = torch.randn(4, 1024, 1)
out = model(x)
print(out.shape)
print("Success")
