# To add a new cell, type '# %%'

# %%
import torch
import torch.nn as nn
from torch.nn import modules, functional as F
from DataGenerator import get_distance_matrices


class DMLoss(nn.Module):
    
    def __init__(self):
        super(DMLoss, self).__init__()
        self.reduction = 'sum'

    def forward(self, x, target):
        return F.mse_loss(get_distance_matrices(x), target,
         reduction=self.reduction)


# %%
from DataGenerator import DataGenerator

dg = DataGenerator((5, 2))
dgl = dg.get_dm_loader(N=3, batch=5)

dml = DMLoss()


# %%
conv1 = torch.nn.Conv2d(
in_channels=1,           # input shape 
out_channels=2,          # output shape
kernel_size=2
)

conv2 = torch.nn.Conv2d(
in_channels=2,           # input shape 
out_channels=2,          # output shape
kernel_size=2
)

r_conv1 = torch.nn.ConvTranspose2d(
in_channels=2,           # input shape 
out_channels=1,          # output shape
kernel_size=3
)

ll = torch.nn.Linear(in_features=5, out_features=2, bias=True) 


# %%
for step, target in enumerate(dgl):
    a1 = target 
    a2 = conv1(a1)
    a3 = conv2(a2)
    a4 = r_conv1(a3)
    a5 = ll(a4)
    a6 = a5.view((3, 5, 2))
    
    d1 = get_distance_matrices(a6)
    d2 = target
    
    break
    # print(dml(target[0], get_distance_matrices(target[0])))


# %%
from bad_grad_viz import view_gradient

