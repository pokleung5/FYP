
# %%
from DataGenerator import *
from bad_grad_viz import *
import torch


def dm(xx: Tensor):
    return get_distance_matrices(xx)


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

dg = DataGenerator((5, 2))

dgl = dg.get_dm_loader(N=3, batch=5)

for step, target in enumerate(dgl):
    a1 = target 
    a2 = conv1(a1)
    a3 = conv2(a2)
    a4 = r_conv1(a3)
    a5 = ll(a4)
    # print(a1, a2, a3, a4, a5, sep='\n\n')
    a6 = dm(a5.view(3, 5, 2))
    print(sum(a1), a6, sep='\n\n')

    ls = torch.sum((a6 - sum(a1)) ** 2)

    print(ls)
    break

view_gradient(torch.sum(a6))

# %% Just test the loss function 

dg2 = dg.get_pair_loader(N=3, batch=5)

for step, target in enumerate(dg2):
    b1 = dm(target)
    break

view_gradient(torch.sum(b1))



# %%
