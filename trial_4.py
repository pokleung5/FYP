# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import torch
import torch.nn as nn
from AutoEncoder import AutoEncoder
from DataGenerator import DataGenerator
from bad_grad_viz import *  


# %%
from importlib import reload
import DataGenerator as dg
from bad_grad_viz import *  

a = dg.get_coordinates((5, 2))
b = dg.get_distance_matrices(a)

# view_gradient(b)
# %% 

EPOCH = 500
dg = DataGenerator((5, 2))
dgl = dg.get_dm_loader(N=100, batch=5)

ae = AutoEncoder(5, 2)
optim = torch.optim.Adam(ae.parameters(), lr=1e-5)

dml = nn.MSELoss(reduction='sum')

for epoch in range(EPOCH):
    for step, target in enumerate(dgl):
        rs = ae(target)

        loss = dml(rs, target)
        # view_gradient(loss)
        # break
        
        print(epoch, "\t\t", step, "\t\t| train loss: ", loss)

        optim.zero_grad()
        # print(target)
        loss.backward()
        optim.step()
    # break
# %%
