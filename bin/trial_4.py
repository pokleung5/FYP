#%% 
import torch
import torch.nn as nn
import scipy.linalg as linalg

from AutoEncoder import AutoEncoder
from DataGenerator import *
from gradient_check.bad_grad_viz import *  
from DataGenerator import *
from model import VAE, Linear, DynParam

# %% 
EPOCH = 10000 
dg = DataGenerator((6, 2), maxXY=1, minXY=0)
dgl = dg.get_dm_loader(N=20, batch=5)

ae =  DynParam.DynParam([6, 4], [4, 2])
optim = torch.optim.Adam(ae.parameters(), lr=1e-16)

# DataLoader(get_distance_matrices(
            # get_coordinates(dim=self.dim, batch=N, maxXY=self.max_XY, minXY=self.min_XY)),
            # batch_size=batch, shuffle=True)
            
dml = nn.MSELoss() #reduction='sum')
#%%
c = 0 
for epoch in range(EPOCH):
    for step, target in enumerate(dgl):
        
        t = target.view(
            (target.size()[0], 1, target.size()[1], target.size()[2])
            ) # torch.inverse(target)
        t = torch.tensor(t.data, requires_grad=True)
        
        rs = ae(t)
        rs = rs.view(
            (rs.size()[0], rs.size()[2], rs.size()[3])
        )
        # b, _, n, m = rs.size()
        
        dm = get_distance_matrices(rs) 
        loss = dml(dm, target)       

        if torch.sum(dm) == 0:
            raise Exception('Fucked !!')

        if loss != loss:
            c = c + 1
            print("Gradient disconnected")
            raise Exception('Gradient disconnected !!')
            continue

        print(rs[0], sep='\n\n')
        # view_gradient(loss)
        # exit(1)

        print(epoch, "\t", step, "\t| train loss: ", loss)

        optim.zero_grad()
        loss.backward()
        optim.step()
