from importlib import reload
reload(sys.modules['mds.lmds'])
from mds.lmds import landmarkMDS

# %%
from mds.fastmap import fastmap
from mds.cmds import classicalMDS
from mds.lmds import landmarkMDS

from model.VAE import VAE
from model.Linear import Linear
from model.DynParam import DynParam
from model.AutoEncoder import AutoEncoder

import utils

from matplotlib import pyplot as plt

import numpy
from torch import nn
import torch
from torch.utils.data import DataLoader

torch.set_default_tensor_type('torch.DoubleTensor')

# %%
panel = (100, 10)
sample, N, d = 160, 50, 5

coords = utils.get_rand_coords((sample, N, d), *panel, isInt=True)
# coords = utils.minmax_norm(coords, *panel)

dms = utils.get_distance_matrix(coords)
dms = utils.minmax_norm(dms, dmax=panel[0] ** 2, dmin=0)

data = dms.view(sample, 1, N, N) # coords
data = torch.tensor(data.data, requires_grad=True)

# %%
batch = 16
dlr = DataLoader(data, batch_size=batch, shuffle=True)

records = {}
# %%
model = Linear([N, 32, 32, 2], 
                final_activation=nn.Tanh)

model_name = type(model).__name__

dmloss = nn.MSELoss()

lr = 1e-4
optim = torch.optim.Adam(model.parameters(), lr=lr)

# %%
EPOCH = 100
records[model_name] = []

for epoch in range(EPOCH):

    loss_values = []

    for step, t in enumerate(dlr):
        optim.zero_grad()

        rs = model(t)
        rs = rs.view(batch, N, 2)

        dm = utils.get_distance_matrix(rs)
 
        loss = dmloss(dm, t.data)

        if loss != loss or torch.sum(rs) == 0:
            raise Exception("Gradient Disconnected !")
        
        loss.backward()
        optim.step()

        loss_values.append(loss.data)
        
        # print(step, "\t| Total loss:", loss)

    loss_values = torch.tensor(loss_values)

    records[model_name].append([
        loss_values.mean(),
        loss_values.max(),
        loss_values.min()
    ])

    print(epoch, "\t| Total loss:", records[model_name][epoch])

# %%
linear_rc = numpy.array(records['Linear'])
VAE_rc = numpy.array(records['VAE'])
AE_rc = numpy.array(records['AutoEncoder'])
# Dynet_rc = numpy.array(records['DynParam'])

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.plot(range(epoch), linear_rc[:epoch, 0], label="Linear", color='R')
plt.plot(range(epoch), VAE_rc[:epoch, 0], label="VAE", color='G')
plt.plot(range(epoch), AE_rc[:epoch, 0], label="AE", color='B')
# plt.plot(range(epoch), Dynet_rc[:epoch, 0], label="DynParam", color='y')

plt.show()


# %%

test_sampe = 100

test_coords = utils.get_rand_coords((test_sampe, N, 5), maxXY=1, minXY=0)
test_dms = utils.get_distance_matrix(test_coords)
test_dms = test_dms.view(test_sampe, 1, N, N)  # zip(coords, dms)
test_data = torch.tensor(test_dms.data, requires_grad=True)

#%%
with torch.no_grad():
    rs = model(test_data).view(test_sampe, N, 2)
    dm = utils.get_distance_matrix(rs)

    loss = dmloss(dm, test_data.data)
    print(loss)

# %%

result = []
for d in test_data:

    d1 = numpy.array(d[0].data)
    
    rs = fastmap(d1, 2)
    rs = torch.tensor(rs)

    dm = utils.get_distance_matrix(rs)
    result.append(
        float(torch.sum((dm - d[0])** 2))
    )
    
numpy.mean(result)


# %%
