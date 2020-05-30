from importlib import reload
reload(sys.modules['utils'])

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
sample_space = (100, 10)
ss, N, d = 1600, 10, 5

data, scaling = utils.generate_data(ss, N, d, sample_space=sample_space, isInt=True)

# %%
batch = 16
dlr = DataLoader(data, batch_size=batch, shuffle=True)
dmloss = nn.MSELoss(reduction='sum')

records = {}
# %%
model = Linear([N, 32, 32, 2], 
                final_activation=nn.Tanh)

model_name = type(model).__name__

lr = 1e-4
optim = torch.optim.Adam(model.parameters(), lr=lr)

# %%
EPOCH = 1000
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

plt.xlabel("Epoch")
plt.ylabel("Loss")

for key in records.keys():
    r = numpy.array(records[key])
    plt.plot(range(epoch), r[:epoch, 0], label=key)

plt.show()

# %%

test_ss = 100
test_sample_space = (100, 10)

test_data, scaling = utils.generate_data(
                            test_ss, N, 2000,
                            sample_space=test_sample_space,
                            isInt=True
                            )

#%%

cmds_loss = []
model_loss = []

with torch.no_grad():

    for d in test_data[:100]:

        d1 = numpy.array(d[0].data)
        
        cmds_rs = classicalMDS(d1, 2)
        cmds_rs = torch.tensor(cmds_rs)

        model_rs = model(d).view(1, N, 2)

        cmds_dm = utils.get_distance_matrix(cmds_rs)
        cmds_dm = (cmds_dm + 1e-30).sqrt() * (scaling)

        model_dm = utils.get_distance_matrix(model_rs)
        model_dm = (model_dm + 1e-30).sqrt() * (scaling)
        
        target_dm = (d + 1e-30).sqrt() * (scaling)

        cmds_loss.append(
            dmloss(cmds_dm, target_dm)
        )
        
        model_loss.append(
            dmloss(model_dm, target_dm)
        )
        
    print(torch.tensor(cmds_loss).mean())
    print(torch.tensor(model_loss).mean())
    


# %%


# %%
