import sys
from importlib import reload
reload(sys.modules['utils'])
reload(sys.modules['trainHelper'])
reload(sys.modules['lossFunction'])

#%%
import trainHelper
import utils
from mds.lmds import landmarkMDS
from mds.cmds import classicalMDS
from mds.fastmap import fastmap
import numpy
from torch.utils.data import DataLoader
import torch
from torch import nn
import lossFunction as lossF
from model.AutoEncoder import AutoEncoder
from model.DynParam import DynParam
from model.Linear import Linear
from model.VAE import VAE

torch.set_default_tensor_type('torch.DoubleTensor')

# %%
sample_space = (10000, 10)
ss, N, d = 800, 50, 2

# %%
euclidean_data = utils.generate_euclidean_DM(
    N=N, d=d,
    sample_size=ss,
    sample_space=sample_space, isInt=True)

rand_data = utils.generate_rand_DM(
    N=N,
    sample_size=ss,
    sample_space=sample_space, isInt=True)

utils.dump_variable(euclidean_data, 'data/euclidean_data.pkl')
utils.dump_variable(rand_data, 'data/rand_data.pkl')

#%%
euclidean_data = utils.load_variable('data/euclidean_data.pkl')
rand_data = utils.load_variable('data/rand_data.pkl')

#%%

data = torch.stack([
            euclidean_data.view(ss, 1, N*N),
            rand_data.view(ss, 1, N*N)
        ]).view(ss * 2, 1, N * N)

data = torch.tensor(data.data, requires_grad=True)

batch = 16
dlr = DataLoader(data, batch_size=batch, shuffle=True)

# %% 
model = Linear([N * N, 32, 32, N * 2], final_activation=None)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

lossFun = lossF.CoordsToDMLoss(
    target_dim=(N, 2),
    lossFun=nn.MSELoss(reduction='sum'))

helper = trainHelper.TrainHelper(
    id="Linear_with_two_32_layers",
    model=model,
    optimizer=optimizer,
    lossFun=lossFun,
    lr_factor=0.1)

# %%

_, time_used = utils.time_measure(
    helper.train, [dlr, 500, 1])

helper.plot(['loss_mean', 'loss_max', 'loss_min'])
helper.plot(['train_time'], value_label='train_time')
helper.plot(['lr'], value_label='lr')

print("Time used for the training: ", time_used, "s")

utils.dump_variable(helper, helper.id + '.model')

#%%

test_data = utils.generate_rand_DM(
    N=N,
    sample_size=200,
    sample_space=sample_space, isInt=True)

cmds_loss = []
fastmap_loss = []
model_loss = []

with torch.no_grad():

    for d in test_data:

        d1 = numpy.array(d[0].data)
        d2 = d.view(1, 1, N * N)

        cmds_rs = classicalMDS(d1, 2)
        cmds_rs = torch.tensor(cmds_rs)
        cmds_dm, _ = utils.get_distance_matrix(cmds_rs)
        cmds_loss.append(torch.sum((cmds_dm - d)** 2))

        fastmap_rs = fastmap(d1, 2)
        fastmap_rs = torch.tensor(fastmap_rs)
        fastmap_dm, _ = utils.get_distance_matrix(fastmap_rs)
        fastmap_loss.append(torch.sum((fastmap_dm - d)** 2))

        model_rs = model(d2).view(1, N, 2)
        model_dm, _ = utils.get_distance_matrix(model_rs)
        model_loss.append(torch.sum((model_dm - d) ** 2))
        
    print("cmds_loss: \t", torch.tensor(cmds_loss).mean())
    print("fastmap_loss: \t", torch.tensor(fastmap_loss).mean())
    print("model_loss: \t", torch.tensor(model_loss).mean())    

# %%


# 