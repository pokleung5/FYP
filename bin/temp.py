import sys
from importlib import reload

try:
    reload(sys.modules['utils'])
    reload(sys.modules['mds.cmds'])
    reload(sys.modules['mds.lmds'])
    reload(sys.modules['trainHelper'])
    reload(sys.modules['lossFunction'])
except:
    pass

# %%
import trainHelper
import utils
import numpy
from torch.utils.data import DataLoader
import torch
from torch import nn, Tensor
import lossFunction as lossF
from model.AutoEncoder import AutoEncoder
from model.DynParam import DynParam
from model.Linear import Linear
from model.VAE import VAE
import os.path
import glob

# %%

sample_space = (1000, 1)
ss, N, d = 6400, 10, 2

try:
    euclidean_data1 = utils.load_variable('data/euclidean_data1.pkl')
    euclidean_data2 = utils.load_variable('data/euclidean_data2.pkl')

    rand_data1 = utils.load_variable('data/rand_data.pkl')
    rand_data2 = utils.load_variable('data/rand_data.pkl')

    if euclidean_data.size() != (ss, 1, N, N):
        print("Updated data for requirement !")
        raise Exception("Previous data not match requirement !")

except:
    euclidean_data1 = utils.generate_euclidean_DM(
        N=N, d=d,
        sample_size=ss,
        sample_space=sample_space, isInt=True)

    euclidean_data2 = utils.generate_euclidean_DM(
        N=N, d=d,
        sample_size=ss,
        sample_space=sample_space, isInt=True)

    rand_data1 = utils.generate_rand_DM(
        N=N,
        sample_size=ss,
        sample_space=sample_space, isInt=True)

    rand_data2 = utils.generate_rand_DM(
        N=N,
        sample_size=ss,
        sample_space=sample_space, isInt=True)

    utils.dump_variable(euclidean_data1, 'data/euclidean_data1.pkl')
    utils.dump_variable(euclidean_data2, 'data/euclidean_data2.pkl')

    utils.dump_variable(rand_data1, 'data/rand_data1.pkl')
    utils.dump_variable(rand_data2, 'data/rand_data2.pkl')

# %%
data = torch.stack([
    euclidean_data1,
    euclidean_data2,
    # rand_data1,
    # rand_data2,
])

data = euclidean_data1[:3200]
data = data.view(3200, 1, N, N)

data = utils.minmax_norm(data, dmin=0)[0]
data = data.clone().detach().requires_grad_(True)

batch = 32
dlr = DataLoader(data, batch_size=batch, shuffle=True)

# %%


class ReconLoss(nn.Module):

    def __init__(self):

        super(ReconLoss, self).__init__()

    def forward(self, rs, target_dm):

        model_dm = rs
        model_dm = utils.minmax_norm(model_dm, dmin=0)[0]

        target_dm = torch.pow(target_dm, 2)
        target_dm = utils.minmax_norm(target_dm, dmin=0)[0]

        target_dm = utils.vectorize_distance_from_DM(target_dm)

        loss = model_dm - target_dm.view_as(model_dm)
        loss = torch.pow(loss, 2)

        loss = loss.view(loss.size()[0], 1, -1)
        return torch.sum(torch.sum(loss, dim=2) ** 2)


# %%


def preprocess(x):
    return x.clone().detach().requires_grad_(True)


def DM_distance(x):
    x = utils.vectorize_distance_from_DM(x)
    return preprocess(x)

# %%

for neuron in [96]:

    for i in range(2, 6):

        model_id = "Recon_AE_" + str(i) + "_" + str(neuron) + "_distance_LReLU"

        in_dim = int((N * N - N) / 2)
        out_dim = N * 2

        model = AutoEncoder([in_dim, *[neuron for j in range(i)], out_dim],
                            activation=nn.LeakyReLU,
                            final_activation=None)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)

        lossFun = ReconLoss()

        helper = trainHelper.TrainHelper(id=model_id,
                                         model=model,
                                         optimizer=optimizer,
                                         preprocess=DM_distance,
                                         lossFun=lossFun, lr_factor=0.1)

        for i in range(3):

            EPOCH = 300

            print("Training ", helper.id)

            try:
                helper.train(dlr, EPOCH, print_on_each=100)
                helper.backup()
            except:
                break

            print("Time used for the training: ",
                  helper.records['train_time'].sum(), "s", "| Last lr: ", helper.records['lr'].min())
