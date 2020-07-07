# %%
import sys
from importlib import reload

# reload(sys.modules['utils']);
# reload(sys.modules['lossFunction']);
# reload(sys.modules['model.RNN']);
# reload(sys.modules['model.EignModel']);

# %%
from preprocess import *
from trainUtils import *
from lossFunction import CustomLoss, CoordsToDMLoss, ReconLoss, MultiLoss, sammon_loss

from torch.utils.data import DataLoader

from model.Linear import Linear, ReuseLinear, StepLinear
from model.EignModel import EignModel

import model.AutoEncoder as ae
import model.RNN as rnn

from shutil import copyfile
import os

# %%
data = dataSource.load_data(shape=(ss, N, d))[0]
data = utils.minmax_norm(data, dmin=0)[0]

dlr = DataLoader(data, batch_size=batch, shuffle=True)

init_lr = 1e-3

train_id = '_'.join(['Expand', 'Linear', 'MF', 'MSE'])

coordLoss = CoordsToDMLoss(N, lossFun=nn.MSELoss(reduction='sum'))
reconLoss = ReconLoss(lossFun=nn.MSELoss(reduction='sum'))
mseLoss = CustomLoss(lossFun=nn.MSELoss(reduction='sum'))

lossFun = MultiLoss(lossFunList=[coordLoss, coordLoss])

preprocess = PrepMatrix(N, flatten=True)
in_dim, out_dim = preprocess.get_inShape(), N * 2

def get_model(neuron, i, in_dim, out_dim):

    n = int(neuron / i)

    mid1 = int((n + in_dim) / 2)
    mid2 = int((n + out_dim) / 2)

    mid = [int(n)] * (i - 2)

    return StepLinear(
        dim_list=[
            [in_dim, mid1, *mid, mid2, in_dim],
            [in_dim, mid1, *mid, mid2, out_dim]
        ], activation=nn.LeakyReLU)

# %%
bkup_dest = 'backup/%s.py' % (train_id)

if os.path.exists(bkup_dest):
    raise Exception("Destination file exists!")

copyfile('train.py', bkup_dest)

#%%

param = [(N + in_dim, L)  for L in range(2, 6) for N in range(8, 73, 16)]

for neuron, i in param:

    model_id = train_id + "_" + str(i) + "_" + str(neuron)

    model = get_model(neuron, i, in_dim, out_dim)

    helper = trainHelper.TrainHelper(
        id=model_id, model=model, lossFun=lossFun,
        optimizer=torch.optim.Adam(model.parameters(), lr=init_lr),
        preprocess=preprocess)

    train(helper, dlr, 'log/%s.log' % (train_id))

# %%
