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
from lossFunction import CoordsToDMLoss, ReconLoss, MultiLoss, sammon_loss

from torch.utils.data import DataLoader

from model.Linear import Linear
from model.EignModel import EignModel

import model.AutoEncoder as ae
import model.RNN as rnn

from shutil import copyfile

# %%
data = dataSource.load_data(shape=(ss, N, d))[1]
data = utils.minmax_norm(data, dmin=0)[0]

dlr = DataLoader(data, batch_size=batch, shuffle=True)

init_lr = 1e-3

train_id = '_'.join(['Recon', 'AE', 'dNoise', '2MSE'])

coordLoss = CoordsToDMLoss(N, 2, lossFun=nn.MSELoss(reduction='sum'))
reconLoss = ReconLoss(lossFun=nn.MSELoss(reduction='sum'))

lossFun = MultiLoss(lossFunList=[coordLoss, reconLoss])

preprocess = preprocess_d_noise

def get_model(neuron, i):
    
    mid1 = int((neuron + in_dim) / 2)
    mid2 = int((neuron + out_dim) / 2)

    mid = [neuron] * (i - 2)

    return ae.AutoEncoder(
        encode_dim=[in_dim, mid1, *mid, mid2, out_dim],
        activation=nn.LeakyReLU)
        
copyfile('train.py', 'backup/%s.py' % (train_id))

#%%

in_dim, out_dim = n_dist, N * 2

param = [(N + in_dim, L) for N in range(8, 73, 16) for L in range(2, 6)]

for neuron, i in param:

    model_id = train_id + "_" + str(i) + "_" + str(neuron)

    model = get_model(neuron, i)

    helper = trainHelper.TrainHelper(
        id=model_id, model=model, lossFun=lossFun,
        optimizer=torch.optim.Adam(model.parameters(), lr=init_lr),
        preprocess=preprocess)

    train(helper, dlr, 'log/%s.log' % (train_id))

# %%
