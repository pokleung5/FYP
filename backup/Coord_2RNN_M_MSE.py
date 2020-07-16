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
# from model.EignModel import EignModel
import model.AutoEncoder as ae
import model.RNN as rnn

from shutil import copyfile
import os

# %%
data = dataSource.load_data(shape=(ss, N, d))[0]
data = utils.minmax_norm(data, dmin=0)[0]

dlr = DataLoader(data, batch_size=batch, shuffle=True)

init_lr = 1e-3

train_id = '_'.join(['Coord', '2RNN', 'M', 'MSE'])

coordLoss = CoordsToDMLoss(N, lossFun=nn.MSELoss(reduction='mean'))
reconLoss = ReconLoss(lossFun=nn.MSELoss(reduction='mean'))
mseLoss = CustomLoss(lossFun=nn.MSELoss(reduction='mean'))

lossFun = MultiLoss(lossFunList=[coordLoss])

preprocess = PrepMatrix(N)
in_dim = preprocess.get_inShape()
out_dim = 2

def get_model(nNeuron, nLayer, in_dim, out_dim):

    mid1 = int(nNeuron + in_dim)
    
    return rnn.LRNN(
        N=N,
        dim=[in_dim, mid1, out_dim],
        num_rnn_layers=nLayer, 
        activation=nn.LeakyReLU)

# %%
bkup_dest = 'backup/%s.py' % (train_id)

if os.path.exists(bkup_dest):
    raise Exception("Destination file exists!")

copyfile('train.py', bkup_dest)

#%%

param = [(N, L)  for L in range(1, 2) for N in range(8, 73, 16)]

for neuron, i in param:

    model_id = train_id + "_" + str(i) + "_" + str(neuron)

    model = get_model(neuron, i, in_dim, out_dim)

    helper = trainHelper.TrainHelper(
        id=model_id, model=model, lossFun=lossFun,
        optimizer=torch.optim.Adam(model.parameters(), lr=init_lr),
        preprocess=preprocess)

    train(helper, dlr, 'log/%s.log' % (train_id))

# %%
