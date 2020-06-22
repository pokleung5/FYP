#%%
import sys
from importlib import reload

#reload(sys.modules['utils']);
#reload(sys.modules['dataSource']);
#reload(sys.modules['mds.cmds']);

#%%
import trainHelper, utils, dataSource

import numpy, torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.Linear import Linear
import model.AutoEncoder as ae
import model.RNN as rnn

from datetime import datetime

from lossFunction import CoordsToDMLoss, ReconLoss

torch.set_default_tensor_type('torch.DoubleTensor')

#%% 

batch = 32
test_size = 1000
ss, N, d = 3200, 10, 2

#%%
# sample_space=(1000, 1)
data = dataSource.load_data(shape=(ss, N, d))[0]
data = utils.minmax_norm(data, dmin=0)[0]

dlr = DataLoader(data, batch_size=batch, shuffle=True)

#%%

test_data = dataSource.generate_rand_DM(N, sample_size=test_size, isInt=True, sample_space=(1000, 1))
test_data = utils.minmax_norm(test_data, dmin=0)[0]

def test(helper, test_data):

    rs = helper._predict(test_data)
    loss = helper.lossFun(rs, test_data)

    return loss / test_data.size()[0]

def train(helper):

    EPOCH = 100

    print("Training ", helper.id)
    
    with open('log/train.log', "a+") as f:
        f.write("Training %s\n" % (helper.id))
    
    test_loss_records = []

    for i in range(10):

        helper.train(dlr, EPOCH)
        helper.backup()
            
        min_lr = helper.records['lr'].min()

        train_loss = helper.records['loss_mean'].iloc[-1] / 32
        test_loss = float(test(helper, test_data))

        output = " | Epoch %d | Train Loss: %f | Test Loss: %f | Lr: 10e%d" % (
                        helper.epoch, train_loss, test_loss, int(numpy.log10(min_lr))
                    )

        print(output)
        
        with open('log/train.log', "a+") as f:
            f.write("%s%s\n" % (str(datetime.today()), output))

        test_loss_records.append(test_loss)

        if i > 2 and numpy.polyfit(range(i + 1), test_loss_records, 1)[-2] > 0:
            print("Overfitting !")
        
            with open('log/train.log', "a+") as f:
                f.write("Overfitting !\n\n")

            utils.dump_variable(helper, 'trained_model/overfit/' + helper.id + '_' + str(helper.epoch))
            break          

        if 1e-7 > min_lr and helper.records['lr'][helper.records['lr'] == min_lr].count() >= EPOCH:
            print("Reach Low learning Rate !")
                        
            with open('log/train.log', "a+") as f:
                f.write("Reach Low Learning Rate !\n\n")

            utils.dump_variable(helper, 'trained_model/' + helper.id + '_' + str(helper.epoch))
            break

# %%

def preprocess_e(x):
    N = x.size()[-1]
    ex = utils.eignmatrix(x.view(-1, N, N)).view(-1, 1, N, N * 2)
    return ex.clone().detach().requires_grad_(True)

def preprocess_d(x):
    v = utils.vectorize_distance_from_DM(x)
    return v.clone().detach().requires_grad_(True)
    
def preprocess_m(x):
    return x.clone().detach().requires_grad_(True)

#%%

init_lr = 1e-3
lossFun = CoordsToDMLoss(N, 2, lossFun=nn.MSELoss(reduction='sum'))

for neuron in range(8, 89, 16):

    for i in range(1, 6):

        model_id = "Coords_Linear_" + str(i)+ "_" + str(neuron) + "_E_MSE"

        in_dim, out_dim = N * 2, 2

        model = Linear([in_dim, *[neuron for j in range(i - 1)], int(neuron / 2),  out_dim],
                 activation=nn.LeakyReLU, final_activation=None)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_e, lossFun=lossFun)
        
        train(helper)

        ###########################################################################

        model_id = "Coord_Linear_" + str(i)+ "_" + str(neuron) + "_D_MSE"

        in_dim, out_dim = int((N * N - N) / 2), N * 2

        model = Linear([in_dim, *[neuron for j in range(i - 1)], int(neuron / 2),  out_dim],
                 activation=nn.LeakyReLU, final_activation=None)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
        
        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_d, lossFun=lossFun)
        
        train(helper)
        
        ###########################################################################

        model_id = "Coord_Linear_" + str(i)+ "_" + str(neuron) + "_M_MSE"

        in_dim, out_dim = N, 2

        model = Linear([in_dim, *[neuron for j in range(i - 1)], int(neuron / 2), out_dim],
                 activation=nn.LeakyReLU, final_activation=None)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
        
        helper = trainHelper.TrainHelper(id=model_id,
            model=model, optimizer=optimizer, preprocess=preprocess_m, lossFun=lossFun)
        
        train(helper)
        
# %%
