import dataSource
import utils
import trainHelper

import numpy, torch
from torch import nn, Tensor
from datetime import datetime

torch.set_default_tensor_type('torch.DoubleTensor')

batch = 16
test_size = 1000
ss, N, d = 3200, 10, 2

n_dist = int(N * (N - 1) / 2)

test_data = dataSource.generate_rand_DM(N, sample_size=test_size, isInt=True, sample_space=(1000, 1))
test_data = utils.minmax_norm(test_data, dmin=0)[0]

def test(helper, test_data):

    rs, target = helper._predict(test_data)
    loss = helper.lossFun(rs, target)

    return loss / test_data.size()[0]

def train(helper, dlr, logFilePath):

    EPOCH = 100

    print("Training ", helper.id)
    
    with open(logFilePath, "a+") as f:
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
        
        with open(logFilePath, "a+") as f:
            f.write("%s%s\n" % (str(datetime.today()), output))

        test_loss_records.append(test_loss)

        if i > 2 and numpy.polyfit(range(i + 1), test_loss_records, 1)[-2] > 0:
            print("Overfitting !")
        
            with open(logFilePath, "a+") as f:
                f.write("Overfitting !\n\n")

            utils.dump_variable(helper, 'result/%s_%s.model' % (helper.id, 'Overfit'))
            break          

        if 1e-7 > min_lr and helper.records['lr'][helper.records['lr'] == min_lr].count() >= EPOCH:
            print("Reach Low learning Rate !")
                        
            with open(logFilePath, "a+") as f:
                f.write("Reach Low Learning Rate !\n\n")

            utils.dump_variable(helper, 'result/%s_%d.model' % (helper.id, helper.epoch))
            break


# def model(neuron, i, in_dim, out_dim):

#     return rnn.aRNN(
#         N=in_dim,
#         in_dim=[*[neuron for j in range(i - 1)], int(neuron / 2), out_dim],
#         num_rnn_layers=1)

#     return rnn.bRNN(
#         N=N,
#         in_layer_dim=[in_dim, *[neuron for j in range(i - 1)], int(neuron / 2)],
#         out_dim=out_dim,
#         num_rnn_layers=1)

#     return EignModel(
#         dim=[in_dim,  *[neuron for j in range(i - 1)], int(neuron / 2), 4],
#         concat_dim=[4, 16, 32, 32, 16, out_dim],
#         activation=nn.LeakyReLU)

#     return Linear(
#         dim=[in_dim, *[neuron for j in range(i - 1)], int(neuron / 2), out_dim],
#         activation=nn.LeakyReLU)

#     return ae.AutoEncoder(
#         encode_dim=[in_dim, *[neuron for j in range(i - 1)], int(neuron / 2), out_dim],
#         activation=nn.LeakyReLU)