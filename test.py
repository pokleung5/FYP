# %%
import trainHelper
import utils
import dataSource
from mds.cmds import classicalMDS

import numpy
import torch
import pandas

from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.Linear import Linear
import model.AutoEncoder as ae
import model.RNN as rnn

from datetime import datetime

from lossFunction import CoordsToDMLoss, ReconLoss, sammon_loss
from mds.fastmap import fastmap
from preprocess import *
from manifold import Algorithm

import glob

torch.set_default_tensor_type('torch.DoubleTensor')


class Test:

    def __init__(self, N, d, test_size):
        self.N = N
        self.d = d
        self.n_d = int(N * (N - 1) / 2)

        self.test_size = test_size
        self.etm = Algorithm(N, d)

        self.lossFun = CoordsToDMLoss(
            N=N, lossFun=nn.L1Loss(reduction='mean'))

        self.test_data = dataSource.generate_rand_DM(
            self.N, self.test_size, isInt=True, sample_space=(1000, 1))
        self.test_data = utils.minmax_norm(self.test_data, dmin=0)[0]

    def reload_rand(self):

        self.test_data = dataSource.generate_rand_DM(
            self.N, self.test_size, isInt=True, sample_space=(1000, 1))
        self.test_data = utils.minmax_norm(self.test_data, dmin=0)[0]

    def reload_custom(self, dist_func, n_arg):

        self.test_data = dataSource.custom_distance(
            self.N, n_arg, self.test_size, isInt=True, sample_space=(1000, 1), dist_func=dist_func)
        self.test_data = utils.minmax_norm(self.test_data, dmin=0)[0]

    def test(self, method, data): 
            
        rs, time = [], []

        for a in data.detach().numpy():
            
            b, t = utils.time_measure(method, [a])

            rs.append(torch.tensor(b))
            time.append(t)

        rs = torch.stack(rs).view(-1, self.N, self.d)
        loss = float(self.lossFun(rs, self.test_data))
        time = sum(time)

        return loss, time


    def classicSolution(self):

        method_itr = {
            'classicalMDS': self.etm.classicalMDS,
            'nonMetricMDS': self.etm.nonMetricMDS,
            'landmarkMDS': self.etm.landmarkMDS,
            'isomap': self.etm.isomap,
            'fastmap': self.etm.fastmap,
            # 't-SNE': self.etm.tsne
        }

        record = []

        data = self.test_data.view(-1, self.N, self.N)

        for name, method in method_itr.items():
            loss, time = self.test(method, data)
            record.append([name, loss, time])

        return pandas.DataFrame(record, columns=['Method', 'Loss', 'Time'])

    def tabulate(self, paths):

        linear_result_score = []

        data = self.test_data.view(-1, 1, self.N, self.N)

        for filepath in paths:

            self.etm.use_pretrained_model(filepath)
            loss, time = self.test(self.etm.deepMDS, data)

            info = filepath.split('\\')[-1].split('.')[0].split('_')
            linear_result_score.append([*info, loss, time])

        linear_result_score = sorted(linear_result_score, key=lambda x: x[-2])
        # return linear_result_score

        return pandas.DataFrame(linear_result_score, columns=[
            'Method', 'Model', 'Input', 'LossFun', 'Layer', 'Neuron', 'Epoch', 'Loss', 'Time'
        ])
#%%

if __name__ == "__main__":

    test = Test(10, 2, 5)

    # test.reload_custom(lambda a, b, _: torch.sum(torch.abs(a - b)**2)**0.5, n_arg=2)

    print(test.classicSolution())
    print('==========================================')

    records = test.tabulate(
        set(glob.glob('result/Coord*.model'))
        - set(glob.glob('result/*Overfit*.model'))
        - set(glob.glob('result/*(M)*.model'))
        )

    r1 = records[records['Epoch'] != 'Overfit']

    print(r1.iloc[r1.groupby(['Model', 'Input', 'LossFun'])['Loss'].idxmin()].sort_values(['Loss']))
    # print('==========================================')

    # records.to_csv('rank.csv')

    # Test(10, 2, 1000).printTopN(
    # glob.glob('trained_model\Coord*'), -1)


# %%
