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
from mds.cmds import classicalMDS
from mds.lmds import landmarkMDS
from mds.fastmap import fastmap
from preprocess import *

import glob

torch.set_default_tensor_type('torch.DoubleTensor')


class Test:

    def __init__(self, N, d, test_size):
        self.N = N
        self.d = d
        self.n_d = int(N * (N - 1) / 2)
        
        self.test_size = test_size
        
        self.lossFun = CoordsToDMLoss(N=N, d=d, lossFun=nn.L1Loss(reduction='mean'))

        self.test_data = dataSource.generate_rand_DM(
            self.N, sample_size=self.test_size, isInt=True, sample_space=(1000, 1))
        self.test_data = utils.minmax_norm(self.test_data, dmin=0)[0]

    def reload_rand(self):

        self.test_data = dataSource.generate_rand_DM(
            self.N, sample_size=self.test_size, isInt=True, sample_space=(1000, 1))
        self.test_data = utils.minmax_norm(self.test_data, dmin=0)[0]

    def reload_euclidean(self):

        self.test_data = dataSource.generate_euclidean_DM(
            self.N, self.d, sample_size=self.test_size, isInt=True, sample_space=(1000, 1))
        self.test_data = utils.minmax_norm(self.test_data, dmin=0)[0]

    def test(self, helper, test_data=None):
        
        if test_data is None:
            test_data = self.test_data

        rs = helper._predict(test_data)
        rs = rs[-1] if type(rs) is tuple else rs

        if rs.size()[-1] == self.n_d:  # special handle for recon

            rs = utils.unvectorize_distance(rs) 

            rs = torch.stack([
                    torch.tensor(
                        classicalMDS(dm)
                     ) for dm in rs.detach().numpy()
                ])

        loss = self.lossFun(rs, test_data)

        return loss

    def classicSolution(self):

        cmds_rs, fastmap_rs, lmds_rs = [], [], []

        for d in self.test_data:

            d1 = numpy.array(d[0].data)

            cmds_rs.append(torch.tensor(classicalMDS(d1, 2)))
            lmds_rs.append(torch.tensor(landmarkMDS(d1, L=7, D=2)))
            fastmap_rs.append(torch.tensor(fastmap(d1, 2)))

        cmds_rs = torch.stack(cmds_rs)
        lmds_rs = torch.stack(lmds_rs)
        fastmap_rs = torch.stack(fastmap_rs)
        
        print("cmds_loss: \t", self.lossFun(cmds_rs, self.test_data))
        print("lmds_loss: \t", self.lossFun(lmds_rs, self.test_data))
        print("fastmap_loss: \t", self.lossFun(fastmap_rs, self.test_data))


    def tabulate(self, paths, N):

        linear_result_score = []

        for filepath in paths:
            t = filepath.split('\\')[-1].split('.')[0].split('_')
            
            h: trainHelper.TrainHelper = utils.load_variable(filepath)
            linear_result_score.append([*t, self.test(h).data])

        linear_result_score = sorted(linear_result_score, key=lambda x: x[-1])
        # return linear_result_score

        return pandas.DataFrame(linear_result_score, columns=[
            'Method', 'Model', 'Input', 'LossFun', 'Layer', 'Neuron', 'Epoch', 'Loss'
            ])


if __name__ == "__main__":

    test = Test(10, 2, 1000)

    # test.reload_euclidean()
    
    test.classicSolution()
    records = test.tabulate(glob.glob('result/*.model'), -1)
    print(records)
    
    print('==========================================')

    #Test(10, 2, 1000).printTopN(
    #glob.glob('trained_model\Coord*'), -1) 

    