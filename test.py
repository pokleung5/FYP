import trainHelper
import utils
import dataSource

import numpy
import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from model.Linear import Linear
import model.AutoEncoder as ae
import model.RNN as rnn

import datatime

from lossFunction import CoordsToDMLoss, ReconLoss
import glob

torch.set_default_tensor_type('torch.DoubleTensor')


class Test:

    def __init__(self):

        self.test_data = dataSource.generate_rand_DM(
            N, sample_size=test_size, isInt=True, sample_space=(1000, 1))
        self.test_data = utils.minmax_norm(test_data, dmin=0)[0]

    def reload(self):
        self.test_data = dataSource.generate_rand_DM(
            N, sample_size=test_size, isInt=True, sample_space=(1000, 1))
        self.test_data = utils.minmax_norm(test_data, dmin=0)[0]

    def test(helper, test_data=None):
        if test_data is None:
            test_data = self.test_data

        rs = helper._predict(test_data)
        loss = helper.lossFun(rs, test_data)

        return loss / test_data.size()[0]

    def printTopN(self, paths, N)
      linear_result_score = []

       for filepath in paths:
            h: trainHelper.TrainHelper = utils.load_variable(filepath)
            linear_result_score.append([
                filepath, self.test(h)
            ])

        linear_result_score = sorted(linear_result_score, key=lambda x: x[1])

        for rss in linear_result_score[:N]:
            print(rss[0], "   \t\t\t|", rss[1].data)
