import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F

import utils


def get_Linear_Sequential(dim: list, activation):

    nL = len(dim)

    return nn.Sequential(
        *sum([[
            nn.Linear(dim[i], dim[i + 1]),
            activation()
        ] for i in range(0, nL - 2, 1)], []),
        nn.Linear(dim[nL - 2], dim[nL - 1])
    )


class Linear(nn.Module):
    def __init__(self, dim: list,
                 activation=nn.ReLU, final_activation=None):
        super(Linear, self).__init__()

        self.encoder = get_Linear_Sequential(dim, activation)

        if final_activation is not None:
            self.final_act = final_activation()
        else:
            self.final_act = None

    def encode(self, x):

        e = self.encoder(x)

        if self.final_act is not None:
            e = self.final_act(e)

        return e

    def forward(self, x):

        return self.encode(x)


class ReuseLinear(nn.Module):

    def __init__(self, N, dim: list, n_reuse, preprocess,
                 activation=nn.ReLU, final_activation=None):
        super(ReuseLinear, self).__init__()

        self.encoder = get_Linear_Sequential(dim, activation)

        self.N = N
        self.n_reuse = n_reuse
        self.preprocess = preprocess

        if final_activation is not None:
            self.final_act = final_activation()
        else:
            self.final_act = None

    def forward(self, x):

        batch = x.size()[0]
        dm = x
        rs = []

        for i in range(self.n_reuse):

            e = self.encoder(dm)
            rs.append(e)

            e = e.view(batch, self.N, -1)
            dm = utils.get_distanceSq_matrix(e)
            dm = torch.sqrt(dm)
            dm = self.preprocess(dm)

        if self.final_act is not None:
            rs[-1] = self.final_act(e)

        return rs


class StepLinear(nn.Module):

    def __init__(self, dim_list: list,
                 activation=nn.ReLU, final_activation=None):
        super(StepLinear, self).__init__()

        self.encoder = get_Linear_Sequential(dim_list[0], activation=activation)

        if len(dim_list) > 1:
            self.nextStep = StepLinear(dim_list[1:], activation, None)
        else:
            self.nextStep = None

        if final_activation is not None:
            self.final_act = final_activation()
        else:
            self.final_act = None

    def forward(self, x):

        e = self.encoder(x)
        rs = [e]

        if self.nextStep is not None:
            rs = [e, *self.nextStep(e)]
            
        if self.final_act is not None:
            rs[-1] = self.final_act(rs[-1])

        return rs
