import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F


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
        return e

    def forward(self, x):
        e = self.encode(x)

        if self.final_act is not None:
            e = self.final_act(e)

        return e