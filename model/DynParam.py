#%% 
import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F


class DynParam(nn.Module):
    def __init__(self, dim: list, paramDim: list,
                 activation=nn.ReLU, final_activation=nn.Tanh):
        super(DynParam, self).__init__()
        
        dimNL = len(dim)
        parNL = len(paramDim)
        
        self.encoder = nn.Sequential(
            *sum([[
                nn.Linear(dim[i - 1], dim[i]),
                activation()
            ] for i in range(1, dimNL)], [])
        )

        self.param = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=dim[0] - paramDim[0] + 1),
            nn.Softmax2d(),
            *sum([[
                nn.Linear(paramDim[i - 1], paramDim[i]),
                activation()
            ] for i in range(1, parNL) if parNL > 1], [])
        )

        if final_activation is not None:
            self.final_act = final_activation()
        else:
            self.final_act = None

    def encode(self, x1, x2):
        e = self.encoder(x1)
        pm = self.param(x2)

        return e, pm

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1
            
        e, p = self.encode(x1, x2)
        z = torch.matmul(e, p)

        if self.final_act is not None:
            return self.final_act(z)

        return z