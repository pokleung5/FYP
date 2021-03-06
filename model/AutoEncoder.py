import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F

from . import Linear

class AutoEncoder(nn.Module):
    
    def __init__(self, dim: list, decode_dim=None,
                 activation=nn.ReLU, final_activation=None):
                 
        super(AutoEncoder, self).__init__()
        
        if decode_dim is None:
            decode_dim = dim.copy()
            decode_dim.reverse()
  
        self.encoder = Linear.get_Linear_Sequential(
                                dim=dim, activation=activation)

        self.decoder = Linear.get_Linear_Sequential(
                                dim=decode_dim, activation=activation)

        if final_activation is not None:
            self.final_act = final_activation()
        else:
            self.final_act = None

    def encode(self, x):

        e = self.encoder(x)
        return e 

    def decode(self, x):

        d = self.decoder(x)
        return d
    
    def forward(self, x):

        e = self.encode(x)
        
        e2 = F.leaky_relu(e)
        d = self.decode(e2)

        if self.final_act is not None:
            d = self.final_act(d)

        return d, e