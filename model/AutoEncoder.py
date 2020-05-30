import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    def __init__(self, encode_dim: list, decode_dim=None, same_encoder=True,
                 activation=nn.ReLU, final_activation=nn.Tanh):
        super(AutoEncoder, self).__init__()
        
        if decode_dim is None:
            decode_dim = encode_dim.copy()
        
        encode_dim.reverse()
        
        encode_nL = len(encode_dim)
        decode_nL = len(decode_dim)

        self.encoder = nn.Sequential(
            *sum([[
                nn.Linear(encode_dim[i], encode_dim[i - 1]),
                activation()
            ] for i in range(encode_nL - 1, 0, -1)], [])
        )

        self.decoder = nn.Sequential(
            *sum([[
                nn.Linear(decode_dim[i], decode_dim[i - 1]),
                activation()
            ] for i in range(decode_nL - 1, 0, -1)], [])
        )

        if same_encoder: 
            self.encoder2 = nn.Sequential(
                *sum([[
                    nn.Linear(encode_dim[i], encode_dim[i - 1]),
                    activation()
                ] for i in range(encode_nL - 1, 0, -1)], [])
            )
        else:
            self.encoder2 = None
        
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
    
    def forward(self, x, same_encoder=True):
        e = self.encode(x)
        d = self.decode(e)

        if self.encoder2 is not None:
            e = self.encoder2(d)
        else:
            e = self.encoder(d)

        if self.final_act is not None:
            return self.final_act(e)

        return e