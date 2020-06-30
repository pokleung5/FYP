import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F

from . import AutoEncoder as ae

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE(ae.AutoEncoder):

    def __init__(self, encode_dim: list, decode_dim=None,
                    activation=nn.ReLU, final_activation=None):

        if decode_dim is None:
            decode_dim = encode_dim.copy()
            decode_dim.reverse()

        super(VAE, self).__init__(
            encode_dim=encode_dim[:-2], decode_dim=decode_dim,
            activation=activation, final_activation=final_activation)

        self.e1 = nn.Linear(encode_dim[-2], encode_dim[-1])
        self.e2 = nn.Linear(encode_dim[-2], encode_dim[-1])


    def encode(self, x):
        e = self.encoder(x)
        return self.e1(e), self.e2(e)

    def forward(self, x):
        mu, logvar = self.encode(x)

        z = reparameterize(mu, logvar)
        d = self.decode(z)

        if self.final_act is not None:
            d = self.final_act(d)
        
        return d, mu, logvar
