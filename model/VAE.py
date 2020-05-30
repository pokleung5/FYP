import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class VAE(nn.Module):
    def __init__(self, dim: list,
                 activation=nn.ReLU, final_activation=nn.Tanh):
        super(VAE, self).__init__()
        
        nL = len(dim)
        dim.reverse()

        self.encoder = nn.Sequential(
            *sum([[
                nn.Linear(dim[i], dim[i - 1]),
                activation()
            ] for i in range(nL - 1, 1, -1)], [])
        )

        self.e1 = nn.Linear(dim[1], dim[0])
        self.e2 = nn.Linear(dim[1], dim[0])

        if final_activation is not None:
            self.final_act = final_activation()
        else:
            self.final_act = None

    def encode(self, x):
        e = self.encoder(x)
        return self.e1(e), self.e2(e)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        
        if self.final_act is not None:
            return self.final_act(z)
        
        return z
