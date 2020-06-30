# %%
import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F

from . import Linear


def invert_e(e):

    N2 = e.size()[-1]
    val_id = list(range(0, N2, 2))
    vect_id = list(range(1, N2 + 1, 2))

    inv_x = torch.zeros(e.size())
    inv_x[..., val_id] = -1 * torch.flip(e[..., val_id], dims=[-1, -2])
    inv_x[..., vect_id] = torch.flip(e[..., vect_id], dims=[-1, -2])

    return inv_x.detach().requires_grad_(True)


class EignModel(nn.Module):

    def __init__(self,
                 dim: list, concat_dim: list, inv_dim=None,
                 activation=nn.ReLU, final_activation=None):

        super(EignModel, self).__init__()

        if inv_dim is None:
            inv_dim = dim

        self.encoder = Linear.get_Linear_Sequential(
            dim, activation=activation)
        self.encoder_inv = Linear.get_Linear_Sequential(
            inv_dim, activation=activation)

        self.final_encoder = Linear.get_Linear_Sequential(
            concat_dim, activation=activation)

        self.final_act = final_activation

    def forward(self, x):

        batch = x.size()[0]
        N = x.size()[-2]

        e1 = self.encoder(x)
        e2 = self.encoder_inv(invert_e(x))
        
        e_concat = torch.stack([e1, e2], dim=-1).view(batch, N, -1)
        
        rs = self.final_encoder(e_concat)

        if self.final_act is not None:
            e3 = self.final_act(e3)

        return e1, e2, e3


# %%
