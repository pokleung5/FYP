# %%
from DataGenerator import DataGenerator
from bad_grad_viz import *
import torch
from torch import Tensor


zo = torch.tensor(1e-8, requires_grad=True)

def _get_distance_matrix(pairs: Tensor):
    m = pairs.mm(pairs.t())
    diag = torch.diag(m, diagonal=0).expand_as(m)
    
    rs2 = diag + diag.t() - m - m
    return torch.sqrt(rs2 + zo).view(1, pairs.size()[0], pairs.size()[0])

def get_coordinates(dim=tuple, maxXY=1, minXY=0, batch=1):
    maxXY = torch.tensor(maxXY, requires_grad=True)
    minXY = torch.tensor(minXY, requires_grad=True)

    if batch < 2:
        return torch.rand(dim, requires_grad=True) * (maxXY - minXY) + minXY
    return torch.rand((batch, *dim), requires_grad=True) * (maxXY - minXY) + minXY


def get_distance_matrices(pl: Tensor):
    batch = pl.size()[0] if len(pl.size()) > 2 else 1

    if batch < 2:
        return _get_distance_matrix(pl)

    return torch.stack([_get_distance_matrix(pairs) for pairs in pl])

# %%

dg = DataGenerator((5, 2))
dgl = dg.get_pair_loader(N=3, batch=5)

for step, target in enumerate(dgl):
    a1 = target 
    a2 = get_distance_matrices(a1)
    break

