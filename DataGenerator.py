import torch
from torch import Tensor, rand
from torch.utils.data import DataLoader

zo = torch.tensor(1e-8, requires_grad=True)

def _get_distance_matrix(pairs: Tensor):
    # m = pairs.mm(pairs.t())
    m = torch.matmul(pairs, pairs.t())
    diag = torch.diag(m, diagonal=0).expand_as(m)
    # diag = m.diag().expand_as(m)

    rs2 = diag + diag.t() - m - m + zo.expand_as(m)
    return torch.sqrt(rs2).view(1, pairs.size()[0], pairs.size()[0])


def get_coordinates(dim=tuple, maxXY=1, minXY=0, batch=1):
    if batch < 2:
        return torch.rand(dim, requires_grad=True) # * (maxXY - minXY) + minXY
    return torch.rand((batch, *dim), requires_grad=True) # * (maxXY - minXY) + minXY


def get_distance_matrices(pl: Tensor):
    batch = pl.size()[0] if len(pl.size()) > 2 else 1

    if batch < 2:
        return _get_distance_matrix(pl)

    return torch.stack([_get_distance_matrix(pairs) for pairs in pl])


class DataGenerator():

    def __init__(self, dim: tuple, maxXY=1, minXY=0):
        if dim.__class__.__name__ != 'tuple':
            raise Exception(
                'Please input dim as (# of Objects, Graph Dimension)')

        self.max_XY = torch.tensor(maxXY, dtype=float, requires_grad=True)
        self.min_XY = torch.tensor(minXY, dtype=float, requires_grad=True)
        self.dim = dim

    def get_dm_loader(self, N: int, batch: int):
        return DataLoader(get_distance_matrices(
            get_coordinates(dim=self.dim, batch=N)),
            batch_size=batch, shuffle=True)

    def get_pair_loader(self, N: int, batch: int):
        return DataLoader(
            get_coordinates(dim=self.dim, batch=N),
            batch_size=batch, shuffle=True)
