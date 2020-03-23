import torch
from torch import Tensor, rand
from torch.utils.data import DataLoader


def _get_distance_matrix(pairs: Tensor):
    m = torch.mm(pairs, pairs.t())
    diag = m.diag().unsqueeze(0).expand_as(m)

    rs2 = m * -2 + diag + diag.t()
    return rs2.sqrt()


def get_coordinates(dim=tuple, maxXY=1, minXY=0, batch=1):
    if batch < 2:
        return rand(dim) * (maxXY - minXY) + minXY
    return rand((batch, *dim)) * (maxXY - minXY) + minXY


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

        self.max_XY = maxXY
        self.min_XY = minXY
        self.dim = dim

    def get_dm_loader(self, N: int, batch: int):
        return DataLoader(
            get_distance_matrices(
                get_coordinates(dim=self.dim, batch=N)),
            batch_size=batch, shuffle=True)

    def get_pair_loader(self, N: int, batch: int):
        return DataLoader(
            get_coordinates(dim=self.dim, batch=N),
            batch_size=batch, shuffle=True)
