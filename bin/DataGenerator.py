#%% 
import torch
from torch import Tensor, rand
from torch.utils.data import DataLoader

zo = torch.tensor(1e-8, requires_grad=True)

def get_coordinates(dim=tuple, maxXY=1, minXY=0, batch=1):
    if batch < 2:
        return torch.rand(dim) * (maxXY - minXY) + minXY
    return torch.rand((batch, *dim)) * (maxXY - minXY) + minXY


def get_distance_matrices(pl: Tensor):
    batch = pl.size()[0] if len(pl.size()) > 2 else 1
    
    if batch < 2:
        pl = pl.view(1, pl.size()[0], pl.size()[1])
    
    n = pl.size()[1]

    dm = torch.matmul(pl, pl.permute(0, 2, 1).clone())
    diag = torch.stack([torch.diag(m).expand_as(m) for m in dm])
    rs2 = diag + diag.permute(0, 2, 1) - dm - dm # + zo.expand_as(dm)
    # rs3 = torch.sqrt(rs2) # .view(batch, n, n)
    # rs3.grad = pl.grad # ??

    return rs2


class DataGenerator():

    def __init__(self, dim: tuple, maxXY=1, minXY=0):
        if dim.__class__.__name__ != 'tuple':
            raise Exception(
                'Please input dim as (# of Objects, Graph Dimension)')

        self.max_XY = maxXY
        self.min_XY = minXY
        self.dim = dim

    def get_dm_loader(self, N: int, batch: int):
        return DataLoader(get_distance_matrices(
            get_coordinates(dim=self.dim, batch=N, maxXY=self.max_XY, minXY=self.min_XY)),
            batch_size=batch, shuffle=True)

    def get_pair_loader(self, N: int, batch: int):
        return DataLoader(
            get_coordinates(dim=self.dim, batch=N, maxXY=self.max_XY, minXY=self.min_XY),
            batch_size=batch, shuffle=True)


# %%
