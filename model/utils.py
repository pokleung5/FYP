import torch
from torch import Tensor, tensor

def get_distance_matrix(pl: Tensor) -> Tensor:

    pl = pl.view(-1, pl.size()[-2], pl.size()[-1])

    N = pl.size()[1]

    dm = torch.matmul(pl, pl.permute(0, 2, 1))
    diag = torch.diagonal(dm, dim1=1, dim2=2).view(dm.size()[0], 1, N)
    rs = diag + diag.permute(0, 2, 1) - dm - dm

    return rs


def minmax_norm(data: Tensor, flat=None, dim=None, dmax=None, dmin=None, epsilon=0) -> Tensor:
    data = data.double()

    if dim is None:
        dim = len(data.size())

    if flat is None:
        flat = [1 for i in range(dim + 1)]
        flat[0] = data.size()[0]
        flat[-1] = -1
        flat = tuple(flat)

    if dmin is None:
        dmin = torch.min(data.view(flat), dim=dim)[0]

    if dmax is None:
        dmax = torch.max(data.view(flat), dim=dim)[0] + epsilon

    return (data - dmin) / (dmax - dmin), dmax, dmin


