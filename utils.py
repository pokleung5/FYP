
# %%

from datetime import datetime
import torch
from torch import Tensor, tensor
# import pickle
import dill as pickle
import math
import numpy
from matplotlib import pyplot as plt
# %%

def doubleCentering(dm: Tensor) -> Tensor:

    batch, N = dm.size()[0], dm.size()[-1]

    J = torch.eye(N) -  torch.ones((N, N)) / N
    B = -0.5 * torch.matmul(torch.matmul(J, dm ** 2), J)

    return B

def eignmatrix2(dm: Tensor) -> Tensor:
    
    batch, N = dm.size()[0], dm.size()[-1]

    B = doubleCentering(dm)
    vals, vects = numpy.linalg.eigh(B.detach().numpy())
    
    rank = numpy.argsort(-1 * vals)

    for i in range(batch):
        vals[i] = vals[i][rank[i]]
        vects[i] = vects[i][:, rank[i]]

    vals = numpy.sign(vals) * numpy.sqrt(numpy.abs(vals))

    return torch.tensor(vals.reshape(batch, 1, N) * vects)


def eignmatrix(dm: Tensor) -> Tensor:
    
    batch, N = dm.size()[0], dm.size()[-1]

    B = doubleCentering(dm)
    vals, vects = numpy.linalg.eigh(B.detach().numpy())
    
    rank = numpy.argsort(-1 * vals)

    for i in range(batch):
        vals[i] = vals[i][rank[i]]
        vects[i] = vects[i][:, rank[i]]

    vals = numpy.sign(vals) * numpy.sqrt(numpy.abs(vals))
    gap = numpy.arange(0, N * 2, 2)

    dm = numpy.zeros((batch, N, N * 2))
    dm[:, :, gap] = vals.reshape(batch, 1, N)
    dm[:, :, gap + 1] = vects
    
    return torch.tensor(dm)

# %%

def vectorize_distance_from_DM(dm: Tensor) -> Tensor:
    
    upper_indices = torch.triu(torch.ones(dm.size()), diagonal=1)
    
    batch = dm.size()[0]
    N = dm.size()[-1]

    n = int((N * (N - 1)) / 2)
    
    return dm[upper_indices == 1].view((batch, 1, n))


def unvectorize_distance(vect: Tensor) -> Tensor:
    
    batch = vect.size()[0]
    N = math.ceil((2 * vect.size()[-1]) ** 0.5)

    rs = torch.zeros(batch, N, N)
    triu_indices = torch.ones(N, N).triu(diagonal=1).nonzero().transpose(1, 0)

    rs[:, triu_indices[0], triu_indices[1]] = vect.view(batch, vect.size()[-1])

    rs = rs + rs.permute(0, 2, 1)

    return rs


def dump_variable(data, path: str):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_variable(path: str):
    data = None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def get_distanceSq_matrix(pl: Tensor) -> Tensor:

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
        dmin = torch.min(data.view(flat), dim=dim)[0] - epsilon

    if dmax is None:
        dmax = torch.max(data.view(flat), dim=dim)[0] + epsilon

    return (data - dmin) / (dmax - dmin), dmax, dmin


def time_measure(fun, arg: list):

    t = datetime.now()
    return fun(*arg), (datetime.now() - t).total_seconds()
