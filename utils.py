
# %%

from datetime import datetime
import torch
from torch import Tensor, tensor
import pickle
import math
import numpy
from matplotlib import pyplot as plt

# %%
# def generate_DM(N: int, sample_size: int, func: callable, isInt=False, sample_space=(1, 0)):


def generate_rand_DM(N: int, sample_size: int, isInt=False, sample_space=(1, 0)) -> Tensor:
    """
        Param: 
            **N** for number of object
        Return:
            **random value matrix**^2
            - generate_euclidean_DM also ^2
    """
    
    dim = (sample_size, N, N)

    data = get_rand_data(dim=dim, isInt=isInt,
                         maxXY=sample_space[0],
                         minXY=sample_space[1])
                         
    upper = torch.triu(data, diagonal=1)
    lower = upper.permute(0, 2, 1)
    
    rs = upper + lower
    
    s = rs.size()
    rs_norm, dmax, _ = minmax_norm(
        rs ** 2, flat=(s[0], 1, 1, N * N), dmin=0
    )
    
    rs_norm = rs_norm.view(sample_size, 1, N, N)

    return torch.tensor(rs_norm, requires_grad=True)


def generate_euclidean_DM(N: int, d: int, sample_size: int, isInt=False, sample_space=(1, 0)) -> Tensor:
    """
        Param: 
            **N** for number of object
            **d** for dimension of the coordinatecs
            **sample_space** for controlling the decimal places
        Return:
            **distance^2 matrix**
    """

    coords = get_rand_data((sample_size, N, d), isInt=isInt,
                            maxXY=sample_space[0],
                            minXY=sample_space[1])

    dms, _ = get_distance_matrix(coords)
    dms = dms.view(sample_size, 1, N, N)

    return tensor(dms.data, requires_grad=True)

# %%


def dump_variable(data, path: str):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_variable(path: str):
    data = None

    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

# %%


def get_rand_data(dim=tuple, isInt=False, maxXY=1, minXY=0) -> Tensor:

    if isInt:
        return torch.randint(minXY, maxXY, dim)
    return torch.rand(dim) * (maxXY - minXY) + minXY


def get_distance_matrix(pl: Tensor) -> Tensor:

    if len(pl.size()) < 3:
        pl = pl.view(1, pl.size()[0], pl.size()[1])

    N = pl.size()[1]

    dm = torch.matmul(pl, pl.permute(0, 2, 1).clone())
    diag = torch.stack([torch.diag(m).expand_as(m) for m in dm])
    rs = diag + diag.permute(0, 2, 1) - dm - dm

    s = rs.size()
    rs_norm, dmax, _ = minmax_norm(
        rs, flat=(s[0], 1, 1, N * N), dmin=0
    )

    # rs2 = torch.sqrt(rs2) # backpropagation might fail if 0 involved

    return rs_norm, dmax.sqrt()

# %%


def minmax_norm(data: Tensor, flat=None, dim=None, dmax=None, dmin=None, epsilon=0) -> Tensor:
    data = data.double()

    if dim is None:
        dim = len(flat) - 1

    if dmin is None:
        dmin = torch.min(data.view(flat), dim=dim)[0]

    if dmax is None:
        dmax = torch.max(data.view(flat), dim=dim)[0] + epsilon

    return (data - dmin) / (dmax - dmin), dmax, dmin


# %%


def time_measure(fun, arg: dict):

    t = datetime.now()
    return fun(*arg), (datetime.now() - t).total_seconds()

# %%


def plot_records(records: dict, epoch: int, value_label="Loss"):

    plt.clf()

    plt.xlabel('Epoch')
    plt.ylabel(value_label)

    for key in records.keys():
        r = numpy.array(records[key])
        plt.plot(range(epoch), r[:epoch], label=key)
    
    plt.legend()
    plt.show()


def plot_2D_coords(coords: dict):
    
    plt.clf()

    plt.xlabel('X')
    plt.ylabel('Y')

    for key in coords.keys():
        r = numpy.array(coords[key])
        plt.scatter(r[:, 0], r[:, 1], label=key)
    
    plt.legend()
    plt.show()

# %%

# def flatten(data: Tensor):
    