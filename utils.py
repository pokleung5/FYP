
#%%

import torch
from torch import Tensor, tensor
import pickle
import math 

#%%

def generate_data(sample_size, N, d, isInt=False, sample_space=(1, 0)):
    """
        Param: 
            **N** for number of object
            **d** for dimension of the coordinate
            **sample_space** for controlling the decimal places
        Return:
            1. **distance^2 matrix**
            2. **scaling factor** for result coordinate
    """
    coords = get_rand_coords((sample_size, N, d), *sample_space, isInt=isInt)
    dms = get_distance_matrix(coords)
    
    dmax = torch.max(dms)
    
    dms = minmax_norm(dms, dmax=dmax, dmin=0)

    data = dms.view(sample_size, 1, N, N)
    
    return torch.tensor(data.data, requires_grad = True), math.sqrt(dmax)
    

def dump_variable(data: tensor, path: str):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_variable(path: str):
    data = None

    with open(path, 'rb') as f:
        data = pickle.load(f)
        
    return data

#%%
def get_rand_coords(dim=tuple, maxXY=1, minXY=0, isInt=False) -> Tensor:

    if isInt:
        return torch.randint(minXY, maxXY, dim)
    return torch.rand(dim) * (maxXY - minXY) + minXY


def get_distance_matrix(pl: Tensor) -> Tensor:
    
    if len(pl.size()) < 3:
        pl = pl.view(1, pl.size()[0], pl.size()[1])
    
    n = pl.size()[1]

    dm = torch.matmul(pl, pl.permute(0, 2, 1).clone())
    diag = torch.stack([torch.diag(m).expand_as(m) for m in dm])
    rs2 = diag + diag.permute(0, 2, 1) - dm - dm
    
    # rs2 = torch.sqrt(rs2) # backpropagation might fail if 0 involved
    
    return rs2

# %%

def _minmax_norm(data: Tensor, dmax=None, dmin=None) -> Tensor:
    if dmin is None:
        dmin = torch.min(data)
        
    if dmax is None:
        dmax = torch.max(data)

    return (data.double() - dmin)/(dmax - dmin)


def minmax_norm(data: Tensor, dmax=None, dmin=None) -> Tensor:
    if len(data.size()) < 3:
        return _minmax_norm(data, dmax, dmin) 

    return torch.stack([_minmax_norm(d, dmax, dmin) for d in data])