import utils

import torch
from torch import Tensor, tensor

def load_data(shape, sample_space=None) -> tuple:
    """
    Param: 
        **sample_space** for 
    Return:
        **(random DM, euclidean DM)**
    """
    
    ss, N, d = shape

    try:
        euclidean_data = utils.load_variable('data/euclidean.dat')
        
        if sample_space is not None or euclidean_data.size() !=  (ss, 1, N, N):
            raise Exception("Previous data not match requirement !")

    except:
        coords_space = (int(sample_space[0] / 2 ** 0.5), sample_space[1]) # to limit the max distance value

        euclidean_data = generate_euclidean_DM(N=N, d=d, sample_size=ss, sample_space=coords_space, isInt=True)
        utils.dump_variable(euclidean_data, 'data/euclidean.dat')

    try:
        rand_data = utils.load_variable('data/rand.dat')
        
        if sample_space is not None or rand_data.size() !=  (ss, 1, N, N):
            raise Exception("Previous data not match requirement !")

    except:
        rand_data = generate_rand_DM(N=N, sample_size=ss, sample_space=sample_space, isInt=True)
        utils.dump_variable(rand_data, 'data/rand.dat')

    return rand_data, euclidean_data


def generate_rand_DM(N: int, sample_size: int, isInt=False, sample_space=(1, 0)) -> Tensor:
    """
        Param: 
            **N** for number of object
        Return:
            **random value matrix**
    """

    dim = (sample_size, N, N)

    data = get_rand_data(dim=dim, isInt=isInt,
                         maxXY=sample_space[0],
                         minXY=sample_space[1])

    upper = torch.triu(data, diagonal=1)
    lower = upper.permute(0, 2, 1)

    rs = upper + lower

    rs = rs.view(sample_size, 1, N, N)

    return rs


def generate_euclidean_DM(N: int, d: int, sample_size: int, isInt=False, sample_space=(1, 0)) -> Tensor:
    """
        Param: 
            **N** for number of object
            **d** for dimension of the coordinatecs
            **sample_space** for controlling the decimal places
        Return:
            **distance matrix**
    """

    coords = get_rand_data((sample_size, N, d), isInt=isInt,
                           maxXY=sample_space[0],
                           minXY=sample_space[1])

    dms = torch.sqrt(utils.get_distanceSq_matrix(coords))
    
    dms = dms.view(sample_size, 1, N, N)

    return dms


def get_rand_data(dim=tuple, isInt=False, maxXY=1, minXY=0) -> Tensor:

    if isInt:
        return torch.randint(minXY, maxXY, dim).double()

    return torch.rand(dim) * (maxXY - minXY) + minXY


def custom_distance(N, n_arg, sample_size: int, isInt=False, sample_space=(1, 0), 
                    dist_func=lambda p1, p2: torch.sum((p2 - p1)** 2)** 0.5):
        
    dm = torch.zeros(sample_size, N, N)

    coords = get_rand_data((sample_size, N, n_arg), isInt=isInt,
                           maxXY=sample_space[0],
                           minXY=sample_space[1])

    for b in range(sample_size):
        for i in range(N):
            for j in range(i + 1, N):
                dm[b][j][i] = dm[b][i][j] = dist_func(coords[b][i], coords[b][j], b)

    return dm.view(sample_size, 1, N, N)


if __name__ == "__main__":

    import numpy 
    from torch import nn
    from mds.cmds import classicalMDS
    from lossFunction import CoordsToDMLoss
    torch.set_default_tensor_type('torch.DoubleTensor')


    dist_func = lambda p1, p2: torch.sum(torch.abs((p2 - p1)))
    
    cus = custom_distance(10, 3, 1000, isInt=True, sample_space=(1000, 1), dist_func=dist_func)

    lossFun = CoordsToDMLoss(N=10, d=2, lossFun=nn.L1Loss(reduction='mean'))
    
    coords, dms = cus

    dms = utils.minmax_norm(dms, dmin=0)[0] + 1e-8
    dms = dms.detach().requires_grad_(True)

    cmds_rs = []

    for d in dms:

        d1 = numpy.array(d.data)
        cmds_rs.append(torch.tensor(classicalMDS(d1, 2)))

    cmds_rs = torch.stack(cmds_rs)
    print("cmds_loss: \t", lossFun(cmds_rs, dms))
