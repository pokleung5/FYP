#%%
import scipy.linalg as linalg
import numpy
from . import cmds

def landmarkMDS(dist: numpy.array, L=0, D=2, r=None):

    distSqM = dist ** 2

    N = len(distSqM)

    if L == 0:
        L = int(N * 0.5)

    if r is None:
        r = range(L)  # r = rand_distinct(range(N), L)
    else:
        L = len(r)

    dM = distSqM[r, :]
    lmD = dM[:, r]

    rank, vals, vects = cmds.classicalMDS(lmD, 0, allowZero=False)
    
    rank = rank[:D]
    vals = vals[rank]
    vects = vects[:, rank]

    Lh = vects.dot(numpy.diag(1 / numpy.sqrt(vals))).T
    Dm = dM ** 0.5 - numpy.tile(numpy.mean(lmD ** 0.5, axis=1), (N, 1)).T

    R = -0.5 * Lh.dot(Dm)
    R = R - numpy.tile(numpy.mean(R, axis=1), (N, 1)).T
    _, vects = linalg.eigh(R.dot(R.T))

    return vects[:, ::-1].T.dot(R).T

#%%
import torch
import utils

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.DoubleTensor')

    pts = torch.tensor([
            [0, 2],
            [2, 5],
            [6, 8],
            [8, 7],
            [9, 10]
        ])
    
    dm = utils.get_distanceSq_matrix(pts)[0] ** 0.5
    dm = utils.minmax_norm(dm)[0]
    print(dm)
    
    dm = numpy.array(dm)
    rs = landmarkMDS(dm, 5, 2)
    rs = torch.tensor(rs)

    rs_dm = utils.get_distanceSq_matrix(rs)[0] ** 0.5
    rs_dm = utils.minmax_norm(rs_dm)[0]
    print(rs_dm)

    print(torch.sum(((rs_dm - torch.tensor(dm)) ** 2)))

# %%
