#%%
import scipy.linalg as linalg
import numpy
from .cmds import classicalMDS

def landmarkMDS(distSqM: numpy.array, L=3, D=2, r=None):
    N = len(distSqM)

    if r is None:
        r = range(L)  # r = rand_distinct(range(N), L)
    else:
        L = len(r)

    dM = distSqM[r, :]
    lmD = dM[:, r]

    vals, vects = classicalMDS(lmD, D, wR=True)

    Lh = vects.dot(numpy.diag(1 / numpy.sqrt(vals))).T
    Dm = distSqM ** 0.5 - numpy.tile(numpy.mean(lmD ** 0.5, axis=1), (N, 1)).T

    R = -0.5 * Lh.dot(Dm)
    R = R - numpy.tile(numpy.mean(R, axis=1), (N, 1)).T
    _, vects = linalg.eigh(R.dot(R.T))

    return vects[:, ::-1].T.dot(R).T


import torch

def get_distance_matrices(pl: torch.Tensor):
    batch = pl.size()[0] if len(pl.size()) > 2 else 1
    
    if batch < 2:
        pl = pl.view(1, pl.size()[0], pl.size()[1])
    
    n = pl.size()[1]
    zo = torch.tensor(1e-16, requires_grad=True)
    
    dm = torch.matmul(pl, pl.permute(0, 2, 1).clone())
    diag = torch.stack([torch.diag(m).expand_as(m) for m in dm])
    rs2 = diag + diag.permute(0, 2, 1) - dm - dm # + zo.expand_as(dm)
    rs3 = torch.sqrt(rs2)
    
    if batch > 1:
        rs3 = rs3.view(batch, 1, n, n)

    return rs3.detach()

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.DoubleTensor')

    pts = torch.tensor(
        [
            [-0.3393,  0.9407],
            [-0.9269, -0.6886],
            [1.2661, -0.2521]
        ], requires_grad=True)
    
    dm = get_distance_matrices(pts)[0]
 
    dm = numpy.array(dm.data)
    rs = landmarkMDS(dm ** 2)
    rs = torch.tensor(rs)
    
    print(dm)
    print(get_distance_matrices(rs))


# %%
