# %% 
# import scipy.linalg as linalg
import numpy as np

def classicalMDS(dist: np.array, D=2, J=None, allowZero=True):

    distSqM = dist ** 2
    
    if J is None:
        N = len(distSqM)
        J = np.eye(N) - np.ones((N, N)) / N

    distSqM = distSqM.reshape(N, N)

    B = -0.5 * J.dot(distSqM).dot(J)

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    nonN, = np.where(eigenvalues >= 0) if allowZero else np.where(eigenvalues > 0) 

    rank = np.argsort(-1 * eigenvalues)

    if D == 0:
        return rank, eigenvalues, eigenvectors

    if D == -1:
        D = len(nonN)

    if len(nonN) < D:
        print(eigenvalues)
        raise Exception('Not enough non-zero eigenvalues for Classical MDS !!')

    rank = rank[:D][:len(nonN)]
    eigenvalues = eigenvalues[rank]
    eigenvectors = eigenvectors[:, rank]

    return eigenvectors.dot(np.diag(np.sqrt(eigenvalues)))

    # return None

#%%

import torch
import utils

if __name__ == "__main__":
    torch.set_default_tensor_type('torch.DoubleTensor')

    pts = torch.tensor([[
            [0, 2],
            [2, 5],
            [6, 4],
            [8, 7],
            [9, 10]
        ]])
    
    dm = utils.get_distanceSq_matrix(pts) ** 0.5
    dm = torch.tensor([[
            [0, 3, 4],
            [3, 0, 5],
            [4, 5, 0]
        ]])

    dm = utils.minmax_norm(dm)[0]
    print(dm)
    
    dm = np.array(dm[0].data)
    rs = classicalMDS(dm, 2)
    rs = torch.tensor(rs)
    print(rs)

    rs_dm = utils.get_distanceSq_matrix(rs) ** 0.5
    rs_dm = utils.minmax_norm(rs_dm)[0]
    print(rs_dm)
    
    print(torch.sum(((rs_dm - torch.tensor(dm)) ** 2)))
