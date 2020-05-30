# %% 
import scipy.linalg as linalg
import numpy as np

def classicalMDS(distSqM: np.array, D=2, J=None, wR=False):
    if J is None:
        N = len(distSqM)
        J = np.eye(N) - np.ones((N, N)) / N

    B = -0.5 * J.dot(distSqM).dot(J)
    # print(B)

    eigenvalues, eigenvectors = linalg.eigh(B)

    nonN, = np.where(eigenvalues > 0)

    if len(nonN) < D:
        raise Exception('Not enough non-zero eigenvalues for Classical MDS !!')

    rank = np.argsort(-1 * eigenvalues)[:D]

    eigenvalues = eigenvalues[rank]
    eigenvectors = eigenvectors[:, rank]

    if wR:
        return eigenvalues, eigenvectors

    return eigenvectors.dot(np.diag(np.sqrt(eigenvalues)))

    # return None


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
    
    dm = torch.tensor([
        [[ 0.0000, 34.0495, 66.4248],
         [34.0495,  0.0000, 89.3443],
         [66.4248, 89.3443,  0.0000]],

        [[ 0.0000, 46.2954, 74.0742],
         [46.2954,  0.0000, 51.9521],
         [74.0742, 51.9521,  0.0000]],

        [[ 0.0000, 39.8987, 58.9082],
         [39.8987,  0.0000, 60.9002],
         [58.9082, 60.9002,  0.0000]],

        [[ 0.0000, 27.3055,  3.9756],
         [27.3055,  0.0000, 24.8017],
         [ 3.9756, 24.8017,  0.0000]],

        [[ 0.0000, 23.4590, 31.4831],
         [23.4590,  0.0000, 37.6982],
         [31.4831, 37.6982, 0.0000]]])[0]
         
    print(dm)
    rs = classicalMDS(dm**2)
    rs = torch.tensor(rs)
    
    print(rs)
    print(get_distance_matrices(rs))
