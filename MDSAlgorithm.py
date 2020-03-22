import scipy.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def landmarkMDS(distSqM: np.array, L=3, D=2, r=None):
    N = len(distSqM)

    if r is None:
        r = range(L)  # r = rand_distinct(range(N), L)
    else:
        L = len(r)

    dM = distSqM[r, :]
    lmD = dM[:, r]
    
    vals, vects = classicalMDS(lmD, D, wR=True)
    
    Lh = vects.dot(np.diag(1 / np.sqrt(vals))).T
    Dm = distSqM ** 0.5 - np.tile(np.mean(lmD ** 0.5, axis=1), (N, 1)).T

    R = -0.5 * Lh.dot(Dm)
    R = R - np.tile(np.mean(R, axis=1), (N, 1)).T
    _, vects = linalg.eigh(R.dot(R.T))

    return vects[:, ::-1].T.dot(R).T

#%% 

from DataGenerator import DataGenerator
from torch import tensor 

def test_case():

    dg = DataGenerator(dim=(3, 2))
    dms = dg.get_DM_loader(N=6, batch=2)

    for _, target in enumerate(dms):
        for t in target:
            t = np.array(t)

            a = tensor(mds.landmarkMDS(t))
            b = tensor(mds.classicalMDS(t))

            c = get_distance_matrices(a)
            d = get_distance_matrices(b)
            
            print(c, d, tensor(t), sep='\n\n', end='\n-----------------------\n')
