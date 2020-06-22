import numpy
import scipy as sp


def landmark_MDS(D, lands, dim):
	Dl = D[:,lands]
	n = len(Dl)

	# Centering matrix
	H = - numpy.ones((n, n))/n
	numpy.fill_diagonal(H,1-1/n)
	# YY^T
	H = -H.dot(Dl**2).dot(H)/2

	# Diagonalize
	evals, evecs = numpy.linalg.eigh(H)

	# Sort by eigenvalue in descending order
	idx   = numpy.argsort(evals)[::-1]
	evals = evals[idx]
	evecs = evecs[:,idx]

	# Compute the coordinates using positive-eigenvalued components only
	w, = numpy.where(evals > 0)
	if dim:
		arr = evals
		w = arr.argsort()[-dim:][::-1]
		if numpy.any(evals[w]<0):
			print('Error: Not enough positive eigenvalues for the selected dim.')
			return []
	if w.size==0:
		print('Error: matrix is negative definite.')
		return []

	V = evecs[:,w]
	L = V.dot(numpy.diag(numpy.sqrt(evals[w]))).T
	N = D.shape[1]
	Lh = V.dot(numpy.diag(1./numpy.sqrt(evals[w]))).T
	Dm = D - numpy.tile(numpy.mean(Dl,axis=1),(N, 1)).T
	dim = w.size
	X = -Lh.dot(Dm)/2.
	X -= numpy.tile(numpy.mean(X,axis=1),(N, 1)).T

	_, evecs = numpy.linalg.eigh(X.dot(X.T))

	return (evecs[:,::-1].T.dot(X)).T



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
    
    dm = utils.get_distance_matrix(pts)[0]
    dm = utils.minmax_norm(dm)[0]
    print(dm)
    
    dm = numpy.array(dm)
    rs = landmark_MDS(dm, [0,1,2,3,4], 2)
    rs = torch.tensor(rs)
    print(rs)

    rs_dm = utils.get_distance_matrix(rs)[0]
    rs_dm = utils.minmax_norm(rs_dm)[0]
    print(rs_dm)
    print(torch.sum(((rs_dm - torch.tensor(dm)) ** 2)))
