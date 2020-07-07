#%%

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

import mds.lmds as lmds
import mds.fastmap as fmp

import numpy, torch
import utils

class Algorithm:

    def __init__(self, N, d):

        self.N = N
        self.d = d
        
        self.cmds = manifold.MDS(n_components=d, eps=1e-12,
                   dissimilarity="precomputed", n_jobs=1)

        self.nmds = manifold.MDS(n_components=d, metric=False, eps=1e-12,
                    max_iter=3000, 
                    dissimilarity="precomputed", n_jobs=1, n_init=1)

        self.pca = PCA(n_components=d)

    def __norm(self, x):

        # x = x / numpy.sqrt((x ** 2).sum())
        rs = self.pca.fit_transform(x)
        return rs

    def classicalMDS(self, x):

        rs = self.cmds.fit_transform(x)
        rs = self.__norm(rs)
        return rs

    def nonMetricMDS(self, x):

        rs = self.nmds.fit_transform(x)
        rs = self.__norm(rs)
        return rs

    def fastmap(self, x):

        rs = fmp.fastmap(x, self.d)
        rs = self.__norm(rs)
        return rs

    def landmarkMDS(self, x):

        L = int(self.N * 0.5)
        rs = lmds.landmarkMDS(x, L=L, D=self.d)

        while rs is None:
            
            L = L + 1
            rs = lmds.landmarkMDS(x, L=L, D=self.d)

        rs = self.__norm(rs)
        return rs

    def use_pretrained_model(self, filename):
        
        self.deep_model = utils.load_variable(filename)._predict

    def deepMDS(self, x):

        x = torch.tensor(x).view(1, self.N, self.N)

        rs, target = self.deep_model(x)
        rs = rs[-1] if type(rs) is not torch.Tensor else rs

        rs = rs.view(self.N, self.d).detach().numpy()
        rs = self.__norm(rs)
        return rs

#%%
if __name__ == "__main__":
    
    import dataSource

    cus = dataSource.custom_distance(10, 2, 1, isInt=True,
            sample_space=(1000, 1),
            dist_func=lambda a, b, _: torch.sum(torch.abs(a - b)** 2)** 0.5)
    cus = utils.minmax_norm(cus, dmin=0)[0]
    
    skl = Algorithm(10, 2)
    a = numpy.array(cus)[0, 0]
    b1 = skl.classicalMDS(a)
    b2 = skl.nonMetricMDS(a)
    b3 = skl.landmarkMDS(a)
    
    # print(b1, b2, sep='\n')
    skl.use_pretrained_model('result\Coord_StepLinear_E2_3MSE_5_82_400.model')

    b4 = skl.deepMDS(a)
    
    c1 = utils.get_distanceSq_matrix(torch.tensor(b1))** 0.5
    c1 = utils.minmax_norm(c1, dmin=0)[0]
    c1 = c1.detach().numpy()

    c2 = utils.get_distanceSq_matrix(torch.tensor(b2))** 0.5
    c2 = utils.minmax_norm(c2, dmin=0)[0]
    c2 = c2.detach().numpy()

    c3 = utils.get_distanceSq_matrix(torch.tensor(b3))** 0.5
    c3 = utils.minmax_norm(c3, dmin=0)[0]
    c3 = c3.detach().numpy()

    c4 = utils.get_distanceSq_matrix(torch.tensor(b4))** 0.5
    c4 = utils.minmax_norm(c4, dmin=0)[0]
    c4 = c4.detach().numpy()

    print("cmds:", numpy.max(b1), numpy.mean(numpy.abs(a - c1)))
    print("nmds:", numpy.max(b2), numpy.mean(numpy.abs(a - c2)))
    print("lmds:", numpy.max(b3), numpy.mean(numpy.abs(a - c3)))
    print("deep:", numpy.max(b4), numpy.mean(numpy.abs(a - c4)))
