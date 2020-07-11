#%%

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

import mds.lmds as lmds
import mds.fastmap as fmp
import mds.cmds as cmds

import numpy, torch, pandas 
import utils

from lossFunction import CoordsToDMLoss

class Algorithm:

    def __init__(self, N, d):

        self.N = N
        self.d = d
        
        self.__cmds = manifold.MDS(n_components=d, eps=1e-12,
                   dissimilarity="precomputed", n_jobs=1)

        self.__nmds = manifold.MDS(n_components=d, metric=False, eps=1e-12,
                    max_iter=3000, 
                    dissimilarity="precomputed", n_jobs=1, n_init=1)

        self.__isomap = manifold.Isomap(n_components=d,
                    max_iter=3000, 
                    metric='precomputed')

        self.__TSNE = manifold.TSNE(n_components=d,
                    metric='precomputed')

        self.__pca = PCA(n_components=d)

    def norm(self, x):

        # x = x / numpy.sqrt((x ** 2).sum())
        rs = self.__pca.fit_transform(x)

        rs = rs - numpy.min(rs)
        rs = rs / numpy.max(rs)

        return rs

    def classicalMDS(self, x):

        rs = self.__cmds.fit_transform(x)
        # rs = cmds.classicalMDS(x)
        rs = self.norm(rs)
        return rs

    def nonMetricMDS(self, x):

        rs = self.__nmds.fit_transform(x)
        rs = self.norm(rs)
        return rs

    def fastmap(self, x):

        rs = fmp.fastmap(x, self.d)
        rs = self.norm(rs)
        return rs

    def isomap(self, x):

        rs = self.__isomap.fit_transform(x)
        rs = self.norm(rs)
        return rs

    def tsne(self, x):

        rs = self.__TSNE.fit_transform(x)
        rs = self.norm(rs)
        return rs

    def landmarkMDS(self, x):

        L = int(self.N * 0.5)
        rs = lmds.landmarkMDS(x, L=L, D=self.d)

        while rs is None:
            
            L = L + 1
            rs = lmds.landmarkMDS(x, L=L, D=self.d)

        rs = self.norm(rs)
        return rs

    def use_pretrained_model(self, filename):
        
        h = utils.load_variable(filename)
        h.preprocess.add_noise=False
        self.__deep_model = h._predict

    def deepMDS(self, x):

        if x is not torch.Tensor:
            x = torch.tensor(x)
        
        x = x.view(1, self.N, self.N)

        rs, target = self.__deep_model(x)
        rs = rs[-1] if type(rs) is not torch.Tensor else rs

        rs = rs.view(self.N, self.d).detach().numpy()
        rs = self.norm(rs)
        return rs

#%%
if __name__ == "__main__":
    
    import dataSource
    from sklearn import datasets

    N, d = 10, 2
    batch = 500

    test_data = dataSource.custom_distance(N, d, batch, isInt=True,
            sample_space=(1000, 1),
            dist_func=lambda a, b, _: torch.sum(torch.abs(a - b)** 2)** 0.5)
    
    test_data = dataSource.generate_rand_DM(N, batch, isInt=True,
            sample_space=(1000, 1))

    # digits = datasets.load_digits(n_class=6).data
    # digits = torch.tensor(digits)
    # test_data = utils.get_distanceSq_matrix(digits) ** 0.5

    test_data = utils.minmax_norm(test_data, dmin=0)[0]

    N = test_data.size()[-1]

    skl = Algorithm(N, d)

    method_itr = {
        'classicalMDS': skl.classicalMDS,
        # 'nonMetricMDS': skl.nonMetricMDS,
        'landmarkMDS': skl.landmarkMDS,
        # 'isomap': skl.isomap,
        # 'fastmap': skl.fastmap,
        # 't-SNE': skl.tsne,
        'Deep MDS': skl.deepMDS
    }

    skl.use_pretrained_model('backup/Coord_Linear_AEModel_MSE_2_8_200.model')
    lossFun = CoordsToDMLoss(
            N=N, lossFun=torch.nn.MSELoss(reduction='mean'))

    record = []

    for name, method in method_itr.items():

        rs, time = [], []

        for a in test_data:

            a = a.view(N, N).detach().numpy()
            b1, t = utils.time_measure(method, [a])

            rs.append(torch.tensor(b1))
            time.append(t)

        rs = torch.stack(rs)

        loss = float(lossFun(rs, test_data))
        time = sum(time) / len(time)

        record.append([name, loss, time])
        
    tabulate = pandas.DataFrame(record, columns=['Method', 'Loss', 'Time'])
    print(tabulate.sort_values(['Loss']))