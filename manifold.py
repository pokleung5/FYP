#%%

from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

import mds.lmds as lmds
import mds.fastmap as fmp
import mds.cmds as cmds

import numpy, torch, pandas 
import utils

import trainHelper

from preprocess import * 
from lossFunction import *

from model.Linear import Linear
from model.AutoEncoder import AutoEncoder


init_lr = 1e-3

lossFunMap = {
    "L1": nn.L1Loss(reduction='mean'),
    "MSE": nn.MSELoss(reduction='mean'),
    "SML": sammon_loss,
    "SRL": relative_loss
}

modelMap = {
    "Linear": Linear,
    "AE": AutoEncoder
}

prepMap = {
    "MF": PrepMatrix,
    "M": PrepMatrix,
    "D": PrepDist,
    "E2": PrepEign
}

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
        h.isTraining = False
        self.__deep_model = h.predict


    def make_new_model(self, modelKey, lossFunKey, prepKey, nNeuron, nLayer, minEpoch=0, maxEpoch=9999):

        lfn = lossFunKey.split('|')

        lossList = [
            ReconLoss(lossFun=lossFunMap[lfn[0]]),
        ] if modelKey == 'AE' else [
            CoordsToDMLoss(self.N, lossFun=lossFunMap[lfn[0]])
        ]

        lossFun = MultiLoss(lossFunList=lossList)

        if prepKey == 'MF':
            preprocess = prepMap[prepKey](self.N, flatten=True)
        else:
            preprocess = prepMap[prepKey](self.N)

        in_dim = preprocess.get_inShape()
        out_dim = preprocess.get_outShape(self.d)

        nNeuron = nNeuron + in_dim
        
        dim = [in_dim, int((in_dim + nNeuron) / 2), *([nNeuron] * (nLayer - 2)), int((in_dim + nNeuron) / 2), out_dim]
        model = modelMap[modelKey](dim = dim, activation = nn.LeakyReLU)
        
        if modelKey == 'AE' and prepKey == 'MF':

            recon_dim = int(self.N * (self.N - 1) / 2)
            model = modelMap[modelKey](dim = dim, decode_dim = [
                out_dim, int((out_dim + nNeuron) / 2), *([nNeuron]*(nLayer - 2)), int((recon_dim + nNeuron) / 2), recon_dim
                ])


        model_id = '_'.join([modelKey, prepKey, lossFunKey, str(nLayer), str(nNeuron)])

        newTrainer = trainHelper.TrainHelper(id=model_id, model=model, lossFun=lossFun,
        optimizer=torch.optim.Adam(model.parameters(), init_lr),
        preprocess=preprocess)

        newTrainer.config_fit(minEpoch=minEpoch, maxEpoch=maxEpoch)
        self.__deep_model = newTrainer.predict


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
    
    import dataSource, test
    from sklearn import datasets

    N, d = 10, 2
    sample_size = 1

    test_data = dataSource.generate_rand_DM(N, sample_size, isInt=True,
            sample_space=(1000, 1))

    test_data = utils.minmax_norm(test_data, dmin=0)[0]

    alg = Algorithm(N, d)

    alg.make_new_model(modelKey='Linear', lossFunKey='SML', prepKey='MF',
     nNeuron=72, nLayer=3)

    tester = test.Test(10, 2, 500)

    print(tester.test(cmds.classicalMDS, test_data[0]))
    print(tester.test(alg.deepMDS, test_data))
