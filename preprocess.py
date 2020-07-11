import utils
import torch
from torch import Tensor

def add_noise_to_dm(x):

    noise = torch.randn(x.size()) * torch.mean(x) * 0.5
    x = x + torch.abs(noise)
    return utils.minmax_norm(x, dmin=0)[0]


class Preprocess:

    def __init__(self, N, add_noise=False, scale=True, flatten=False):

        self.N = N
        self.add_noise = add_noise
        self.scale = scale
        self.flatten = flatten

    def __call__(self, x):

        prep_x = x

        if self.add_noise:
            prep_x = add_noise_to_dm(prep_x)

        if self.scale:
            prep_x = utils.minmax_norm(prep_x, dmin=0)[0]

        if self.flatten:
            prep_x = prep_x.view(prep_x.size()[0], 1, -1)
        
        prep_x = prep_x.clone().detach().requires_grad_(True)
        return prep_x, x

    def get_inShape(self):

        if self.flatten:
            return self.N * self.N
        
        return self.N


class PrepMatrix(Preprocess):

    def __init__(self, N, scale=True, flatten=False):

        super(PrepMatrix, self).__init__(N, False, scale, flatten)


class PrepEign(Preprocess):

    def __init__(self, N, scale=True):

        super(PrepEign, self).__init__(N, False, scale, False)

    def __call__(self, x):

        x = x.view(-1, self.N, self.N)
        ex = utils.eignmatrix(x)
        ex = ex.view(-1, 1, self.N, self.get_inShape())

        return super(PrepEign, self).__call__(ex)[0], x

    def get_inShape(self):

        return self.N * 2


class PrepPerDist(Preprocess):

    def __init__(self, N, scale=True):

        super(PrepPerDist, self).__init__(N, False, scale, False)

    def __call__(self, x):

        px = x.view(-1, 1, self.N)

        return super(PrepPerDist, self).__call__(px)



class PrepDist(Preprocess):

    def __init__(self, N, add_noise=False, scale=True):

        super(PrepDist, self).__init__(N, add_noise, scale, False)

    def __call__(self, x):

        v = utils.vectorize_distance_from_DM(x)
        return super(PrepDist, self).__call__(v)[0], x

    def get_inShape(self):

        return int((self.N * self.N - self.N) / 2)


class PrepModel(Preprocess):
   
    def __init__(self, outShape: tuple, modelpath: str, index=-1, scale=True):

        super(PrepModel, self).__init__(None, False, scale, False)
        
        self.prepHelper = utils.load_variable(modelpath)
        self.outShape = outShape
        self.index = index

    def __call__(self, x):

        rs, target = self.prepHelper._predict(x)

        rs = rs[self.index] if rs is not Tensor else rs
        rs = rs.view(-1, *self.outShape)

        return super(PrepModel, self).__call__(rs)[0], x

    def get_inShape(self):

        return self.outShape[-1]


# class PrepInvEigen(Preprocess):

#     def __init__(self, N, add_noise=False, scale=True):

#         super(PrepInvEigen, self).__init__(N, add_noise, scale)

#     def __call__(self, x):

#         x = x.view(-1, self.N, self.N)
#         ex = utils.eignmatrix2(x)
#         ex = ex.view(-1, 1, self.N, self.get_inShape())

#         return super(PrepInvEigen, self).__call__(x)[0], ex.data


# class PrepCMDS(Preprocess):

#     def __init__(self, N, scale=True):

#         super(PrepCMDS, self).__init__(N, False, scale, False)

#     def __call__(self, x):

#         x = x.view(-1, self.N, self.N)
#         ex = utils.eignmatrix2(x)
#         ex = ex.view(-1, 1, self.N, self.get_inShape())

#         return super(PrepCMDS, self).__call__(ex)

