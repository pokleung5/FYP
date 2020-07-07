import utils
import torch


def add_noise_to_dm(x):

    noise = torch.randn(x.size()) * 0.1
    x = x + torch.abs(noise)
    return utils.minmax_norm(x, dmin=0)[0]


class Preprocess:

    def __init__(self, N, add_noise=False, scale=True, flatten=False):

        self.N = N
        self.add_noise = add_noise
        self.scale = scale
        self.flatten = flatten

    def __call__(self, x):

        if self.add_noise:
            prep_x = add_noise_to_dm(x)

        if self.scale:
            prep_x = utils.minmax_norm(x, dmin=0)[0]

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


class PrepCMDS(Preprocess):

    def __init__(self, N, scale=True):

        super(PrepCMDS, self).__init__(N, False, scale, False)

    def __call__(self, x):

        x = x.view(-1, self.N, self.N)
        ex = utils.eignmatrix2(x)
        ex = ex.view(-1, 1, self.N, self.get_inShape())

        return super(PrepCMDS, self).__call__(ex)


class PrepDist(Preprocess):

    def __init__(self, N, add_noise=False, scale=True):

        super(PrepDist, self).__init__(N, add_noise, scale, False)

    def __call__(self, x):

        v = utils.vectorize_distance_from_DM(x)
        return super(PrepDist, self).__call__(v)[0], x

    def get_inShape(self):

        return int((self.N * self.N - self.N) / 2)


class PrepDC(Preprocess):

    def __init__(self, N, scale=True):

        super(PrepDC, self).__init__(N, False, scale)

    def __call__(self, x):

        v = utils.doubleCentering(x)
        return super(PrepDC, self).__call__(v)


class PrepInvEigen(Preprocess):

    def __init__(self, N, add_noise=False, scale=True):

        super(PrepInvEigen, self).__init__(N, add_noise, scale)

    def __call__(self, x):

        x = x.view(-1, self.N, self.N)
        ex = utils.eignmatrix2(x)
        ex = ex.view(-1, 1, self.N, self.get_inShape())

        return super(PrepInvEigen, self).__call__(x)[0], ex.data

