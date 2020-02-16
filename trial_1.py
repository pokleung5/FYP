import torch
from torch import rand
from torch import nn
from torch.nn import modules
# from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader


def getDM(pairlist: torch.tensor):
    r = torch.mm(pairlist, pairlist.t())

    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)

    D = diag + diag.t() - 2 * r
    return D.sqrt()


class DataGenerator():

    def __init__(self, MAX_XY=1, MIN_XY=0):
        self.MAX_XY = MAX_XY
        self.MIN_XY = MIN_XY

    def _getPairs(self, size: tuple):
        return rand(size) * (self.MAX_XY - self.MIN_XY) + self.MIN_XY

    def getTrainDataDM(self, size: tuple, N: int):
        self.lastPs = [self._getPairs(size) for i in range(N)]
        self.lastDM = torch.stack([getDM(ps) for ps in self.lastPs])
        return self.lastDM

    def getTrainDataPairs(self, size: tuple, N: int):
        self.lastPs = [self._getPairs(size) for i in range(N)]
        return torch.stack(self.lastPs)


class DMLoss(nn.Module):
    def __init__(self):
        super(DMLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.stack([getDM(r) for r in x]) - y
        return torch.sum(diff ** 2) ** 0.5


class AutoEncoder(nn.Module):
    def __init__(self, in_shape: int, out_shape: int):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_shape, out_shape),
            nn.Tanh(),
            nn.Linear(out_shape, out_shape),
            nn.Sigmoid()
        )

    def decoder(self, pairlist: torch.tensor):
        d, n, m = pairlist.size()
        dat  = pairlist.transpose(1, 2).contiguous().view(d * m, -1).transpose(0, 1)
        return getDM(dat)

    def forward(self, x):
        rs = self.encoder(x)
        dm = self.decoder(rs)
        print(x, rs, dm, sep='\n')
        return rs, dm

    
class Main():

    def __init__(self, size: tuple, MAX_XY=100, MIN_XY=1):
        self.setConfig(size, MAX_XY, MIN_XY)
        self.MSE = nn.MSELoss()

    def setConfig(self, size: tuple, MAX_XY: int, MIN_XY: int):
        self.SIZE = size
        self.dg = DataGenerator(MAX_XY, MIN_XY)

    def test(self, EPOCH=1, BATCH_SIZE=1, LR=0.000005, N=1, DM=True):
        if N > 0:
            if DM:
                self.data = self.dg.getTrainDataDM(self.SIZE, N)
            else:
                self.data = self.dg.getTrainDataPairs(self.SIZE, N)

            self.train_loader = DataLoader(self.data, batch_size=BATCH_SIZE, shuffle=True)

            self.ae = AutoEncoder(self.SIZE[0], 2)
            self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=LR)

        # try:
        for epoch in range(EPOCH):
            for step, target in enumerate(self.train_loader):

                self.e, self.rs = self.ae(target)
                self.loss = self.MSE(self.rs, target)

                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                if step % 100 == 0:
                    print("Epoch: ", epoch, "| train loss: %.4f" % self.loss.data.numpy())

        # except NameError:
        #     print("Please specify the number of data !!")


