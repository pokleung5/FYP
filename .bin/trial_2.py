import torch
from torch import rand
from torch import nn


def getDM(pl: torch.tensor):
    r = torch.mm(pl, pl.t())

    diag = r.diag().unsqueeze(0)
    diag = diag.expand_as(r)

    D = diag + diag.t() - 2 * r
    return D.sqrt()


class DataGenerator():
    def __init__(self, MAX_XY=1, MIN_XY=0):
        self.MAX_XY = MAX_XY
        self.MIN_XY = MIN_XY

        self.lastSize = None
        self.lastPL = None
        self.lastDM = None

    def _getPairs(self, size):
        if size is None:
            raise Exception('Pair size is not specified')

        return rand(size) * (self.MAX_XY - self.MIN_XY) + self.MIN_XY

    def getTrainDataPL(self, size=None, N=0, reset=None):
        if size is None:
            size = self.lastSize
        else:
            self.lastSize = size

        if reset is not None or self.lastPL is None:
            self.lastPL = [self._getPairs(size) for i in range(N)]
            return torch.stack(self.lastPL)
        return self.lastPL

    def getTrainDataDM(self, size=None, N=0, reset=None):
        if size is None:
            PL = self.lastPL
        else:
            PL = self.getTrainDataPL(size, N)
            
        if reset is not None or self.lastDM is None:
            self.lastDM = torch.stack([getDM(ps) for ps in PL])
        return self.lastDM

    
class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = None
        self.decoder = None

        self.updateShape(None, None)

    def _updateShape(self, inShape: int, outShape: int):
        self.inShape = inShape
        self.outShape = outShape

    def update(self, inShape: int, outShape: int, reset=None):
        if reset is not None or self.encoder is None:
            self.updateShape(inShape, outShape)
            self.encoder = nn.Sequential(
                nn.Linear(inShape, outShape),
                nn.Tanh(),
                nn.Linear(outShape, outShape),
                nn.Sigmoid()
            )
            self.decoder = nn.Sequential(
                nn.Linear(outShape, outShape),
                nn.Tanh(),
                nn.Linear(outShape, inShape)
            )
        else:
            if self.inShape != inShape or self.outShape != outShape:
                raise Exception('Please reset the encoder and decoder !!')

    def getInstance(self):
        return AETrainer(self.inShape, self.outShape)


    
class AETrainer():
    def __init__(self, inShape: int, outShape: int):
        self.dg = DataGenerator()
        self.inShape = inShape
        self.outShape = outShape

    def trainDecoder(self, epoch=1, batchSize=1, LR=0.0000005, N=1):
        PL = self.dg.getTrainDataPL(size=(self.inShape, self.outShape), N=N)
        DM = self.dg.getTrainDataDM(PL=PL, N=N)

        # self.train_loader = DataLoader(self.data, batch_size=batchSize, shuffle=True)
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=LR)
        
