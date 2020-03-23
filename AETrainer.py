# %%
import torch
from torch import Tensor
import torch.nn as nn
from AutoEncoder import AutoEncoder, DMLoss
from DataGenerator import DataGenerator, get_distance_matrices


class AETrainer():
    def __init__(self, in_dim: int, out_dim: int, LR=1e-5):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dg = DataGenerator((in_dim, out_dim))
        self.pair_loader = None
        self.dm_loader = None
        self._train_init(in_dim=in_dim, out_dim=out_dim, LR=LR)

    def _train_init(self, in_dim: int, out_dim: int, LR: float):
        self.loss_fn = DMLoss()
        self.ae = AutoEncoder(in_dim, out_dim)
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=LR)

    def refresh_loader(self, N: int, batch: int):
        self.pair_loader = self.dg.get_pair_loader(N=N, batch=batch)
        self.dm_loader = self.dg.get_dm_loader(N=N, batch=batch)

    def train(self, EPOCH: int, LR=1e-5, reset=None):
        if reset is not None:
            self._train_init(in_dim=self.in_dim, out_dim=self.out_dim, LR=LR)

        for epoch in range(EPOCH):
            for step, target in enumerate(self.pair_loader):
                rs = self.ae(torch.stack(
                    [get_distance_matrices(t) for t in target]))
                loss = self.loss_fn(rs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 100 == 0:
                    print("Epoch: ", epoch, "| train loss: %.4f" %
                          loss.data.numpy())
