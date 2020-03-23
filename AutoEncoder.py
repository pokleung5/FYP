# %%
import torch
import torch.nn as nn
from torch.nn import modules


class DMLoss(nn.Module):
    
    def __init__(self):
        super(DMLoss, self).__init__()

    def forward(self, x, target):
        return torch.sum(x - target)** 2

        batch = x.size()[0] if len(x.size()) > 2 else 1
        m = x

        if batch > 2:
            m = sum(m.matmul(m.permute(0, 2, 1)))

        diag = m.diag().unsqueeze(0).expand_as(m)
        rs = (m * -2 + diag + diag.t()) ** 0.5

        return torch.sum((rs - target) ** 2)


class AutoEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        rs = self.encoder(x)
        return rs

# %%
