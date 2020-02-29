#%%
import torch.nn as nn
from torch.nn import modules

class AutoEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Tanh(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        rs = self.encoder(x)
        return rs

# %%
