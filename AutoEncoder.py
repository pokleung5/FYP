#%%
import torch.nn as nn
from torch.nn import modules

class AutoEncoder(nn.Module):

    def __init__(self, in_shape: int, out_shape: int):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_shape, out_shape),
            nn.Tanh(),
            nn.Linear(out_shape, out_shape),
            nn.Sigmoid()
        )

    def forward(self, x):
        rs = self.encoder(x)
        return rs

# %%
