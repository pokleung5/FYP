# %%
import torch
import torch.nn as nn
from torch.nn import modules, functional as F
from DataGenerator import get_distance_matrices

# %%

class AutoEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2),
            nn.ReLU(True),
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2),
            nn.Sigmoid(),
            # nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3),
            nn.ReLU(True),
            nn.Linear(in_features=in_dim, out_features=int(out_dim), bias=True),
            # nn.ReLU(True),
            # nn.Linear(in_features=int(out_dim/2), out_features=out_dim, bias=True),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):

        encoded = self.encoder(x)
        # print("encoded: ", encoded.size())
        decoded = self.decoder(encoded)
        # print("decoded: ", decoded.size())

        b, _, n, m = decoded.size()
        rs = decoded.view((b, n, m))

        return get_distance_matrices(rs)
# %%
