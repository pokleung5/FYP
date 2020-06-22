# %%
import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F

from . import Linear


class aRNN(nn.Module):
    def __init__(self, N, output_layer_dim, output_layer=None, num_rnn_layers=1):

        super(aRNN, self).__init__()

        self.lstm = nn.LSTM(
            input_size=N,
            hidden_size=output_layer_dim[0],
            num_layers=num_rnn_layers)

        if output_layer is None:
            self.encoder = Linear.get_Linear_Sequential(
                output_layer_dim, nn.LeakyReLU)
        else:
            self.encoder = output_layer

    def forward(self, x):

        y1, (hn, cn) = self.lstm(x)
        y2 = self.encoder(y1)

        return y2

# %%


class bRNN(nn.Module):
    def __init__(self, N, d, in_layer_dim, in_layer=None, num_rnn_layers=1):

        super(bRNN, self).__init__()

        if in_layer is None:
            self.encoder = Linear.get_Linear_Sequential(
                in_layer_dim, nn.LeakyReLU)
        else:
            self.encoder = in_layer

        self.lstm = nn.LSTM(
            input_size=in_layer_dim[-1],
            hidden_size=d,
            num_layers=num_rnn_layers)

    def forward(self, x):
        x1 = self.encoder(x)

        y1, (hn, cn) = self.lstm(x1)

        return y1
