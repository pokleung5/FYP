# %%
import torch

from torch import tensor
from torch import nn, optim
from torch.nn import functional as F

from . import Linear


class aRNN(nn.Module):

    def __init__(self, N, in_dim, output_layer_dim, output_layer=None, num_rnn_layers=1):

        super(aRNN, self).__init__()

        self.N = N
        self.in_lstm = in_dim

        self.lstm = nn.LSTM(
            input_size=self.in_lstm,
            hidden_size=output_layer_dim[0],
            num_layers=num_rnn_layers)

        if output_layer is None:
            self.encoder = Linear.get_Linear_Sequential(
                output_layer_dim, nn.LeakyReLU)
        else:
            self.encoder = output_layer

    def forward(self, x):

        batch = x.size()[0]
        x1 = x.view(batch, -1, self.in_lstm)
        
        x1 = x1.permute(1, 0, 2)
        y1, (hn, cn) = self.lstm(x1)
        y1 = y1.permute(1, 0, 2)

        y2 = self.encoder(y1)

        return y2

# %%


class bRNN(nn.Module):

    def __init__(self, N, in_layer_dim, out_dim, in_layer=None, num_rnn_layers=1):

        super(bRNN, self).__init__()

        if in_layer is None:
            self.encoder = Linear.get_Linear_Sequential(
                in_layer_dim, nn.LeakyReLU)
        else:
            self.encoder = in_layer

        self.N = N
        self.in_lstm = in_layer_dim[-1]

        self.lstm = nn.LSTM(
            input_size=self.in_lstm,
            hidden_size=out_dim,
            num_layers=num_rnn_layers)

    def forward(self, x):
        batch = x.size()[0]

        x1 = self.encoder(x)
        x1 = x1.view(batch, self.N, self.in_lstm)

        x1 = x1.permute(1, 0, 2)
        y1, (hn, cn) = self.lstm(x1)
        y1 = y1.permute(1, 0, 2)

        return y1
