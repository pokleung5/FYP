import torch
import torch.nn as nn
from AutoEncoder import AutoEncoder
from DataGenerator import DataGenerator

class DMLoss(nn.Module):
    def __init__(self):
        super(DMLoss, self).__init__()

    def forward(self, x, target):
        batch = x.size()[0] if len(x.size()) > 2 else 1
        m = x
        
        if batch > 2:
            m = sum(m.matmul(m.permute(0, 2, 1)))
        
        diag = m.diag().unsqueeze(0).expand_as(m)
        rs = sqrt(m * -2 + diag + diag.t())

        return sum((rs - target) ** 2)

# class AETrainer():
#     def __init__(self, dim):
        