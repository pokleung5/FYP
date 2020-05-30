
import torch
from torch import tensor

def realMSELoss(d1: tensor, d2: tensor, scaling):
    scaling = scaling ** 2
    
    d1 = (d1 * scaling) ** 0.5
    d2 = (d2 * scaling) ** 0.5

    return torch.mean((d2 - d1) ** 2)