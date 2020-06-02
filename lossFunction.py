
import utils

import torch
from torch import Tensor, tensor
from torch import nn
from torch.nn import functional as F

class CustomLoss:
    def __init__(self, target_dim, lossFun):
        
        self.lossFun = lossFun
        self.target_dim = target_dim
    
    def __call__(self, rs: Tensor, target: Tensor):
        if torch.sum(rs) == 0:
            return torch.nan

        return self.lossFun(rs, target.view_as(rs))


class CoordsToDMLoss(CustomLoss):
    def __init__(self, target_dim, lossFun):        

        self.lastdm = None        
        super(CoordsToDMLoss, self).__init__(target_dim=target_dim, lossFun=lossFun)

    def __call__(self, rs: Tensor, target: Tensor):
        
        dim = rs.size()
        rs = rs.view(dim[0], *self.target_dim)
        dm, _ = utils.get_distance_matrix(rs)

        self.lastdm = dm

        return super(CoordsToDMLoss, self).__call__(dm, target)


class VAELoss(CustomLoss):

    def __init__(self, target_dim, lossFun=None, reduction='sum'):        
        
        if lossFun is None:
            lossFun = nn.BCELoss(reduction=reduction)
        
        self.reduction = reduction
        super(VAELoss, self).__init__(target_dim=target_dim, lossFun=lossFun)
        
    def __call__(self, rs: tuple, target: Tensor): 
        d, mu, logvar = rs
        
        decodeLoss = self.lossFun(d, target)
        
        KLDLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.reduction == 'mean':
            KLDLoss = KLDLoss / d.size()[0]

        return decodeLoss + KLDLoss
