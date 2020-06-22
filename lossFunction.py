
import utils

import torch
from torch import Tensor, tensor
from torch import nn
from torch.nn import functional as F

def sammon_loss(result, target):
    
    r = result.view_as(target)
    
    if result.size()[-1] == result.size()[-2]:
        r = utils.vectorize_distance_from_DM(result)
    
    t = utils.vectorize_distance_from_DM(target)
    
    loss = torch.pow(r - t, 2)

    a = torch.div(loss, t)
    a = torch.sum(a)

    b = torch.sum(t)
    c = torch.div(a, b)

    return torch.sum(c)


class CoordsToDMLoss(nn.Module):

    def __init__(self, N, d, lossFun):
        super(CoordsToDMLoss, self).__init__()

        self.N = N
        self.d = d
        self.lossFun = lossFun

    def forward(self, rs, target_dm):

        batch = target_dm.size()[0]

        rs = rs.view(batch, self.N, self.d)
        rs = utils.minmax_norm(rs)[0]

        rs_dm = utils.get_distanceSq_matrix(rs)
        rs_dm = rs_dm.view_as(target_dm)
        
        rs_dist = utils.vectorize_distance_from_DM(rs_dm)
        rs_dist = torch.sqrt(rs_dist)
        
        target_dist = utils.vectorize_distance_from_DM(target_dm)

        return self.lossFun(rs_dist, target_dist)


class ReconLoss(nn.Module):

    def __init__(self, lossFun):

        super(ReconLoss, self).__init__()

        self.lossFun = lossFun

    def forward(self, rs, target_dm):

        rs_dist = F.softplus(rs) # avoid negative distance value
        rs_dist = utils.minmax_norm(dist_vect, dmin=0)[0]
    
        target_dist = utils.vectorize_distance_from_DM(target)

        return self.lossFun(rs_dist, target_dist)


class VAELoss(nn.Module):

    def __init__(self, lossFun=None, reduction='sum'):        
    
        super(VAELoss, self).__init__(lossFun=lossFun)

        if lossFun is None:
            lossFun = nn.BCELoss(reduction=reduction)
        
        self.reduction = reduction
        

    def __call__(self, rs: tuple, target: Tensor): 
        d, mu, logvar = rs
        
        decodeLoss = self.lossFun(d, target)
        
        KLDLoss = -0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        
        if self.reduction == 'mean':
            KLDLoss = KLDLoss / d.size()[0]

        return decodeLoss + KLDLoss
