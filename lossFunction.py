
import utils

import torch
from torch import Tensor, tensor
from torch import nn
from torch.nn import functional as F


def sammon_loss(result, target):
    
    r = result.view_as(target)
    
    if result.size()[-1] == result.size()[-2]:
        result = utils.vectorize_distance_from_DM(result)

    if target.size()[-1] == target.size()[-2]:
        target = utils.vectorize_distance_from_DM(target)
    
    loss = torch.pow(result - target, 2)

    a = torch.div(loss, target)
    a = torch.sum(a, 2)

    b = torch.sum(target, 2)
    c = torch.div(a, b)

    return torch.sum(c)


class CustomLoss(nn.Module):

    def __init__(self, lossFun: torch.nn.modules.loss._Loss, scale=True):
        
        super(CustomLoss, self).__init__()
        self.lossFun = lossFun
        self.scale = scale
    
    def forward(self, rs, target):

        rs = utils.minmax_norm(rs, dmin=0)[0]
        target = utils.minmax_norm(target, dmin=0)[0]

        return self.lossFun(rs, target)
        


class MultiLoss(nn.Module):

    def __init__(self, lossFunList: list, weight=None):
        
        super(MultiLoss, self).__init__()

        self.lossFunList = lossFunList
        self.weight = torch.ones(len(lossFunList)) if weight is None else torch.tensor(weight)

    def forward(self, rs, target):

        rs = list(rs) if type(rs) is tuple else rs if type(rs) is list else [rs] 

        loss = []
        
        for i in range(len(self.lossFunList)):

            if self.lossFunList[i] is not None:
                loss.append(self.lossFunList[i](rs[i], target) * self.weight[i])
        
        loss = torch.stack(loss)
        loss = torch.abs(loss)
        
        return torch.sum(loss)


class CoordsToDMLoss(CustomLoss):

    def __init__(self, N, lossFun):

        super(CoordsToDMLoss, self).__init__(lossFun)
        self.N = N

    def forward(self, rs, target):

        batch = target.size()[0]

        rs = rs.view(batch, self.N, -1)

        rs_dm = utils.get_distanceSq_matrix(rs)
        rs_dm = rs_dm.view_as(target)
        
        rs_dist = utils.vectorize_distance_from_DM(rs_dm)
        rs_dist = torch.sqrt(rs_dist + 1e-16)
        
        target_dist = utils.vectorize_distance_from_DM(target)

        return super(CoordsToDMLoss, self).forward(rs_dist, target_dist)


class ReconLoss(CustomLoss):

    def __init__(self, lossFun):

        super(ReconLoss, self).__init__(lossFun)

    def forward(self, rs, target):

        rs_dist = F.softplus(rs) # avoid negative distance value    
        target_dist = utils.vectorize_distance_from_DM(target)

        return super(ReconLoss, self).forward(rs_dist, target_dist)


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
