import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].
    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x):
        if (not self.training) or (random.random()) > self.p:
            return x
        mean = x.mean(dim=[2, 3], keepdim=False)
        std = (x.var(dim=[2, 3], keepdim=False) + self.eps).sqrt()
        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)
        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)
        x = (x - mean.reshape(x.shape[0], x.shape[1], 1, 1)) / std.reshape(x.shape[0], x.shape[1], 1, 1)
        x = x * gamma.reshape(x.shape[0], x.shape[1], 1, 1) + beta.reshape(x.shape[0], x.shape[1], 1, 1)
        return x

class DSU_BN(nn.Module):
    def __init__(self, nc, p=0.5, eps=1e-6):
        super(DSU_BN, self).__init__()
        self.dsu = DistributionUncertainty(p=p, eps=eps)
        self.BN = nn.BatchNorm2d(nc)
    def forward(self, x):
        return self.BN(self.dsu(x))

class DSU_IN(nn.Module):
    def __init__(self, nc, p=0.5, eps=1e-6):
        super(DSU_IN, self).__init__()
        self.dsu = DistributionUncertainty(p=p, eps=eps)
        self.IN = nn.InstanceNorm2d(nc)
    def forward(self, x):
        return self.IN(self.dsu(x))
