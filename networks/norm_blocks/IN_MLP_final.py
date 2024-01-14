import torch
import torch.nn as nn
import torch.nn.functional as F

class IN_MLP_Rescaling4_detach_small(nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-7,
        affine: bool = False,
    ) -> None:
        super(IN_MLP_Rescaling4_detach_small, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.fc_mean = nn.Sequential(
            nn.Linear(num_features, num_features//4,),
            nn.LayerNorm((num_features//4,)),
            nn.Linear(num_features//4, num_features,),
            nn.Sigmoid(),
        )
        self.fc_std = nn.Sequential(
            nn.Linear(num_features, num_features//4,),
            nn.LayerNorm((num_features//4,)),
            nn.Linear(num_features//4, num_features,),            
            nn.Sigmoid(),
        )
        self.affine = affine
        if self.affine:
            self.alpha = nn.parameter.Parameter(torch.ones([1,num_features,1,1]))
            self.beta = nn.parameter.Parameter(torch.zeros([1,num_features,1,1]))
        self.hist_info = nn.parameter.Parameter(torch.randn([1, num_features,]), requires_grad=False)
        self.hist_info_update_rate = 0.01
    def forward(self, input, extra_info=None):
        bsz = input.shape[0]
        meanv = input.mean(dim=(-2,-1), keepdim=True)
        stdv = input.std(dim=(-2,-1), keepdim=True)
        # ori_m = meanv.data.view(-1)
        # ori_s = stdv.data.view(-1)
        if extra_info is not None:
            meanv = meanv * (self.fc_mean(meanv.detach().view(bsz, -1) + extra_info)+0.5).view(meanv.shape)
            stdv = stdv * (self.fc_std(stdv.detach().view(bsz, -1) + extra_info)+0.5).view(stdv.shape)
            if self.training:
                self.hist_info.data = self.hist_info.data * (1.0-self.hist_info_update_rate) + extra_info.detach().mean(dim=0, keepdim=True)*self.hist_info_update_rate
        else:
            meanv = meanv * (self.fc_mean(meanv.detach().view(bsz, -1) + self.hist_info.data.repeat(bsz, 1))+0.5).view(meanv.shape)
            stdv = stdv * (self.fc_std(stdv.detach().view(bsz, -1) + self.hist_info.data.repeat(bsz, 1))+0.5).view(stdv.shape)
        output = (input - meanv + self.eps) / (stdv + self.eps)
        if self.affine:
            output = output * self.alpha + self.beta
        return output

