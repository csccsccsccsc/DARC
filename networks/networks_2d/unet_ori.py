import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from networks.norm_blocks.dsu import DistributionUncertainty
import math
from networks.amp.amp_layers import AmpNorm
from torch.autograd import grad as grad_func
from networks.networks_2d.SamplingFeatures import SamplingFeatures


def make_gn(nc):
    return nn.GroupNorm(nc//8, nc)


class residual_conv(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, act=nn.ELU):
        super(residual_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm(out_ch),
            act(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.norm = nn.Sequential(
            norm(out_ch),
            act()
        )
        self.res = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x,):
        x = self.norm(self.conv(x) + self.res(x))
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, act=nn.ELU):
        super(inconv, self).__init__()
        self.conv = residual_conv(in_ch, out_ch, norm=norm, act=act)
    def forward(self, x, ):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=nn.BatchNorm2d, act=nn.ELU):
        super(down, self).__init__()
        self.pooling_conv = nn.Sequential(
            nn.MaxPool2d(2),
            residual_conv(in_ch, out_ch, norm=norm, act=act),
        )
    def forward(self, x, ):
        x = self.pooling_conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, bilinear=True, norm=nn.BatchNorm2d, act=nn.ELU):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch_1, in_ch_2, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch_1, in_ch_2, 2, stride=2)
        self.conv = residual_conv(in_ch_2, out_ch, norm=norm, act=act)
    def forward(self, x1, x2,):
        x1 = self.up(x1)
        _,_,h1,w1 = x1.shape
        _,_,h2,w2 = x2.shape
        if h1<h2 or w1<w2:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2))
        return self.conv(x1+x2)

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):

    def __init__(self, n_channels=3, n_features=32, n_out=2, norm=nn.InstanceNorm2d, sigmoid_prob=True, act=nn.ELU):
        super(UNet, self).__init__()

        self.inc = inconv(n_channels, n_features, norm=norm, act=act)
        self.down1 = down(n_features, n_features*2, norm=norm, act=act)
        self.down2 = down(n_features*2, n_features*4, norm=norm, act=act)
        self.down3 = down(n_features*4, n_features*8, norm=norm, act=act)
        self.down4 = down(n_features*8, n_features*8, norm=norm, act=act)
        self.up1 = up(n_features*8, n_features*8, n_features*8, bilinear=True, norm=norm, act=act)
        self.up2 = up(n_features*8, n_features*4, n_features*8, bilinear=True, norm=norm, act=act)
        self.up3 = up(n_features*8, n_features*2, n_features*4, bilinear=True, norm=norm, act=act)
        self.up4 = up(n_features*4, n_features  , n_features*2, bilinear=True, norm=norm, act=act)
        self.out = outconv(n_features*2, n_out)

        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.n_channels = n_channels

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.normal_(m.bias, 0.0, 1.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.InstanceNorm2d) :
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.sigmoid_prob = sigmoid_prob

    def forward(self, img, sigmoid_prob=True, extract_features=False):
        img = img[:, :self.n_channels]
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        dx1 = self.up1(x5, x4)
        dx2 = self.up2(dx1, x3)
        dx3 = self.up3(dx2, x2)
        dx4 = self.up4(dx3, x1)
        x = self.out(dx4)
        if self.sigmoid_prob or sigmoid_prob:
            x = self.sigmoid(x)

        if extract_features:
            return [x, dx4]

        return [x, ]
