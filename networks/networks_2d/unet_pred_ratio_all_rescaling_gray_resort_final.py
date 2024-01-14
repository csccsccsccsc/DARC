import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import grad as grad_func
from networks.norm_blocks.IN_MLP_final import IN_MLP_Rescaling4_detach_small as IN_MLP_Rescaling

def make_gn(nc):
    return nn.GroupNorm(nc//8, nc)

def resort(x1, x1_norm):
    b,c,h,w = x1.shape
    ref = torch.argsort(torch.argsort(x1.view(b,c,-1), dim=-1), dim=-1)
    val = torch.sort(x1_norm.view(b,c,-1),dim=-1)[0]
    x1 = torch.gather(val, -1, ref)
    x1 = x1.view(b,c,h,w)
    return x1   

class residual_conv(nn.Module):
    def __init__(self, in_ch, out_ch, norm=IN_MLP_Rescaling, ):
        super(residual_conv, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm(out_ch, ),
            nn.ELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        ])
        self.norm = norm(out_ch, )
        self.act = nn.ELU(inplace=True)
        self.res = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, ori_x, extra=None):
        x = self.conv[0](ori_x)
        x = self.conv[1](x, extra) if extra is not None else self.conv[1](x)
        x = self.conv[3](self.conv[2](x))
        x = self.norm(x+self.res(ori_x), extra) if extra is not None else self.norm(x+self.res(ori_x))
        x = self.act(x)
        return x


class residual_conv_woACT(nn.Module):
    def __init__(self, in_ch, out_ch, norm=IN_MLP_Rescaling, ):
        super(residual_conv_woACT, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            norm(out_ch, ),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        ])
        self.norm = norm(out_ch, )
        self.res = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, ori_x, extra=None):
        x = self.conv[0](ori_x)
        x = self.conv[1](x, extra) if extra is not None else self.conv[1](x)
        x = self.conv[2](x)
        x = self.norm(x+self.res(ori_x), extra) if extra is not None else self.norm(x+self.res(ori_x))
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm=IN_MLP_Rescaling):
        super(down, self).__init__()
        self.pooling_conv = nn.ModuleList([
            nn.MaxPool2d(2),
            residual_conv(in_ch, out_ch, norm=norm)
        ])
    def forward(self, x, extra=None):
        x = self.pooling_conv[0](x)
        x = self.pooling_conv[1](x, extra)
        return x

class up(nn.Module):
    def __init__(self, in_ch_1, in_ch_2, out_ch, bilinear=True, norm=IN_MLP_Rescaling):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch_1, in_ch_2, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch_1, in_ch_2, 2, stride=2)
        self.conv = residual_conv(in_ch_2, out_ch, norm=norm)
    def forward(self, x1, x2, extra=None):
        x1 = self.up(x1)
        _,_,h1,w1 = x1.shape
        _,_,h2,w2 = x2.shape
        if h1<h2 or w1<w2:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                            diffY // 2, diffY - diffY//2))
        return self.conv(x1+x2, extra)

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        return x

class Scaling(nn.Module):
    def __init__(self, num_features, biasic_rate=1):
        super(Scaling, self).__init__()
        self.scale = nn.parameter.Parameter(torch.ones([1, num_features,1,1])*biasic_rate, requires_grad=True)
    def forward(self, x):
        return self.scale*x

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_features=32, n_out=2, norm=IN_MLP_Rescaling, sigmoid_prob=True, rev_aug=False, biasic_rate=1, with_gray=True):
        super(UNet, self).__init__()

        self.n_out = n_out
        self.with_gray = with_gray

        if self.with_gray:
            self.inc_gray2rgb = nn.Sequential(
                residual_conv_woACT(1, n_features, norm=nn.BatchNorm2d),
                nn.Conv2d(n_features, n_channels, kernel_size=3, padding=1),
            )

        self.inc = residual_conv(n_channels, n_features, norm=norm)

        self.down1 = down(n_features, n_features*2, norm=norm)
        self.down2 = down(n_features*2, n_features*4, norm=norm)
        self.down3 = down(n_features*4, n_features*8, norm=norm)
        self.down4 = down(n_features*8, n_features*8, norm=norm)
        self.up1 = up(n_features*8, n_features*8, n_features*8, bilinear=True, norm=nn.BatchNorm2d)
        self.up2 = up(n_features*8, n_features*4, n_features*8, bilinear=True, norm=nn.BatchNorm2d)
        self.up3 = up(n_features*8, n_features*2, n_features*4, bilinear=True, norm=nn.BatchNorm2d)
        self.up4 = up(n_features*4, n_features  , n_features*2, bilinear=True, norm=nn.BatchNorm2d)
        self.out = outconv(n_features*2, n_out)

        # self.ratio_feat_inc_gray2rgb = nn.Linear(n_features, n_features)
        self.ratio_feat_inc = nn.Linear(n_features, n_features)
        self.ratio_feat_down1 = nn.Linear(n_features, n_features*2)
        self.ratio_feat_down2 = nn.Linear(n_features, n_features*4)
        self.ratio_feat_down3 = nn.Linear(n_features, n_features*8)
        self.ratio_feat_down4 = nn.Linear(n_features, n_features*8)
        # self.ratio_feat_up1 = nn.Linear(n_features, n_features*2)
        # self.ratio_feat_up2 = nn.Linear(n_features, n_features*4)
        # self.ratio_feat_up3 = nn.Linear(n_features, n_features*8)
        # self.ratio_feat_up4 = nn.Linear(n_features, n_features*8)
        self.rev_aug = rev_aug

        ### Ratio / Sorting
        self.conv_pred_ratio = nn.Sequential(
            residual_conv(n_features*8, n_features, norm=nn.InstanceNorm2d),
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(n_features, self.n_out, kernel_size=1),
            nn.Sigmoid()
        )
        self.feat_ratio = nn.Sequential(
            Scaling(self.n_out, biasic_rate=biasic_rate),
            nn.Conv2d(self.n_out, n_features, kernel_size=1),
        )

        self.sigmoid = nn.Sigmoid()
        self.sigmoid_prob = sigmoid_prob
        for m in self.modules():
            valid_w = hasattr(m, 'weight')
            valid_b = hasattr(m, 'bias')
            if (valid_w or valid_b) and (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear)):
                if valid_w:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if valid_b and m.bias is not None:
                    nn.init.normal_(m.bias, 0.0, 1.0)
            elif (valid_w or valid_b) and isinstance(m, IN_MLP_Rescaling) or isinstance(m, nn.GroupNorm) or isinstance(m, norm) :
                if valid_w and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if valid_b and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, img, sigmoid_prob=True, prior_ratio=None):
        bsz = img.shape[0]
        with torch.no_grad():
            if img.shape[1] > 4:
                img, gt = torch.split(img, (4, self.n_out), dim=1)
                gt_ratio = gt.mean(dim=(-2,-1)).view(gt.shape[0], self.n_out, 1, 1)
            else:
                gt = None
                gt_ratio = None
            if torch.isnan(img).any() or torch.isinf(img).any():
                assert(False)

            if self.with_gray:
                img, img_gray = torch.split(img, (3, 1), dim=1)
                if self.training and self.rev_aug:
                    img_gray_rev = -img_gray
                    mask = (torch.rand(bsz, 1, 1, 1)>=0.5).float().to(img_gray_rev.device)
                    img_gray = mask * img_gray_rev + (1. - mask) * img_gray
            else:
                img = img[:, :3]

        if prior_ratio is None:
            # pseudo_ratio = torch.zeros([img.shape[0], self.n_out, 1, 1], dtype=img.dtype, device=img.device) # w/o prior distribution
            # pseudo_feat_ratio = self.feat_ratio(pseudo_ratio).view(pseudo_ratio.shape[0], -1)
            x1 = self.inc(img, None)
            x2 = self.down1(x1, None)
            x3 = self.down2(x2, None)
            x4 = self.down3(x3, None)
            x5 = self.down4(x4, None)

            pred_ratio = self.conv_pred_ratio(x5)
            pred_feat_ratio = self.feat_ratio(pred_ratio).view(bsz, -1)

        if gt_ratio is not None and self.training:
            gt_feat_ratio = self.feat_ratio(gt_ratio).view(bsz, -1)
            img = img.repeat(2,1,1,1)
            if self.with_gray:
                img_gray = img_gray.repeat(2,1,1,1)
            cur_feat_ratio = torch.cat((pred_feat_ratio, gt_feat_ratio), dim=0)
        else:
            if prior_ratio is None:
                cur_feat_ratio = pred_feat_ratio
            else:
                cur_feat_ratio = self.feat_ratio(prior_ratio).view(bsz, -1)
                pred_ratio = prior_ratio
                pred_feat_ratio = cur_feat_ratio


        if self.with_gray:
            img_gray = self.inc_gray2rgb(img_gray)
            if self.training:
                img_r = torch.cat((img, img_gray - img_gray.detach() + resort(img, img_gray)), dim=0)
                cur_feat_ratio = cur_feat_ratio.repeat(2, 1)
            else:
                img_r = img_gray - img_gray.detach() + resort(img, img_gray)
        else:
            img_r = img

        x1 = self.inc(img_r, self.ratio_feat_inc(cur_feat_ratio))
        x2 = self.down1(x1, self.ratio_feat_down1(cur_feat_ratio))
        x3 = self.down2(x2, self.ratio_feat_down2(cur_feat_ratio))
        x4 = self.down3(x3, self.ratio_feat_down3(cur_feat_ratio))
        x5 = self.down4(x4, self.ratio_feat_down4(cur_feat_ratio))
        x = self.up1(x5, x4, )
        x = self.up2(x, x3, )
        x = self.up3(x, x2, )
        x = self.up4(x, x1, )
        x = self.out(x)

        if self.sigmoid_prob and sigmoid_prob:
            x = self.sigmoid(x)

        if gt_ratio is not None and self.training:
            return [x, pred_ratio, pred_feat_ratio, gt_feat_ratio]
        else:
            return [x, pred_ratio, pred_feat_ratio, pred_feat_ratio]



def print_model_param_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

import time
if __name__=='__main__':
    model = UNet().cuda()
    print_model_param_nums(model)
    st = time.time()
    for _ in range(100):
        model.forward(torch.rand([1,3,224,224]).cuda())
    et = time.time()
    print((et-st)/100)

    exit()
