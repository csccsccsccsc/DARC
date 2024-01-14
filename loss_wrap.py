import torch
import torch.nn.functional as F
import torch.nn as nn
import math

def binary_focal_loss(pred,
                      target,
                      gamma=2.0,
                      alpha=0.25,
                      reduction='mean',
                      if_sigmoid=False):
    if if_sigmoid:
        pred = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(pred, target, reduction='none') * focal_weight
    if reduction == 'none':
        return loss
    elif reduction == 'avg' or reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementError('NOT supported reduction={}'.format(reduction))

def dice_binary(pred, target, eps=1e-5, if_sigmoid=False):
    if if_sigmoid:
        pred = F.sigmoid(pred)
    b, c = pred.shape[0], pred.shape[1]
    pflat = pred.contiguous().view(b, c, -1)
    tflat = target.float().contiguous().view(b, c, -1)
    intersection = (pflat * tflat).sum(dim=-1)
    dice = ((2. * intersection + eps) / (pflat.sum(dim=-1) + tflat.sum(dim=-1) + eps)).mean()
    return dice


def dice_score_binary(pred, target, eps=1e-5, if_sigmoid=False):
    if if_sigmoid:
        pred = F.sigmoid(pred)
    pred = (pred>=0.5).float()
    b, c = pred.shape[0], pred.shape[1]
    pflat = pred.contiguous().view(b, c, -1)
    tflat = target.float().contiguous().view(b, c, -1)
    intersection = (pflat * tflat).sum(dim=-1)
    dice = ((2. * intersection + eps) / (pflat.sum(dim=-1) + tflat.sum(dim=-1) + eps)).mean()
    return dice

# targets: seg, domain_label, domain_confidence

class DomainClsLoss(nn.Module):
    def __init__(self, domain_tgt_loc=1):
        super(DomainClsLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.domain_tgt_loc = domain_tgt_loc
        self.GP = nn.AdaptiveAvgPool2d(1)
    def forward(self, predictions, targets, **kwargs):
        b = predictions[0].shape[0]
        p = self.GP(predictions[0])
        loss = self.loss(p, targets[self.domain_tgt_loc].view(b, 1, 1))
        # print(torch.argmax(predictions[0], dim=1).view(-1).data.cpu().numpy(), targets[self.domain_tgt_loc].view(-1).data.cpu().numpy())
        acc = torch.mean((torch.argmax(p, dim=1).view(-1) == targets[self.domain_tgt_loc].view(-1)).float())
        return loss, {'cls_loss': loss, 'acc': acc, 'n1':torch.mean((targets[self.domain_tgt_loc]==1).float())}


class CntLoss(nn.Module):
    def __init__(self, domain_tgt_loc=1):
        super(CntLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.domain_tgt_loc = domain_tgt_loc
        self.GP = nn.AdaptiveAvgPool2d(1)
    def forward(self, predictions, targets, **kwargs):
        b = predictions[0].shape[0]
        p = self.GP(predictions[0])
        loss = self.loss(p, targets[self.domain_tgt_loc].view(b, 1, 1, 1))
        return loss, {'mse_loss': loss, }

class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
    def forward(self, predictions, targets, **kwargs):
        rec = predictions[0]
        if isinstance(rec, (list, tuple)):
            l1loss = 0.0
            for irec in rec:
                l1loss += self.l1_loss(irec, targets[-1])
        else: 
            l1loss = self.l1_loss(rec, targets[-1])

        return l1loss, {'l1loss': l1loss, }


class SegLoss(nn.Module):
    def __init__(self, all=False):
        super(SegLoss, self).__init__()
        self.all = all
    def forward(self, predictions, targets, **kwargs):
        seg_pred = predictions[0]
        seg_target = targets[0]
        b = seg_target.shape[0]
        if self.all:
            b2 = seg_pred.shape[0]
            assert(b2%b==0)
            n_aug = b2//b
            segloss = 0.0
            for i_aug in range(n_aug):
                segloss += F.binary_cross_entropy(seg_pred[i_aug*b:(i_aug+1)*b], seg_target)
            segloss /= n_aug
        else:
            segloss = F.binary_cross_entropy(seg_pred[0:b], seg_target)

        return segloss, {'segloss': segloss, }

class SegLossDSCRatio(nn.Module):
    def __init__(self, scaling=[1.,1.,1.,1e-3], eps=1.0):
        super(SegLossDSCRatio, self).__init__()
        self.scaling = scaling
        self.eps = eps
    def forward(self, predictions, targets, **kwargs):
        seg_pred = predictions[0]
        ratios_pred = predictions[1]
        ratios_feat_pred = predictions[2]
        ratios_feat_gt = predictions[3]
        seg_target = targets[0]
        b0 = seg_pred.shape[0]
        b = seg_target.shape[0]
        assert(b0 % b == 0)
        segloss = 0.
        segdiceloss = 0.
        n_batch = b0 / b
        for _i in range(int(n_batch)):
            segloss += F.binary_cross_entropy(seg_pred[_i*b:(_i+1)*b], seg_target)
            segdiceloss += 1.0 - dice_binary(seg_pred[_i*b:(_i+1)*b], seg_target)
        segloss /= n_batch
        segdiceloss /= n_batch
        ratio_target = torch.mean(seg_target, dim=(-2, -1), keepdim=True)
        ratioloss_map = F.binary_cross_entropy(ratios_pred, ratio_target, reduction='none')
        cmp_map = (ratios_pred>=ratio_target).float()
        ratioloss = ((ratioloss_map*cmp_map).sum() + self.eps) / (cmp_map.sum() + self.eps) \
                  + ((ratioloss_map*(1-cmp_map)).sum() + self.eps) / ((1-cmp_map).sum() + self.eps)
        # ratioloss = F.binary_cross_entropy(ratios_pred, ratio_target)
        ratiofeat_mseloss = F.mse_loss(ratios_feat_pred, ratios_feat_gt)

        return segloss*self.scaling[0] + segdiceloss*self.scaling[1] + ratioloss*self.scaling[2] + ratiofeat_mseloss*self.scaling[3], \
            {
                'segloss':segloss, 'segdiceloss':segdiceloss, 'ratioloss':ratioloss, \
                'ratiofeat_mseloss':ratiofeat_mseloss, 
            }
