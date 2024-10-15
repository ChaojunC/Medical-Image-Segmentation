
import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(MSELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.mse_loss(inputs, targets, reduction='mean')

        return BCE

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return BCE

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class Boundary_Loss(nn.Module):
    def __init__(self):
        super(Boundary_Loss, self).__init__()
        self.reference_loss = nn.BCELoss()

    def __call__(self, y_pred, y_true, dist_maps):
        #y_pred = F.softmax(y_pred)

        prediction = y_pred.type(torch.float32)
        dist_map = dist_maps.type(torch.float32)
        #print("prediction.shape", prediction.shape)
        #print("dist_map.shape", dist_map.shape)
        boundary_loss = torch.einsum("bkwh,bkwh->bkwh", prediction, dist_map).mean()
        return self.reference_loss(y_pred, y_true) + 0.01 * boundary_loss


class Boundary_Loss_Modified(nn.Module):
    def __init__(self):
        super(Boundary_Loss_Modified, self).__init__()

    def __call__(self, y_pred, y_true, dist_maps):
        #y_pred = torch.nn.Softmax(y_pred)

        dc = dist_maps.type(torch.float32)
        pc = y_pred.type(torch.float32)

        label_target = torch.where(dc <= 0., -dc.type(torch.double),
                                   torch.tensor(0.0).type(torch.double).to(dc.device)).type(torch.float32)
        label_background = torch.where(dc > 0., dc.type(torch.double),
                                       torch.tensor(0.0).type(torch.double).to(dc.device)).type(
            torch.float32)
        # label_background_unweighted = torch.where(dc > 0, 1, 0).type(torch.float32)
        #print("label_target.shape", label_target.shape)

        #print("pc.shape", pc.shape)
        c_fg = torch.einsum("bcwh,bcwh->bcwh", pc, label_target)
        sum_gt_fg = label_target.sum()
        ic_bg = sum_gt_fg - c_fg.sum()

        ic_fg = torch.einsum("bcwh,bcwh->bcwh", pc, label_background)

        boundary_loss = - ((c_fg.sum() / ((ic_bg + ic_fg.sum()) + 1e-10)).mean())

        return boundary_loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        pred = self._tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()
      neg_weights = torch.pow(1 - gt, 4)

      loss = 0
      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss

class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_off = RegL1Loss()
        self.L_wh =  RegL1Loss()

    def forward(self, pr_decs, gt_batch):
        #print(pr_decs)
        #print(gt_batch['hm'])
        hm_loss  = self.L_hm(pr_decs['hm'],  gt_batch['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        loss_dec = hm_loss + off_loss + wh_loss
        return loss_dec
