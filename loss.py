import torch.nn.functional as F
from utils import one_hot_segmentation
import torch.nn as nn
import torch
import math
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as edt

#######  CrossEntropy  ######
class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.Loss = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        if len(target.size()) == 4:
            target = torch.squeeze(target, 1).long()
        return self.Loss(pred, target)

#######  DiceLossSoftmax  ######
class DiceLossSoftmax(nn.Module):
    def __init__(self, smooth=1e-4):
        super().__init__()
        self.smooth = smooth

    def flatten(self, tensor):
        tensor = tensor.transpose(0, 1).contiguous()
        return tensor.view(tensor.size(0), -1)

    def forward(self, pred, target):
        return 1.0 - self.dice_coef(pred, target)

    def dice_coef(self, pred, target):
        n, c = pred.shape[:2]

        pred = F.softmax(pred, dim=1)
        target = one_hot_segmentation(target, c).float()
        pred = pred.view(n, c, -1)
        target = target.view(n, c, -1)

        intersect = torch.sum(target * pred, -1)
        dice = (2 * intersect + self.smooth) / (torch.sum(target, -1) + torch.sum(pred, -1) + self.smooth)
        dice = torch.mean(dice, dim=-1)

        return torch.mean(dice)

#######  ComboLoss  ######
class ComboLoss(nn.Module):
    def __init__(self, weights=[1., 1.]):
        super().__init__()
        self.dice_loss = DiceLossSoftmax()
        self.bce_loss = CrossEntropy()
        self.weights = weights

    def forward(self, pred, target):
        dl = self.dice_loss(pred, target)
        bl = self.bce_loss(pred, target)
        loss = self.weights[0] * dl + self.weights[1] * bl

        return loss

##################  Distance##################
class Distance(nn.Module):

    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha

    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert (pred.dim() == target.dim())

        pred = torch.sigmoid(pred)
        pred_dt = self.distance_field(pred)
        target_dt = self.distance_field(target)

        pred_error = (pred - target) ** 2
        distance = (pred_dt ** self.alpha + target_dt ** self.alpha) / (pred_dt + target_dt) ** self.alpha

        dt_field = (pred_error * distance).cpu()
        loss = torch.mean(dt_field,dim=-1)

        return torch.mean(loss)

################# AreaLoss###########

class AreaLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.distance = Distance()

    def forward(self, pred, target):
        
        intersect = torch.sum(target * pred, dim=-1)
        total = torch.sum(target + pred, dim=-1)
        union = total - intersect             

        loss = self.distance(pred,target) * union

        return torch.mean(loss)
        
####### ComboWithAreaLoss  ######
class ComboWithAreaLoss(nn.Module):

    def __init__(self, weights=[0.5, 0.5]):
        super().__init__()
        self.combo_loss = ComboLoss()
        self.area_loss = AreaLoss()
        self.weights = weights

    def forward(self, pred, target):
        cl = self.combo_loss(pred, target)
        al = self.area_loss(pred, target)
        loss = self.weights[0] * cl + self.weights[1] * al

        return loss