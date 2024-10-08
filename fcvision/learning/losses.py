import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import random

import os
from torchvision.models.segmentation import fcn_resnet50
from torchvision.utils import save_image


def build_loss(loss_cfg):
    loss_class = globals()[loss_cfg["name"]]
    return loss_class()


class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, pos_weight=None):
        if pos_weight is None:
            return F.binary_cross_entropy_with_logits(preds, targets)
        else:
            return F.binary_cross_entropy_with_logits(preds, targets, pos_weight=pos_weight)


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction="none")

    def forward(self, output, target, ignore_mask):
        """
        output: [N,H,W]
        target: [N,H,W]
        ignore_mask: [N,H,W]
        """
        valid_sum = torch.sum(torch.logical_not(ignore_mask))
        loss = self.loss(output, target)
        loss[ignore_mask > 0] = 0.0
        return torch.sum(loss) / valid_sum


class KPVectorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vector_loss = MaskedMSELoss()
        self.segmentation_loss = SegmentationLoss()

    def forward(self, output, target):

        target_seg, target_vec = target[:, 0], target[:, 1]
        pred_seg, pred_vec = output[:, 0], output[:, 1]
        seg_loss = self.segmentation_loss(pred_seg, target_seg)

        ignore_mask = target_seg > 0.1

        vec_loss = self.vector_loss(pred_vec, target_vec, ignore_mask)
        return seg_loss + vec_loss  # equal weighting for now
