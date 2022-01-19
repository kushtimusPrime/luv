import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import random

import os
from torchvision.models.segmentation import fcn_resnet50
from torchvision.utils import save_image



class SegmentationLoss(nn.Module):

	def __init__(self):
	    super().__init__()

	def forward(self, preds, targets):
		return F.binary_cross_entropy_with_logits(preds, targets)


class MaskedMSELoss(nn.Module):

  def __init__(self):
    super().__init__()
    self.loss = nn.MSELoss(reduction='none')

  def forward(self, output, target, ignore_mask):
    '''
        output: [N,H,W]
        target: [N,H,W]
        ignore_mask: [N,H,W]
        '''
    valid_sum = torch.sum(torch.logical_not(ignore_mask))
    loss = self.loss(output, target)
    loss[ignore_mask > 0] = 0.0
    return torch.sum(loss) / valid_sum


class KPVectorLoss(nn.Module):

  def __init__(self):
    super().__init__()
    self.vector_loss = MaskedMSELoss()
    self.segmentation_loss = SegmentationLoss()

  def forward(self, output, target, ignore_mask):

  	target_seg, target_vec = None, None # TODO
  	pred_seg, pred_vec = None, None # TODO
    seg_loss = self.segmentation_loss(pred_seg, target_seg)

    ignore_mask = target_seg > 0.1

    vec_loss = self.vector_loss(pred_vec, target_vec, ignore_mask)
    return seg_loss + vec_loss # equal weighting for now

