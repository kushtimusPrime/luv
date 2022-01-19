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

from fcvision.losses import *

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.25):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(tensor.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class PlModel(pl.LightningModule):

    def __init__(self, params=None, logdir=None):
        super().__init__()

        self.save_hyperparameters()
        self.params = params

        if logdir is not None:
            self.vis_dir = os.path.join(logdir, 'vis')
            self.vis_counter = 0
            try:
                os.makedirs(self.vis_dir)
            except FileExistsError:
                pass
        else:
            self.vis_dir = None
            self.vis_counter = None

        self.model = fcn_resnet50(pretrained=False, progress=False, num_classes=1)

        self.loss_fn = SegmentationLoss()

        t1 = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.15)
        t2 = torchvision.transforms.GaussianBlur(9, sigma=(1, 10.0))
        t3 = torchvision.transforms.RandomErasing(p=0.7, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)
        t4 = AddGaussianNoise()
        t5 = None # No transformation
        self.transforms = [t5]
        # self.segmask_head = DPCVSegMaskHead(params)

    def forward(self, img):
        # print(img_left.dtype)
        encoding_dict = self.model(img)
        out = encoding_dict['out']
        return out

    def transform(self, im):
        transform = random.choice(self.transforms)
        if transform is not None:
            img_width = im.shape[2]
            im = transform(im)
        return im

    def training_step(self, batch, batch_idx):
        ims, targets = batch
        ims = self.transform(ims)

        preds = self(ims)

        loss_sm = self.loss_fn(preds, targets)
        self.log('Loss', loss_sm, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_sm

    def validation_step(self, batch, batch_idx):
        ims = batch
        preds = self(ims)
        return {'ims': ims, 'preds': preds}

    def validation_epoch_end(self, outputs):
        if self.vis_dir is None:
            return

        idx = self.vis_counter
        self.vis_counter += 1
        if idx % 10 or not len(outputs):
            return
        outputs = outputs[0]
        for j in range(len(outputs['ims'])):
            im = outputs['ims'][j]
            pred = torch.sigmoid(outputs['preds'][j])
            save_image(im /255, os.path.join(self.vis_dir, '%d_%d_im.png' % (idx, j)))
            save_image(pred, os.path.join(self.vis_dir, '%d_%d_pred.png' % (idx, j)))
            save_image(pred + im/255, os.path.join(self.vis_dir, '%d_%d_overlayed.png' % (idx, j)))


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params['optim_learning_rate'])
        return optimizer

