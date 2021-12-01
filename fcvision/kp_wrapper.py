import torchvision.transforms as transforms

import numpy as np
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

from fcvision.dataset import KPDataset
from fcvision.model import PlModel
from fcvision.arg_utils import parse_args
# from torchvision import models
import fcvision.run_utils as ru
from fcvision.vision_utils import find_peaks
from matplotlib.patches import Circle
from fcvision.phoxi import prepare_phoxi_image_for_net


class KeypointNetwork:

	def __init__(self, checkpoint, params=None, logdir=None):
		self.model = PlModel.load_from_checkpoint(checkpoint, params=params, logdir=logdir).eval()


	def _prepare_image(self, img):
		return prepare_phoxi_image_for_net(img)


	def __call__(self, img, mode='kp'):
		img = self._prepare_image(img)
		with torch.no_grad():
			pred = torch.sigmoid(self.model(img))[0, 0].numpy()
		if mode == 'kp':
			coords = find_peaks(pred)
			return coords
		else:
			return pred

