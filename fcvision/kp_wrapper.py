from numpy.ma.core import masked
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
from fcvision.vision_utils import find_peaks, get_cable_mask
from matplotlib.patches import Circle
from fcvision.phoxi import prepare_phoxi_image_for_net
import fcvision.pytorch_utils as ptu

class KeypointNetwork:

	def __init__(self, checkpoint, params=None, logdir=None):
		self.params = params
		self.model = PlModel.load_from_checkpoint(checkpoint, params=params, logdir=logdir).cuda().eval()


	def _prepare_image(self, img):
		cpy = img.copy()
		cpy._data[550:] = 0
		cpy._data[:, :180] = 0
		return prepare_phoxi_image_for_net(cpy)


	def __call__(self, img, mode='kp'):
		orig_img = img.color._data.copy()
		img = self._prepare_image(img)
		with torch.no_grad():
			pred = torch.sigmoid(self.model(img))[0, 0].cpu().numpy()
			# plt.imshow(pred)
			# plt.show()
		if mode == 'kp':
			coords = find_peaks(pred)
			masked_image = get_cable_mask(orig_img)

			plt.imshow(img.cpu().numpy().squeeze().transpose(1, 2, 0))
			for coord in coords:
				plt.scatter(coord[1], coord[0], color='red')
			plt.show()

			# if we can't find any real endpoints, trace along outside of image to see if cable overflows, use these as endpoints
			xmin, ymin, xmax, ymax = 175, 0, 1010, 550
			pseudo_endpoints = []
			if np.max(masked_image[ymin]) > 0:
				pseudo_endpoints.append((ymin, np.argmax(masked_image[ymin])))
			if np.max(masked_image[ymax]) > 0:
				pseudo_endpoints.append((ymax, np.argmax(masked_image[ymax])))
			if np.max(masked_image[:, xmin]) > 0:
				pseudo_endpoints.append((np.argmax(masked_image[:, xmin]), xmin))
			if np.max(masked_image[:, xmax]) > 0:
				pseudo_endpoints.append((np.argmax(masked_image[:, xmax]), xmax))
			print(f"Filled in {len(pseudo_endpoints)} endpoints from end of image.")
			pseudo_endpoints = np.array(pseudo_endpoints)
			return coords, pseudo_endpoints
		else:
			return pred


class SegNetwork:

	def __init__(self, checkpoint, params=None, logdir=None):
		self.model = PlModel.load_from_checkpoint(checkpoint, params=params, logdir=logdir).cuda().eval()


	def _prepare_image(self, img):
		im = np.copy(img)
		im = np.transpose(im, (2, 0, 1))
		im = ptu.torchify(im,device='cuda')
		im = torch.unsqueeze(im, 0)
		return im


	def __call__(self, img, mode='kp'):
		img = self._prepare_image(img)
		with torch.no_grad():
			pred = torch.sigmoid(self.model(img))[0, 0].cpu().numpy()
			# plt.imshow(img.squeeze().permute(1, 2, 0).numpy())
			# plt.show()
			# plt.imshow(pred)
			# plt.show()
		if mode == 'kp':
			coords = find_peaks(pred)

			masked_image = get_cable_mask(orig_img)
			plt.imshow(masked_image)
			plt.show()

			if coords == []:
				# if we can't find any real endpoints, trace along outside of image to see if cable overflows, use these as endpoints
				if np.max(masked_image[50]) > 0:
					coords += [(50, np.argmax(masked_image[50]))]
				if np.max(masked_image[640]) > 0:
					coords += [(640, np.argmax(masked_image[640]))]
				if np.max(masked_image[:, 150]) > 0:
					coords += [(np.argmax(masked_image[:, 150]), 0)]
				if np.max(masked_image[:, 940]) > 0:
					coords += [(np.argmax(masked_image[:, 940]), 940)]

			return coords
		else:
			return pred

