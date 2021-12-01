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

from src.dataset import KPDataset
from src.model import PlModel
from src.arg_utils import parse_args
# from torchvision import models
import src.run_utils as ru
from src.vision_utils import find_peaks
from matplotlib.patches import Circle


def main():
	params = parse_args()
	logdir = ru.get_file_prefix(params)
	os.makedirs(os.path.join(logdir, 'lightning_logs'))
	model = PlModel.load_from_checkpoint(params['checkpoint'], params=params, logdir=logdir).eval()
	dataset_val = KPDataset(val=True)

	for idx in range(len(dataset_val)):
		im = torch.unsqueeze(dataset_val[idx], 0)
		with torch.no_grad():
			pred = torch.sigmoid(model(im))[0, 0].numpy()
		coords = find_peaks(pred)
		print(coords)
		im = im.numpy()[0, 0]
		fig = plt.figure(frameon=False)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		ax.imshow(im, aspect='auto')
		for xx, yy in coords:
			circ = Circle((yy, xx), 10, color='r')
			ax.add_patch(circ)
		plt.savefig(osp.join(logdir, "vis", "pred_%d.jpg"%idx))
		# plt.show()
		plt.clf()


if __name__ == '__main__':
	main()