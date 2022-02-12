import os
import os.path as osp
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

from fcvision.model import PlModel
import fcvision.utils.run_utils as ru
from fcvision.utils.arg_utils import parse_yaml


def main():
	cfg, params = parse_yaml(osp.join("cfg", "test_config.yaml"))

	logdir = ru.get_file_prefix(params)
	os.makedirs(os.path.join(logdir, 'lightning_logs'))

	model = params["model"]
	model.logdir = logdir

	dataset_val = params['dataset_val']
	for idx in range(len(dataset_val)):
		im = torch.unsqueeze(dataset_val[idx], 0)
		with torch.no_grad():
			pred = model(im)[0, 0].numpy()


if __name__ == '__main__':
	main()
