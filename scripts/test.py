import torchvision.transforms as transforms

import numpy as np
import os

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



def main():
	params = parse_args()

	logdir = ru.get_file_prefix(params)
	os.makedirs(os.path.join(logdir, 'lightning_logs'))
	model = PlModel.load_from_checkpoint(params['checkpoint'], params=params, logdir=logdir)



if __name__ == '__main__':
	main()