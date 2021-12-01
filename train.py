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

    if params['seed'] != -1:
        pl.seed_everything(params['seed'])
        np.random.seed(params['seed'])

    dataset = KPDataset()
    dataset_val = KPDataset(val=True)

    im = dataset[1]
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=params['loader_n_workers'])
    loader_val = DataLoader(dataset_val, batch_size=12, num_workers=params['loader_n_workers'])

    model = PlModel(params, logdir)

    trainer = pl.Trainer(
        default_root_dir=logdir,
        gpus=params['n_gpus'],
        max_epochs=params['epochs'],
        callbacks=[ModelCheckpoint(dirpath=os.path.join(logdir, 'models'))]
        # plugins=pl.plugins.training_type.ddp.DDPPlugin(find_unused_parameters=False),
    )
    trainer.fit(model, loader, loader_val)


    # total_iters = 100000
    # iters = 0
    # while iters < total_iters:
    #     for i_batch, (imgs_left, imgs_right, keypoints) in enumerate(loader):
    #         # print_mem_usage()
    #         if iters > total_iters:
    #             break
    #
    #         # print(i_batch, imgs_right.shape, imgs_left.shape, keypoints.shape)
    #
    #         encoding = stereo_backbone(imgs_left, imgs_right)
    #         keypoints = keypoint_head(encoding)
    #
    #         # print(encoding.shape)
    #
    #         # print(set(encoding))
    #         # for key in encoding:
    #         #    print(key, encoding[key].shape)
    #
    #         # TODO Forward pass through network
    #         # TODO: NOTE: some of the keypoints will be NAN (for datapoints where the needle
    #         #  was oriented such that it couldnt see all of the keypoints)
    #         #  be sure to account for that
    #
    #         # TODO Compute loss/backward pass/gradient step
    #
    #         if iters % 10 == 0:
    #             print(iters)

            # iters += 1


if __name__ == '__main__':
    main()
