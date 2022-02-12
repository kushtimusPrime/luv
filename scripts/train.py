from json import load
import torchvision.transforms as transforms

import numpy as np
import os

from torch.utils.data import DataLoader
import torch.cuda
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

from fcvision.model import PlModel
from fcvision.arg_utils import parse_args
import fcvision.run_utils as ru



def main():

    params = parse_args()

    logdir = ru.get_file_prefix(params)
    os.makedirs(os.path.join(logdir, 'lightning_logs'))

    if params['seed'] != -1:
        np.random.seed(params['seed'])

    batch_size = params['batch_size']
    loader = DataLoader(params['dataset'], batch_size=batch_size, shuffle=True, num_workers=params['loader_n_workers'],persistent_workers=True)
    loader_val = DataLoader(params['dataset_val'], batch_size=12, num_workers=params['loader_n_workers'],persistent_workers=True)
    print("Amnt of training batches: ",len(loader))
    model = PlModel(params, logdir)

    trainer = pl.Trainer(
        default_root_dir=logdir,
        gpus=params['n_gpus'],
        max_epochs=params['epochs'],
        callbacks=[ModelCheckpoint(dirpath=os.path.join(logdir, 'models'))],auto_lr_find=True
    )
    torch.cuda.empty_cache()
    trainer.fit(model, loader, loader_val)



if __name__ == '__main__':
    main()
