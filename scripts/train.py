import numpy as np
import os
import os.path as osp

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import fcvision.utils.run_utils as ru
from fcvision.utils.arg_utils import parse_yaml



def main():
    cfg, params = parse_yaml(osp.join("cfg", "train_config.yaml"))

    logdir = ru.get_file_prefix(params)
    os.makedirs(os.path.join(logdir, 'lightning_logs'))

    model = params["model"]
    model.logdir = logdir

    if params['seed'] != -1:
        pl.seed_everything(params['seed'])
        np.random.seed(params['seed'])

    batch_size = 2
    loader = DataLoader(params['dataset'], batch_size=params["batch_size"], shuffle=True, num_workers=params['loader_n_workers'])
    loader_val = DataLoader(params['dataset_val'], batch_size=12, num_workers=params['loader_n_workers'])

    trainer = pl.Trainer(
        default_root_dir=logdir,
        gpus=params['n_gpus'],
        max_epochs=params['epochs'],
        callbacks=[ModelCheckpoint(dirpath=os.path.join(logdir, 'models'))]
        # plugins=pl.plugins.training_type.ddp.DDPPlugin(find_unused_parameters=False),
    )
    trainer.fit(model, loader, loader_val)


if __name__ == '__main__':
    main()
