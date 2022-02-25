import numpy as np
import os
import os.path as osp

from torch.utils.data import DataLoader
import torch.cuda
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt


import fcvision.utils.run_utils as ru
from fcvision.utils.arg_utils import load_yaml_recursive, parse_yaml,load_yaml


def main():
    yaml_path=osp.join("cfg", "apps", "train_config.yaml")
    cfg, params = parse_yaml(yaml_path)

    logdir = ru.get_file_prefix(params)
    os.makedirs(os.path.join(logdir, "lightning_logs"))

    model = params["model"]
    model.set_logdir(logdir)

    if params["seed"] != -1:
        pl.seed_everything(params["seed"])
        np.random.seed(params["seed"])


    loader = DataLoader(
        params["dataset"],
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["loader_n_workers"],
    )
    loader_val = DataLoader(
        params["dataset_val"],
        batch_size=12,
        num_workers=params["loader_n_workers"],
    )

    yaml = load_yaml(yaml_path)
    wandb_config = load_yaml_recursive(yaml)
    wandb_logger = WandbLogger(project="all-you-need-is-luv", entity="luv", name=params["experiment"],config=wandb_config)

    trainer = pl.Trainer(
        default_root_dir=logdir,
        gpus=params["n_gpus"],
        logger=wandb_logger,
        # strategy="ddp",
        max_epochs=params["epochs"],
        callbacks=[ModelCheckpoint(dirpath=os.path.join(logdir, "models"))],
        auto_lr_find=True,
        precision=16,
        amp_backend="native"
    )
    torch.cuda.empty_cache()
    trainer.fit(model, loader, loader_val)


if __name__ == "__main__":
    main()
