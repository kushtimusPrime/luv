import numpy as np
import os
import os.path as osp

from torch.utils.data import DataLoader
import torch.cuda
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import wandb
import os
import sys

# Get the current working directory
current_directory = os.getcwd()

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(current_directory):
    for file in files:
        # Check if the file is a Python file and exclude special files (like __init__.py)
        if file.endswith('.py') and not file.startswith('__'):
            # Extract the module name without the extension
            module_name = os.path.splitext(file)[0]
            # Add the module to the list of importable modules
            sys.path.append(root)
            # print(module_name)


import fcvision.utils.run_utils as ru
from fcvision.utils.arg_utils import load_yaml_recursive, parse_yaml,load_yaml

def main():    
    yaml_path=osp.join("cfg", "apps", "diana_test_config.yaml")
    cfg, params = parse_yaml(yaml_path)

    logdir = ru.get_file_prefix(params)
    os.makedirs(os.path.join(logdir, "lightning_logs"))
    print(f"Logdir: {logdir}")
    
    model = params["model"]
    model.set_logdir(logdir)

    if params["seed"] != -1:
        pl.seed_everything(params["seed"])
        np.random.seed(params["seed"])

    loader = DataLoader(
        params["dataset"],
        batch_size=1,
        shuffle=True,
        num_workers=16,
    )

    yaml = load_yaml(yaml_path)
    wandb_config = load_yaml_recursive(yaml)
    wandb.init(project="suture-seg", entity="diana-poplacenel", name=params["experiment"], config=wandb_config)
    wandb_logger = WandbLogger(project="suture-seg", entity="diana-poplacenel", name=params["experiment"], config=wandb_config)

    trainer = pl.Trainer(
        default_root_dir=logdir,
        accelerator="gpu" if True else "cpu",
        # devices=params["n_gpus"],
        # gpus=params["n_gpus"],
        logger=wandb_logger,
        log_every_n_steps=500,
        # strategy="ddp",
        max_epochs=50,
        callbacks=[ModelCheckpoint(dirpath=os.path.join(logdir, "models"))],
        # auto_lr_find=True,
        precision=16,
        # amp_backend="native"
    )
    torch.cuda.empty_cache()

    # model = load_model_from_wandb("suture-seg", "fc-vision")

    # model = wandb_logger.use_model("latest")
    ## training at 7:25 pm 
    # ckpt_path = '/home/kushtimusprime/U-Net/fc-vision/diana-sunset.pt'
    # model.load_state_dict(torch.load(ckpt_path))

    trainer.predict(model, loader)

if __name__ == "__main__":
    # wandb.init(mode = "disabled")
    main()
