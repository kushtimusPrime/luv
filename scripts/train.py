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

def load_model_from_wandb(entity, project, model_name='model'):
    api = wandb.Api()
    runs = api.runs(project, order='-created_at')
    latest_run = next(iter(runs), next)
    # run = api.run(f"{entity}/{project}/{run_id}")
    run_file = latest_run.file(f"{model_name}.h5")  # Adjust the file name if needed
    model_content = run_file.download(replace=True)
    loaded_model = torch.load(model_content.name)  # Load the model
    return loaded_model

import fcvision.utils.run_utils as ru
from fcvision.utils.arg_utils import load_yaml_recursive, parse_yaml,load_yaml

def main():    
    wandb.finish()

    yaml_path=osp.join("cfg", "apps", "prime_train_config.yaml")
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
    wandb.init(project="suture-seg", name=params["experiment"], config=wandb_config)
    wandb_logger = WandbLogger(project="suture-seg", name=params["experiment"], config=wandb_config)
    print(f"GPUS: {params['n_gpus']}")
    trainer = pl.Trainer(
        default_root_dir=logdir,
        accelerator="gpu" if params["n_gpus"] > 0 else "cpu",
        devices=params["n_gpus"],
        logger=wandb_logger,
        log_every_n_steps=500,
        strategy="ddp",
        max_epochs=params["epochs"],
        callbacks=[ModelCheckpoint(dirpath=os.path.join(logdir, "models"))],
        precision=16,
    )
    torch.cuda.empty_cache()

    # model = load_model_from_wandb("suture-seg", "needle-combo")

    # model = wandb_logger.use_model("latest")
    # training at 7:25 pm 
    # ckpt_path = '/home/kushtimusprime/U-Net/fc-vision/diana_04_16.pt'
    # print(model)
    # model.load_state_dict(torch.load(ckpt_path))
    # print(f"---------BRK--------")
    # print(model)
    trainer.fit(model, loader, loader_val)

if __name__ == "__main__":
    # wandb.init(mode = "disabled")
    main()
