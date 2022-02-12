from argparse import ArgumentParser
import yaml

from fcvision.tasks import get_task_parameters
from fcvision.dataset import build_dataset
from fcvision.losses import build_loss
from fcvision.model import build_PL_model

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--exper-name', default=None)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--n-gpus', type=int, default=1)
    parser.add_argument('--loader-n-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--task', type=str, default="cable_endpoints")

    optim_group = parser.add_argument_group("optim")
    optim_group.add_argument("--optim-learning-rate", default=0.002, type=float)

    args = parser.parse_args()
    args = vars(args)
    args = get_task_parameters(args)

    return args


def parse_yaml(fname):
    ret = {}
    with open(fname, 'r') as file:
        cfg = yaml.safe_load(file)

    if "dataset" in cfg:
        dataset = build_dataset(cfg["dataset"])
        ret["dataset"] = dataset
    if "dataset_val" in cfg:
        dataset_val = build_dataset(cfg["dataset_val"])
        ret["dataset_val"] = dataset_val
    if "train" in cfg:
        loss = build_loss(cfg["train"]["loss"])
        ret["loss"] = loss
        pl_model = build_PL_model(cfg["train"], ret["loss"])
        ret["model"] = pl_model
        ret["seed"] = cfg["train"]["seed"]
        ret["loader_n_workers"] = cfg["train"]["loader_n_workers"]
    if "experiment" in cfg:
        ret["experiment"] = cfg["experiment"]

    return cfg, ret
