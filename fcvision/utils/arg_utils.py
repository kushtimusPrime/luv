from argparse import ArgumentParser
import yaml

from fcvision.dataset import build_dataset
from fcvision.losses import build_loss
from fcvision.model import build_PL_model


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
        pl_model = build_PL_model(cfg["train"], train=True, loss=ret["loss"])
        ret["model"] = pl_model
        ret["seed"] = cfg["train"]["seed"]
        ret["loader_n_workers"] = cfg["train"]["loader_n_workers"]
        ret["n_gpus"] = cfg["train"]["n_gpus"]
        ret["epochs"] = cfg["train"]["epochs"]
        ret["batch_size"] = cfg["train"]["batch_size"]
    if "experiment" in cfg:
        ret["experiment"] = cfg["experiment"]
    if "test" in cfg:
        assert "train" not in cfg
        pl_model = build_PL_model(cfg["train"], train=False, loss=ret["loss"], checkpoint=cfg["test"]["checkpoint"])
        ret["model"] = pl_model
        ret["seed"] = cfg["train"]["seed"]
    return cfg, ret
