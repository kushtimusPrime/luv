from argparse import ArgumentParser
import yaml

from fcvision.learning.dataset import build_dataset
from fcvision.learning.losses import build_loss
from fcvision.learning.model import build_PL_model


def parse_yaml(fname):
    ret = {}
    with open(fname, "r") as file:
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
        pl_model = build_PL_model(
            cfg["test"], train=False, loss=None, checkpoint=cfg["test"]["checkpoint"]
        )
        ret["model"] = pl_model
        ret["seed"] = cfg["test"]["seed"]
        if "wrapper" in cfg["test"]:
            ret["model"] = build_model_wrapper(cfg["test"], pl_model)
    return cfg, ret
