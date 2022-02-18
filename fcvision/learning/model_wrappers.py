import numpy as np
import torch
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

from fcvision.utils.vision_utils import find_peaks
import fcvision.utils.pytorch_utils as ptu


def build_model_wrapper(cfg, model):
    wrapper_class = globals()[cfg["wrapper"]]
    return wrapper_class(model)


def prepare_image(img):
    im = np.copy(img)
    if im.max() > 1:
        im = im / 255
    im = np.transpose(im, (2, 0, 1))
    im = ptu.torchify(im, device="cuda")
    im = torch.unsqueeze(im, 0)
    return im


class ModelWrapper(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, img):
        pass


class KeypointNetwork(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, img, mode="kp", vis=False, prep=False):
        if prep:
            img = prepare_image(img)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img))[0, 0].cpu().numpy()
        if mode == "kp":
            coords_list = find_peaks(pred)
            return coords_list
        else:
            return pred


class SegNetwork(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, img, prep=False):
        if prep:
            img = prepare_image(img)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img))[0, 0].cpu().numpy()
        return pred
