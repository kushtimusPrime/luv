import numpy as np
import torch
import matplotlib.pyplot as plt

from fcvision.utils.vision_utils import find_peaks


def build_model_wrapper(cfg, model):
    wrapper_class = globals()[cfg["wrapper_class"]]


def prepare_image(img):
    im = np.copy(img)
    im = np.transpose(im, (2, 0, 1))
    im = ptu.torchify(im, device="cuda")
    im = torch.unsqueeze(im, 0)
    return im


# TODO: clean up this class
class KeypointNetwork:
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


class SegNetwork:
    def __init__(self, model):
        self.model = model

    def __call__(self, img, prep=False):
        if prep:
            img = prepare_image(img)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img))[0, 0].cpu().numpy()
        return pred
