import os
import numpy as np
import os.path as osp
import fcvision.utils.pytorch_utils as ptu

import torch
import torchvision.transforms.functional as TF
import random


def build_dataset(dataset_cfg):
    dataset_dir = dataset_cfg["dataset_dir"]
    dataset_val = dataset_cfg["val"]
    transform = dataset_cfg["transform"]
    cache = dataset_cfg["cache"] if "cache" in dataset_cfg else False
    dataset = FCDataset(
        dataset_dir=dataset_dir, val=dataset_val, transform=transform, cache=cache
    )
    return dataset


def target_transforms(image, target):
    image = torch.Tensor(image)
    target = torch.Tensor(target)
    if random.random() > 0.8:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        target = TF.rotate(target, angle)
    if random.random() > 0.8:
        angle = random.randint(0, 90)
        translate = list(np.random.uniform(0.1, 0.3, 2))
        scale = np.random.uniform(0.75, 0.99)
        image = TF.affine(image, angle, translate, scale, [0, 0])
        target = TF.affine(target, angle, translate, scale, [0, 0])
    if random.random() > 0.8:
        image = TF.hflip(image)
        target = TF.hflip(target)
    if random.random() > 0.8:
        image = TF.vflip(image)
        target = TF.vflip(target)

    # image = image + np.random.uniform(0, 0.1, image.shape)
    image = TF.adjust_brightness(image, np.random.uniform(0.5, 1.5))
    image = TF.adjust_contrast(image, np.random.uniform(0.85, 1.15))

    return image, target


class FCDataset:

    """
    From now on, make sure that dataset_dir has two directories: images and targets.
    Each contain numpy arrays of the form "image_X.npy" and "target_X.npy", where X
    is an integer.
    """

    def __init__(self, dataset_dir, val, transform, cache=False):
        self.dataset_dir = dataset_dir
        self.val = val
        self.transform = transform
        if cache:
            self.cache = {}
        else:
            self.cache = None

        self.image_fnames = os.listdir(osp.join(self.dataset_dir, "images"))
        if self.val:
            self.image_fnames = self.image_fnames[:10]
        else:
            self.image_fnames = self.image_fnames[10:]
        self.mask_fnames = [f.replace("image", "target") for f in self.image_fnames]

    def __getitem__(self, idx):
        im_file = self.image_fnames[idx]
        target_file = self.mask_fnames[idx]

        if self.cache and idx in self.cache:
            im, target = self.cache[idx]
        else:
            im = np.load(osp.join(self.dataset_dir, "images", im_file))
            target = np.load(osp.join(self.dataset_dir, "targets", target_file))
            self.cache[idx] = im, target

        im = np.transpose(im, (2, 0, 1))
        if im.max() > 1.0:
            im = im / 255.0
        if target.max() > 1.0:
            target = target / 255.0

        if len(target.shape) == 2:
            target = target[np.newaxis, :, :]

        if self.transform:
            im, target = target_transforms(im, target)

        if self.val:
            return ptu.torchify(im)
        return ptu.torchify(im, target)

    def __len__(self):
        return len(self.image_fnames)
