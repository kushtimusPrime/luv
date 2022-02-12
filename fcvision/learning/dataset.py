import os
import numpy as np
import os.path as osp
import fcvision.utils.pytorch_utils as ptu

import torch
import torchvision.transforms.functional as TF
import random


def build_dataset(dataset_cfg):
    dataset_class = globals()[dataset_cfg["name"]]
    dataset_dir = dataset_cfg["dataset_dir"]
    dataset_val = dataset_cfg["val"]
    transform = dataset_cfg["transform"]
    dataset = dataset_class(dataset_dir=dataset_dir, val=dataset_val, transform=transform)
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
        image =  TF.affine(image, angle, translate, scale, [0, 0])
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
    From now now, make sure that dataset_dir has two directories: images and targets.
    Each contains numpy arrays of the form "image_X.npy" and "target_X.npy", where X
    is an integer.
    """

    def __init__(self, dataset_dir, val, transform):
        self.dataset_dir = dataset_dir
        self.val = val
        self.transform = transform

        self.image_fnames = os.listdir(osp.join(self.dataset_dir, "images"))
        if self.val:
            self.image_fnames = self.image_fnames[:10]
        else:
            self.image_fnames = self.image_fnames[10:]
        self.mask_fnames = [f.replace("image", "target") for f in self.image_fnames]



    def __getitem__(self, idx):
        im_file = self.image_fnames[idx]
        target_file = self.mask_fnames[idx]

        im = np.load(osp.join(self.dataset_dir, "images", im_file))
        target = np.load(osp.join(self.dataset_dir, "targets", target_file))

        im = np.transpose(im, (2, 0, 1))

        if len(target.shape) == 2:
            target = target[np.newaxis,:,:]

        if self.transform:
            im, target = target_transforms(im, target)

        if self.val:
            return ptu.torchify(im)
        return ptu.torchify(im, target)


    def __len__(self):
        return len(self.image_fnames)


class CableSegDataset:

    def __init__(self, dataset_dir="data/cable_red_painted_images", val=False):
        self.dataset_dir = dataset_dir
        self.datapoints = [f for f in os.listdir(self.dataset_dir) if "image" in f and "uv" not in f]
        self.val = val
        if self.val:
            self.datapoints = [f for f in self.datapoints if int(f.split(".")[0].split("_")[1]) < 5]
            # self.datapoints = self.datapoints[:10]
        else:
            self.datapoints = [f for f in self.datapoints if int(f.split(".")[0].split("_")[1]) >= 5]
            # self.datapoints = self.datapoints[10:]


    def __getitem__(self, idx):
        im_file = self.datapoints[idx]
        im = np.array(Image.open(osp.join(self.dataset_dir, im_file)))
        # new_im = np.zeros([3, im.shape[1], im.shape[2]])
        target_file = im_file.replace("image", "mask").replace("_", "_full_") # train on full red/green masks
        target = np.array(Image.open(osp.join(self.dataset_dir, target_file)))
        # target = np.load(osp.join(self.dataset_dir, target_file))
        im = np.transpose(im, (2, 0, 1))
        if len(target.shape) == 2:
            target = target[np.newaxis,:,:]
        target[target > 0] = 1.0
        if im.max() > 1.0:
            im = im /255.
        im, target = target_transforms(im, target)

        if self.val:
            return ptu.torchify(im)
        return ptu.torchify(im, target)

    def __len__(self):
        return len(self.datapoints)


