import os
import numpy as np
import os.path as osp
import fcvision.utils.pytorch_utils as ptu

import torch
import torchvision.transforms.functional as TF
import random
import cv2


def build_dataset(dataset_cfg):
    dataset_dir = dataset_cfg["dataset_dir"]
    dataset_val = dataset_cfg["val"]
    transform = dataset_cfg["transform"]
    cache = dataset_cfg["cache"] if "cache" in dataset_cfg else False
    print(f"Cache: {cache}")
    grayscale = dataset_cfg["grayscale"] if "grayscale" in dataset_cfg else False
    dataset = FCDataset(
        dataset_dir=dataset_dir, val=dataset_val, transform=transform, cache=cache, grayscale=grayscale
    )
    return dataset


def target_transforms(image, target, grayscale=False):
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

    image = TF.adjust_brightness(image, np.random.uniform(0.5, 1.5))
    image = TF.adjust_contrast(image, np.random.uniform(0.85, 1.15))

    if grayscale:
        image = TF.rgb_to_grayscale(image, 1)

    # if random.random() > 0.8:
    #     image = TF.rgb_to_grayscale(image, 3)

    return image, target


class FCDataset:

    """
    From now on, make sure that dataset_dir has two directories: images and targets.
    Each contain numpy arrays of the form "image_X.npy" and "target_X.npy", where X
    is an integer.
    """

    def __init__(self, dataset_dir, val, transform, test=False, cache=False, grayscale=False):
        self.dataset_dir = dataset_dir
        self.val = val
        self.transform = transform
        self.grayscale = grayscale
        if cache:
            self.cache = {}
        else:
            self.cache = None

        # self.image_fnames = os.listdir(osp.join(self.dataset_dir, "images"))
        if not test:
            self.mask_fnames = os.listdir(osp.join(self.dataset_dir, "targets"))
        # EDITED
        # if changing naming convention, do below 
        if not test:
            self.image_fnames = [f.replace("lb", "im") for f in self.mask_fnames]
        else: 
            self.image_fnames = os.listdir(self.dataset_dir)

        self.test = test
        # self.image_fnames = [f.replace("label", "image") for f in self.mask_fnames]

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        im_file = self.image_fnames[idx]
        if not self.test:
            target_file = self.mask_fnames[idx]

        if self.cache and idx in self.cache and not self.test:
            im, target = self.cache[idx]
        elif self.cache and idx in self.cache:
            im = self.cache[idx]
        else:
            # im = cv2.imread(osp.join(self.dataset_dir, "images", im_file))
            im = cv2.imread(osp.join(self.dataset_dir, im_file))
            # im =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            if not self.test:
                target = cv2.imread(osp.join(self.dataset_dir, "targets", target_file))
            # target = cv2.imread(osp.join(self.dataset_dir, "targets", target_file))[...,0]
         
            if self.cache and not self.test:
                self.cache[idx] = im, target
            elif self.test and self.cache is not None: 
                self.cache[idx] = im 
        # DIANA EDITED: changed 640, 400 -> 256, 256
        im=cv2.resize(im,(640, 400))
        if not self.test:
            target=cv2.resize(target,(640, 400))
        # im = cv2.resize(im, (300, 300)) 
        # target = cv2.resize(target, (300, 300))
        # im = cv2.resize(im, (256, 256)) 
        # target = cv2.resize(target, (256, 256))

        # NOTE: REMOVE BELOW–ONLY FOR 2 CHANNEL IMAGES
        if not self.test: 
            target = np.stack([
                target[..., 0],
                np.logical_or(target[..., 1], target[..., 2]),
                np.zeros_like(target[..., 0])],
                axis=-1)
        
        im = np.transpose(im, (2, 0, 1))
        if im.max() > 1.0:
            im = im / 255.0
        if not self.test and target.max() > 1.0:
            target = target / 255.0
        if not self.test and len(target.shape) == 2:
            target = target[np.newaxis, :, :]
        elif not self.test:
            target = np.transpose(target, (2, 0, 1))

        if self.transform and not self.test:
            im, target = target_transforms(im, target, self.grayscale)
        elif self.grayscale:
            image = torch.Tensor(im)
            im = TF.rgb_to_grayscale(image, 1)

        if not self.test: 
            return ptu.torchify(im, target)
        else: 
            return ptu.torchify(im)

    def __len__(self):
        return len(self.image_fnames)
