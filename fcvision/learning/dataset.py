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
    print(f"grayscale: {grayscale}")
    dataset = FCDataset(
        dataset_dir=dataset_dir, val=dataset_val, transform=transform, cache=cache, grayscale=grayscale
    )
    return dataset


def target_transforms(image, target, grayscale=False):
    print(f"Targeting transforms!")
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

    def __init__(self, dataset_dir, val, transform, cache=False, grayscale=False):
        self.dataset_dir = dataset_dir
        self.val = val
        self.transform = transform
        self.grayscale = grayscale
        print(f"cache: {cache}")
        if cache:
            self.cache = {}
        else:
            self.cache = None
        images_folder = os.path.join(self.dataset_dir, 'images')
        targets_folder = os.path.join(self.dataset_dir, 'targets')
        # self.image_fnames = os.listdir(osp.join(self.dataset_dir, "images"))
        self.image_fnames = sorted(os.listdir(images_folder))
        self.mask_fnames = sorted(os.listdir(targets_folder))

    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        im_file = self.image_fnames[idx]
        target_file = self.mask_fnames[idx]
        
        if self.cache and idx in self.cache:
            im, target = self.cache[idx]
        else:
            print(osp.join(self.dataset_dir, "images", im_file))
            im = cv2.imread(osp.join(self.dataset_dir, "images", im_file))
            
            im =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            target = cv2.imread(osp.join(self.dataset_dir, "targets", target_file))[...,0]
            ret,target = cv2.threshold(target,127,255,cv2.THRESH_BINARY)
         
            if self.cache:
                self.cache[idx] = im, target

        im=cv2.resize(im,(960,600))
        target=cv2.resize(target,(960,600))
        
        im = np.transpose(im, (2, 0, 1))
        if im.max() > 1.0:
            im = im / 255.0
        if target.max() > 1.0:
            target = target / 255.0
        if len(target.shape) == 2:
            target = target[np.newaxis, :, :]
        else:
            target = np.transpose(target, (2, 0, 1))

        if self.transform:
            im, target = target_transforms(im, target)

        return ptu.torchify(im, target)

    def __len__(self):
        return len(self.image_fnames)
