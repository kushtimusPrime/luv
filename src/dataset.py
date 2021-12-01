import os
import numpy as np
import os.path as osp
import src.pytorch_utils as ptu


class KPDataset:

    def __init__(self, dataset_dir="data/cable_images_labeled", val=False):
        self.dataset_dir = dataset_dir
        self.datapoints = [f for f in os.listdir(self.dataset_dir) if "image" in f]
        self.val = val
        if self.val:
            self.datapoints = self.datapoints[:10]
        else:
            self.datapoints = self.datapoints[10:]


    def __getitem__(self, idx):
        im_file = self.datapoints[idx]
        im = np.load(osp.join(self.dataset_dir, im_file))
        new_im = np.zeros([3, im.shape[1], im.shape[2]])
        new_im[0] = np.copy(im[0])
        new_im[1:] = im
        im = new_im
        target_file = im_file.replace("image", "target")
        target = np.load(osp.join(self.dataset_dir, target_file))[np.newaxis,:,:]

        if self.val:
            return ptu.torchify(im)
        return ptu.torchify(im, target)


    def __len__(self):
        return len(self.datapoints)
