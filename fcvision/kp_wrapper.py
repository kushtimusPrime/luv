from numpy.ma.core import masked
import torchvision.transforms as transforms

import numpy as np
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np

from fcvision.dataset import KPDataset
from fcvision.model import PlModel
from fcvision.arg_utils import parse_args
import fcvision.run_utils as ru
from fcvision.vision_utils import find_peaks, get_cable_mask
from matplotlib.patches import Circle
from fcvision.phoxi import prepare_phoxi_image_for_net
import fcvision.pytorch_utils as ptu
from fcvision.tasks import get_task_parameters


class KeypointNetwork:

    def __init__(self, checkpoint, params=None, logdir=None):
        ### for just endpoint prediction, use task = "cable_endpoints",
        ### for endpoint + direction prediction, use "cable_kp_vecs"
        if params is not None:
            self.params = get_task_parameters(params)
        else:
            self.params = params
        self.model = PlModel.load_from_checkpoint(checkpoint, params=params, logdir=logdir).cuda().eval()


    def _prepare_image(self, img):
        cpy = img.copy()
        cpy._data[550:] = 0
        cpy._data[:, :180] = 0
        return prepare_phoxi_image_for_net(cpy)


    def __call__(self, img, mode='kp', vis=True, prep=True):
        if prep:
            orig_img = img.color._data.copy()
            img = self._prepare_image(img)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img))[0].cpu().numpy()
            # plt.imshow(pred)
            # plt.show()
        if mode == 'kp':
            coords_list = []
            for cur_class in range(self.params['num_classes']):
                coords_list.append(find_peaks(pred[cur_class])) # do something with these

            # TODO: rearrange second list to match first in cartesian coordinates
            reversed_dirs = coords_list[1][::-1]
            reg_dist = np.linalg.norm(coords_list[0] - coords_list[1], axis=1).sum()
            rev_dist = np.linalg.norm(coords_list[0] - reversed_dirs, axis=1).sum()
            if rev_dist < reg_dist:
                coords_list[1] = reversed_dirs
            coords_list = np.array(coords_list)
            print(coords_list.shape)
            coords_list = coords_list.reshape((2, -1, 2)).transpose(1, 0, 2)

            masked_image = get_cable_mask(orig_img)

            # plt.imshow(img.cpu().numpy().squeeze().transpose(1, 2, 0))
            # for coord in coords:
            #     plt.scatter(coord[1], coord[0], color='red')
            # plt.show()

            # trace along outside of image to see if cable overflows, use these as pseudo endpoints
            xmin, ymin, xmax, ymax = 175, 50, 1010, 550
            pseudo_endpoints = []
            delta = 5
            if np.max(masked_image[ymin, xmin:xmax]) > 0:
                aym = np.argmax(masked_image[ymin, xmin:xmax]) + xmin
                pseudo_endpoints.append(((ymin, aym), (ymin - delta, aym)))
            if np.max(masked_image[ymax, xmin:xmax]) > 0:
                aym = np.argmax(masked_image[ymax, xmin:xmax]) + xmin
                pseudo_endpoints.append(((ymax, aym), (ymax + delta, aym)))
            if np.max(masked_image[ymin:ymax, xmin]) > 0:
                axm = np.argmax(masked_image[ymin:ymax, xmin]) + ymin
                pseudo_endpoints.append(((axm, xmin), (axm, xmin - delta)))
            if np.max(masked_image[ymin:ymax, xmax]) > 0:
                axm = np.argmax(masked_image[ymin:ymax, xmax]) + ymin
                pseudo_endpoints.append(((axm, xmax), (axm, xmax + delta)))
            print(f"Filled in {(pseudo_endpoints)} endpoints from end of image.")
            pseudo_endpoints = np.array(pseudo_endpoints)

            # coords_list = coords_list.reshape((-1, self.params['num_classes'], 2))
            pseudo_endpoints = pseudo_endpoints.reshape((-1, 2, 2))

            if vis:
                print(coords_list.shape, pseudo_endpoints.shape)
                all_endpoints = np.concatenate((coords_list, pseudo_endpoints), axis=0)
                plt.scatter(all_endpoints[:, :, 1], all_endpoints[:, :, 0], color='red')
                plt.imshow(masked_image)
                plt.show()

            return coords_list, pseudo_endpoints
        else:
            return pred
class SegNetwork:

    def __init__(self, checkpoint, params=None, logdir=None):
        self.model = PlModel.load_from_checkpoint(checkpoint, params=params, logdir=logdir).cuda().eval()


    def _prepare_image(self, img):
        im = np.copy(img)
        im = np.transpose(im, (2, 0, 1))
        im = ptu.torchify(im,device='cuda')
        im = torch.unsqueeze(im, 0)
        return im


    def __call__(self, img, mode='kp'):
        img = self._prepare_image(img)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img))[0, 0].cpu().numpy()
            # plt.imshow(img.squeeze().permute(1, 2, 0).numpy())
            # plt.show()
            # plt.imshow(pred)
            # plt.show()
        if mode == 'kp':
            coords = find_peaks(pred)

            masked_image = get_cable_mask(orig_img)
            plt.imshow(masked_image)
            plt.show()

            if coords == []:
                # if we can't find any real endpoints, trace along outside of image to see if cable overflows, use these as endpoints
                if np.max(masked_image[50]) > 0:
                    coords += [(50, np.argmax(masked_image[50]))]
                if np.max(masked_image[640]) > 0:
                    coords += [(640, np.argmax(masked_image[640]))]
                if np.max(masked_image[:, 150]) > 0:
                    coords += [(np.argmax(masked_image[:, 150]), 0)]
                if np.max(masked_image[:, 940]) > 0:
                    coords += [(np.argmax(masked_image[:, 940]), 940)]

            return coords
        else:
            return pred

