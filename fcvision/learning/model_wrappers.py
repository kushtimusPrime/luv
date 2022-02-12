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

from fcvision.learning.dataset import KPDataset
from fcvision.learning.model import PlModel
from fcvision.utils.arg_utils import parse_args
from fcvision.utils.vision_utils import find_peaks, get_cable_mask
from matplotlib.patches import Circle
from fcvision.cameras.phoxi import prepare_phoxi_image_for_net
import fcvision.utils.pytorch_utils as ptu
import time



# TODO: clean up this class
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


    def __call__(self, img, mode='kp', vis=False, prep=True):
        if prep:
            orig_img = img.color._data.copy()
            img = self._prepare_image(img)
        else:
#            img = np.transpose(img, (2, 0, 1))
            img = ptu.torchify(img,device='cuda')
            img = torch.unsqueeze(img, 0)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img))[0].cpu().numpy()
            # plt.imshow(pred)
            # plt.show()
        if mode == 'kp':
            if self.params['num_classes'] == 2:
                coords_list = []
                # TODO: REVERSED OR NOT???
                for cur_class in reversed(range(self.params['num_classes'])):
                    coords_list.append(find_peaks(pred[cur_class])) # do something with these
                    print(find_peaks(pred[cur_class]))

                res = []
                if coords_list[0].shape[0] == 0 or coords_list[1].shape[0] == 0:
                    print("No actual endpoints found")
                else:
                    for i in range(coords_list[0].shape[0]):
                        cur_endpoint = coords_list[0][i]
                        diffs = np.linalg.norm(coords_list[1] - cur_endpoint[None, :], axis=-1)
                        closest_neck = coords_list[1][np.argmin(diffs)]
                        res.append([cur_endpoint, closest_neck])
                coords_list = np.array(res)

                try:
                    coords_list = coords_list.reshape((2, -1, 2)).transpose(1, 0, 2)
                except:
                    coords_list = np.zeros((0, 2, 2))
            else:
                # plt.imshow(pred[0])
                # plt.show()
                coords_list = np.expand_dims(find_peaks(pred[0]), axis=1)
                necks_list = coords_list.copy()
                # each is a Nx2 array of coords, we want Nx2x2
                coords_list = np.concatenate([coords_list, necks_list], axis=1)

            masked_image = get_cable_mask(orig_img)

            # trace along outside of image to see if cable overflows, use these as pseudo endpoints
            xmin, ymin, xmax, ymax = 175, 50, 1010, 540
            pseudo_endpoints = []
            delta = 5
            if np.max(masked_image[ymin, xmin:xmax]) > 0:
                aym = np.argmax(masked_image[ymin, xmin:xmax]) + xmin
                pseudo_endpoints.append(((ymin, aym), (ymin - delta, aym)))
            # if np.max(masked_image[ymax, xmin:xmax]) > 0:
            #     aym = np.argmax(masked_image[ymax, xmin:xmax]) + xmin
            #     pseudo_endpoints.append(((ymax, aym), (ymax + delta, aym)))
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
                plt.clf()
                # plt.scatter(all_endpoints[:, :, 1], all_endpoints[:, :, 0], color='red')
                plt.imshow(masked_image)
                plt.savefig(f"/home/jkerr/yumi/cable-untangling/logs/{mode}_{int(time.time())}_vis.png")

            return coords_list, pseudo_endpoints
        else:
            return pred


class SegNetwork:

    def __init__(self, checkpoint, params=None, logdir=None):
        self.model = PlModel.load_from_checkpoint(checkpoint, params=params, logdir=logdir).cuda().eval()


    def _prepare_image(self, img):
        im = np.copy(img)
        im = np.transpose(im, (2, 0, 1))
        im = ptu.torchify(im, device='cuda')
        im = torch.unsqueeze(im, 0)
        return im


    def __call__(self, img):
        # img = self._prepare_image(img)
        with torch.no_grad():
            pred = torch.sigmoid(self.model(img))[0, 0].cpu().numpy()
        return pred

