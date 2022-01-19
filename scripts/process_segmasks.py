from torchvision.utils import save_image
from tqdm import tqdm
from skimage.morphology import thin

import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import random

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from torchvision import models
# from simnetneedles.vision.mask_smoother import smooth_mask
from PIL import Image
import cv2



def custom_processing(task):
    if task == 'cable_segmasks':
        def crop_segmask(mask, image):
            mask[600:] = 0
            mask[:,:200] = 0
            mask[450:,1000:] = 0
            return mask
        return crop_segmask
    else:
        return lambda x, y: x

def smooth_mask(mask, remove_small_artifacts=False, hsv=None):
    '''
    mask: a binary image
    returns: a filtered segmask smoothing feathered images
    '''
    paintval=np.max(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)#CHAIN_APPROX_TC89_L1,CHAIN_APPROX_TC89_KCOS
    smoothedmask=np.zeros(mask.shape,dtype=mask.dtype)
    cv2.drawContours(smoothedmask, contours, -1, float(paintval), -1)
    smoothedmask=thin(smoothedmask,max_iter=1).astype(np.uint8)



    if remove_small_artifacts:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(smoothedmask, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 200

        #your answer image
        img2 = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if (sizes[i] >= min_size) or (hsv is not None and hsv[output == i + 1][:,1].mean() > 120 and sizes[i] >= 100):
                smoothedmask[output == i + 1] = 255

    return smoothedmask


def get_segmasks(img_left, img_right, plot=False):
    hsv_left=cv2.cvtColor(img_left,cv2.COLOR_RGB2HSV)
    hsv_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2HSV)

    lower1 = np.array([0, 50, 100])
    upper1 = np.array([30, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([150, 50, 100])
    upper2 = np.array([180, 255, 255])

    lower_mask = cv2.inRange(hsv_left, lower1, upper1)
    upper_mask = cv2.inRange(hsv_left, lower2, upper2)
    mask_left = lower_mask + upper_mask

    lower_mask = cv2.inRange(hsv_right, lower1, upper1)
    upper_mask = cv2.inRange(hsv_right, lower2, upper2)
    mask_right = lower_mask + upper_mask

    mask_left = smooth_mask(mask_left, True, hsv_left)
    mask_right = smooth_mask(mask_right, True, hsv_right)

    if plot:
        _,axs=plt.subplots(3,2)
        axs[0,0].imshow(img_left)
        axs[0,1].imshow(img_right)
        axs[1,0].imshow(hsv_left)
        axs[1,1].imshow(hsv_right)
        axs[2,0].imshow(mask_left)
        axs[2,1].imshow(mask_right)
        plt.show()
    return mask_left,mask_right, img_left, img_right


def main():
    data_dir = osp.join("data", "cable_uv_images")
    right_images = [f for f in os.listdir(data_dir) if ("imager_" in f and "uv" not in f)]

    data_outdir = data_dir + "_processed"

    if not osp.exists(data_outdir):
        os.mkdir(data_outdir)
    # random.shuffle(right_images)

    custom_processor = custom_processing("cable_segmasks")

    for r_img_name in tqdm(right_images):
        
        l_img_name = r_img_name.replace("imager", "imagel")
        r_uv_img_name = r_img_name.replace("imager", "imager_uv")
        l_uv_img_name = l_img_name.replace("imagel", "imagel_uv")
        r_mask_name = r_img_name.replace("imager", "maskr")
        l_mask_name = l_img_name.replace("imagel", "maskl")


        r_img = np.array(Image.open(osp.join(data_dir, r_img_name)))
        r_uv_img = np.array(Image.open(osp.join(data_dir, r_uv_img_name)))
        l_uv_img = np.array(Image.open(osp.join(data_dir, l_uv_img_name)))
        l_img = np.array(Image.open(osp.join(data_dir, l_img_name)))
        r_mask = np.array(Image.open(osp.join(data_dir, r_mask_name)))
        l_mask = np.array(Image.open(osp.join(data_dir, l_mask_name)))

        # # l_mask, r_mask, _, _ = get_segmasks(l_uv_img, r_uv_img, False)
        l_mask = smooth_mask(l_mask, remove_small_artifacts=True)
        r_mask = smooth_mask(r_mask, remove_small_artifacts=True)

        l_mask = custom_processor(l_mask, l_uv_img) > 0
        r_mask = custom_processor(r_mask, r_uv_img) > 0


        np.save(osp.join(data_outdir, l_mask_name.replace(".png", ".npy")), l_mask)
        np.save(osp.join(data_outdir, r_mask_name.replace(".png", ".npy")), r_mask)
        Image.fromarray(l_mask).save(osp.join(data_outdir, l_mask_name))
        Image.fromarray(r_mask).save(osp.join(data_outdir, r_mask_name))
        Image.fromarray(r_img/255.).save(osp.join(data_outdir, r_img_name))
        Image.fromarray(l_img/255.).save(osp.join(data_outdir, l_img_name))
        Image.fromarray(l_uv_img).save(osp.join(data_outdir, l_uv_img_name))
        Image.fromarray(r_uv_img).save(osp.join(data_outdir, r_uv_img_name))

        # _,axs=plt.subplots(3,2)
        # # l_img[l_mask>.1,1]=255
        # # r_img[r_mask > .1, 1] = 255
        # axs[0,0].imshow(l_img)
        # axs[0,1].imshow(r_img)
        # axs[1,0].imshow(l_uv_img)
        # axs[1,1].imshow(r_uv_img)
        # axs[2, 0].imshow(l_mask)
        # axs[2, 1].imshow(r_mask)
        # plt.show()


        l_img[l_mask>.1]= 0
        r_img[r_mask > .1] = 0
        l_img[l_mask>.1,1]=255
        r_img[r_mask > .1, 1] = 255
        im = Image.fromarray(l_img)
        im.save(osp.join(data_outdir, l_mask_name.replace(".png", "overlayed.png")))
        im = Image.fromarray(r_img)
        im.save(osp.join(data_outdir, r_mask_name.replace(".png", "overlayed.png")))



if __name__ == '__main__':
    main()