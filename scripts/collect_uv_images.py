from autolab_core import RigidTransform
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import time
import cv2
from PIL import Image

from fcvision.phoxi import Phoxi
from fcvision.plug import Plug
from fcvision.zed import ZedImageCapture

N_COLLECT = 1000
START_ID = 0
OUTPUT_DIR = "data/cable_uv_images"

colors = {
    'green': (np.array([60 - 30, 150, 40]), np.array([60 + 30, 255, 255])),
    'red': (np.array([0, 180, 40]), np.array([255, 255, 255])),
    'blue': (np.array([0, 180, 40]), np.array([255, 255, 255]))
}


def get_rgb(zed):
    zed.set_exposure(80)
    time.sleep(1)
    iml,imr=zed.capture_image()
    return iml,imr



def get_segmasks(zed, plug, plot=True):
    zed.set_exposure(30)
	plug.turn_on()
    time.sleep(1)
    img_left, img_right = zed.capture_image()
	plug.turn_off()
    img_left = img_left[:,:,::-1]
    img_right = img_right[:,:,::-1]
    # plt.imshow(img_left[:,:,::-1]); plt.show(); assert 0
    hsv_left = cv2.cvtColor(img_left,cv2.COLOR_RGB2HSV)
    hsv_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2HSV)


    lower1 = np.array([95, 50, 100])
    upper1 = np.array([130, 200, 255])
    # upper boundary RED color range values; Hue (160 - 180)
    lower2 = np.array([95, 200, 0])
    upper2 = np.array([130, 255, 100])

    lower_mask = cv2.inRange(hsv_left, lower1, upper1)
    # upper_mask = cv2.inRange(hsv_left, lower2, upper2)
    mask_left = lower_mask

    lower_mask = cv2.inRange(hsv_right, lower1, upper1)
    # upper_mask = cv2.inRange(hsv_right, lower2, upper2)
    mask_right = lower_mask



    # mask_left = remove_small_blobs(mask_left)
    # mask_right = remove_small_blobs(mask_right)


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


if __name__ == '__main__':

	if not osp.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	zed = ZedImageCapture()
	plug = Plug()

	for idx in range(START_ID, N_COLLECT):
		input("Press enter when ready to take a new image.")
		plug.turn_off()
		iml, imr = get_rgb(zed)
		# plt.imshow(iml); plt.show()
		# plt.imshow(imr); plt.show()
		ml, mr, iml_uv, imr_uv = get_segmasks(zed, plug)
		# plt.imshow(img[:,:,0]); plt.show()


		Image.fromarray(iml_uv).save(osp.join(OUTPUT_DIR, "imagel_uv_%d.png"%idx))
		Image.fromarray(imr_uv).save(osp.join(OUTPUT_DIR, "imager_uv_%d.png"%idx))
		Image.fromarray(iml).save(osp.join(OUTPUT_DIR, "imagel_%d.png"%idx))
		Image.fromarray(imr).save(osp.join(OUTPUT_DIR, "imager_%d.png"%idx))
		Image.fromarray(ml).save(osp.join(OUTPUT_DIR, "maskl_%d.png"%idx))
		Image.fromarray(mr).save(osp.join(OUTPUT_DIR, "maskr_%d.png"%idx))
