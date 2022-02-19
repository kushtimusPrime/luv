from functools import partial
from pickletools import stringnl_noescape
import numpy as np
import matplotlib.pyplot as plt
import os
from fcvision.utils.visualization import get_mask_vis
from fcvision.utils.mask_utils import COMMON_THRESHOLDS, smooth_mask,get_mask
import os.path as osp
import cv2
from PIL import Image
from skimage.morphology import thin

import glob
DATA_DIR = 'data/green_towel'
# DATA_DIR = 'data/cable_painted_red_images'
# colors = {
# 	'green': [(np.array([80, 100, 80]), np.array([110, 255, 220]))],
# 	'red': [(np.array([0, 100, 80]), np.array([30, 255, 255])),  (np.array([150, 100, 80]), np.array([180, 255, 255]))],
# 	'blue': [(np.array([110, 100, 150]), np.array([130, 255, 255]))]
# }
# DATA_DIR = 'data/cable_red_painted_images'
# colors = {
# 	'green': [(np.array([65, 100, 80]), np.array([100, 255, 255]))],
# 	'red': [(np.array([0, 110, 100]), np.array([15, 255, 255])),  (np.array([150, 110, 100]), np.array([180, 255, 255]))],
# 	'blue': [(np.array([110, 100, 150]), np.array([130, 255, 255]))]
# }
# DATA_DIR = 'data/cable_blue_painted_images'
# colors = {
# 	'green': [(np.array([80, 100, 80]), np.array([110, 255, 220]))],
# 	'red': [(np.array([0, 90, 80]), np.array([30, 255, 255])),  (np.array([150, 90, 80]), np.array([180, 255, 255]))],
# 	'blue': [(np.array([90, 100, 200]), np.array([130, 255, 255]))]
# }


def create_masks():
	uvimgs = glob.glob(f"{DATA_DIR}/*uv*.png")
	for f in uvimgs:
		uvname = f[f.rfind('/')+1:]
		mask_name = uvname.replace('imagel','maskl').replace('imager','maskr').replace("_uv","")
		uv_im=np.array(Image.open(f"{DATA_DIR}/{uvname}"))
		# mask = np.maximum(get_mask(uv_im,color='green',plot=False), get_mask(uv_im, color='red', plot=False))
		mask = get_mask(uv_im, COMMON_THRESHOLDS[DATA_DIR], plot=False,smooth=True)
		save_file=f"{DATA_DIR}/{mask_name}"
		vis=get_mask_vis(uv_im,mask,channel=2,strength=10)
		# _,axs=plt.subplots(1,2)
		# axs[0].imshow(vis)
		# axs[1].imshow(mask)
		# plt.show()
		# print("WARNING NOT SAVING, UNCOMMENT LINES BELOW")
		print(f"saving {save_file}")
		Image.fromarray(mask).save(save_file)

if __name__=='__main__':
	create_masks()
