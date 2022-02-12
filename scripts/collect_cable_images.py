from autolab_core import RigidTransform
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import time
import cv2
from PIL import Image

from yumiplanning.yumi_kinematics import YuMiKinematics as YK
from fcvision.phoxi import Phoxi
from fcvision.plug import Plug
from fcvision.zed import ZedImageCapture

N_COLLECT = 100
START_ID = 0
OUTPUT_DIR = "data/iphone_red_painted_images"
RGB_EXP = 100
UV_EXP = 50
UV_GAIN=5
RGB_GAIN=15
RES='2K'

colors = {
	'green': (np.array([80, 100, 80]), np.array([110, 255, 220])),
	'red': (np.array([130, 100, 100]), np.array([180, 230, 230])),
	'blue': (np.array([110, 100, 150]), np.array([130, 255, 255]))
}


def get_rgb(zed):
	iml,imr=zed.capture_image()
	return iml,imr

def get_segmasks(zed, plug, color='red', plot=True):
	zed.set_exposure(UV_EXP)
	zed.set_gain(UV_GAIN)
	plug.turn_on()
	time.sleep(1)
	img_left, img_right, img_depth = zed.capture_image(depth=True)
	plug.turn_off()
	zed.set_exposure(RGB_EXP)
	zed.set_gain(RGB_GAIN)
	hsv_left = cv2.cvtColor(img_left,cv2.COLOR_RGB2HSV)
	hsv_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2HSV)


	lower1, upper1 = colors[color]

	# upper boundary RED color range values; Hue (160 - 180)

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
	return mask_left,mask_right, img_left, img_right, img_depth



if __name__ == '__main__':
	if not osp.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	zed = ZedImageCapture(resolution=RES,exposure=RGB_EXP,gain=RGB_GAIN)
	plug = Plug()
	plug.turn_off()
	idx = START_ID
	while idx < N_COLLECT:
		print(idx)
		input("Press enter when ready to take a new image.")

		plug.turn_off()
		iml, imr = get_rgb(zed)
		# plt.imshow(iml)
		# plt.show()
		ml, mr, iml_uv, imr_uv, imd = get_segmasks(zed, plug, color='red',plot=True)

		# action = input("Enter s to save image, q to discard image.")
		action='s'
		if action == 's':
			Image.fromarray(iml_uv).save(osp.join(OUTPUT_DIR, "imagel_uv_%d.png"%idx))
			Image.fromarray(imr_uv).save(osp.join(OUTPUT_DIR, "imager_uv_%d.png"%idx))
			Image.fromarray(iml).save(osp.join(OUTPUT_DIR, "imagel_%d.png"%idx))
			Image.fromarray(imr).save(osp.join(OUTPUT_DIR, "imager_%d.png"%idx))
			# np.save(osp.join(OUTPUT_DIR, "depth_%d.npy"%idx), imd)
			# Image.fromarray(ml).save(osp.join(OUTPUT_DIR, "maskl_%d.png"%idx))
			# Image.fromarray(mr).save(osp.join(OUTPUT_DIR, "maskr_%d.png"%idx))
			idx += 1
		
