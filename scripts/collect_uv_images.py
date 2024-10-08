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
OUTPUT_DIR = "data/cable_uv_endpoint_images"

colors = {
	'green': (np.array([60 - 30, 150, 40]), np.array([60 + 30, 255, 255])),
	'red': (np.array([0, 50, 150]), np.array([20, 255, 255])),
	'blue': (np.array([110, 100, 150]), np.array([130, 255, 255]))
}


def get_rgb(zed):
	zed.set_exposure(80)
	time.sleep(1)
	iml,imr=zed.capture_image()
	return iml,imr



def get_segmasks(zed, plug, color='blue', plot=True):
	zed.set_exposure(10)
	plug.turn_on()
	time.sleep(1)
	img_left, img_right, img_depth = zed.capture_image(depth=True)
	plug.turn_off()
	img_left = img_left[:,:,::-1]
	img_right = img_right[:,:,::-1]
	hsv_left = cv2.cvtColor(img_left,cv2.COLOR_RGB2HSV)
	hsv_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2HSV)


	lower1, upper1 = colors[color]

	lower_mask = cv2.inRange(hsv_left, lower1, upper1)
	mask_left = lower_mask

	lower_mask = cv2.inRange(hsv_right, lower1, upper1)
	mask_right = lower_mask

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
    cfg, params = parse_yaml(osp.join("cfg", "apps", "uv_data_collection.yaml"))

	zed = params["camera"]
	plug = params["plug"]

	idx = START_ID
	while idx < N_COLLECT:
		print(idx)
		input("Press enter when ready to take a new image.")

		plug.turn_off()
		iml, imr = get_rgb(zed)
		# plt.imshow(iml); plt.show()
		# plt.imshow(imr); plt.show()
		ml, mr, iml_uv, imr_uv, imd = get_segmasks(zed, plug, color='red', plot=False)

		action = input("Enter s to save image, q to discard image.")
			if action == 's':
				Image.fromarray(iml_uv).save(osp.join(OUTPUT_DIR, "imagel_uv_%d.png"%idx))
				Image.fromarray(imr_uv).save(osp.join(OUTPUT_DIR, "imager_uv_%d.png"%idx))
				Image.fromarray(iml).save(osp.join(OUTPUT_DIR, "imagel_%d.png"%idx))
				Image.fromarray(imr).save(osp.join(OUTPUT_DIR, "imager_%d.png"%idx))
				np.save(osp.join(OUTPUT_DIR, "image_depth_%d.npy"%idx), imd)
				Image.fromarray(ml).save(osp.join(OUTPUT_DIR, "maskl_%d.png"%idx))
				Image.fromarray(mr).save(osp.join(OUTPUT_DIR, "maskr_%d.png"%idx))
				idx += 1
