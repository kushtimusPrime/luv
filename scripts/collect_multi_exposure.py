from autolab_core import RigidTransform
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import time
import cv2
from PIL import Image
from scipy import ndimage

from fcvision.phoxi import Phoxi
from fcvision.plug import Plug
from fcvision.zed import ZedImageCapture

N_COLLECT = 1000
START_ID = 0
OUTPUT_DIR = "data/cable_multi_test"

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


def get_uv_images(zed, plug):
	plug.turn_on()
	right_ims = []
	left_ims = []
	for exposure in [5, 10]:
		zed.set_exposure(exposure)
		time.sleep(0.25)
		img_left, img_right, img_depth = zed.capture_image(depth=True)
		# Image.fromarray(img_left).save(osp.join(OUTPUT_DIR, "imagel_uv_%d.png"%exposure))
		# Image.fromarray(img_right).save(osp.join(OUTPUT_DIR, "imager_uv_%d.png"%exposure))
		right_ims.append(img_right)
		left_ims.append(img_left)
	plug.turn_off()
	return left_ims, right_ims


def dist_to_mask(im):
	dist = ndimage.distance_transform_edt(1 - im)
	return dist


def get_segmasks(im5, im10, color='blue', plot=False):




	# im = np.array(Image.open(osp.join(OUTPUT_DIR, "imagel_uv_5.png")))
	im = im5
	imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(imgray, 50, 255, 0)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	a = np.zeros_like(im)
	cv2.drawContours(a, contours, -1, (0,255,0), 3)
	a = a.sum(2)
	contour_mask = a
	# print(a.sum())
	# plt.imshow(a); plt.show()
	# assert 0

	# img_left = np.array(Image.open(osp.join(OUTPUT_DIR, "imagel_uv_5.png")))
	hsv_left = cv2.cvtColor(im5,cv2.COLOR_BGR2HSV)
	lower1, upper1 = np.array([100, 150, 150]), np.array([120, 255, 255])
	lower_mask = cv2.inRange(np.copy(hsv_left), lower1, upper1)
	dist = dist_to_mask(lower_mask == 255)
	dist_mask = dist < 20

	_,axs=plt.subplots(3,3)

	axs[0,0].imshow(hsv_left)
	axs[0,1].imshow(lower_mask)



	# img_left = np.array(Image.open(osp.join(OUTPUT_DIR, "imagel_uv_10.png")))
	hsv_left = cv2.cvtColor(im10,cv2.COLOR_BGR2HSV)
	lower1, upper1 = np.array([110, 100, 100]), np.array([135, 255, 255])
	lower_mask = cv2.inRange(np.copy(hsv_left), lower1, upper1)
	cur_mask = np.multiply(dist_mask, lower_mask) + lower_mask

	for i in range(10):

		dist = dist_to_mask(cur_mask == 255)
		dist_mask = dist < 20
		lower1, upper1 = np.array([100, 100, 100]), np.array([135, 255, 255])
		lower_mask = cv2.inRange(np.copy(hsv_left), lower1, upper1)
		cur_mask = np.multiply(dist_mask, lower_mask)
	cur_mask = np.multiply(cur_mask, contour_mask)

	cur_mask[650:] = 0




	axs[2,0].imshow(im5)
	axs[2,1].imshow(hsv_left)
	axs[2,2].imshow(cur_mask)
	axs[1,0].imshow(contour_mask)
	# axs[1,1].imshow(dist_mask)
	# axs[1,2].imshow(hsv_left)
	# axs[2,0].imshow(cur_mask)
	plt.show()




if __name__ == '__main__':

	if not osp.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	zed = ZedImageCapture()
	plug = Plug()

	idx = START_ID
	while idx < N_COLLECT:
		print(idx)
		input("Press enter when ready to take a new image.")

		plug.turn_off()
		iml, imr = get_rgb(zed)

		left_ims, right_ims = get_uv_images(zed, plug)
		get_segmasks(left_ims[0], left_ims[1], plug)


		# plt.imshow(iml); plt.show()
		# plt.imshow(imr); plt.show()
		# ml, mr, iml_uv, imr_uv, imd = get_segmasks(zed, plug, color='red', plot=False)
		# plt.imshow(img[:,:,0]); plt.show()

		# action = input("Enter s to save image, q to discard image.")
		# if action == 's':
		# Image.fromarray(iml_uv).save(osp.join(OUTPUT_DIR, "imagel_uv_%d.png"%idx))
		# Image.fromarray(imr_uv).save(osp.join(OUTPUT_DIR, "imager_uv_%d.png"%idx))
		# Image.fromarray(iml).save(osp.join(OUTPUT_DIR, "imagel_%d.png"%idx))
		# Image.fromarray(imr).save(osp.join(OUTPUT_DIR, "imager_%d.png"%idx))
		# np.save(osp.join(OUTPUT_DIR, "image_depth_%d.npy"%idx), imd)
		# Image.fromarray(ml).save(osp.join(OUTPUT_DIR, "maskl_%d.png"%idx))
		# Image.fromarray(mr).save(osp.join(OUTPUT_DIR, "maskr_%d.png"%idx))
		# idx += 1
