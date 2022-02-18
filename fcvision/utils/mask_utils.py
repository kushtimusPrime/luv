from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import os
from fcvision.cameras.zed import ZedImageCapture
import os.path as osp
import cv2
from PIL import Image
from fcvision.utils.vision_utils import get_hdr_capture, get_high_sat_capture, get_high_sat_img,get_multi_exposure_img
from skimage.morphology import thin
import time

COMMON_THRESHOLDS = {
	'green': [(np.array([65, 100, 80]), np.array([100, 255, 255]))],
	'red': [(np.array([0, 80, 100]), np.array([20, 255, 255])),  
			(np.array([130, 80, 100]), np.array([180, 255, 255]))],
	'data/bright_green_towel': [(np.array([0, 50, 100]), np.array([30, 255, 255])),  
			(np.array([150, 50, 100]), np.array([180, 255, 255]))],
	'data/white_towel':[(np.array([134, 60, 100]), np.array([150, 255, 255]))],
	'data/blue_towel':[(np.array([140, 60, 100]), np.array([180, 255, 255]))],
	'data/green_towel':[(np.array([140, 60, 100]), np.array([180, 255, 255])),(np.array([0, 80, 100]), np.array([20, 255, 255]))],
	'data/bright_green_towel':[(np.array([140, 60, 100]), np.array([180, 255, 255])),(np.array([0, 80, 100]), np.array([20, 255, 255]))],
	'data/yellow_towel':[(np.array([140, 60, 100]), np.array([170, 255, 255]))],
	'data/misc_towels':[(np.array([140, 60, 50]), np.array([180, 255, 255])),(np.array([0, 60, 50]), np.array([20, 255, 255])),(np.array([134, 60, 100]), np.array([150, 255, 255]))],
	'blue': [(np.array([110, 100, 150]), np.array([130, 255, 255]))]
}

def smooth_mask(mask, remove_small_artifacts=False):
	'''
	mask: a binary image
	returns: a filtered segmask smoothing feathered images
	'''
	paintval=np.max(mask)
	contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)#CHAIN_APPROX_TC89_L1,CHAIN_APPROX_TC89_KCOS
	smoothedmask=np.zeros(mask.shape,dtype=mask.dtype)
	cv2.drawContours(smoothedmask, contours, -1, float(paintval), -1)
	smoothedmask=thin(smoothedmask,max_num_iter=1).astype(np.uint8) * paintval


	if remove_small_artifacts:
		nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(smoothedmask, connectivity=8)
		#connectedComponentswithStats yields every seperated component with information on each of them, such as size
		#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
		sizes = stats[1:, -1]; nb_components = nb_components - 1

		# minimum size of particles we want to keep (number of pixels)
		#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
		min_size = 200

		#for every component in the image, you keep it only if it's above min_size
		for i in range(0, nb_components):
			if sizes[i] < min_size:
				smoothedmask[output == i + 1] = 0

	return smoothedmask

def get_mask(im, color_bounds, plot=False,smooth=False):
	hsv = cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
	mask=np.zeros_like(im)[:,:,0]
	for lower1,upper1 in color_bounds:
		partial_mask = cv2.inRange(hsv, lower1, upper1)
		mask=np.maximum(mask,partial_mask)
	if smooth:
		mask = smooth_mask(mask, remove_small_artifacts=True)
	if plot:
		_,axs=plt.subplots(3,1)
		axs[0,0].imshow(im)
		axs[1,0].imshow(hsv)
		axs[2,0].imshow(mask)
		plt.show()
	return mask

def get_segmasks(zed, plug, color_bounds, GAIN, EXPS, plot=True,capture_fn=get_hdr_capture):
	zed.set_gain(GAIN)
	plug.turn_on()
	img_left,img_right=capture_fn(zed,EXPS)
	plug.turn_off()
	hsv_left = cv2.cvtColor(img_left,cv2.COLOR_RGB2HSV)
	hsv_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2HSV)
	mask_left=get_mask(img_left,color_bounds,smooth=False)
	mask_right=get_mask(img_right,color_bounds,smooth=False)
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


def get_rgb(zed:ZedImageCapture,EXP,GAIN):
	zed.set_exposure(EXP)
	zed.set_gain(GAIN)
	time.sleep(.7)
	iml,imr=zed.capture_image()
	return iml,imr