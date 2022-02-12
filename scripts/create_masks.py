from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import cv2
from PIL import Image
from skimage.morphology import thin

import glob

# DATA_DIR = 'data/cable_painted_red_images'
# colors = {
# 	'green': [(np.array([80, 100, 80]), np.array([110, 255, 220]))],
# 	'red': [(np.array([0, 100, 80]), np.array([30, 255, 255])),  (np.array([150, 100, 80]), np.array([180, 255, 255]))],
# 	'blue': [(np.array([110, 100, 150]), np.array([130, 255, 255]))]
# }
DATA_DIR = 'data/cable_red_painted_images'
colors = {
	'green': [(np.array([65, 100, 80]), np.array([100, 255, 255]))],
	'red': [(np.array([0, 110, 100]), np.array([15, 255, 255])),  (np.array([150, 110, 100]), np.array([180, 255, 255]))],
	'blue': [(np.array([110, 100, 150]), np.array([130, 255, 255]))]
}
# DATA_DIR = 'data/cable_blue_painted_images'
# colors = {
# 	'green': [(np.array([80, 100, 80]), np.array([110, 255, 220]))],
# 	'red': [(np.array([0, 90, 80]), np.array([30, 255, 255])),  (np.array([150, 90, 80]), np.array([180, 255, 255]))],
# 	'blue': [(np.array([90, 100, 200]), np.array([130, 255, 255]))]
# }


def smooth_mask(mask, remove_small_artifacts=False, hsv=None):
    '''
    mask: a binary image
    returns: a filtered segmask smoothing feathered images
    '''
    paintval=np.max(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)#CHAIN_APPROX_TC89_L1,CHAIN_APPROX_TC89_KCOS
    smoothedmask=np.zeros(mask.shape,dtype=mask.dtype)
    cv2.drawContours(smoothedmask, contours, -1, float(paintval), -1)
    smoothedmask=thin(smoothedmask,max_num_liter=1).astype(np.uint8) * paintval


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
            if sizes[i] < min_size:
                smoothedmask[output == i + 1] = 0

    return smoothedmask



def get_mask(im, color='red', plot=True):
	hsv = cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
	bounds=colors[color]
	mask=np.zeros_like(im)[:,:,0]
	for lower1,upper1 in bounds:
		partial_mask = cv2.inRange(hsv, lower1, upper1)
		mask=np.maximum(mask,partial_mask)
	mask = smooth_mask(mask, True)
	if plot:
		_,axs=plt.subplots(3,2)
		axs[0,0].imshow(im)
		axs[1,0].imshow(hsv)
		axs[2,0].imshow(mask)
		plt.show()
	return mask

def create_masks():
	uvimgs = glob.glob(f"{DATA_DIR}/*uv*.png")
	for f in uvimgs:
		uvname = f[f.rfind('/')+1:]
		mask_name = uvname.replace('imagel','maskl').replace('imager','maskr').replace("_uv","")
		uv_im=np.array(Image.open(f"{DATA_DIR}/{uvname}"))
		# mask = np.maximum(get_mask(uv_im,color='green',plot=False), get_mask(uv_im, color='red', plot=False))
		mask = get_mask(uv_im, color='red', plot=False)
		save_file=f"{DATA_DIR}/{mask_name}"
		print(f"saving {save_file}")
		Image.fromarray(mask).save(save_file)

if __name__=='__main__':
	create_masks()