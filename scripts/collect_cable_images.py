import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
from fcvision.utils.arg_utils import parse_yaml
import time
import cv2
from fcvision.utils.mask_utils import get_rgb,get_segmasks,COMMON_THRESHOLDS
from PIL import Image

from fcvision.plugs import Plug
from fcvision.cameras.zed import ZedImageCapture

N_COLLECT = 100
START_ID = 0
OUTPUT_DIR = "data/iphone_red_painted_images"
RGB_EXP = 100
UV_EXPS = [5,10,20,40,80]
UV_GAIN=15
RGB_GAIN=15
RES='2K'


if __name__ == '__main__':
	if not osp.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	cfg, params = parse_yaml(osp.join("cfg", "apps", "uv_data_collection.yaml"))

	zed = params["camera"]
	plug = params["plug"]
	plug.turn_off()
	idx = START_ID
	while idx < N_COLLECT:
		print("Collecting id ",idx)
		input("Press enter when ready to take a new image.")
		iml, imr = get_rgb(zed,RGB_EXP,RGB_GAIN)
		ml, mr, iml_uv, imr_uv = get_segmasks(zed, plug, COMMON_THRESHOLDS['red'],UV_GAIN,UV_EXPS,plot=True)
		Image.fromarray(iml_uv).save(osp.join(OUTPUT_DIR, "imagel_uv_%d.png"%idx))
		Image.fromarray(imr_uv).save(osp.join(OUTPUT_DIR, "imager_uv_%d.png"%idx))
		Image.fromarray(iml).save(osp.join(OUTPUT_DIR, "imagel_%d.png"%idx))
		Image.fromarray(imr).save(osp.join(OUTPUT_DIR, "imager_%d.png"%idx))
		# np.save(osp.join(OUTPUT_DIR, "depth_%d.npy"%idx), imd)
		# Image.fromarray(ml).save(osp.join(OUTPUT_DIR, "maskl_%d.png"%idx))
		# Image.fromarray(mr).save(osp.join(OUTPUT_DIR, "maskr_%d.png"%idx))
		idx += 1
		
