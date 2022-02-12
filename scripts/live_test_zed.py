import matplotlib.pyplot as plt
import os
from fcvision.arg_utils import parse_args
import fcvision.run_utils as ru
from fcvision.kp_wrapper import SegNetwork
from fcvision.zed import ZedImageCapture
from autolab_core import DepthImage
import open3d
import numpy as np
import time

"""
python scripts/live_test_zed.py --checkpoint outputs/2021-12-09/07-53-38/models/epoch\=199-step\=7599.ckpt
"""

def get_rgb(zed):
	zed.set_exposure(18)
	time.sleep(.5)
	iml,imr=zed.capture_image()
	return iml,imr

def main():
	params = parse_args()
	logdir = ru.get_file_prefix(params)
	os.makedirs(os.path.join(logdir, 'lightning_logs'))
	model = SegNetwork(params['checkpoint'], params=params, logdir=logdir)
	zed = ZedImageCapture(resolution='1080p')

	for idx in range(10):
		iml, imr = get_rgb(zed)
		predl = model(iml)
		predr = model(imr)


		_,axs=plt.subplots(2,2)
		axs[0,0].imshow(iml)
		axs[0,1].imshow(imr)
		# axs[1,1].imshow(hsv_right)
		axs[1,0].imshow(predl)
		axs[1,1].imshow(predr)
		plt.show()
		

if __name__ == '__main__':
	main()