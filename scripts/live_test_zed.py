import matplotlib.pyplot as plt
import os
from fcvision.utils.arg_utils import parse_yaml
import fcvision.utils.run_utils as ru
from fcvision.learning.model_wrappers import SegNetwork
from fcvision.cameras.zed import ZedImageCapture
from autolab_core import DepthImage
import open3d
import os.path as osp
import numpy as np
import time
from fcvision.utils.mask_utils import get_rgb,get_segmasks,COMMON_THRESHOLDS


"""
python scripts/live_test_zed.py --checkpoint outputs/2021-12-09/07-53-38/models/epoch\=199-step\=7599.ckpt
"""

RGB_EXP = 100
RGB_GAIN=20

# def get_rgb(zed):
# 	zed.set_exposure(18)
# 	time.sleep(.5)
# 	iml,imr=zed.capture_image()
# 	return iml,imr

def main():
	cfg, params = parse_yaml(osp.join("cfg", "apps", "live_test_config.yaml"))
	logdir = ru.get_file_prefix(params)
	os.makedirs(os.path.join(logdir, 'vis'))
	# model = SegNetwork(params['checkpoint'], params=params, logdir=logdir)
	zed = params['camera']
	model = params['model']

	for idx in range(10):
		# iml, imr = get_rgb(zed)
		iml, imr = get_rgb(zed,RGB_EXP,RGB_GAIN)

		predl = model(iml, prep=True)
		predr = model(imr, prep=True)


		_,axs=plt.subplots(2,2)
		axs[0,0].imshow(iml)
		axs[0,1].imshow(imr)
		# axs[1,1].imshow(hsv_right)
		axs[1,0].imshow(predl)
		axs[1,1].imshow(predr)
		plt.show()
		

if __name__ == '__main__':
	main()