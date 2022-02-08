import os
import os.path as osp
import matplotlib.pyplot as plt
import time

from fcvision.dataset import KPDataset
from fcvision.arg_utils import parse_args
import fcvision.run_utils as ru
from fcvision.vision_utils import find_peaks
from matplotlib.patches import Circle
from fcvision.phoxi import Phoxi,prepare_phoxi_image_for_net
from fcvision.kp_wrapper import SegNetwork
from fcvision.zed import ZedImageCapture
import fcvision.pytorch_utils as ptu
import cv2



"""
python scripts/live_test_zed.py --checkpoint outputs/2021-12-09/07-53-38/models/epoch\=199-step\=7599.ckpt
"""

def get_rgb(zed):
	zed.set_exposure(80)
	time.sleep(1)
	iml,imr=zed.capture_image()
	return iml,imr

def main():
	params = parse_args()
	logdir = ru.get_file_prefix(params)
	os.makedirs(os.path.join(logdir, 'lightning_logs'))
	phoxi = Phoxi()
	model = SegNetwork(params['checkpoint'], params=params, logdir=logdir)
	zed = ZedImageCapture()

	for idx in range(100):
		input("Press enter when ready to take a new image.")
		iml, imr = get_rgb(zed)

		predl = model(iml, mode='seg')
		predr = model(imr, mode='seg')


		_,axs=plt.subplots(2,2)
		axs[0,0].imshow(iml)
		axs[0,1].imshow(imr)
		# axs[1,0].imshow(hsv_left)
		# axs[1,1].imshow(hsv_right)
		axs[1,0].imshow(predl)
		axs[1,1].imshow(predr)
		plt.show()
		
		im=phoxi.capture().color._data
		imseg=model(im,mode='seg')
		_,axs=plt.subplots(1,2)
		axs[0].imshow(im)
		axs[1].imshow(imseg)
		plt.show()


		# plt.imshow(pred + np_im); plt.show()

		# coords = find_peaks(pred)
		# fig = plt.figure(frameon=False)
		# ax = plt.Axes(fig, [0., 0., 1., 1.])
		# ax.set_axis_off()
		# fig.add_axes(ax)
		# ax.imshow(np_im, aspect='auto')
		# for xx, yy in coords:
		# 	circ = Circle((yy, xx), 3, color='r')
		# 	ax.add_patch(circ)
		# plt.savefig(osp.join(logdir, "vis", "pred_%d.jpg"%idx))
		# plt.show()
		# plt.clf()


if __name__ == '__main__':
	main()