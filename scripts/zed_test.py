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

def get_rgb(zed):
	iml,imr=zed.capture_image()
	return iml,imr

def main():
	visdepth=False
	# params = parse_args()
	# logdir = ru.get_file_prefix(params)
	# os.makedirs(os.path.join(logdir, 'lightning_logs'))
	# model = SegNetwork(params['checkpoint'], params=params, logdir=logdir)
	zed = ZedImageCapture(resolution='2K')
	time.sleep(.5)

	for idx in range(10):
		iml, imr = get_rgb(zed)
		if visdepth:
			iml,imr,depth=zed.capture_image(True)
			dimg=DepthImage(depth,'zed')
			pc=zed.intrinsics.deproject(dimg)
			pc.remove_infinite_points()
			pc.remove_zero_points()
			q:open3d.PointCloud=open3d.geometry.PointCloud(open3d.utility.Vector3dVector((pc.data).T))
			print(np.asarray(o3dpc.points)[:100,:])
			open3d.visualization.draw_geometries([o3dpc])
		# predl = model(iml, mode='seg')
		# predr = model(imr, mode='seg')


		_,axs=plt.subplots(2,2)
		axs[0,0].imshow(iml)
		axs[0,1].imshow(imr)
		# axs[1,1].imshow(hsv_right)
		# axs[1,0].imshow(predl)
		# axs[1,1].imshow(predr)
		plt.show()
		

if __name__ == '__main__':
	main()
