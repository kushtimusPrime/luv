from autolab_core import RigidTransform
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import time
import cv2
from PIL import Image

from untangling.utils.interface_rws import Interface
from untangling.utils.tcps import ABB_WHITE
from yumiplanning.yumi_kinematics import YuMiKinematics as YK
from fcvision.phoxi import Phoxi
from fcvision.plug import Plug
from fcvision.zed import ZedImageCapture

N_COLLECT = 1000
START_ID = 0
OUTPUT_DIR = "data/cable_uv_endpoint_images"

colors = {
	'green': (np.array([60 - 30, 150, 40]), np.array([60 + 30, 255, 255])),
	'red': (np.array([0, 180, 40]), np.array([255, 255, 255])),
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
	# plt.imshow(img_left[:,:,::-1]); plt.show(); assert 0
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




def random_arm_positions(iface:Interface):
	random_left  = np.random.uniform([.3,.05,.15],[.5,.3,.3])
	random_right = np.random.uniform([.3,-.3,.15],[.5,-.05,.3])
	iface.go_pose('left',l_p(random_left),linear=True)
	iface.go_pose('right',r_p(random_right),linear=True)
	iface.sync()


if __name__ == '__main__':
	def l_p(trans, rot=Interface.GRIP_DOWN_R):
		return RigidTransform(translation=trans, rotation=rot, from_frame=YK.l_tcp_frame, to_frame=YK.base_frame)

	def r_p(trans, rot=Interface.GRIP_DOWN_R):
		return RigidTransform(translation=trans, rotation=rot, from_frame=YK.r_tcp_frame, to_frame=YK.base_frame)
	SPEED = (.3, 6*np.pi)
	iface = Interface("1703005", ABB_WHITE.as_frames(YK.l_tcp_frame, YK.l_tip_frame),
					  ABB_WHITE.as_frames(YK.r_tcp_frame, YK.r_tip_frame), speed=SPEED)
	iface.open_grippers()
	iface.home()
	if not osp.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	zed = ZedImageCapture()
	plug = Plug()

	idx = START_ID
	while idx < N_COLLECT:

		random_arm_positions(iface)

		print(idx)
		input("Press enter when ready to take a new image.")

		plug.turn_off()
		iml, imr = get_rgb(zed)
		# plt.imshow(iml); plt.show()
		# plt.imshow(imr); plt.show()
		ml, mr, iml_uv, imr_uv, imd = get_segmasks(zed, plug, color='red')

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