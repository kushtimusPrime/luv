from autolab_core import RigidTransform
import numpy as np
import matplotlib.pyplot as plt
from fcvision.phoxi import Phoxi
import os
import os.path as osp
import time

N_COLLECT = 1000
START_ID = 0
OUTPUT_DIR = "data/cable_images"

TIME_DELAY = 5




if __name__ == '__main__':

	if not osp.exists(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)
	cam = Phoxi()

	for idx in range(START_ID, N_COLLECT):
		# time.sleep(TIME_DELAY)
		input("Press enter when ready to take a new image.")
		img = cam.capture()._data
		# plt.imshow(img[:,:,0]); plt.show()
		np.save(osp.join(OUTPUT_DIR, "image_%d"%idx), img)

	

