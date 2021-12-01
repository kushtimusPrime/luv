from autolab_core import RigidTransform
import numpy as np
from phoxipy.phoxi_sensor import PhoXiSensor
import matplotlib.pyplot as plt
from src.phoxi import Phoxi

if __name__ == '__main__':
	cam = Phoxi()
	img = cam.capture()._data
	print(img.shape)
	for i in range(4):
		plt.imshow(img[:,:,i]); plt.show()

	

