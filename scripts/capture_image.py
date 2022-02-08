from autolab_core import RigidTransform
import numpy as np
from phoxipy.phoxi_sensor import PhoXiSensor
import matplotlib.pyplot as plt
from fcvision.phoxi import Phoxi
from PIL import Image

if __name__ == '__main__':
	cam = Phoxi()
	img = cam.capture()._data; np.save("test.npy", img)
	for i in range(4):
		plt.imshow(img[:,:,i]); plt.show()
