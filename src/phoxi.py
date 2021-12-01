from autolab_core import RigidTransform
import numpy as np
from phoxipy.phoxi_sensor import PhoXiSensor
import matplotlib.pyplot as plt

class Phoxi:

	def __init__(self):
		self.cam = PhoXiSensor("1703005")
		self.cam.start()
		img=self.cam.read()
		self.cam.intrinsics=self.cam.create_intr(img.width,img.height)


	def capture(self):
		return self.cam.read()