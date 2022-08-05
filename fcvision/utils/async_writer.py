import threading
import os.path as osp
from PIL import Image
class AsyncWrite(threading.Thread):

	def __init__(self, iml, imr, uvl, uvr, i, data_folder):
		# calling superclass init
		threading.Thread.__init__(self)
		self.img_left = iml
		self.img_right = imr
		self.uvl = uvl
		self.uvr = uvr
		self.i = i
		self.OUTPUT_DIR = data_folder

	def run(self):
		Image.fromarray(self.uvl).save(osp.join(self.OUTPUT_DIR, "imagel_uv_%d.png" % self.i))
		Image.fromarray(self.uvr).save(osp.join(self.OUTPUT_DIR, "imager_uv_%d.png" % self.i))
		Image.fromarray(self.img_left).save(osp.join(self.OUTPUT_DIR, "imagel_%d.png" % self.i))
		Image.fromarray(self.img_right).save(osp.join(self.OUTPUT_DIR, "imager_%d.png" % self.i))