from fcvision.phoxi import Phoxi
from fcvision.vision_utils import get_shake_point


if __name__ == '__main__':
	cam = Phoxi()
	im = cam.capture()._data
	get_shake_point(im, vis=True)
