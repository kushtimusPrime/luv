from .phoxi import Phoxi
from .zed import ZedImageCapture

def build_camera(cfg):
	return globals()[cfg["name"]](cfg["params"])
