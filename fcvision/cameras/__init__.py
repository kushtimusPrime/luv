try:
    from .phoxi import Phoxi
except:
    pass  # camera libraries not installed
try:
    from .zed import ZedImageCapture
except:
    pass  # camera libraries not installed


def build_camera(cfg):
    return globals()[cfg["name"]](**cfg["params"])
