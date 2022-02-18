try:
    from .plug import Plug
except:
    pass  # kasa not installed


def build_plug(cfg):
    return Plug(cfg["ip"])
