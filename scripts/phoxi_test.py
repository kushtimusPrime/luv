import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

from fcvision.utils.arg_utils import parse_yaml


def main():
    cfg, params = parse_yaml(osp.join("cfg", "apps", "phoxi_test.yaml"))
    cam = ret["camera"]
    img = cam.capture_image()._data
    for i in range(4):
        plt.imshow(img[:, :, i])
        plt.show()


if __name__ == "__main__":
    main()
