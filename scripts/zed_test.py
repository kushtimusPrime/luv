import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

from fcvision.utils.arg_utils import parse_yaml



def main():
    cfg, params = parse_yaml(osp.join("cfg", "apps", "zed.yaml"))


if __name__ == '__main__':
    main()
