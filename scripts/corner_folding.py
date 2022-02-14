from autolab_core import RigidTransform
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import time
import cv2
from PIL import Image

from fcvision.cameras.phoxi import Phoxi
from fcvision.plug import Plug
from fcvision.cameras.zed import ZedImageCapture
from fcvision.utils.mask_utils import get_rgb,get_segmasks

from untangling.utils.interface_rws import Interface
from untangling.utils.grasp import GraspSelector
from untangling.utils.tcps import ABB_WHITE
from yumiplanning.yumi_kinematics import YuMiKinematics as YK

from scripts.collect_cable_images import N_COLLECT
OUTPUT_DIR='data/live_execution_folding'
START_ID=0
N_COLLECT=100
if __name__ == "__main__":
    if not osp.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    zed = ZedImageCapture()
    plug = Plug()
    iface = Interface(speed=(0.6, 4 * np.pi))
    iface.open_grippers()
    iface.home()
    iface.sync()
    idx = START_ID
    while idx < N_COLLECT:
        print(f"Taking image {idx}")
        iml, imr = get_rgb(zed)
        ml, mr, iml_uv, imr_uv, imd = get_segmasks(zed, plug, color="green", plot=True)

        print("WARNING, SAVING IMAGES DISABLED")
        # Image.fromarray(iml_uv).save(osp.join(OUTPUT_DIR, "imagel_uv_%d.png" % idx))
        # Image.fromarray(imr_uv).save(osp.join(OUTPUT_DIR, "imager_uv_%d.png" % idx))
        # Image.fromarray(iml).save(osp.join(OUTPUT_DIR, "imagel_%d.png" % idx))
        # Image.fromarray(imr).save(osp.join(OUTPUT_DIR, "imager_%d.png" % idx))
        np.save(osp.join(OUTPUT_DIR, "image_depth_%d.npy" % idx), imd)
        Image.fromarray(ml).save(osp.join(OUTPUT_DIR, "maskl_%d.png" % idx))
        Image.fromarray(mr).save(osp.join(OUTPUT_DIR, "maskr_%d.png" % idx))
        idx += 1
