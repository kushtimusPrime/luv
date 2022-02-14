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
from fcvision.utils.vision_utils import find_center_of_mass
from fcvision.utils.mask_utils import get_rgb, get_segmasks, COMMON_THRESHOLDS
from untangling.utils.interface_rws import Interface
from untangling.utils.grasp import GraspSelector
from untangling.utils.tcps import ABB_WHITE
from yumiplanning.yumi_kinematics import YuMiKinematics as YK

N_COLLECT = 1000
START_ID = 0
OUTPUT_DIR = "data/cloth_images_auto_test"
RES='2K'
UV_EXPS=[5,10,20,40,80]
RGB_EXP=100
RGB_GAIN=15
UV_GAIN=15
AUTOMATIC=False


def l_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(
        translation=trans,
        rotation=rot,
        from_frame=YK.l_tcp_frame,
        to_frame=YK.base_frame,
    )


def r_p(trans, rot=Interface.GRIP_DOWN_R):
    return RigidTransform(
        translation=trans,
        rotation=rot,
        from_frame=YK.r_tcp_frame,
        to_frame=YK.base_frame,
    )


def sample_cloth_point(cloth_seg):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        cloth_seg, connectivity=8
    )
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 2000
    cloth_seg = np.zeros_like(cloth_seg)
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            cloth_seg[output == i + 1] = 255
    #choose a point only along the edge
    for i in range(5):
        #remove some from border for safer grasps
        eroded = cv2.erode(cloth_seg, np.ones((3, 3)))
        cloth_seg &= eroded
    eroded = cv2.erode(cloth_seg, np.ones((3, 3)))
    cloth_seg -= eroded
    ids = np.vstack(np.nonzero(cloth_seg))
    randchoice = np.random.randint(ids.shape[1])
    coords = ids[0, randchoice], ids[1, randchoice]
    coords = (int(coords[1]), int(coords[0]))
    return coords


def reset_cloth(iface: Interface):
    # input("Press enter when ready to take a new image.")
    img = iface.take_image()
    colorim = img.color._data[:, :, 0]
    h, w = colorim.shape
    g = GraspSelector(img, iface.cam.intrinsics, iface.T_PHOXI_BASE)
    cloth_seg = colorim
    cloth_seg[cloth_seg < 80] = 0
    crop_r = 0.1#zeros out the border to remove robot
    cloth_seg[:, : int(crop_r * w)] = 0
    cloth_seg[int((1 - 2 * crop_r) * h) :, :] = 0
    cloth_seg[:, int((1 - crop_r) * w) :] = 0
    cloth_seg[cloth_seg != 0] = 255
    coords = sample_cloth_point(cloth_seg)
    grasp = g.top_down_grasp(coords, 0.02, iface.R_TCP)
    grasp.pose.translation[2] = 0.04
    grasp.speed = (0.3, np.pi)
    iface.grasp(r_grasp=grasp)
    iface.sync()
    iface.go_delta_single("right", [0, 0, 0.05], reltool=False)
    dx, dy = np.random.uniform((-0.1, -0.1), (0.1, 0.1))
    iface.go_pose("left", l_p([0.4 + dx, 0 + dy, 0.3]), linear=False)
    iface.sync()
    iface.shake_J('right',[1],1)
    iface.open_gripper("right")
    iface.home()
    iface.sync()


if __name__ == "__main__":
    if not osp.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    zed = ZedImageCapture(resolution=RES)
    plug = Plug()
    iface = Interface(speed=(0.6, 4 * np.pi))
    iface.open_grippers()
    iface.home()
    iface.sync()
    idx = START_ID
    while idx < N_COLLECT:
        if AUTOMATIC:
            reset_cloth(iface)
        else:
            input("Enter to take a pic")
        print(f"Taking image {idx}")
        iml, imr = get_rgb(zed,RGB_EXP,RGB_GAIN)
        ml, mr, iml_uv, imr_uv, imd = get_segmasks(zed, plug, COMMON_THRESHOLDS['red'],UV_GAIN,UV_EXPS,plot=True)

        print("WARNING, SAVING IMAGES DISABLED")
        # Image.fromarray(iml_uv).save(osp.join(OUTPUT_DIR, "imagel_uv_%d.png" % idx))
        # Image.fromarray(imr_uv).save(osp.join(OUTPUT_DIR, "imager_uv_%d.png" % idx))
        # Image.fromarray(iml).save(osp.join(OUTPUT_DIR, "imagel_%d.png" % idx))
        # Image.fromarray(imr).save(osp.join(OUTPUT_DIR, "imager_%d.png" % idx))
        # np.save(osp.join(OUTPUT_DIR, "image_depth_%d.npy" % idx), imd)
        # Image.fromarray(ml).save(osp.join(OUTPUT_DIR, "maskl_%d.png" % idx))
        # Image.fromarray(mr).save(osp.join(OUTPUT_DIR, "maskr_%d.png" % idx))
        idx += 1
