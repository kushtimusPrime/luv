from attr import asdict
from autolab_core import RigidTransform
import numpy as np
import matplotlib.pyplot as plt
from yumirws.yumi import YuMi
import os
import os.path as osp
import cv2

from fcvision.utils.arg_utils import parse_yaml
from fcvision.utils.vision_utils import get_hdr_capture, get_high_sat_capture
from fcvision.utils.mask_utils import get_rgb, get_segmasks, COMMON_THRESHOLDS
from untangling.utils.interface_rws import Interface
from untangling.utils.grasp import GraspSelector
from yumiplanning.yumi_kinematics import YuMiKinematics as YK
from fcvision.utils.async_writer import AsyncWrite

N_COLLECT = 10
START_ID = 0
OUTPUT_DIR = "data/tshirt"
COLOR_BOUNDS=COMMON_THRESHOLDS.get(OUTPUT_DIR,COMMON_THRESHOLDS['red'])
RES='2K'
UV_EXPS={'data/white_towel':[7],
		'data/blue_towel':[15],'data/yellow_towel':[20],'data/green_towel':[20],'data/bright_green_towel':[9],
		'data/misc_towels':[20,15,10,5],'data/tshirt':[100,90,80]}.get(OUTPUT_DIR,[20])
RGB_EXP=100
RGB_GAIN=20
UV_GAIN=20
AUTOMATIC=False
cap_fn = get_hdr_capture


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
    # choose a point only along the edge
    for i in range(5):
        # remove some from border for safer grasps
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
    cloth_seg[cloth_seg < 130] = 0
    # zeros out the border to remove robot
    crop_lr = 0.1
    crop_bottom = 0.2
    crop_top = 0.2
    cloth_seg[:, : int(crop_lr * w)] = 0
    cloth_seg[:, int((1 - crop_lr) * w) :] = 0
    cloth_seg[int((1 - crop_bottom) * h) :, :] = 0
    cloth_seg[: int(crop_top * h), :] = 0
    cloth_seg[cloth_seg != 0] = 255
    while True:
        coords = sample_cloth_point(cloth_seg)
        grasp = g.top_down_grasp(coords, 0.02, iface.R_TCP)
        grasp.pose.translation[2] = 0.04
        grasp.speed = (1, 2 * np.pi)
        try:
            iface.grasp(r_grasp=grasp)
            break
        except:
            iface.y = YuMi(l_tcp=iface.L_TCP, r_tcp=iface.R_TCP)
            iface.open_grippers()
            iface.home()
            iface.sync()
    iface.sync()
    iface.go_delta_single("right", [0, 0, 0.05], reltool=False)
    dx, dy = np.random.uniform((-0.1, -0.1), (0.1, 0.1))
    iface.go_pose_plan_single("right", r_p([0.4 + dx, 0 + dy, 0.4]))
    iface.sync()
    iface.shake_J("right", [1], 2)
    iface.open_gripper("right")
    iface.home()
    iface.sync()


if __name__ == "__main__":
    if not osp.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    cfg, params = parse_yaml(osp.join("cfg", "apps", "uv_data_collection.yaml"))

    zed = params["camera"]
    plug = params["plug"]
    plug.turn_off()
    if AUTOMATIC:
        iface = Interface(speed=(1.5, 6 * np.pi))
        iface.open_grippers()
        iface.home()
        iface.sync()
    idx = START_ID
    while idx < N_COLLECT:
        if AUTOMATIC:
            try:
                reset_cloth(iface)
            except Exception as e:
                print(e)
                iface.y = YuMi(l_tcp=iface.L_TCP, r_tcp=iface.R_TCP)
                iface.open_grippers()
                iface.home()
                iface.sync()
        else:
            input("Enter to take a pic")
        print(f"Taking image {idx}")
        iml, imr = get_rgb(zed, RGB_EXP, RGB_GAIN)
        ml, mr, iml_uv, imr_uv = get_segmasks(
            zed, plug, COLOR_BOUNDS, UV_GAIN, UV_EXPS, plot=True, capture_fn=cap_fn
        )
        zed.set_exposure(RGB_EXP)
        zed.set_exposure(RGB_GAIN)
        writer = AsyncWrite(iml, imr, iml_uv, imr_uv, idx, OUTPUT_DIR)
        writer.start()
        idx += 1
        plt.close("all")
