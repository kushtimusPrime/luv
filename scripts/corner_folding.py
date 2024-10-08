from autolab_core import RigidTransform,CameraIntrinsics
import numpy as np
from yumirws.yumi import YuMi
from fcvision.utils.arg_utils import parse_yaml
from fcvision.utils.async_writer import AsyncWrite
import os
import os.path as osp
import time
import cv2
from autolab_core.image import DepthImage
import matplotlib.pyplot as plt
from fcvision.plugs import Plug
from fcvision.cameras.zed import ZedImageCapture
from fcvision.utils.mask_utils import get_rgb,get_segmasks,COMMON_THRESHOLDS

from untangling.utils.interface_rws import Interface
from untangling.utils.grasp import GraspSelector,Grasp
from untangling.utils.tcps import ABB_WHITE
from yumiplanning.yumi_kinematics import YuMiKinematics as YK

from scripts.collect_cable_images import N_COLLECT
OUTPUT_DIR='data/live_execution_folding'
RGB_EXP=100
RGB_GAIN=20
UV_EXPS=[20,15,10,5]
UV_GAIN=20
COLOR_BOUNDS=COMMON_THRESHOLDS['data/white_towel']
NETWORK_ACTIONS = True
SAVE_IMGS=False
area_thres = 100
Y_thre = 150

START_ID=0
N_COLLECT=100
T_world_zed = RigidTransform.load('/home/justin/yumi/phoxipy/tools/zed_to_world_bww.tf')

intrc = CameraIntrinsics(frame='zed',
    fx=1.36817798e+03,
    fy=1.36817798e+03,
    cx=1.07913940e+03,
    cy=6.10996582e+02,
    skew=0.0,
    height=1242,
    width=2208)

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


def grasp_point_filter(p3d,center_list,iface):
    p3d_center_list = []
    for i,center in enumerate(center_list):
        center_x,center_y = center
        Tcam_p = RigidTransform(translation=p3d[center_y,center_x],from_frame="corner",to_frame="zed")
        p3d_center = (T_world_zed*Tcam_p).translation
        
        if p3d_center[0] < 0.6 and p3d_center[2]<0.3:
            p3d_center[-1] = 0.04
            p3d_center_list.append(p3d_center)

    frame = [iface.L_TCP.from_frame,iface.L_TCP.to_frame,iface.R_TCP.from_frame,iface.R_TCP.to_frame]

    if len(p3d_center_list)==4:
        
        p3d_center_list.sort(reverse = True, key = lambda x: x[0])#rank accoridng to x
        pair_f = p3d_center_list[:2]
        pair_f.sort(reverse = True, key = lambda x: x[1])
        pair_c = p3d_center_list[2:]
        pair_c.sort(reverse = True, key = lambda x: x[1])

        grasp_start_list_1 = []
        for i,p3d_point in enumerate(pair_c):
            H = RigidTransform(rotation =RigidTransform.x_axis_rotation(np.pi), translation=p3d_point,from_frame= frame[2*i], to_frame=frame[2*i+1])
            grasp = Grasp(H)
            grasp_start_list_1.append(grasp)
        
        grasp_end_list_1 = []
        for i,p3d_point in enumerate(pair_f):
            H = RigidTransform(rotation =RigidTransform.x_axis_rotation(np.pi), translation=p3d_point,from_frame= frame[2*i], to_frame=frame[2*i+1])
            grasp = Grasp(H)
            grasp_end_list_1.append(grasp)

        pair_m = [np.mean(np.array([pf,pc]),axis=0) for pf,pc in zip(pair_f,pair_c)]
        pair_m2 = [np.mean(np.array([pf,pm]),axis=0) for pf,pm in zip(pair_f,pair_m)]
        # pair_l = [pair_c[0],pair_m[0]]
        # pair_r = [pair_c[1],pair_m[1]]
        # pair_l = [pair_m[0],pair_f[0]]
        # pair_r = [pair_m[1],pair_f[1]]
        pair_l = [pair_m2[0]]
        pair_r = [pair_m2[1]]
        
        grasp_start_list_2 = []
        for i,p3d_point in enumerate(pair_l):
            H = RigidTransform(rotation =RigidTransform.x_axis_rotation(np.pi), translation=p3d_point,from_frame= frame[2*i], to_frame=frame[2*i+1])
            grasp = Grasp(H)
            grasp_start_list_2.append(grasp)
        
        grasp_end_list_2 = []
        for i,p3d_point in enumerate(pair_r):
            H = RigidTransform(rotation =RigidTransform.x_axis_rotation(np.pi), translation=p3d_point,from_frame= frame[2*i], to_frame=frame[2*i+1])
            grasp = Grasp(H)
            grasp_end_list_2.append(grasp)
    else:
        print("less than 4 corners")
        return None, None, None, None

    return grasp_start_list_1,grasp_end_list_1,grasp_start_list_2,grasp_end_list_2


def grasp_corner(ml,iface,zed):
    _,_,depth_img= zed.capture_image(depth=True)
    dimg=DepthImage(depth_img,frame='zed')
    p3d = intrc.deproject_to_image(dimg)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(ml, connectivity=8)
    #rank by size
    area_list = np.array([s[-1] for s in stats])
    sort_index = np.argsort(area_list)[::-1]
    #the first is the entire one, use the second one
    center_list = []
    area_list = area_list[sort_index]

    for i,area in enumerate(area_list):
        if i ==0:
            continue
        if area>area_thres:
            center_x,center_y = int(centroids[sort_index[i]][0]),int(centroids[sort_index[i]][1])
            if center_y > Y_thre: 
                center_list.append([center_x,center_y])
            
        else:
            break

    print("Grasping 2")
    grasp_start_list_1,grasp_end_list_1,grasp_start_list_2,grasp_end_list_2 = grasp_point_filter(p3d,center_list,iface)
    if grasp_start_list_1 is not None:
        yumi_grasp_two(iface,grasp_start_list_1,grasp_end_list_1,0)
        yumi_grasp_one(iface,grasp_start_list_2,grasp_end_list_2,1)
    else:
        print("error! less than two points avaliable")


def yumi_grasp_one(iface:Interface,grasp_start,grasp_end,step):
    if step == 0:
        rotation_l = RigidTransform.x_axis_rotation(-np.pi/6)@grasp_start[0].pose.rotation
    else:
        rotation_l = RigidTransform.z_axis_rotation(-np.pi/2)@grasp_start[0].pose.rotation

    grasp_start[0].pose.rotation=rotation_l

    iface.grasp(l_grasp=grasp_start[0])
    iface.sync()
    
    start_ee_l = grasp_start[0].pose.translation

    end_ee_l = grasp_end[0].pose.translation

    hold_height=.08
    up_pose_l = l_p([start_ee_l[0],start_ee_l[1],hold_height],rotation_l)

    down_pose_l = l_p([end_ee_l[0],end_ee_l[1]+0.03,0.07],rotation_l)

    l_ps=[up_pose_l,down_pose_l]
    iface.go_cartesian(l_targets=l_ps)
    iface.sync()
    iface.open_gripper('left')
    iface.open_gripper('right')
    time.sleep(.2)
    iface.home()
    iface.sync()

def yumi_grasp_two(iface:Interface,grasp_start,grasp_end,step):
    if step == 0:
        rotation_l = RigidTransform.x_axis_rotation(-np.pi/12)@grasp_start[0].pose.rotation
        rotation_r = RigidTransform.x_axis_rotation(np.pi/12)@grasp_start[0].pose.rotation
        
    else:
        rotation_l = RigidTransform.y_axis_rotation(-np.pi/6)@grasp_start[0].pose.rotation
        rotation_r = RigidTransform.y_axis_rotation(np.pi/6)@grasp_start[0].pose.rotation

    grasp_start[0].pose.rotation=rotation_l
    grasp_start[1].pose.rotation=rotation_r

    iface.grasp(l_grasp=grasp_start[0],r_grasp=grasp_start[1])
    iface.sync()
    
    start_ee_l = grasp_start[0].pose.translation
    start_ee_r = grasp_start[1].pose.translation

    end_ee_l = grasp_end[0].pose.translation
    end_ee_r = grasp_end[1].pose.translation

    hold_height=.08
    up_pose_l = l_p([start_ee_l[0],start_ee_l[1],hold_height],rotation_l)
    up_pose_r = r_p([start_ee_r[0],start_ee_r[1],hold_height],rotation_r)

    down_pose_l = l_p([end_ee_l[0]-0.03,end_ee_l[1],0.07],rotation_l)
    down_pose_r = r_p([end_ee_r[0]-0.03,end_ee_r[1],0.07],rotation_r)

    l_ps=[up_pose_l,down_pose_l]
    r_ps=[up_pose_r,down_pose_r]
    iface.go_cartesian(l_targets=l_ps,r_targets=r_ps)
    iface.sync()
    iface.open_gripper('left')
    iface.open_gripper('right')
    time.sleep(.2)
    iface.go_pose_plan(up_pose_l,up_pose_r,table_z=.02,mode='Distance')
    iface.home()
    iface.sync()

if __name__ == "__main__":
    if not osp.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    cfg, params = parse_yaml(osp.join("cfg", "apps", "live_test_config.yaml"))
    zed = params['camera']
    plug=params['plug']
    if NETWORK_ACTIONS:
        model = params['model']
    iface = Interface(speed=(0.8, 4 * np.pi))
    iface.open_grippers()
    iface.home()
    iface.sync()
    idx = START_ID
    while idx < N_COLLECT:
        print(f"Taking image {idx}")
        iml, imr = get_rgb(zed,RGB_EXP,RGB_GAIN)
        if NETWORK_ACTIONS:
            ml = model(cv2.resize(iml,(1280,720)), prep=True)
            ml=cv2.resize(ml,(iml.shape[1],iml.shape[0]))
            _,axs=plt.subplots(1,3)
            axs[0].imshow(iml)
            axs[1].imshow(ml)
            ml=(ml>.8).astype(np.uint8)
            axs[2].imshow(ml)
            plt.show()
        else:
            ml, mr, iml_uv, imr_uv = get_segmasks(zed, plug, COLOR_BOUNDS,UV_GAIN,UV_EXPS,plot=False)
        if SAVE_IMGS:
            writer = AsyncWrite(iml,imr,iml_uv,imr_uv,idx,OUTPUT_DIR)
            writer.start()
        idx += 1
            
        grasp_list = grasp_corner(ml,iface,zed)
