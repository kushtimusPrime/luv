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
area_thres = 400
Y_thre = 230

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

def grasp_point(p3d,center_list,iface):
    grasp_list = []
    frame = [iface.L_TCP.from_frame,iface.L_TCP.to_frame,iface.R_TCP.from_frame,iface.R_TCP.to_frame]
    for i,center in enumerate(center_list):
        center_x,center_y = center
        Tcam_p = RigidTransform(translation=p3d[center_y,center_x],from_frame="corner",to_frame="zed")
        corner_robot = T_world_zed*Tcam_p

        H = RigidTransform(rotation =RigidTransform.x_axis_rotation(np.pi), translation=corner_robot.translation,from_frame= frame[2*i], to_frame=frame[2*i+1])

        grasp = Grasp(H)
        grasp.pose.translation[2]=.04
        grasp_list.append(grasp)
    return grasp_list

def grasp_point_filter(p3d,center_list,iface):
    p3d_center_list = []
    for i,center in enumerate(center_list):
        center_x,center_y = center
        Tcam_p = RigidTransform(translation=p3d[center_y,center_x],from_frame="corner",to_frame="zed")
        p3d_center = (T_world_zed*Tcam_p).translation
        if p3d_center[0] < 0.52 and p3d_center[1]<0.3:
            p3d_center[-1] = 0.04 if p3d_center[-1]<0.05 else p3d_center[-1]
            p3d_center_list.append(p3d_center)

    frame = [iface.L_TCP.from_frame,iface.L_TCP.to_frame,iface.R_TCP.from_frame,iface.R_TCP.to_frame]
    grasp_list = []

    if len(p3d_center_list)>=2:
        min_dist = 1000000
        p3d_center_list = np.array(p3d_center_list)
        for i in range(len(p3d_center_list)-1):
            dist_list = p3d_center_list[i+1:]-p3d_center_list[i]
            dist_list_2 = [c[0]**2+c[1]**2 for c in dist_list]
            if np.min(dist_list_2)< min_dist:
                min_dist = np.min(dist_list_2)
                if p3d_center_list[i][1] < p3d_center_list[np.argmin(dist_list_2)+i+1][1]:
                    min_pair = [p3d_center_list[np.argmin(dist_list_2)+i+1].tolist(),p3d_center_list[i].tolist()]
                else:
                    min_pair = [p3d_center_list[i].tolist(),p3d_center_list[np.argmin(dist_list_2)+i+1].tolist()]
        print(min_pair)
        # import pdb;pdb.set_trace()
        for i,p3d_point in enumerate(min_pair):
            # Tcam_p = RigidTransform(translation=p3d_point,from_frame="corner",to_frame="zed")
            # corner_robot = T_world_zed*Tcam_p
            H = RigidTransform(rotation =RigidTransform.x_axis_rotation(np.pi), translation=p3d_point,from_frame= frame[2*i], to_frame=frame[2*i+1])
            grasp = Grasp(H)
            grasp_list.append(grasp)
    else:
        print("grasp one after filtering")
        H = RigidTransform(rotation =RigidTransform.x_axis_rotation(np.pi), translation=p3d_center_list[0],from_frame= frame[0], to_frame=frame[1])
        grasp = Grasp(H)
        grasp_list.append(grasp)

    return grasp_list

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
	cloth_seg[cloth_seg < 130] = 0
	#zeros out the border to remove robot
	crop_lr = 0.1
	crop_bottom=.2
	crop_top=.2
	cloth_seg[:, : int(crop_lr * w)] = 0
	cloth_seg[:, int((1 - crop_lr) * w) :] = 0
	cloth_seg[int((1 - crop_bottom) * h) :, :] = 0
	cloth_seg[:int(crop_top*h), :] = 0
	cloth_seg[cloth_seg != 0] = 255
	while True:
		coords = sample_cloth_point(cloth_seg)
		grasp = g.top_down_grasp(coords, 0.02, iface.R_TCP)
		grasp.pose.translation[2] = 0.04
		grasp.speed = (1, 2*np.pi)
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
	iface.shake_J('right',[1],2)
	iface.open_gripper("right")
	iface.home()
	iface.sync()

def grasp_corner(ml,iface,zed):
    _,_,depth_img= zed.capture_image(depth=True)
    # depth_img = np.load("/home/justin/yumi/fc-vision/data/yellow_towel/image_depth_0.npy")
    dimg=DepthImage(depth_img,frame='zed')
    # dimg=dimg.inpaint()
    p3d = intrc.deproject_to_image(dimg)
    # row,col = np.where(ml)
    # center_y,center_x =int(np.mean(row)), int(np.mean(col)) 
    b_components, output, stats, centroids = cv2.connectedComponentsWithStats(ml, connectivity=8)
    # import pdb;pdb.set_trace()
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

    # if area_list[sort_index[1]]>area_thres:
    #     center_x,center_y = int(centroids[sort_index[1]][0]),int(centroids[sort_index[1]][1])
    #     center_list.append([center_x,center_y])
    
    # if area_list[sort_index[2]]>area_thres:
    #     center_x,center_y = int(centroids[sort_index[2]][0]),int(centroids[sort_index[2]][1])
    #     center_list.append([center_x,center_y])
    print(center_list)
    if len(center_list)==0:
        reset_cloth(iface)
        return
    
    if len(center_list)==1:
        print("Grasping 1")
        grasp_list = grasp_point(p3d,center_list,iface)
        yumi_grasp_one(iface,grasp_list[0])
    else:
        print("Grasping 2")
        grasp_list = grasp_point_filter(p3d,center_list,iface)
        # min_dist = 1000000
        # center_list = np.array(center_list)
        # for i in range(len(center_list)-1):
        #     dist_list = center_list[i+1:]-center_list[i]
        #     dist_list_2 = [c[0]**2+c[1]**2 for c in dist_list]
        #     if np.min(dist_list_2)< min_dist:
        #         min_dist = np.min(dist_list_2)
        #         if center_list[i][0] < center_list[np.argmin(dist_list_2)+i+1][0]:
        #             min_pair = [center_list[i].tolist(),center_list[np.argmin(dist_list_2)+i+1].tolist()]
        #         else:
        #             min_pair = [center_list[np.argmin(dist_list_2)+i+1].tolist(),center_list[i].tolist()]
        
        # grasp_list = grasp_point(p3d,min_pair,iface)
        # import pdb;pdb.set_trace()
        if len(grasp_list) >1:
            yumi_grasp_two(iface,grasp_list)
        else:
            yumi_grasp_one(iface,grasp_list[0])



def yumi_grasp_one(iface:Interface,grasp):
    arm = 'left'
    iface.grasp(l_grasp=grasp)
    iface.sync()
    up_pose = l_p([.4,-.2,0.35])
    down_pose = l_p([.4,0.1,0.06])
    iface.go_pose_plan_single(arm, up_pose,table_z=0.02)
    iface.sync()
    iface.shake_J(arm,[.3],2)
    iface.sync()
    iface.set_speed((.3,np.pi))
    iface.go_cartesian(l_targets=[down_pose])
    # iface.go_pose_plan_single(arm, down_pose, table_z=.02)
    iface.sync()
    iface.set_speed(iface.default_speed)
    iface.open_gripper(arm)
    iface.home()
    iface.sync()

def yumi_grasp_two(iface:Interface,grasp):
    grasp[0].pose.rotation=RigidTransform.x_axis_rotation(-np.pi/5)@grasp[0].pose.rotation
    grasp[1].pose.rotation=RigidTransform.x_axis_rotation(np.pi/5)@grasp[1].pose.rotation
    iface.grasp(l_grasp=grasp[0],r_grasp=grasp[1])
    iface.sync()
    hold_dist=.2
    forward_dist=.2
    hold_height=.35
    up_pose_l = l_p([.35,hold_dist/2,hold_height])
    up_pose_r = r_p([.35,-hold_dist/2,hold_height])
    forward_pose_l = l_p([.35+forward_dist,hold_dist/2,hold_height])
    forward_pose_r = r_p([.35+forward_dist,-hold_dist/2,hold_height])
    down_pose_l = l_p([.35+forward_dist,hold_dist/2,.12])
    down_pose_r = r_p([.35+forward_dist,-hold_dist/2,.12])
    back_pose_l = l_p([.35,hold_dist/2,.07])
    back_pose_r = r_p([.35,-hold_dist/2,.07])
    iface.go_pose_plan(up_pose_l,up_pose_r,table_z=.02,mode='Distance')
    l_ps=[forward_pose_l,down_pose_l,back_pose_l]
    r_ps=[forward_pose_r,down_pose_r,back_pose_r]
    iface.go_cartesian(l_targets=l_ps,r_targets=r_ps)
    iface.sync()
    iface.open_gripper('left')
    iface.open_gripper('right')
    time.sleep(.2)
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
