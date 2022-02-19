import os
import os.path as osp
import imageio
import numpy as np
import pathlib

dataset_directories = ["data/white_towel", "data/blue_towel", "data/green_towel", "data/yellow_towel"]

OUTPUT_DIR = "data/towel_seg"

pathlib.Path(f"{OUTPUT_DIR}/images").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{OUTPUT_DIR}/targets").mkdir(parents=True, exist_ok=True)


images = []
targets = []

for data_dir in dataset_directories:
	cur_images = [osp.join(data_dir, f) for f in os.listdir(data_dir) if "image" in f and "uv" not in f and "depth" not in f]
	images += cur_images
	if data_dir != "data/cable_red_painted_images":
		targets += [f.replace("imagel", "maskl").replace("imager", "maskr") for f in cur_images]
	else:
		targets += [f.replace("imagel", "maskl_full").replace("imager", "maskr_full") for f in cur_images]

for i, (imfile, targfile) in enumerate(zip(images, targets)):
	print(i)
	im = imageio.imread(imfile) / 255
	targ = imageio.imread(targfile)
	np.save(osp.join(OUTPUT_DIR, "images", "image_%d.npy"%i), im)
	np.save(osp.join(OUTPUT_DIR, "targets", "target_%d.npy"%i), targ)

