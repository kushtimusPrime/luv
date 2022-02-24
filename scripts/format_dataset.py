import os
import os.path as osp
import imageio
import numpy as np
import pathlib
import tqdm

# dataset_directories = ["data/white_towel", "data/blue_towel", "data/green_towel", "data/yellow_towel"]
# # dataset_directories = ["data/white_towel"]

# OUTPUT_DIR = "data/towel_seg"


dataset_directories = ["data/unlabeled_test_blue_towel", "data/unlabeled_test_green_towel", "data/unlabeled_test_towels", "data/unlabeled_test_white_towel", "data/unlabeled_test_yellow_towel"]
# dataset_directories = ["data/white_towel"]

OUTPUT_DIR = "data/towel_seg_test"


def format_training_dataset(dataset_directories, output_dir):
	pathlib.Path(f"{output_dir}/images").mkdir(parents=True, exist_ok=True)
	pathlib.Path(f"{output_dir}/targets").mkdir(parents=True, exist_ok=True)
	images = []
	targets = []

	for data_dir in dataset_directories:
		cur_images = [osp.join(data_dir, f) for f in os.listdir(data_dir) if "image" in f and "uv" not in f and "depth" not in f]
		images += cur_images
		if data_dir != "data/cable_red_painted_images":
			targets += [f.replace("imagel", "maskl").replace("imager", "maskr") for f in cur_images]
		else:
			targets += [f.replace("imagel", "maskl_full").replace("imager", "maskr_full") for f in cur_images]

	for i, (imfile, targfile) in tqdm.tqdm(enumerate(zip(images, targets))):
		im = imageio.imread(imfile) / 255
		targ = imageio.imread(targfile)
		np.savez_compressed(osp.join(OUTPUT_DIR, "images", "image_%d.npz"%i), im)
		np.savez_compressed(osp.join(OUTPUT_DIR, "targets", "target_%d.npz"%i), targ)

def format_test_dataset(dataset_directories, output_dir):
	pathlib.Path(f"{output_dir}/images").mkdir(parents=True, exist_ok=True)
	pathlib.Path(f"{output_dir}/targets").mkdir(parents=True, exist_ok=True)
	images = []
	for data_dir in dataset_directories:
		# TODO: glob
		cur_images = [osp.join(data_dir, f) for f in os.listdir(data_dir) if "image" in f and "uv" not in f and "depth" not in f and "png" in f]
		images += cur_images
	for i, imfile in enumerate(tqdm.tqdm(images)):
		im = imageio.imread(imfile)
		if im.max() > 1.0:
			im = im / 255
		np.savez_compressed(osp.join(OUTPUT_DIR, "images", "image_%d.npz"%i), im)


if __name__ ==  "__main__":
	format_test_dataset(dataset_directories, OUTPUT_DIR)
