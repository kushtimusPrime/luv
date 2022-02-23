import numpy as np
import os
import os.path as osp
from pqdm.processes import pqdm


DATASET_DIR = "data/towel_seg"

def compress_image(fn):
	im = np.load(fn)
	np.savez_compressed(fn.replace(".npy", ".npz"), im)

	with np.load(fn.replace(".npy", ".npz")) as f:
		im2 = f["arr2"]
	if np.linalg.norm(im2 - im) > 1e-4:
		compress_image(fn)

if __name__ == '__main__':
	images = [osp.join(DATASET_DIR, "images", f) for f in os.listdir(osp.join(DATASET_DIR, "images")) if ".npy" in f]

	# for f in images:
	# 	if ".npz" in f:
	# 		continue
	# 	else:
	# 		print(f)
	# 		im = np.load(f)
	# 		im2 = np.load(f.replace(".npy", ".npz"))
	# 		im2_data = im2["arr_0"]
	# 		print(im2_data.shape, im2_data.dtype, im.shape, im.dtype)
	# 		im2.close()
	# 		print(np.linalg.norm(im2_data - im), f)
	# assert 0

	targets = [osp.join(DATASET_DIR, "targets", f) for f in os.listdir(osp.join(DATASET_DIR, "targets")) if ".npy" in f]
	pqdm(images + targets, compress_image, n_jobs=12)
