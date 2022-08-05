import numpy as np
import os
import os.path as osp
from pqdm.processes import pqdm


DATASET_DIR = "data/towel_seg"

def compress_image(fn):
	im = np.load(fn)
	np.savez_compressed(fn.replace(".npy", ".npz"), im)

	with np.load(fn.replace(".npy", ".npz")) as f:
		im2 = f["arr_0"]
	if np.linalg.norm(im2 - im) > 1e-4:
		compress_image(fn)

def verify_compression(fn):
	assert not ".npz" in fn
	im = np.load(fn)
	im2 = np.load(fn.replace(".npy", ".npz"))
	im2_data = im2["arr_0"]
	im2.close()
	diff = np.linalg.norm(im2_data - im)
	if diff > 1e-5:
		return False
	return True

def delete_file(fn):
	assert ".npy" in fn
	os.remove(fn)


def load_file(fn):
	assert ".npz" in fn
	with np.load(fn) as f:
		im = f["arr_0"]
	return True



NJOBS = 12

if __name__ == '__main__':
	images = [osp.join(DATASET_DIR, "images", f) for f in os.listdir(osp.join(DATASET_DIR, "images")) if ".npy" in f]
	targets = [osp.join(DATASET_DIR, "targets", f) for f in os.listdir(osp.join(DATASET_DIR, "targets")) if ".npy" in f]

	pqdm(images + targets, compress_image, n_jobs=NJOBS)
	ret = pqdm(images + targets, verify_compression, n_jobs=NJOBS)
	assert np.all(ret)
	pqdm(images + targets, delete_file, n_jobs=NJOBS)


	# images = [osp.join(DATASET_DIR, "images", f) for f in os.listdir(osp.join(DATASET_DIR, "images")) if ".npy" in f]
	# targets = [osp.join(DATASET_DIR, "targets", f) for f in os.listdir(osp.join(DATASET_DIR, "targets")) if ".npy" in f]
	# ret = pqdm(images + targets, delete_file, n_jobs=NJOBS)
	# print(ret)
	# print(len(ret), np.sum(ret))

