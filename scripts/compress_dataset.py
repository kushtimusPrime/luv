import numpy as np
import os
import os.path as osp


DATASET_DIR = "towel_seg"

if __name__ == '__main__':
	images = os.listdir(osp.join(DATASET_DIR, "images"))
	for im_fn in images:
		im_fn = osp.join(DATASET_DIR, "images", im_fn)
		im = np.load(im_fn)
		np.savez_compressed(im_fn, im)
		im2 = np.load(im_fn)
		print(np.linalg.norm(im - im2))

	targets = os.listdir(osp.join(DATASET_DIR, "targets"))
	for targ_fn in targets:
		targ_fn = osp.join(DATASET_DIR, "targets", targ_fn)
		targ = np.load(targ_fn)
		np.savez_compressed(targ_fn, targ)
		targ2 = np.load(targ_fn)
		print(np.linalg.norm(targ - targ2))
