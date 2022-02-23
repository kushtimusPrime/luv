import numpy as np
import os
import os.path as osp
from pqdm.processes import pqdm


DATASET_DIR = "data/towel_seg"

def compress_image(fn):
	im = np.load(fn)
	np.savez_compressed(fn.replace(".npy", ".npz"), im)

if __name__ == '__main__':
	images = [osp.join(DATASET_DIR, "images", f) for f in os.listdir(osp.join(DATASET_DIR, "images"))]
	targets = [osp.join(DATASET_DIR, "targets", f) for f in os.listdir(osp.join(DATASET_DIR, "targets"))]
	pqdm(images + targets, compress_image, n_jobs=12)
