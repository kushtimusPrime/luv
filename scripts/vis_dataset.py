import os.path as osp
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from fcvision.utils.arg_utils import parse_yaml

cfg, ret = parse_yaml(osp.join("cfg", "datasets", "cable_seg.yaml"))
dataset = ret["dataset"]

im, target = dataset[0]
print(im)
plt.imshow(np.transpose(im, (1, 2, 0)))
plt.show()
plt.imshow(target[0])
plt.show()


# idx = 0
# for f in os.listdir(osp.join("data", "cable_red_painted_images")):
# 	if not ("image" in f and "uv" not in f):
# 		continue
# 	print(idx)
# 	mask_fname = f.replace("image", "mask")
# 	im = np.array(Image.open(osp.join("data/cable_red_painted_images", f))) / 255.
# 	mask = np.array(Image.open(osp.join("data/cable_red_painted_images", mask_fname))) / 255.

# 	np.save(osp.join("data/cable_seg", "images", "image_%d.npy"%idx), im)
# 	np.save(osp.join("data/cable_seg", "targets", "target_%d.npy"%idx), mask)

# 	idx += 1
