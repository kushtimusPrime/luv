import os
import os.path as osp
import imageio
import numpy as np
import pathlib

dataset_directories = [
    "data/cable_red_painted_images",
    "data/cable_blue_painted_images",
    "data/iphone_red_painted_images",
]

OUTPUT_DIR = "data/cable_seg"

pathlib.Path(f"{OUTPUT_DIR}/{images}").mkdir(parents=True, exist_ok=True)
pathlib.Path(f"{OUTPUT_DIR}/{target}").mkdir(parents=True, exist_ok=True)


images = []
targets = []

for data_dir in dataset_directories:
    cur_images = [
        osp.join(data_dir, f)
        for f in os.listdir(data_dir)
        if "image" in f and "uv" not in f
    ]
    images += cur_images
    if data_dir != "data/cable_red_painted_images":
        targets += [
            f.replace("imagel", "maskl").replace("imager", "maskr") for f in cur_images
        ]
    else:
        targets += [
            f.replace("imagel", "maskl_full").replace("imager", "maskr_full")
            for f in cur_images
        ]

for i, (imfile, targfile) in enumerate(zip(images, targets)):
    print(i)
    im = imageio.imread(imfile)
    targ = imageio.imread(targfile)
    np.save(osp.join(OUTPUT_DIR, "images", "image_%d.npy" % i), im)
    np.save(osp.join(OUTPUT_DIR, "targets", "target_%d.npy" % i), targ)
