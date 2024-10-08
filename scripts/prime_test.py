import os
import os.path as osp
from PIL import Image
import torch
import numpy as np
import fcvision.utils.pytorch_utils as ptu
import os
import sys
import cv2
# Get the current working directory
current_directory = os.getcwd()

# Walk through the directory and its subdirectories
for root, dirs, files in os.walk(current_directory):
    for file in files:
        # Check if the file is a Python file and exclude special files (like __init__.py)
        if file.endswith('.py') and not file.startswith('__'):
            # Extract the module name without the extension
            module_name = os.path.splitext(file)[0]
            # Add the module to the list of importable modules
            sys.path.append(root)
            # print(module_name)


import fcvision.utils.run_utils as ru
from fcvision.utils.arg_utils import parse_yaml


def main():
    cfg, params = parse_yaml(osp.join("cfg", "apps", "prime_test_config.yaml"))
    images = str(osp.join(cfg['dataset_dir'],'images'))
    targets = str(osp.join(cfg['dataset_dir'],'targets'))
    logdir = ru.get_file_prefix(params)
    os.makedirs(os.path.join(logdir, "vis"))
    model = params["model"]
    model.logdir = logdir

    image_files = sorted(os.listdir(images))
    target_files = sorted(os.listdir(targets))
    idx = 1
    for img_file, target_file in zip(image_files, target_files):
        # Construct full file paths
        img_path = os.path.join(images, img_file)
        target_path = os.path.join(targets, target_file)

        im = cv2.imread(img_path)
        print(f"IM FILE: {img_path}")
        im =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        target = cv2.imread(target_path)[...,0]
        original_height,original_width,_ = im.shape
        im=cv2.resize(im,(960,600))
        target=cv2.resize(target,(960,600))
        
        im = np.transpose(im, (2, 0, 1))
        if im.max() > 1.0:
            im = im / 255.0
        if target.max() > 1.0:
            target = target / 255.0
        if len(target.shape) == 2:
            target = target[np.newaxis, :, :]
        else:
            target = np.transpose(target, (2, 0, 1))
        im = torch.unsqueeze(ptu.torchify(im),0).cuda()
        with torch.no_grad():
            pred = torch.sigmoid(model(im))[0, 0].cpu().numpy()
        pred = (pred*255).astype(np.uint8)
        target = (target*255)[0].astype(np.uint8)
        im = np.transpose(im.cpu().numpy()[0], (1, 2, 0))
        pred = cv2.resize(pred,(original_width,original_height))
        target = cv2.resize(target,(original_width,original_height))
        im = cv2.resize(im,(original_width,original_height))
        overlayed = np.copy(im)
        overlayed[pred > 0.25 * 255] = [0, 0, 1]
        im = Image.fromarray((im*255).astype(np.uint8))
        im.save(osp.join(logdir, "vis", "%d_im.png" % idx))
        pred = Image.fromarray(pred).convert("L")
        pred.save(osp.join(logdir, "vis", "%d_pred.png" % idx))
        overlayed = Image.fromarray((overlayed*255).astype(np.uint8))
        overlayed.save(osp.join(logdir, "vis", "%d_overlayed.png" % idx))
        gt = Image.fromarray(target)
        gt.save(osp.join(logdir, "vis", "%d_gt.png" % idx))
        import pdb
        pdb.set_trace()
        idx += 1


if __name__ == "__main__":
    main()
