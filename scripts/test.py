import os
import os.path as osp
from PIL import Image
import torch
import numpy as np


import fcvision.utils.run_utils as ru
from fcvision.utils.arg_utils import parse_yaml


def main():
    cfg, params = parse_yaml(osp.join("cfg", "apps", "test_config.yaml"))

    logdir = ru.get_file_prefix(params)
    os.makedirs(os.path.join(logdir, "vis"))

    model = params["model"]
    model.logdir = logdir

    dataset_val = params["dataset_val"]
    for idx in range(len(dataset_val)):
        im = torch.unsqueeze(dataset_val[idx], 0).cuda()
        with torch.no_grad():
            pred = torch.sigmoid(model(im))[0, 0].cpu().numpy()
        im = np.transpose(im.cpu().numpy()[0], (1, 2, 0))
        import IPython; IPython.embed(); assert 0
        overlayed = np.copy(im)
        overlayed[pred > 0.25] = [1, 0, 0]
        im = Image.fromarray((im*255).astype(np.uint8))
        im.save(osp.join(logdir, "vis", "%d_im.jpg" % idx))
        pred = Image.fromarray((pred*255).astype(np.uint8)).convert("L")
        pred.save(osp.join(logdir, "vis", "%d_pred.jpg" % idx))
        overlayed = Image.fromarray((overlayed*255).astype(np.uint8))
        overlayed.save(osp.join(logdir, "vis", "%d_overlayed.jpg" % idx))


if __name__ == "__main__":
    main()
