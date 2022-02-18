import os
import os.path as osp
from PIL import Image
import torch

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
            pred = model(im)[0, 0].cpu().numpy()
        im = Image.fromarray(pred).convert("L")
        im.save(osp.join(logdir, "vis", "pred_%d.jpg" % idx))


if __name__ == "__main__":
    main()
