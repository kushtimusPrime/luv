import torch
import numpy as np


TORCH_DEVICE = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
# TORCH_DEVICE = torch.device('cpu')
#
#
# def setup(idx=0):
#     global TORCH_DEVICE
#     if idx < 0:
#         TORCH_DEVICE = torch.device('cpu')
#     else:
#         TORCH_DEVICE = torch.device('cuda', idx) if torch.cuda.is_available() else torch.device('cpu')
#     print('Using', TORCH_DEVICE)


def torchify(*args, cls=torch.FloatTensor, device=None):
    out = []
    for x in args:
        if type(x) is not torch.Tensor and type(x) is not np.ndarray:
            x = np.array(x)
        if type(x) is not torch.Tensor:
            x = cls(x)
        if device is not None:
            x = x.to(device)
        out.append(x)
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)


def torchify_fast(*args, cls=torch.FloatTensor, device=None):
    x = np.array(args)
    x = cls(x).to(device)
    return x


def to_numpy(x):
    if x is None:
        return x
    return x.detach().cpu().numpy()


def process_checkpoint(ckpt):
    """
    Required because models trained using DataParallel will have "module" in the checkpoint.
    """
    # original saved file with DataParallel
    # old_dict = torch.load('models/towelNet.ckpt')
    old_dict = torch.load(ckpt)
    state_dict = old_dict['state_dict']
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        print(name, k)
        new_state_dict[name] = v
    old_dict['state_dict'] = new_state_dict
    return old_dict

