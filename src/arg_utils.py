from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--exper-name', default=None)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--n-gpus', type=int, default=1)
    parser.add_argument('--loader-n-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='real')
    parser.add_argument('--dataset-val', type=str, default='real_val')
    parser.add_argument('--checkpoint', type=str, default=None)

    optim_group = parser.add_argument_group("optim")
    optim_group.add_argument("--optim-type", default='sgd', type=str)
    optim_group.add_argument("--optim-learning-rate", default=0.002, type=float)
    optim_group.add_argument("--optim-momentum", default=0.9, type=float)
    optim_group.add_argument("--optim-weight-decay", default=1e-4, type=float)
    optim_group.add_argument("--optim-poly-exp", default=0.9, type=float)
    optim_group.add_argument("--optim-warmup-epochs", default=None, type=int)
    # parser.add_argument("--model-file", type=str, required=True)
    # parser.add_argument("--model-name", type=str, required=True)
    # parser.add_argument("--checkpoint", default=None, type=str)
    # parser.add_argument("--wandb-name", type=str, required=True)
    # Ignore Mask Search.
    # parser.add_argument("--min-height", default=0.0, type=float)
    # parser.add_argument("--min-occlusion", default=0.0, type=float)
    # parser.add_argument("--min-truncation", default=0.0, type=float)
    # # Backbone configs
    # parser.add_argument("--model-norm", default='BN', type=str)
    # parser.add_argument("--num-filters-scale", default=4, type=int)

    args = parser.parse_args()
    return vars(args)
