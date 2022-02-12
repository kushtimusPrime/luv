from argparse import ArgumentParser

from fcvision.tasks import get_task_parameters

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--exper-name', default=None)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--n-gpus', type=int, default=1)
    parser.add_argument('--loader-n-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size',type=int,default=10)
    # parser.add_argument('--dataset', type=str, default='real')
    # parser.add_argument('--dataset-val', type=str, default='real_val')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--dataset_dir',type=str,default=None)
    parser.add_argument('--num_classes',type=int,default=None)
    parser.add_argument('--max_size',type=int,default=1280)

    optim_group = parser.add_argument_group("optim")
    optim_group.add_argument("--optim-learning-rate", default=1e-3, type=float)
    optim_group.add_argument("--optim-weight-decay", default=0, type=float)
    # parser.add_argument("--model-file", type=str, required=True)
    # parser.add_argument("--model-name", type=str, required=True)
    # parser.add_argument("--checkpoint", default=None, type=str)
    # parser.add_argument("--wandb-name", type=str, required=True)

    args = parser.parse_args()
    args = vars(args)
    args = get_task_parameters(args)

    return args
