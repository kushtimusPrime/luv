import os
import torch
import numpy as np
import random
from datetime import datetime
from collections.abc import Iterable
import logging

log = logging.getLogger("utils")


def seed(s, envs=None):
    if s == -1:
        return

    # torch.set_deterministic(True)
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)

    if envs is not None:
        if isinstance(envs, Iterable):
            for env in envs:
                env.seed(s)
                env.action_space.seed(s)
        else:
            envs.seed(s)
            envs.action_space.seed(s)


def get_file_prefix(params):
    if params['exper_name'] is not None:
        folder = os.path.join('outputs', params['exper_name'])
    else:
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d/%H-%M-%S")
        folder = os.path.join('outputs', date_string)
    if params['seed'] != -1:
        folder = os.path.join(folder, str(params['seed']))
    return folder


def init_logging(folder, file_level=logging.INFO, console_level=logging.DEBUG):
    # set up logging to file
    logging.basicConfig(level=file_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=os.path.join(folder, 'log.txt'),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(console_level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    color2num = dict(
        gray=30,
        red=31,
        green=32,
        yellow=33,
        blue=34,
        magenta=35,
        cyan=36,
        white=37,
        crimson=38
    )

    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)
