import random
import socket
from os import path as osp

import numpy as np
import torch


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_base_run_dir() -> str:
    socket.gethostname()
    # Get the ZSC-Eval directory path
    current_file = osp.abspath(__file__)
    # Navigate from zsceval/utils/train_util.py to ZSC-Eval/
    zsceval_root = osp.dirname(osp.dirname(current_file))
    project_root = osp.dirname(zsceval_root)
    base = osp.join(project_root, "results")
    return base
