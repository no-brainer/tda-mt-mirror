from itertools import repeat
import json
import random

import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader


def parse_config(config_path):
    with open(config_path, "r") as in_file:
        config = json.load(in_file)
    return config


def init_obj(obj_dict, module, *args, **kwargs):
    obj_name = obj_dict["type"]
    obj_args = dict(obj_dict["args"])
    if len(set(obj_args.keys()) & set(kwargs.keys())) > 0:
        raise RuntimeError("Overwriting arguments from config file is not allowed")
    obj_args.update(kwargs)
    return getattr(module, obj_name)(*args, **obj_args)
