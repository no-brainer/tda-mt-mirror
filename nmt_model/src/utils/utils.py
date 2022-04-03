from itertools import repeat
import json
import random
from typing import Dict

import numpy as np
import torch

import src.datasets


def set_seed(seed: int):
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


def init_obj(module, obj_dict, *args, **kwargs):
    obj_name = obj_dict["type"]
    obj_args = dict(obj_dict["args"])
    if len(set(obj_args.keys()) & set(kwargs.keys())) > 0:
        raise RuntimeError("Overwriting arguments from config file is not allowed")
    obj_args.update(kwargs)
    return getattr(module, obj_name)(*args, **obj_args)


def prepare_dataloaders(data_params: Dict) -> Dict[str, torch.utils.data.DataLoader]:
    dataloaders = dict()
    for split, split_params in data_params.items():
        datasets = [
            init_obj(src.datasets, dataset_params) for dataset_params in split_params["datasets"]
        ]

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = torch.utils.data.ConcatDataset(datasets)

        dataloaders[split] = torch.utils.data.DataLoader(dataset, **split_params["dataloader"])

    return dataloaders
