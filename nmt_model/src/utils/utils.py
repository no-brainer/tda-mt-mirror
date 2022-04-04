from itertools import repeat
import json
import random
from typing import Dict

import numpy as np
import torch

import src.collators
import src.datasets
import src.samplers


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


def init_obj(lookup_modules, obj_dict, *args, **kwargs):
    if not isinstance(lookup_modules, list):
        lookup_modules = [lookup_modules]

    obj_name = obj_dict["type"]
    obj_args = dict(obj_dict["args"])
    if len(set(obj_args.keys()) & set(kwargs.keys())) > 0:
        raise RuntimeError("Overwriting arguments from config file is not allowed")
    obj_args.update(kwargs)

    obj = None
    for module in lookup_modules:
        if hasattr(module, obj_name):
            obj = getattr(module, obj_name)(*args, **obj_args)
            break

    return obj


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

        dataloader_params = split_params["dataloader"]
        if "collate_fn" in dataloader_params:
            dataloader_params["collate_fn"] = init_obj(src.collators, dataloader_params["collate_fn"])
        if "batch_sampler" in dataloader_params:
            dataloader_params["batch_sampler"] = init_obj(src.samplers, dataloader_params["batch_sampler"])

        dataloaders[split] = torch.utils.data.DataLoader(dataset, **dataloader_params)

    return dataloaders
