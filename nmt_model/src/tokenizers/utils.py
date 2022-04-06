import os
from typing import Dict

import tokenizers
from tokenizers.pre_tokenizers import Whitespace


def train_new_model(model_name, model_args, trainer_args) -> tokenizers.Tokenizer:
    tokenizer = tokenizers.Tokenizer(
        getattr(tokenizers.models, model_name)(**model_args)
    )

    trainer = tokenizer.model.get_trainer()
    for arg_name, value in trainer_args.items():
        if arg_name == "train_files":
            continue
        setattr(trainer, arg_name, value)

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(trainer_args["train_files"], trainer)

    return tokenizer


def prepare_tokenizer(save_dirpath: str, model_name: str, model_args: Dict, trainer_args: Dict) -> tokenizers.Tokenizer:
    tokenizer_file = os.path.join(save_dirpath, "tokenizer.json")

    if os.path.exists(save_dirpath):
        return tokenizers.Tokenizer.from_file(tokenizer_file)

    tokenizer = train_new_model(model_name, model_args, trainer_args)
    os.makedirs(save_dirpath, exist_ok=True)
    tokenizer.save(tokenizer_file)

    return tokenizer
