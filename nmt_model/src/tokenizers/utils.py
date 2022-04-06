import os
from typing import Dict

import tokenizers
from tokenizers.pre_tokenizers import Whitespace


def load_tokenizer_from_disk(model_name, save_dirpath) -> tokenizers.Tokenizer:
    saved_files = list(
        map(lambda filename: os.path.join(save_dirpath, filename), os.listdir(save_dirpath))
    )

    if model_name == "BPE":
        assert len(saved_files) == 2, f"Incorrect number of files in directory {save_dirpath}"

        vocab_filepath, merge_filepath = saved_files
        if not vocab_filepath.endswith(".json"):
            vocab_filepath, merge_filepath = merge_filepath, vocab_filepath

        saved_data = dict(
            vocab=vocab_filepath,
            merge=merge_filepath,
        )

    else:
        assert len(saved_files) == 1, f"Incorrect number of files in directory {save_dirpath}"

        vocab_filepath = saved_files[0]
        saved_data = dict(vocab=vocab_filepath)

    return tokenizers.Tokenizer(
        getattr(tokenizers.models, model_name)(**saved_data)
    )


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
    if os.path.exists(save_dirpath):
        return load_tokenizer_from_disk(model_name, save_dirpath)

    tokenizer = train_new_model(model_name, model_args, trainer_args)
    os.makedirs(save_dirpath, exist_ok=True)
    tokenizer.save(save_dirpath)

    return tokenizer
