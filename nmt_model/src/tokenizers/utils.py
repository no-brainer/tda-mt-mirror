import os
from typing import Dict

import youtokentome as yttm


def dump_lines(dest, src_path):
    with open(src_path, "r") as f:
        for line in f:
            dest.write(line)


def prepare_tokenizer(save_dirpath: str, model_args: Dict) -> yttm.BPE:
    tokenizer_file = os.path.join(save_dirpath, "tokenizer.model")

    if os.path.exists(save_dirpath):
        return yttm.BPE(model=tokenizer_file)

    has_multiple_files = isinstance(model_args["data"], list)
    tmp_path = None
    if has_multiple_files:
        tmp_path = "./tmp_data"
        with open(tmp_path, "w") as tmp_file:
            for file_path in model_args["data"]:
                dump_lines(tmp_file, file_path)
        model_args["data"] = tmp_path

    os.makedirs(save_dirpath, exist_ok=True)
    tokenizer = yttm.BPE.train(model=tokenizer_file, **model_args)

    if has_multiple_files:
        os.remove(tmp_path)

    return tokenizer
