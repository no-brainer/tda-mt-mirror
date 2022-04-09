import os
from typing import Dict

import youtokentome as yttm


def prepare_tokenizer(save_dirpath: str, model_args: Dict) -> yttm.BPE:
    tokenizer_file = os.path.join(save_dirpath, "tokenizer.model")

    if os.path.exists(save_dirpath):
        return yttm.BPE(model=tokenizer_file)

    os.makedirs(save_dirpath, exist_ok=True)
    tokenizer = yttm.BPE.train(model=tokenizer_file, **model_args)

    return tokenizer
